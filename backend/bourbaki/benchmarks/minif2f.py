"""miniF2F benchmark runner for Bourbaki.

Runs the agent on miniF2F problems and reports pass rates, enabling
comparison with SOTA systems (ReProver 26.5%, DeepSeek 60.2%,
Goedel-V2 90.4%, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bourbaki.benchmarks.loader import (
    MiniF2FProblem,
    get_problem_stats,
    load_minif2f_problems,
)
from bourbaki.tools.lean_prover import lean_prover

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / ".bourbaki" / "benchmarks" / "results"


@dataclass
class ProblemResult:
    """Result of attempting a single miniF2F problem."""
    problem_id: str
    source: str
    solved: bool
    proof_code: str | None = None
    error: str | None = None
    tactics_used: int = 0
    duration_seconds: float = 0.0
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "solved": self.solved,
            "proof_code": self.proof_code,
            "error": self.error,
            "tactics_used": self.tactics_used,
            "duration_seconds": round(self.duration_seconds, 2),
            "attempts": self.attempts,
        }


@dataclass
class BenchmarkResult:
    """Aggregate result of a benchmark run."""
    total: int = 0
    solved: int = 0
    pass_rate: float = 0.0
    results: list[ProblemResult] = field(default_factory=list)
    total_time_seconds: float = 0.0
    avg_duration_per_problem: float = 0.0
    avg_tactics_per_solve: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "solved": self.solved,
            "pass_rate": round(self.pass_rate, 4),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_duration_per_problem": round(self.avg_duration_per_problem, 2),
            "avg_tactics_per_solve": round(self.avg_tactics_per_solve, 2),
            "config": self.config,
            "timestamp": self.timestamp,
            "by_source": self.by_source,
            "results": [r.to_dict() for r in self.results],
        }


async def attempt_proof(
    problem: MiniF2FProblem,
    prove_fn: ProveFunction | None = None,
    timeout: int = 300,
) -> ProblemResult:
    """Attempt to prove a single miniF2F problem.

    Args:
        problem: The miniF2F problem to attempt.
        prove_fn: Custom proving function. If None, uses lean_prover directly
                  with common automation tactics.
        timeout: Timeout in seconds per problem.

    Returns:
        ProblemResult with success/failure details.
    """
    start = time.monotonic()

    if prove_fn is not None:
        # Use custom proving function (e.g., search tree, agent-based)
        try:
            result = await asyncio.wait_for(
                prove_fn(problem),
                timeout=timeout,
            )
            result.duration_seconds = time.monotonic() - start
            return result
        except asyncio.TimeoutError:
            return ProblemResult(
                problem_id=problem.id,
                source=problem.source,
                solved=False,
                error="Timeout",
                duration_seconds=time.monotonic() - start,
            )
        except Exception as e:
            return ProblemResult(
                problem_id=problem.id,
                source=problem.source,
                solved=False,
                error=str(e),
                duration_seconds=time.monotonic() - start,
            )

    # Default: try common automation tactics
    automation_tactics = [
        "simp",
        "omega",
        "ring",
        "norm_num",
        "decide",
        "linarith",
        "nlinarith",
        "aesop",
        "simp_all",
        "norm_num [Nat.Prime]",
    ]

    attempts = 0
    for tactic in automation_tactics:
        attempts += 1
        # Build proof attempt: replace sorry with the tactic
        proof_code = problem.full_lean_code.replace("sorry", tactic)

        try:
            result = await asyncio.wait_for(
                lean_prover(code=proof_code, mode="check", timeout=60),
                timeout=timeout,
            )
            if result.get("proofComplete"):
                return ProblemResult(
                    problem_id=problem.id,
                    source=problem.source,
                    solved=True,
                    proof_code=proof_code,
                    tactics_used=1,
                    attempts=attempts,
                    duration_seconds=time.monotonic() - start,
                )
        except (asyncio.TimeoutError, Exception):
            continue

    return ProblemResult(
        problem_id=problem.id,
        source=problem.source,
        solved=False,
        error="All automation tactics failed",
        attempts=attempts,
        duration_seconds=time.monotonic() - start,
    )


# Type for custom proving functions
ProveFunction = Any  # Callable[[MiniF2FProblem], Awaitable[ProblemResult]]


async def run_minif2f(
    split: str = "valid",
    source_filter: str | None = None,
    problem_ids: list[str] | None = None,
    prove_fn: ProveFunction | None = None,
    timeout: int = 300,
    concurrency: int = 1,
    minif2f_dir: Path | None = None,
) -> BenchmarkResult:
    """Run Bourbaki on miniF2F problems and report pass rate.

    Args:
        split: "valid", "test", or "all".
        source_filter: Filter by source (e.g., "aime", "imo").
        problem_ids: Specific problem IDs to run.
        prove_fn: Custom proving function. If None, tries automation tactics.
        timeout: Timeout per problem in seconds.
        concurrency: Number of parallel proof attempts.
        minif2f_dir: Path to miniF2F-lean4 checkout.

    Returns:
        BenchmarkResult with aggregate and per-problem results.
    """
    problems = load_minif2f_problems(
        split=split,
        source_filter=source_filter,
        problem_ids=problem_ids,
        minif2f_dir=minif2f_dir,
    )

    if not problems:
        return BenchmarkResult(
            config={"split": split, "source_filter": source_filter},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    stats = get_problem_stats(problems)
    logger.info("Running miniF2F benchmark: %d problems", stats["total"])

    overall_start = time.monotonic()
    results: list[ProblemResult] = []

    if concurrency <= 1:
        # Sequential execution
        for i, problem in enumerate(problems):
            logger.info("[%d/%d] %s", i + 1, len(problems), problem.id)
            result = await attempt_proof(problem, prove_fn, timeout)
            results.append(result)
            status = "SOLVED" if result.solved else "FAILED"
            logger.info("  %s (%.1fs)", status, result.duration_seconds)
    else:
        # Parallel execution with semaphore
        sem = asyncio.Semaphore(concurrency)

        async def bounded_attempt(p: MiniF2FProblem) -> ProblemResult:
            async with sem:
                return await attempt_proof(p, prove_fn, timeout)

        results = await asyncio.gather(
            *(bounded_attempt(p) for p in problems)
        )
        results = list(results)

    # Compute aggregate stats
    total_time = time.monotonic() - overall_start
    solved = [r for r in results if r.solved]
    solved_count = len(solved)

    by_source: dict[str, dict[str, int]] = {}
    for r in results:
        if r.source not in by_source:
            by_source[r.source] = {"total": 0, "solved": 0}
        by_source[r.source]["total"] += 1
        if r.solved:
            by_source[r.source]["solved"] += 1

    benchmark = BenchmarkResult(
        total=len(results),
        solved=solved_count,
        pass_rate=solved_count / len(results) if results else 0.0,
        results=results,
        total_time_seconds=total_time,
        avg_duration_per_problem=total_time / len(results) if results else 0.0,
        avg_tactics_per_solve=(
            sum(r.tactics_used for r in solved) / solved_count
            if solved_count else 0.0
        ),
        config={
            "split": split,
            "source_filter": source_filter,
            "timeout": timeout,
            "concurrency": concurrency,
            "prove_fn": prove_fn.__name__ if prove_fn and hasattr(prove_fn, "__name__") else "default",
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
        by_source=by_source,
    )

    # Save results
    _save_results(benchmark, split)

    # Log summary
    logger.info(
        "miniF2F %s: %d/%d solved (%.1f%%) in %.1fs",
        split, solved_count, len(results),
        benchmark.pass_rate * 100, total_time,
    )
    for source, counts in sorted(by_source.items()):
        pct = counts["solved"] / counts["total"] * 100 if counts["total"] else 0
        logger.info("  %s: %d/%d (%.0f%%)", source, counts["solved"], counts["total"], pct)

    return benchmark


def _save_results(benchmark: BenchmarkResult, split: str) -> None:
    """Save benchmark results to a JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{date_str}_minif2f_{split}.json"
    path = RESULTS_DIR / filename
    path.write_text(
        json.dumps(benchmark.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", path)
