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
from bourbaki.autonomous.search_tree import prove_with_search
from bourbaki.tools.lean_prover import lean_prover
from bourbaki.tools.lean_repl import LeanREPLSession, stop_session

logger = logging.getLogger(__name__)

# Default automation tactics (ordered by likelihood of solving miniF2F problems)
AUTOMATION_TACTICS = [
    "norm_num",
    "omega",
    "ring",
    "simp",
    "linarith",
    "decide",
]

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

    # Default: try common automation tactics via lean_prover (slow path,
    # ~90s per attempt due to Mathlib import; prefer attempt_proof_repl)
    attempts = 0
    for tactic in AUTOMATION_TACTICS:
        attempts += 1
        # Build proof attempt: replace sorry with the tactic
        proof_code = problem.full_lean_code.replace("sorry", tactic)

        try:
            result = await asyncio.wait_for(
                lean_prover(code=proof_code, mode="check", timeout=180),
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


async def attempt_proof_repl(
    problem: MiniF2FProblem,
    session: LeanREPLSession,
    tactics: list[str] | None = None,
    timeout: int = 60,
) -> ProblemResult:
    """Attempt to prove a problem using the REPL (fast path).

    Uses a pre-initialized REPL session with Mathlib loaded, so each tactic
    attempt takes ~0.01-0.1s instead of ~90s.

    Args:
        problem: The miniF2F problem to attempt.
        session: Pre-initialized LeanREPLSession with Mathlib loaded.
        tactics: List of tactics to try. Defaults to AUTOMATION_TACTICS.
        timeout: Timeout in seconds for the entire problem.

    Returns:
        ProblemResult with success/failure details.
    """
    start = time.monotonic()
    tactics = tactics or AUTOMATION_TACTICS

    # Build the problem command: preamble (without imports) + theorem + sorry
    # The REPL session already has Mathlib imported, so we skip import lines
    preamble_lines = []
    for line in problem.full_lean_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import"):
            continue  # Already imported in session
        if stripped.startswith("set_option") or stripped.startswith("open"):
            preamble_lines.append(stripped)
        elif stripped.startswith("theorem") or stripped.startswith("lemma"):
            break

    preamble = "\n".join(preamble_lines) if preamble_lines else ""
    cmd = f"{preamble}\n{problem.statement} := by sorry" if preamble else f"{problem.statement} := by sorry"

    try:
        # Send theorem to get proof state (uses base Mathlib env)
        result = await asyncio.wait_for(
            session.send_cmd(cmd, env=session.env_id),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return ProblemResult(
            problem_id=problem.id,
            source=problem.source,
            solved=False,
            error="Timeout setting up proof state",
            duration_seconds=time.monotonic() - start,
        )

    sorries = result.get("sorries", [])
    if not sorries:
        # Check for errors in the response
        messages = result.get("messages", [])
        error_msgs = [m.get("data", "") for m in messages if m.get("severity") == "error"]
        return ProblemResult(
            problem_id=problem.id,
            source=problem.source,
            solved=False,
            error=f"No proof state: {'; '.join(error_msgs) if error_msgs else 'unknown'}",
            duration_seconds=time.monotonic() - start,
        )

    ps = sorries[0]
    ps_id = ps.get("proofState", 0) if isinstance(ps, dict) else 0

    # Try each tactic against the proof state
    attempts = 0
    for tactic in tactics:
        attempts += 1
        try:
            tactic_result = await asyncio.wait_for(
                session.send_tactic(tactic, ps_id),
                timeout=30,
            )
        except asyncio.TimeoutError:
            continue

        goals = tactic_result.get("goals", [])
        proof_complete = (
            tactic_result.get("proofStatus") == "Completed"
            or (isinstance(goals, list) and len(goals) == 0
                and "error" not in tactic_result
                and "message" not in tactic_result)
        )

        if proof_complete:
            return ProblemResult(
                problem_id=problem.id,
                source=problem.source,
                solved=True,
                proof_code=f"{problem.statement} := by {tactic}",
                tactics_used=1,
                attempts=attempts,
                duration_seconds=time.monotonic() - start,
            )

    return ProblemResult(
        problem_id=problem.id,
        source=problem.source,
        solved=False,
        error="All automation tactics failed",
        attempts=attempts,
        duration_seconds=time.monotonic() - start,
    )


async def attempt_proof_search(
    problem: MiniF2FProblem,
    session: LeanREPLSession,
    budget: int = 64,
    timeout: int = 60,
    use_mathlib: bool = False,
) -> ProblemResult:
    """Attempt to prove a problem using best-first search tree.

    First tries automation tactics via REPL (fast). If those fail, runs a
    best-first search over tactic sequences.

    Args:
        problem: The miniF2F problem to attempt.
        session: Pre-initialized LeanREPLSession with Mathlib loaded.
        budget: Max tactic attempts for the search tree.
        timeout: Timeout in seconds for the entire problem.
        use_mathlib: Whether to query Mathlib search during tree expansion.

    Returns:
        ProblemResult with success/failure details.
    """
    start = time.monotonic()

    # Phase 1: Try automation tactics first (instant)
    auto_result = await attempt_proof_repl(
        problem, session, timeout=min(timeout, 30),
    )
    if auto_result.solved:
        return auto_result

    # Phase 2: Best-first search tree
    # Build the theorem statement for the search tree
    preamble_lines = []
    for line in problem.full_lean_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import"):
            continue
        if stripped.startswith("set_option") or stripped.startswith("open"):
            preamble_lines.append(stripped)
        elif stripped.startswith("theorem") or stripped.startswith("lemma"):
            break

    preamble = "\n".join(preamble_lines) if preamble_lines else ""
    theorem = f"{preamble}\n{problem.statement}" if preamble else problem.statement

    remaining_time = timeout - (time.monotonic() - start)
    if remaining_time <= 5:
        return auto_result  # Not enough time for search

    try:
        search_result = await asyncio.wait_for(
            prove_with_search(
                theorem=theorem,
                budget=budget,
                timeout=remaining_time,
                max_depth=15,
                use_mathlib=use_mathlib,
                session=session,
            ),
            timeout=remaining_time + 5,
        )

        if search_result.success:
            return ProblemResult(
                problem_id=problem.id,
                source=problem.source,
                solved=True,
                proof_code=search_result.proof_code,
                tactics_used=len(search_result.proof_tactics),
                attempts=search_result.nodes_explored,
                duration_seconds=time.monotonic() - start,
            )
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("Search tree failed for %s: %s", problem.id, e)

    return ProblemResult(
        problem_id=problem.id,
        source=problem.source,
        solved=False,
        error=f"Automation + search failed ({auto_result.attempts} auto + search)",
        attempts=auto_result.attempts,
        duration_seconds=time.monotonic() - start,
    )


async def run_minif2f(
    split: str = "valid",
    source_filter: str | None = None,
    problem_ids: list[str] | None = None,
    prove_fn: ProveFunction | None = None,
    timeout: int = 300,
    concurrency: int = 1,
    minif2f_dir: Path | None = None,
    use_repl: bool = True,
    use_search: bool = False,
    search_budget: int = 64,
    use_mathlib_search: bool = False,
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
        use_repl: Use REPL for automation tactics (default True, ~40x faster).
                  Falls back to lean_prover if REPL not available.
        use_search: Use best-first search tree after automation fails.
        search_budget: Max tactic attempts per problem for search tree.
        use_mathlib_search: Query Mathlib APIs during search tree expansion.

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

    # Set up REPL session if requested (one-time Mathlib import)
    repl_session: LeanREPLSession | None = None
    if use_repl and prove_fn is None:
        try:
            repl_session = LeanREPLSession(import_full_mathlib=True)
            await repl_session.start()
            logger.info("Initializing REPL with full Mathlib import...")
            await repl_session.ensure_initialized()
            if not repl_session._initialized:
                logger.warning("REPL init failed, falling back to lean_prover")
                repl_session = None
        except Exception as e:
            logger.warning("REPL not available (%s), using lean_prover", e)
            repl_session = None

    try:
        if concurrency <= 1:
            # Sequential execution
            for i, problem in enumerate(problems):
                logger.info("[%d/%d] %s", i + 1, len(problems), problem.id)
                if repl_session is not None and use_search:
                    result = await attempt_proof_search(
                        problem, repl_session,
                        budget=search_budget,
                        timeout=timeout,
                        use_mathlib=use_mathlib_search,
                    )
                elif repl_session is not None:
                    result = await attempt_proof_repl(
                        problem, repl_session, timeout=timeout,
                    )
                else:
                    result = await attempt_proof(problem, prove_fn, timeout)
                results.append(result)
                status = "SOLVED" if result.solved else "FAILED"
                logger.info("  %s (%.2fs)", status, result.duration_seconds)
        else:
            # Parallel execution with semaphore
            # Note: REPL is sequential (single process), so parallel only
            # works with lean_prover or custom prove_fn
            if repl_session is not None:
                logger.warning(
                    "REPL is sequential; ignoring concurrency=%d", concurrency,
                )
                for i, problem in enumerate(problems):
                    logger.info("[%d/%d] %s", i + 1, len(problems), problem.id)
                    if use_search:
                        result = await attempt_proof_search(
                            problem, repl_session,
                            budget=search_budget,
                            timeout=timeout,
                            use_mathlib=use_mathlib_search,
                        )
                    else:
                        result = await attempt_proof_repl(
                            problem, repl_session, timeout=timeout,
                        )
                    results.append(result)
                    status = "SOLVED" if result.solved else "FAILED"
                    logger.info("  %s (%.2fs)", status, result.duration_seconds)
            else:
                sem = asyncio.Semaphore(concurrency)

                async def bounded_attempt(p: MiniF2FProblem) -> ProblemResult:
                    async with sem:
                        return await attempt_proof(p, prove_fn, timeout)

                results = await asyncio.gather(
                    *(bounded_attempt(p) for p in problems)
                )
                results = list(results)
    finally:
        if repl_session is not None:
            await repl_session.stop()

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
            "use_repl": repl_session is not None,
            "use_search": use_search,
            "search_budget": search_budget if use_search else None,
            "use_mathlib_search": use_mathlib_search if use_search else None,
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
