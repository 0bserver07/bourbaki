"""PutnamBench benchmark runner for Bourbaki.

Runs the agent on PutnamBench problems (672 Putnam competition problems,
1962-2025) and reports pass rates, enabling comparison with SOTA systems
(HILBERT 70.0%, etc.).

Usage:
    # Quick test (5 problems)
    python -m bourbaki.benchmarks.putnam --quick

    # Full run with 60s timeout
    python -m bourbaki.benchmarks.putnam --timeout 60

    # Filter by year range
    python -m bourbaki.benchmarks.putnam --year-from 2020 --year-to 2025

    # Filter by section
    python -m bourbaki.benchmarks.putnam --section a

    # Specific problems
    python -m bourbaki.benchmarks.putnam --problems putnam_2023_a1 putnam_2024_a1
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bourbaki.benchmarks.putnam_loader import (
    PutnamProblem,
    get_putnam_stats,
    load_putnam_problems,
)
from bourbaki.autonomous.search_tree import prove_with_search
from bourbaki.tools.lean_repl import LeanREPLSession

logger = logging.getLogger(__name__)

# Default automation tactics (same as miniF2F, ordered by success likelihood)
AUTOMATION_TACTICS = [
    "norm_num",
    "omega",
    "ring",
    "simp",
    "linarith",
    "decide",
    "nlinarith",
    "positivity",
    "norm_cast",
    "field_simp",
    "polyrith",
    "simp_all",
    "push_cast",
    "ring_nf",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / ".bourbaki" / "benchmarks" / "results"

# Quick-test problems: a mix of years and difficulty levels
QUICK_TEST_IDS = [
    "putnam_1988_a1",
    "putnam_2000_a1",
    "putnam_2010_a1",
    "putnam_2020_a1",
    "putnam_2024_a1",
]


@dataclass
class ProblemResult:
    """Result of attempting a single Putnam problem."""

    problem_id: str
    year: int
    section: str
    solved: bool
    proof_code: str | None = None
    error: str | None = None
    tactics_used: int = 0
    duration_seconds: float = 0.0
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "year": self.year,
            "section": self.section,
            "solved": self.solved,
            "proof_code": self.proof_code,
            "error": self.error,
            "tactics_used": self.tactics_used,
            "duration_seconds": round(self.duration_seconds, 2),
            "attempts": self.attempts,
        }


@dataclass
class PutnamBenchmarkResult:
    """Aggregate result of a PutnamBench run."""

    total: int = 0
    solved: int = 0
    pass_rate: float = 0.0
    results: list[ProblemResult] = field(default_factory=list)
    total_time_seconds: float = 0.0
    avg_duration_per_problem: float = 0.0
    avg_tactics_per_solve: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    by_year: dict[int, dict[str, int]] = field(default_factory=dict)
    by_section: dict[str, dict[str, int]] = field(default_factory=dict)
    by_decade: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "putnam",
            "total": self.total,
            "solved": self.solved,
            "pass_rate": round(self.pass_rate, 4),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_duration_per_problem": round(self.avg_duration_per_problem, 2),
            "avg_tactics_per_solve": round(self.avg_tactics_per_solve, 2),
            "config": self.config,
            "timestamp": self.timestamp,
            "by_year": {str(k): v for k, v in sorted(self.by_year.items())},
            "by_section": self.by_section,
            "by_decade": self.by_decade,
            "results": [r.to_dict() for r in self.results],
        }


async def attempt_putnam_repl(
    problem: PutnamProblem,
    session: LeanREPLSession,
    tactics: list[str] | None = None,
    timeout: int = 60,
) -> ProblemResult:
    """Attempt to prove a Putnam problem using the REPL (fast path).

    Uses a pre-initialized REPL session with Mathlib loaded, so each tactic
    attempt takes ~0.01-0.1s instead of ~90s.
    """
    start = time.monotonic()
    tactics = tactics or AUTOMATION_TACTICS

    # Build the command: preamble (without imports) + setup + theorem + sorry
    # The REPL session already has Mathlib imported, so skip import lines.
    cmd_parts: list[str] = []

    # Add open/set_option directives
    if problem.preamble:
        cmd_parts.append(problem.preamble)

    # Add setup block (abbrev, def, variable, etc.)
    if problem.setup_block:
        cmd_parts.append(problem.setup_block)

    # Add theorem with sorry
    cmd_parts.append(f"{problem.statement} := by sorry")

    cmd = "\n".join(cmd_parts)

    try:
        result = await asyncio.wait_for(
            session.send_cmd(cmd, env=session.env_id),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return ProblemResult(
            problem_id=problem.id,
            year=problem.year,
            section=problem.section,
            solved=False,
            error="Timeout setting up proof state",
            duration_seconds=time.monotonic() - start,
        )

    sorries = result.get("sorries", [])
    if not sorries:
        messages = result.get("messages", [])
        error_msgs = [
            m.get("data", "") for m in messages if m.get("severity") == "error"
        ]
        return ProblemResult(
            problem_id=problem.id,
            year=problem.year,
            section=problem.section,
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
            or (
                isinstance(goals, list)
                and len(goals) == 0
                and "error" not in tactic_result
                and "message" not in tactic_result
            )
        )

        if proof_complete:
            return ProblemResult(
                problem_id=problem.id,
                year=problem.year,
                section=problem.section,
                solved=True,
                proof_code=f"{problem.statement} := by {tactic}",
                tactics_used=1,
                attempts=attempts,
                duration_seconds=time.monotonic() - start,
            )

    return ProblemResult(
        problem_id=problem.id,
        year=problem.year,
        section=problem.section,
        solved=False,
        error="All automation tactics failed",
        attempts=attempts,
        duration_seconds=time.monotonic() - start,
    )


async def attempt_putnam_search(
    problem: PutnamProblem,
    session: LeanREPLSession,
    budget: int = 64,
    timeout: int = 60,
    use_mathlib: bool = False,
) -> ProblemResult:
    """Attempt to prove a Putnam problem using best-first search tree.

    First tries automation tactics via REPL (fast). If those fail, runs a
    best-first search over tactic sequences.
    """
    start = time.monotonic()

    # Phase 1: Try automation tactics first (instant)
    auto_result = await attempt_putnam_repl(
        problem, session, timeout=min(timeout, 30),
    )
    if auto_result.solved:
        return auto_result

    # Phase 2: Best-first search tree
    # Build the theorem statement for the search tree (without imports)
    parts: list[str] = []
    if problem.preamble:
        parts.append(problem.preamble)
    if problem.setup_block:
        parts.append(problem.setup_block)
    parts.append(problem.statement)
    theorem = "\n".join(parts)

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
                year=problem.year,
                section=problem.section,
                solved=True,
                proof_code=search_result.proof_code,
                tactics_used=len(search_result.proof_tactics),
                attempts=search_result.nodes_explored,
                duration_seconds=time.monotonic() - start,
            )
        else:
            search_attempts = search_result.nodes_explored
            return ProblemResult(
                problem_id=problem.id,
                year=problem.year,
                section=problem.section,
                solved=False,
                error=f"Automation + search failed ({auto_result.attempts} auto + {search_attempts} search nodes)",
                attempts=auto_result.attempts + search_attempts,
                duration_seconds=time.monotonic() - start,
            )
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("Search tree failed for %s: %s", problem.id, e)

    return ProblemResult(
        problem_id=problem.id,
        year=problem.year,
        section=problem.section,
        solved=False,
        error=f"Automation + search failed ({auto_result.attempts} auto + search)",
        attempts=auto_result.attempts,
        duration_seconds=time.monotonic() - start,
    )


async def run_putnam(
    year_filter: int | None = None,
    section_filter: str | None = None,
    problem_ids: list[str] | None = None,
    timeout: int = 60,
    putnam_dir: Path | None = None,
    use_search: bool = True,
    search_budget: int = 64,
    use_mathlib_search: bool = True,
    year_range: tuple[int, int] | None = None,
) -> PutnamBenchmarkResult:
    """Run Bourbaki on PutnamBench problems and report pass rate.

    Args:
        year_filter: Filter by specific year.
        section_filter: Filter by section ("a" or "b").
        problem_ids: Specific problem IDs to run.
        timeout: Timeout per problem in seconds.
        putnam_dir: Path to PutnamBench checkout.
        use_search: Use best-first search tree after automation fails.
        search_budget: Max tactic attempts per problem for search tree.
        use_mathlib_search: Query Mathlib APIs during search tree expansion.
        year_range: Filter by year range (inclusive).

    Returns:
        PutnamBenchmarkResult with aggregate and per-problem results.
    """
    problems = load_putnam_problems(
        year_filter=year_filter,
        section_filter=section_filter,
        problem_ids=problem_ids,
        putnam_dir=putnam_dir,
        year_range=year_range,
    )

    if not problems:
        return PutnamBenchmarkResult(
            config={
                "year_filter": year_filter,
                "section_filter": section_filter,
                "year_range": year_range,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    stats = get_putnam_stats(problems)
    logger.info(
        "Running PutnamBench: %d problems (%d-%d)",
        stats["total"],
        stats["year_range"][0],
        stats["year_range"][1],
    )

    overall_start = time.monotonic()
    results: list[ProblemResult] = []

    # Set up REPL session (one-time Mathlib import)
    repl_session: LeanREPLSession | None = None
    try:
        repl_session = LeanREPLSession(import_full_mathlib=True)
        await repl_session.start()
        logger.info("Initializing REPL with full Mathlib import...")
        await repl_session.ensure_initialized()
        if not repl_session._initialized:
            logger.error("REPL init failed -- cannot proceed without REPL")
            return PutnamBenchmarkResult(
                config={"error": "REPL initialization failed"},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
    except Exception as e:
        logger.error("REPL not available (%s) -- aborting", e)
        return PutnamBenchmarkResult(
            config={"error": f"REPL not available: {e}"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    try:
        for i, problem in enumerate(problems):
            logger.info(
                "[%d/%d] %s (year=%d)",
                i + 1, len(problems), problem.id, problem.year,
            )
            if use_search:
                result = await attempt_putnam_search(
                    problem,
                    repl_session,
                    budget=search_budget,
                    timeout=timeout,
                    use_mathlib=use_mathlib_search,
                )
            else:
                result = await attempt_putnam_repl(
                    problem, repl_session, timeout=timeout,
                )
            results.append(result)
            status = "SOLVED" if result.solved else "FAILED"
            logger.info("  %s (%.2fs)", status, result.duration_seconds)
    finally:
        if repl_session is not None:
            await repl_session.stop()

    # Compute aggregate stats
    total_time = time.monotonic() - overall_start
    solved = [r for r in results if r.solved]
    solved_count = len(solved)

    # Per-year breakdown
    by_year: dict[int, dict[str, int]] = {}
    for r in results:
        if r.year not in by_year:
            by_year[r.year] = {"total": 0, "solved": 0}
        by_year[r.year]["total"] += 1
        if r.solved:
            by_year[r.year]["solved"] += 1

    # Per-section breakdown
    by_section: dict[str, dict[str, int]] = {}
    for r in results:
        if r.section not in by_section:
            by_section[r.section] = {"total": 0, "solved": 0}
        by_section[r.section]["total"] += 1
        if r.solved:
            by_section[r.section]["solved"] += 1

    # Per-decade breakdown
    by_decade: dict[str, dict[str, int]] = {}
    for r in results:
        decade = f"{(r.year // 10) * 10}s"
        if decade not in by_decade:
            by_decade[decade] = {"total": 0, "solved": 0}
        by_decade[decade]["total"] += 1
        if r.solved:
            by_decade[decade]["solved"] += 1

    benchmark = PutnamBenchmarkResult(
        total=len(results),
        solved=solved_count,
        pass_rate=solved_count / len(results) if results else 0.0,
        results=results,
        total_time_seconds=total_time,
        avg_duration_per_problem=total_time / len(results) if results else 0.0,
        avg_tactics_per_solve=(
            sum(r.tactics_used for r in solved) / solved_count
            if solved_count
            else 0.0
        ),
        config={
            "year_filter": year_filter,
            "section_filter": section_filter,
            "year_range": year_range,
            "timeout": timeout,
            "use_search": use_search,
            "search_budget": search_budget if use_search else None,
            "use_mathlib_search": use_mathlib_search if use_search else None,
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
        by_year=by_year,
        by_section=by_section,
        by_decade=by_decade,
    )

    # Save results
    _save_results(benchmark)

    # Log summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PutnamBench Results")
    logger.info("=" * 60)
    logger.info(
        "Total: %d/%d solved (%.1f%%) in %.1fs",
        solved_count, len(results), benchmark.pass_rate * 100, total_time,
    )
    logger.info("")

    # Decade breakdown
    logger.info("By decade:")
    for decade, counts in sorted(by_decade.items()):
        pct = counts["solved"] / counts["total"] * 100 if counts["total"] else 0
        logger.info(
            "  %s: %d/%d (%.1f%%)",
            decade, counts["solved"], counts["total"], pct,
        )

    # Section breakdown
    logger.info("")
    logger.info("By section:")
    for sec, counts in sorted(by_section.items()):
        pct = counts["solved"] / counts["total"] * 100 if counts["total"] else 0
        logger.info(
            "  %s: %d/%d (%.1f%%)",
            sec, counts["solved"], counts["total"], pct,
        )

    logger.info("=" * 60)

    return benchmark


def _save_results(benchmark: PutnamBenchmarkResult) -> None:
    """Save benchmark results to a JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{date_str}_putnam.json"
    path = RESULTS_DIR / filename
    path.write_text(
        json.dumps(benchmark.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", path)


def main() -> None:
    import argparse

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pydantic_ai").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="PutnamBench benchmark runner for Bourbaki"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Quick test: run {len(QUICK_TEST_IDS)} problems",
    )
    parser.add_argument("--year", type=int, help="Filter by specific year")
    parser.add_argument(
        "--year-from", type=int, help="Year range start (inclusive)"
    )
    parser.add_argument(
        "--year-to", type=int, help="Year range end (inclusive)"
    )
    parser.add_argument(
        "--section", choices=["a", "b"], help="Filter by section"
    )
    parser.add_argument(
        "--problems", nargs="+", help="Specific problem IDs"
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Timeout per problem (seconds)"
    )
    parser.add_argument(
        "--budget", type=int, default=64, help="Search tree budget per problem"
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip search tree (automation tactics only)",
    )
    parser.add_argument(
        "--no-mathlib",
        action="store_true",
        help="Disable Mathlib search during tree expansion",
    )
    args = parser.parse_args()

    # Determine problem IDs
    problem_ids = args.problems
    if args.quick and not problem_ids:
        problem_ids = QUICK_TEST_IDS

    # Determine year range
    year_range = None
    if args.year_from or args.year_to:
        year_range = (args.year_from or 1962, args.year_to or 2025)

    asyncio.run(
        run_putnam(
            year_filter=args.year,
            section_filter=args.section,
            problem_ids=problem_ids,
            timeout=args.timeout,
            use_search=not args.no_search,
            search_budget=args.budget,
            use_mathlib_search=not args.no_mathlib,
            year_range=year_range,
        )
    )


if __name__ == "__main__":
    main()
