"""PutnamBench benchmark runner for Bourbaki.

Runs the agent on PutnamBench problems (672 Putnam competition problems,
1962-2025) and reports pass rates, enabling comparison with SOTA systems
(HILBERT 70.0%, etc.).

Usage:
    # Quick test (10 problems, mix of answer and non-answer)
    python -m bourbaki.benchmarks.putnam --quick

    # Full run with 60s timeout
    python -m bourbaki.benchmarks.putnam --timeout 60

    # Filter by year range
    python -m bourbaki.benchmarks.putnam --year-from 2020 --year-to 2025

    # Filter by section
    python -m bourbaki.benchmarks.putnam --section a

    # Include answer-sorry problems (excluded by default)
    python -m bourbaki.benchmarks.putnam --include-answer

    # Attempt to solve answer-type problems via the answer generation pipeline
    python -m bourbaki.benchmarks.putnam --attempt-answers

    # Attempt answers with a specific LLM model
    python -m bourbaki.benchmarks.putnam --attempt-answers --answer-model glm:glm-5

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
from bourbaki.tools.lean_prover import lean_prover
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

# ---------------------------------------------------------------------------
# Tactic filtering: reject proofs that use suspicious non-proof tactics.
# These are Lean internals, typeclass instances, or other non-mathematical
# terms that satisfy the type checker but are not valid proofs.
# ---------------------------------------------------------------------------
INVALID_TACTICS = [
    "exact Lean.defaultMaxRecDepth",
    "exact Float.toRatParts",
    "exact instDecidableEqRat",
    "exact Real.commRing",
    "exact Real.instAdd",
    "exact trapezoidal_error",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / ".bourbaki" / "benchmarks" / "results"

# Quick-test problems: a mix of years, difficulty, and answer/non-answer types
QUICK_TEST_IDS = [
    "putnam_1962_a1",
    "putnam_1988_a1",
    "putnam_2000_a1",
    "putnam_2010_a1",
    "putnam_2020_a1",
    "putnam_2023_a1",  # answer problem
    "putnam_2023_b1",  # answer problem
    "putnam_2024_a1",  # answer problem
    "putnam_2024_b1",
    "putnam_2024_a2",
]


def _contains_invalid_tactic(proof_code: str) -> str | None:
    """Check if proof code contains a suspicious non-proof tactic.

    Returns the matched invalid tactic string, or None if clean.
    """
    for invalid in INVALID_TACTICS:
        if invalid in proof_code:
            return invalid
    return None


async def _verify_whole_file(
    problem: PutnamProblem,
    proof_code: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Verify a proof via whole-file compilation with lean_prover.

    Builds the full Lean file (imports + preamble + setup + proved theorem)
    and checks it compiles without errors or sorry.

    Returns:
        Dict with 'verified' (bool) and 'error' (str | None).
    """
    # Build the complete Lean source replacing ``sorry`` with the discovered proof.
    # proof_code typically looks like: ``<statement> := by <tactic>``
    parts: list[str] = ["import Mathlib"]
    if problem.preamble:
        parts.append(problem.preamble)
    if problem.setup_block:
        parts.append(problem.setup_block)
    parts.append(proof_code)

    full_code = "\n\n".join(parts)

    try:
        result = await asyncio.wait_for(
            lean_prover(code=full_code, mode="check", timeout=timeout),
            timeout=timeout + 5,
        )
        verified = result.get("proofComplete", False)
        if not verified:
            errors = result.get("errors") or []
            err_msgs = [e.get("message", "") for e in errors if isinstance(e, dict)]
            return {
                "verified": False,
                "error": "; ".join(err_msgs) if err_msgs else "Verification failed",
            }
        return {"verified": True, "error": None}
    except (asyncio.TimeoutError, Exception) as e:
        return {"verified": False, "error": f"Verification error: {e}"}


@dataclass
class ProblemResult:
    """Result of attempting a single Putnam problem."""

    problem_id: str
    year: int
    section: str
    solved: bool
    has_answer: bool = False
    verified: bool = False
    proof_code: str | None = None
    error: str | None = None
    tactics_used: int = 0
    duration_seconds: float = 0.0
    attempts: int = 0
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "problem_id": self.problem_id,
            "year": self.year,
            "section": self.section,
            "solved": self.solved,
            "has_answer": self.has_answer,
            "verified": self.verified,
            "proof_code": self.proof_code,
            "error": self.error,
            "tactics_used": self.tactics_used,
            "duration_seconds": round(self.duration_seconds, 2),
            "attempts": self.attempts,
        }
        if self.skipped:
            d["skipped"] = True
            d["skip_reason"] = self.skip_reason
        return d


@dataclass
class PutnamBenchmarkResult:
    """Aggregate result of a PutnamBench run."""

    total: int = 0
    solved: int = 0
    pass_rate: float = 0.0
    # Separate reporting for theorem-only vs answer problems
    theorem_only_total: int = 0
    theorem_only_solved: int = 0
    answer_total: int = 0
    answer_skipped: int = 0
    answer_attempted: int = 0
    answer_solved: int = 0
    verified_count: int = 0
    tactic_filtered_count: int = 0
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
        theorem_only_rate = (
            self.theorem_only_solved / self.theorem_only_total
            if self.theorem_only_total
            else 0.0
        )
        return {
            "benchmark": "putnam",
            "total": self.total,
            "solved": self.solved,
            "pass_rate": round(self.pass_rate, 4),
            "theorem_only_total": self.theorem_only_total,
            "theorem_only_solved": self.theorem_only_solved,
            "theorem_only_pass_rate": round(theorem_only_rate, 4),
            "answer_total": self.answer_total,
            "answer_skipped": self.answer_skipped,
            "answer_attempted": self.answer_attempted,
            "answer_solved": self.answer_solved,
            "verified_count": self.verified_count,
            "tactic_filtered_count": self.tactic_filtered_count,
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
            has_answer=problem.has_answer,
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
            has_answer=problem.has_answer,
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
            candidate_code = f"{problem.statement} := by {tactic}"

            # Filter suspicious non-proof tactics
            bad_tactic = _contains_invalid_tactic(candidate_code)
            if bad_tactic:
                logger.info(
                    "  Rejected invalid tactic for %s: %s",
                    problem.id, bad_tactic,
                )
                continue

            return ProblemResult(
                problem_id=problem.id,
                year=problem.year,
                section=problem.section,
                has_answer=problem.has_answer,
                solved=True,
                proof_code=candidate_code,
                tactics_used=1,
                attempts=attempts,
                duration_seconds=time.monotonic() - start,
            )

    return ProblemResult(
        problem_id=problem.id,
        year=problem.year,
        section=problem.section,
        has_answer=problem.has_answer,
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
            candidate_code = search_result.proof_code

            # Filter suspicious non-proof tactics
            if candidate_code:
                bad_tactic = _contains_invalid_tactic(candidate_code)
                if bad_tactic:
                    logger.info(
                        "  Rejected invalid tactic for %s: %s",
                        problem.id, bad_tactic,
                    )
                    return ProblemResult(
                        problem_id=problem.id,
                        year=problem.year,
                        section=problem.section,
                        has_answer=problem.has_answer,
                        solved=False,
                        error=f"Proof rejected: contains invalid tactic '{bad_tactic}'",
                        attempts=search_result.nodes_explored,
                        duration_seconds=time.monotonic() - start,
                    )

            return ProblemResult(
                problem_id=problem.id,
                year=problem.year,
                section=problem.section,
                has_answer=problem.has_answer,
                solved=True,
                proof_code=candidate_code,
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
                has_answer=problem.has_answer,
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
        has_answer=problem.has_answer,
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
    exclude_answer: bool = True,
    verify_proofs: bool = True,
    verify_timeout: int = 30,
    attempt_answers: bool = False,
    answer_model: str = "glm:glm-5",
    answer_max_attempts: int = 3,
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
        exclude_answer: Skip problems where has_answer=True and answer is
            still ``sorry`` (default True).  These can't be validly solved
            without filling the answer placeholder.  Ignored when
            ``attempt_answers`` is True.
        verify_proofs: After REPL/search finds a proof, verify it with
            whole-file lean_prover compilation (default True).
        verify_timeout: Timeout in seconds for whole-file verification
            (default 30).
        attempt_answers: When True, run the answer generation pipeline on
            answer-sorry problems instead of skipping them.  Overrides
            ``exclude_answer`` for answer-sorry problems.
        answer_model: LLM model string for answer generation (default
            "glm:glm-5").
        answer_max_attempts: Maximum LLM answer attempts per problem
            (default 3).

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
                "exclude_answer": exclude_answer,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    stats = get_putnam_stats(problems)

    # Count answer-sorry problems for reporting
    answer_sorry_problems = [p for p in problems if p.answer_is_sorry]
    answer_sorry_count = len(answer_sorry_problems)

    # When attempt_answers is set, override exclude_answer for answer-sorry problems
    effective_exclude_answer = exclude_answer and not attempt_answers

    if effective_exclude_answer and answer_sorry_count > 0:
        logger.info(
            "Excluding %d answer-sorry problems (answer placeholder is sorry)",
            answer_sorry_count,
        )
    elif attempt_answers and answer_sorry_count > 0:
        logger.info(
            "Will attempt answer generation for %d answer-sorry problems "
            "(model=%s, max_attempts=%d)",
            answer_sorry_count, answer_model, answer_max_attempts,
        )

    logger.info(
        "Running PutnamBench: %d problems total (%d-%d), %d answer-sorry%s",
        stats["total"],
        stats["year_range"][0],
        stats["year_range"][1],
        answer_sorry_count,
        " (attempting)" if attempt_answers else (
            " (excluded)" if effective_exclude_answer else ""
        ),
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

    tactic_filtered = 0

    try:
        for i, problem in enumerate(problems):
            logger.info(
                "[%d/%d] %s (year=%d, answer=%s)",
                i + 1, len(problems), problem.id, problem.year,
                "sorry" if problem.answer_is_sorry else (
                    "yes" if problem.has_answer else "no"
                ),
            )

            # Handle answer-sorry problems
            if problem.answer_is_sorry:
                if attempt_answers:
                    # Run the answer generation pipeline
                    from bourbaki.benchmarks.answer_generator import generate_answer

                    logger.info("  Running answer generation pipeline...")
                    answer_start = time.monotonic()
                    try:
                        answer_result = await generate_answer(
                            problem=problem,
                            model=answer_model,
                            max_attempts=answer_max_attempts,
                            prove_timeout=timeout,
                            search_budget=search_budget,
                            verify_timeout=verify_timeout,
                        )
                    except Exception as e:
                        logger.error("  Answer generation failed: %s", e)
                        answer_result = None

                    answer_duration = time.monotonic() - answer_start

                    if answer_result is not None and answer_result.verified:
                        result = ProblemResult(
                            problem_id=problem.id,
                            year=problem.year,
                            section=problem.section,
                            has_answer=problem.has_answer,
                            solved=True,
                            verified=True,
                            proof_code=(
                                f"-- answer: {answer_result.answer_code}\n"
                                + (answer_result.proof_tactics[0] if answer_result.proof_tactics else "")
                            ),
                            tactics_used=len(answer_result.proof_tactics),
                            duration_seconds=answer_duration,
                        )
                    elif answer_result is not None and answer_result.theorem_solved:
                        result = ProblemResult(
                            problem_id=problem.id,
                            year=problem.year,
                            section=problem.section,
                            has_answer=problem.has_answer,
                            solved=True,
                            verified=False,
                            proof_code=f"-- answer: {answer_result.answer_code}",
                            tactics_used=len(answer_result.proof_tactics),
                            duration_seconds=answer_duration,
                            error="Answer found but verification failed",
                        )
                    else:
                        error_msg = (
                            answer_result.error
                            if answer_result is not None
                            else "Answer generation returned None"
                        )
                        result = ProblemResult(
                            problem_id=problem.id,
                            year=problem.year,
                            section=problem.section,
                            has_answer=problem.has_answer,
                            solved=False,
                            error=f"Answer pipeline: {error_msg}",
                            duration_seconds=answer_duration,
                        )

                    results.append(result)
                    status = "SOLVED" if result.solved else "FAILED"
                    verified_tag = " (verified)" if result.verified else ""
                    logger.info(
                        "  ANSWER %s%s (%.2fs)",
                        status, verified_tag, result.duration_seconds,
                    )
                    continue

                elif effective_exclude_answer:
                    result = ProblemResult(
                        problem_id=problem.id,
                        year=problem.year,
                        section=problem.section,
                        has_answer=problem.has_answer,
                        solved=False,
                        skipped=True,
                        skip_reason="answer_is_sorry",
                        error="Skipped: answer placeholder is sorry",
                        duration_seconds=0.0,
                    )
                    results.append(result)
                    logger.info("  SKIPPED (answer-sorry)")
                    continue

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

            # Whole-file verification for solved problems
            if result.solved and verify_proofs and result.proof_code:
                logger.info("  Verifying proof with lean_prover...")
                vresult = await _verify_whole_file(
                    problem, result.proof_code, timeout=verify_timeout,
                )
                if vresult["verified"]:
                    result.verified = True
                    logger.info("  Verification: PASSED")
                else:
                    # REPL said solved but whole-file check failed
                    logger.info(
                        "  Verification: FAILED (%s)", vresult["error"]
                    )
                    result.solved = False
                    result.verified = False
                    result.error = f"Verification failed: {vresult['error']}"

            # Track tactic filtering (already done inside attempt_*, but count
            # rejections visible through the error field)
            if result.error and "invalid tactic" in (result.error or ""):
                tactic_filtered += 1

            results.append(result)
            status = "SOLVED" if result.solved else "FAILED"
            verified_tag = " (verified)" if result.verified else ""
            logger.info(
                "  %s%s (%.2fs)", status, verified_tag, result.duration_seconds,
            )
    finally:
        if repl_session is not None:
            await repl_session.stop()

    # Compute aggregate stats
    total_time = time.monotonic() - overall_start
    non_skipped = [r for r in results if not r.skipped]
    solved = [r for r in non_skipped if r.solved]
    solved_count = len(solved)
    verified_count = sum(1 for r in results if r.verified)

    # Separate theorem-only vs answer reporting
    theorem_only = [r for r in non_skipped if not r.has_answer]
    theorem_only_solved = [r for r in theorem_only if r.solved]
    answer_attempted = [r for r in non_skipped if r.has_answer]
    answer_skipped = [r for r in results if r.skipped]

    # Per-year breakdown (non-skipped only)
    by_year: dict[int, dict[str, int]] = {}
    for r in non_skipped:
        if r.year not in by_year:
            by_year[r.year] = {"total": 0, "solved": 0}
        by_year[r.year]["total"] += 1
        if r.solved:
            by_year[r.year]["solved"] += 1

    # Per-section breakdown (non-skipped only)
    by_section: dict[str, dict[str, int]] = {}
    for r in non_skipped:
        if r.section not in by_section:
            by_section[r.section] = {"total": 0, "solved": 0}
        by_section[r.section]["total"] += 1
        if r.solved:
            by_section[r.section]["solved"] += 1

    # Per-decade breakdown (non-skipped only)
    by_decade: dict[str, dict[str, int]] = {}
    for r in non_skipped:
        decade = f"{(r.year // 10) * 10}s"
        if decade not in by_decade:
            by_decade[decade] = {"total": 0, "solved": 0}
        by_decade[decade]["total"] += 1
        if r.solved:
            by_decade[decade]["solved"] += 1

    # Answer-specific accounting
    answer_attempted_results = [r for r in non_skipped if r.has_answer]
    answer_solved_results = [r for r in answer_attempted_results if r.solved]

    non_skipped_count = len(non_skipped)
    benchmark = PutnamBenchmarkResult(
        total=non_skipped_count,
        solved=solved_count,
        pass_rate=solved_count / non_skipped_count if non_skipped_count else 0.0,
        theorem_only_total=len(theorem_only),
        theorem_only_solved=len(theorem_only_solved),
        answer_total=len(answer_attempted_results) + len(answer_skipped),
        answer_skipped=len(answer_skipped),
        answer_attempted=len(answer_attempted_results),
        answer_solved=len(answer_solved_results),
        verified_count=verified_count,
        tactic_filtered_count=tactic_filtered,
        results=results,
        total_time_seconds=total_time,
        avg_duration_per_problem=(
            total_time / non_skipped_count if non_skipped_count else 0.0
        ),
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
            "exclude_answer": exclude_answer,
            "attempt_answers": attempt_answers,
            "answer_model": answer_model if attempt_answers else None,
            "answer_max_attempts": answer_max_attempts if attempt_answers else None,
            "verify_proofs": verify_proofs,
            "verify_timeout": verify_timeout,
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
        "Theorem-only: %d/%d solved (%.1f%%)",
        len(theorem_only_solved),
        len(theorem_only),
        len(theorem_only_solved) / len(theorem_only) * 100 if theorem_only else 0,
    )
    if answer_attempted_results:
        logger.info(
            "Answer problems attempted: %d (solved: %d)",
            len(answer_attempted_results),
            len(answer_solved_results),
        )
    if answer_skipped:
        logger.info(
            "Answer-sorry problems skipped: %d", len(answer_skipped),
        )
    logger.info(
        "Verified (whole-file): %d/%d", verified_count, solved_count,
    )
    if tactic_filtered:
        logger.info("Tactic-filtered (rejected): %d", tactic_filtered)
    logger.info(
        "Total: %d/%d solved (%.1f%%) in %.1fs",
        solved_count, non_skipped_count,
        benchmark.pass_rate * 100, total_time,
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
    parser.add_argument(
        "--include-answer",
        action="store_true",
        help="Include answer-sorry problems (excluded by default)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip whole-file verification of solved proofs",
    )
    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=30,
        help="Timeout for whole-file verification (seconds, default 30)",
    )
    parser.add_argument(
        "--attempt-answers",
        action="store_true",
        help="Run the answer generation pipeline on answer-sorry problems",
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default="glm:glm-5",
        help="LLM model for answer generation (default: glm:glm-5)",
    )
    parser.add_argument(
        "--answer-max-attempts",
        type=int,
        default=3,
        help="Max LLM attempts per answer problem (default: 3)",
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
            exclude_answer=not args.include_answer,
            verify_proofs=not args.no_verify,
            verify_timeout=args.verify_timeout,
            attempt_answers=args.attempt_answers,
            answer_model=args.answer_model,
            answer_max_attempts=args.answer_max_attempts,
        )
    )


if __name__ == "__main__":
    main()
