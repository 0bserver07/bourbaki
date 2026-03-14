"""Enhanced miniF2F benchmark runner with new features.

Runs the baseline (REPL + search tree + mathlib search) then falls back
to the multi-agent coordinator for problems that automation can't solve.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set up logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("pydantic_ai").setLevel(logging.WARNING)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = _PROJECT_ROOT / ".bourbaki" / "benchmarks" / "results"


async def run_enhanced_benchmark(
    split: str = "valid",
    use_multi_agent: bool = True,
    multi_agent_timeout: float = 90.0,
    multi_agent_retries: int = 2,
    model: str | None = None,
    problem_ids: list[str] | None = None,
    skip_baseline: bool = False,
) -> None:
    """Run enhanced benchmark: baseline + multi-agent fallback.

    Args:
        split: "valid" or "test"
        use_multi_agent: Try multi-agent on failed problems
        multi_agent_timeout: Timeout per problem for multi-agent
        multi_agent_retries: Max retries for multi-agent coordinator
        model: Model string for multi-agent (e.g. "glm:glm-5")
        problem_ids: Optional specific problem IDs to test
        skip_baseline: Skip baseline, only run multi-agent on specified problems
    """
    from bourbaki.benchmarks.minif2f import (
        run_minif2f,
        ProblemResult,
        BenchmarkResult,
        _save_results,
        _verify_with_lean_prover,
    )
    from bourbaki.benchmarks.loader import load_minif2f_problems

    overall_start = time.monotonic()

    # Phase 1: Baseline (REPL + search tree + mathlib search)
    if not skip_baseline:
        logger.info("=" * 60)
        logger.info("PHASE 1: Baseline (automation + search + mathlib)")
        logger.info("=" * 60)

        baseline = await run_minif2f(
            split=split,
            use_repl=True,
            use_search=True,
            search_budget=64,
            use_mathlib_search=True,
            timeout=60,
            problem_ids=problem_ids,
            verify=True,
            verify_timeout=150,
        )

        logger.info("")
        logger.info("Baseline: %d/%d (%.1f%%)", baseline.solved, baseline.total, baseline.pass_rate * 100)

        failed_ids = [r.problem_id for r in baseline.results if not r.solved]
        solved_ids = {r.problem_id for r in baseline.results if r.solved}
    else:
        # Load problems and use all as "failed" for multi-agent
        problems = load_minif2f_problems(split=split, problem_ids=problem_ids)
        failed_ids = [p.id for p in problems]
        solved_ids = set()
        baseline = None

    if not failed_ids:
        logger.info("All problems solved by baseline!")
        return

    # Phase 2: Multi-agent coordinator on failed problems
    if use_multi_agent and model:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 2: Multi-agent coordinator on %d unsolved problems", len(failed_ids))
        logger.info("Model: %s", model)
        logger.info("=" * 60)

        from bourbaki.agent.coordinator import ProofCoordinator
        from bourbaki.benchmarks.loader import load_minif2f_problems

        failed_problems = load_minif2f_problems(
            split=split, problem_ids=failed_ids,
        )

        multi_agent_results: list[ProblemResult] = []
        coordinator = ProofCoordinator(model=model)

        for i, problem in enumerate(failed_problems):
            logger.info("[%d/%d] Multi-agent: %s", i + 1, len(failed_problems), problem.id)
            start = time.monotonic()

            try:
                result = await asyncio.wait_for(
                    coordinator.prove(
                        theorem=problem.statement,
                        timeout=multi_agent_timeout,
                        max_retries=multi_agent_retries,
                    ),
                    timeout=multi_agent_timeout + 10,
                )

                if result.success and result.proof_code:
                    pr = ProblemResult(
                        problem_id=problem.id,
                        source=problem.source,
                        solved=True,
                        repl_reported=True,
                        proof_code=result.proof_code,
                        duration_seconds=time.monotonic() - start,
                        attempts=sum(result.agent_stats.values()),
                    )
                    # Mandatory verification for multi-agent results too
                    pr = await _verify_with_lean_prover(pr, verify_timeout=150)
                    if pr.verified:
                        logger.info("  SOLVED by multi-agent (verified)! (%.1fs)", pr.duration_seconds)
                    elif pr.solved:
                        logger.info("  SOLVED by multi-agent (unverified)! (%.1fs)", pr.duration_seconds)
                    else:
                        logger.info("  Multi-agent false positive (%.1fs): %s", pr.duration_seconds, pr.error)
                else:
                    pr = ProblemResult(
                        problem_id=problem.id,
                        source=problem.source,
                        solved=False,
                        error=result.error or "Multi-agent failed",
                        duration_seconds=time.monotonic() - start,
                    )
                    logger.info("  FAILED (%.1fs): %s", pr.duration_seconds, pr.error)

            except asyncio.TimeoutError:
                pr = ProblemResult(
                    problem_id=problem.id,
                    source=problem.source,
                    solved=False,
                    error="Timeout",
                    duration_seconds=time.monotonic() - start,
                )
                logger.info("  TIMEOUT (%.1fs)", pr.duration_seconds)
            except Exception as e:
                pr = ProblemResult(
                    problem_id=problem.id,
                    source=problem.source,
                    solved=False,
                    error=str(e),
                    duration_seconds=time.monotonic() - start,
                )
                logger.info("  ERROR: %s", e)

            multi_agent_results.append(pr)

        # Merge results
        ma_solved = [r for r in multi_agent_results if r.solved]
        logger.info("")
        logger.info("Multi-agent: solved %d/%d additional problems", len(ma_solved), len(failed_ids))

        if ma_solved:
            for r in ma_solved:
                logger.info("  NEW SOLVE: %s", r.problem_id)

    else:
        multi_agent_results = []

    # Final summary with honest verification reporting
    total_time = time.monotonic() - overall_start

    # Baseline: use verified count (the headline number from run_minif2f)
    baseline_verified = baseline.verified if baseline else 0
    baseline_repl = baseline.repl_reported if baseline else 0
    baseline_fp = baseline.false_positives if baseline else 0

    # Multi-agent: count verified and REPL-reported separately
    ma_verified_count = len([r for r in multi_agent_results if r.verified])
    ma_repl_count = len([r for r in multi_agent_results if r.repl_reported])
    ma_fp_count = ma_repl_count - ma_verified_count

    total_problems = (baseline.total if baseline else len(failed_ids))
    total_verified = baseline_verified + ma_verified_count
    total_repl = baseline_repl + ma_repl_count
    total_fp = baseline_fp + ma_fp_count

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS (enhanced miniF2F)")
    logger.info("=" * 60)
    logger.info(
        "Verified:          %d/%d (%.1f%%)",
        total_verified, total_problems,
        total_verified / total_problems * 100 if total_problems else 0,
    )
    logger.info(
        "REPL-reported:     %d/%d (%.1f%%)",
        total_repl, total_problems,
        total_repl / total_problems * 100 if total_problems else 0,
    )
    logger.info(
        "False positives:   %d (%.1f%% of REPL-reported)",
        total_fp,
        total_fp / total_repl * 100 if total_repl else 0,
    )
    logger.info("")
    logger.info("  Baseline verified:     %d", baseline_verified)
    logger.info("  Multi-agent verified:  %d additional", ma_verified_count)
    logger.info("Time:              %.1fs", total_time)
    logger.info("=" * 60)

    # Save combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    result_data = {
        "total": total_problems,
        "verified": total_verified,
        "repl_reported": total_repl,
        "false_positives": total_fp,
        "false_positive_rate": (
            round(total_fp / total_repl, 4) if total_repl else 0.0
        ),
        "pass_rate": total_verified / total_problems if total_problems else 0,
        "baseline_verified": baseline_verified,
        "baseline_repl_reported": baseline_repl,
        "multi_agent_verified": ma_verified_count,
        "multi_agent_repl_reported": ma_repl_count,
        "total_time_seconds": total_time,
        "model": model,
        "config": {
            "split": split,
            "use_multi_agent": use_multi_agent,
            "multi_agent_timeout": multi_agent_timeout,
            "multi_agent_retries": multi_agent_retries,
        },
        "multi_agent_results": [
            r.to_dict() for r in multi_agent_results
        ] if multi_agent_results else [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = RESULTS_DIR / f"{date_str}_minif2f_enhanced_{split}.json"
    path.write_text(json.dumps(result_data, indent=2, default=str), encoding="utf-8")
    logger.info("Results saved to %s", path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced miniF2F benchmark")
    parser.add_argument("--split", default="valid", choices=["valid", "test", "all"])
    parser.add_argument("--model", default=None, help="Model for multi-agent (e.g. glm:glm-5)")
    parser.add_argument("--no-multi-agent", action="store_true", help="Skip multi-agent phase")
    parser.add_argument("--timeout", type=float, default=90, help="Multi-agent timeout per problem")
    parser.add_argument("--retries", type=int, default=2, help="Multi-agent retries")
    parser.add_argument("--problems", nargs="+", help="Specific problem IDs")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline, only multi-agent")
    args = parser.parse_args()

    asyncio.run(run_enhanced_benchmark(
        split=args.split,
        use_multi_agent=not args.no_multi_agent,
        multi_agent_timeout=args.timeout,
        multi_agent_retries=args.retries,
        model=args.model,
        problem_ids=args.problems,
        skip_baseline=args.skip_baseline,
    ))


if __name__ == "__main__":
    main()
