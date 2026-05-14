#!/usr/bin/env python3
"""Standalone runner for the full miniF2F valid split (244 problems).

Designed to be launched outside of a bash-tool-managed session: the long-
running asyncio loop in :func:`bourbaki.benchmarks.minif2f.run_minif2f` lives
inside this single Python process, so killing the launching shell will NOT
kill the proof loop.

Usage
-----
``python3 backend/scripts/run_full_244.py``

Flags
-----
``--split {valid,test,all}``   Which miniF2F split (default valid).
``--max-iter INT``             Proposer-builder-reviewer iterations per problem.
``--timeout INT``              Per-problem timeout in seconds.
``--model STR``                Loop model (default ``glm:glm-5.1``).
``--memory STR``               Loop memory class name (default ``MemorylessMemory``).
``--memory-k INT``             Memory window for ``PreviousKMemory``.
``--pass-n INT``               Pass@N sampling (default 1).
``--enable-mathlib-search``    Wire ``mathlib_search`` as a proposer tool.

The script ALWAYS uses ``use_loop=True`` (Phase 2 architecture).  The
underlying ``run_minif2f`` writes a timestamped JSON to
``.bourbaki/benchmarks/results/`` — this script does not re-save it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Bootstrap import paths and logging.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _runner_common as common  # noqa: E402

common.add_backend_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full miniF2F valid split (244 problems) "
                    "through the proposer-builder-reviewer loop.",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test", "all"],
        default="valid",
        help="miniF2F split (default: valid; 'all' doubles cost).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Loop max iterations per problem (default: 20).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-problem timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--model",
        default="glm:glm-5.1",
        help="Loop model (default: glm:glm-5.1).",
    )
    parser.add_argument(
        "--memory",
        default="MemorylessMemory",
        help="Loop memory class (default: MemorylessMemory).",
    )
    parser.add_argument(
        "--memory-k",
        type=int,
        default=2,
        help="Memory window for PreviousKMemory (default: 2).",
    )
    parser.add_argument(
        "--pass-n",
        type=int,
        default=1,
        help="Pass@N sampling (default: 1).",
    )
    parser.add_argument(
        "--enable-mathlib-search",
        action="store_true",
        help="Register mathlib_search as a proposer tool.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace, logger) -> None:
    from bourbaki.benchmarks.minif2f import run_minif2f

    result = await run_minif2f(
        split=args.split,
        timeout=args.timeout,
        use_loop=True,
        loop_max_iterations=args.max_iter,
        loop_model=args.model,
        loop_memory=args.memory,
        loop_memory_k=args.memory_k,
        loop_enable_mathlib_search=args.enable_mathlib_search,
        pass_n=args.pass_n,
    )

    # Final summary printout — supplements the per-problem log lines and
    # the _log_results_summary call inside run_minif2f.
    logger.info("")
    logger.info("Final results:")
    logger.info(
        "  verified: %d/%d (%.1f%%)",
        result.verified,
        result.total,
        100 * result.verified / result.total if result.total else 0.0,
    )
    logger.info("  false positives: %d", result.false_positives)
    logger.info("  total time: %.1fs", result.total_time_seconds)
    common.print_per_source_breakdown(logger, result.by_source)
    logger.info("")
    logger.info(
        "SUMMARY: miniF2F %s verified=%d/%d (%.1f%%) time=%.0fs",
        args.split,
        result.verified,
        result.total,
        100 * result.verified / result.total if result.total else 0.0,
        result.total_time_seconds,
    )


def main() -> None:
    args = parse_args()
    logger, log_path = common.setup_logging("run_full_244")

    # Estimate wall time: ~5 min/problem average × 244 problems for valid.
    # `--split all` (488) is ~2x.
    n_problems = (
        244 if args.split == "valid" else 244 if args.split == "test" else 488
    )
    est_minutes = n_problems * args.timeout * args.pass_n / 60.0
    est_str = f"~{est_minutes / 60.0:.1f}h ({n_problems} problems × {args.timeout}s × pass_n={args.pass_n}, worst case)"

    if args.split == "all":
        logger.warning(
            "WARNING: --split all doubles cost vs. --split valid. "
            "Consider running valid and test separately.",
        )

    common.require_glm_api_key(logger)

    cfg = {
        "split": args.split,
        "max_iter": args.max_iter,
        "timeout_s": args.timeout,
        "model": args.model,
        "memory": args.memory,
        "memory_k": args.memory_k,
        "pass_n": args.pass_n,
        "enable_mathlib_search": args.enable_mathlib_search,
        "minif2f_dir": common.MINIF2F_DIR,
    }
    common.banner(
        logger,
        "run_full_244",
        cfg,
        log_path,
        expected_wall_time=est_str,
        expected_result_glob=str(common.RESULTS_DIR / f"YYYY-MM-DD_HHMM_minif2f_{args.split}.json"),
    )

    if not common.MINIF2F_DIR.is_dir():
        logger.error(
            "miniF2F checkout missing at %s — run scripts/run-benchmark.sh "
            "once or clone manually: git clone https://github.com/yangky11/miniF2F-lean4 %s",
            common.MINIF2F_DIR,
            common.MINIF2F_DIR,
        )
        sys.exit(3)

    asyncio.run(_run(args, logger))


if __name__ == "__main__":
    main()
