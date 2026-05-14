#!/usr/bin/env python3
"""A/B runner: ``loop_enable_mathlib_search`` toggle on an 8-problem hard subset (#17).

Runs ``run_minif2f`` twice on the same problem list — once with
``loop_enable_mathlib_search=False`` (control) and once with True
(treatment) — and compares verified counts and per-problem deltas.

The default 8-problem hard subset is every problem that ALL failed at
attempts=0 on the May 9 35-problem run, i.e. the loop never even produced
a candidate.  These are the cases where mathlib_search has the most
plausible upside.

Usage
-----
``python3 backend/scripts/ab_mathlib_search.py``                                  # default 8 problems
``python3 backend/scripts/ab_mathlib_search.py --problems id1,id2``               # custom set
``python3 backend/scripts/ab_mathlib_search.py --timeout 300 --max-iter 20``
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _runner_common as common  # noqa: E402

common.add_backend_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B: loop_enable_mathlib_search on a hard miniF2F subset.",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help=f"Comma-separated problem IDs. Default: 8-problem hard subset "
             f"({', '.join(common.AB_MATHLIB_DEFAULT_IDS[:2])}...).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-problem timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Loop max iterations per problem (default: 20).",
    )
    parser.add_argument(
        "--model",
        default="glm:glm-5.1",
        help="Loop model (default: glm:glm-5.1).",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="miniF2F split (default: valid).",
    )
    return parser.parse_args()


async def _one_run(
    problem_ids: list[str],
    enable_mathlib_search: bool,
    args: argparse.Namespace,
    logger,
) -> Any:
    from bourbaki.benchmarks.minif2f import run_minif2f

    label = "TREATMENT (mathlib_search=on)" if enable_mathlib_search else "CONTROL (mathlib_search=off)"
    bar = "-" * 60
    logger.info(bar)
    logger.info(label)
    logger.info(bar)

    return await run_minif2f(
        split=args.split,
        problem_ids=problem_ids,
        timeout=args.timeout,
        use_loop=True,
        loop_max_iterations=args.max_iter,
        loop_model=args.model,
        loop_enable_mathlib_search=enable_mathlib_search,
    )


def _per_problem_table(
    control: Any,
    treatment: Any,
    logger,
) -> None:
    c_rows = {r.problem_id: r for r in control.results}
    t_rows = {r.problem_id: r for r in treatment.results}
    ids = sorted(set(c_rows) | set(t_rows))
    logger.info("")
    logger.info("Per-problem A/B (CONTROL vs TREATMENT):")
    logger.info(
        "  %-50s  %-8s  %-8s  %-6s",
        "PROBLEM_ID", "CONTROL", "TREATMT", "DELTA",
    )
    for pid in ids:
        c = c_rows.get(pid)
        t = t_rows.get(pid)
        c_pass = "PASS" if (c and c.verified) else "FAIL"
        t_pass = "PASS" if (t and t.verified) else "FAIL"
        if c_pass == "FAIL" and t_pass == "PASS":
            delta = "+1"
        elif c_pass == "PASS" and t_pass == "FAIL":
            delta = "-1"
        else:
            delta = "0"
        logger.info("  %-50s  %-8s  %-8s  %-6s", pid, c_pass, t_pass, delta)


async def _run(args: argparse.Namespace, problem_ids: list[str], logger) -> None:
    # Control: mathlib_search off
    control = await _one_run(problem_ids, False, args, logger)
    control_path = common.newest_result_json()

    # Treatment: mathlib_search on
    treatment = await _one_run(problem_ids, True, args, logger)
    treatment_path = common.newest_result_json()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("A/B SUMMARY: enable_mathlib_search")
    logger.info("=" * 60)
    logger.info("  CONTROL   verified=%d/%d  time=%.0fs  json=%s",
                control.verified, control.total, control.total_time_seconds,
                control_path.name if control_path else "?")
    logger.info("  TREATMENT verified=%d/%d  time=%.0fs  json=%s",
                treatment.verified, treatment.total, treatment.total_time_seconds,
                treatment_path.name if treatment_path else "?")
    delta = treatment.verified - control.verified
    logger.info("  Δ verified: %+d", delta)

    _per_problem_table(control, treatment, logger)

    logger.info("")
    logger.info(
        "Both result JSONs saved under %s (use view_result.py --diff to compare).",
        common.RESULTS_DIR,
    )


def main() -> None:
    args = parse_args()
    logger, log_path = common.setup_logging("ab_mathlib_search")
    common.require_glm_api_key(logger)

    if args.problems:
        problem_ids = [pid.strip() for pid in args.problems.split(",") if pid.strip()]
    else:
        problem_ids = list(common.AB_MATHLIB_DEFAULT_IDS)

    n = len(problem_ids)
    # Two runs, so 2x cost.
    est_minutes = 2 * n * args.timeout / 60.0
    est_str = f"~{est_minutes:.0f}m (2 runs × {n} problems × {args.timeout}s)"

    cfg = {
        "n_problems": n,
        "split": args.split,
        "timeout_s": args.timeout,
        "max_iter": args.max_iter,
        "model": args.model,
        "default_subset": args.problems is None,
    }
    common.banner(
        logger,
        "ab_mathlib_search",
        cfg,
        log_path,
        expected_wall_time=est_str,
        expected_result_glob="2 files in .bourbaki/benchmarks/results/",
    )
    logger.info("Problem IDs: %s", ", ".join(problem_ids))

    if not common.MINIF2F_DIR.is_dir():
        logger.error("miniF2F checkout missing at %s", common.MINIF2F_DIR)
        sys.exit(3)

    asyncio.run(_run(args, problem_ids, logger))


if __name__ == "__main__":
    main()
