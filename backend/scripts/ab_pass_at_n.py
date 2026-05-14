#!/usr/bin/env python3
"""A/B runner: ``pass_n`` toggle on the 35-problem stratified subset (#18).

Runs ``run_minif2f`` twice on the same 35-problem subset — once with
``pass_n=1`` (control) and once with ``pass_n=N`` (treatment).  Reports the
verified-count delta and a per-problem A vs B table.

Note the cost asymmetry: the treatment side runs up to N independent loop
attempts per problem (early-exit on first success).  Worst-case it takes
N× the control's wall time; on solved problems with N=4 it usually takes
~1.5-2x because the first attempt often succeeds.

Usage
-----
``python3 backend/scripts/ab_pass_at_n.py``                # default N=4
``python3 backend/scripts/ab_pass_at_n.py --n 8``
``python3 backend/scripts/ab_pass_at_n.py --ids id1,id2``  # custom subset
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
        description="A/B: pass_n=1 vs pass_n=N on the 35-problem subset.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Pass@N for the treatment arm (default: 4).",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated problem IDs. Default: 35-problem stratified subset.",
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
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="miniF2F split (default: valid).",
    )
    return parser.parse_args()


async def _one_run(
    problem_ids: list[str],
    pass_n: int,
    args: argparse.Namespace,
    logger,
) -> Any:
    from bourbaki.benchmarks.minif2f import run_minif2f

    label = f"TREATMENT (pass_n={pass_n})" if pass_n > 1 else "CONTROL (pass_n=1)"
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
        pass_n=pass_n,
    )


def _per_problem_table(
    control: Any,
    treatment: Any,
    n: int,
    logger,
) -> None:
    c_rows = {r.problem_id: r for r in control.results}
    t_rows = {r.problem_id: r for r in treatment.results}
    ids = sorted(set(c_rows) | set(t_rows))
    logger.info("")
    logger.info("Per-problem A/B (CONTROL pass_n=1 vs TREATMENT pass_n=%d):", n)
    logger.info(
        "  %-45s  %-8s  %-13s  %-6s",
        "PROBLEM_ID", "CTRL", f"TREAT(att/N)", "DELTA",
    )
    for pid in ids:
        c = c_rows.get(pid)
        t = t_rows.get(pid)
        c_pass = "PASS" if (c and c.verified) else "FAIL"
        t_pass = "PASS" if (t and t.verified) else "FAIL"
        t_att = (t.attempts_pass_n if t else "-")
        t_label = f"{t_pass} ({t_att}/{n})"
        if c_pass == "FAIL" and t_pass == "PASS":
            delta = "+1"
        elif c_pass == "PASS" and t_pass == "FAIL":
            delta = "-1"
        else:
            delta = "0"
        logger.info("  %-45s  %-8s  %-13s  %-6s", pid, c_pass, t_label, delta)


async def _run(args: argparse.Namespace, problem_ids: list[str], logger) -> None:
    # Control: pass_n=1
    control = await _one_run(problem_ids, 1, args, logger)
    control_path = common.newest_result_json()

    # Treatment: pass_n=N
    treatment = await _one_run(problem_ids, args.n, args, logger)
    treatment_path = common.newest_result_json()

    logger.info("")
    logger.info("=" * 60)
    logger.info("A/B SUMMARY: pass_n=1 vs pass_n=%d", args.n)
    logger.info("=" * 60)
    logger.info(
        "  CONTROL   verified=%d/%d  time=%.0fs  json=%s",
        control.verified, control.total, control.total_time_seconds,
        control_path.name if control_path else "?",
    )
    logger.info(
        "  TREATMENT verified=%d/%d  time=%.0fs  json=%s",
        treatment.verified, treatment.total, treatment.total_time_seconds,
        treatment_path.name if treatment_path else "?",
    )
    delta = treatment.verified - control.verified
    logger.info("  Δ verified: %+d", delta)

    _per_problem_table(control, treatment, args.n, logger)

    logger.info("")
    logger.info(
        "Both result JSONs saved under %s (use view_result.py --diff to compare).",
        common.RESULTS_DIR,
    )


def main() -> None:
    args = parse_args()
    if args.n < 2:
        print("error: --n must be >= 2 for a meaningful A/B (control is pass_n=1)", file=sys.stderr)
        sys.exit(2)

    logger, log_path = common.setup_logging("ab_pass_at_n")
    common.require_glm_api_key(logger)

    if args.ids:
        problem_ids = [pid.strip() for pid in args.ids.split(",") if pid.strip()]
    else:
        try:
            problem_ids = common.load_default_subset_ids()
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(2)

    n_problems = len(problem_ids)
    # Worst case: control 1x + treatment Nx
    est_minutes = (1 + args.n) * n_problems * args.timeout / 60.0
    est_str = f"~{est_minutes:.0f}m worst case (control + N={args.n}× treatment × {n_problems} × {args.timeout}s)"

    cfg = {
        "n_problems": n_problems,
        "split": args.split,
        "pass_n_treatment": args.n,
        "timeout_s": args.timeout,
        "max_iter": args.max_iter,
        "model": args.model,
        "default_subset": args.ids is None,
    }
    common.banner(
        logger,
        "ab_pass_at_n",
        cfg,
        log_path,
        expected_wall_time=est_str,
        expected_result_glob="2 files in .bourbaki/benchmarks/results/",
    )
    head = problem_ids[:5]
    logger.info(
        "Problem IDs (first 5 of %d): %s%s",
        n_problems, ", ".join(head), "..." if n_problems > 5 else "",
    )

    if not common.MINIF2F_DIR.is_dir():
        logger.error("miniF2F checkout missing at %s", common.MINIF2F_DIR)
        sys.exit(3)

    asyncio.run(_run(args, problem_ids, logger))


if __name__ == "__main__":
    main()
