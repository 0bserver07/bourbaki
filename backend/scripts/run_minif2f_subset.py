#!/usr/bin/env python3
"""Standalone runner for an arbitrary miniF2F subset.

Defaults to the 35-problem stratified subset that was used for the May 9
post-refactor A/B (62.9% verified).  The IDs are read from the baseline
result file ``.bourbaki/benchmarks/results/2026-03-19_1516_minif2f_valid.json``.

Usage
-----
``python3 backend/scripts/run_minif2f_subset.py``                       # 35-problem subset
``python3 backend/scripts/run_minif2f_subset.py --ids a,b,c``           # specific IDs
``python3 backend/scripts/run_minif2f_subset.py --from-file ids.txt``   # IDs from a file (one per line)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _runner_common as common  # noqa: E402

common.add_backend_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a subset of miniF2F problems through the "
                    "proposer-builder-reviewer loop.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated list of problem IDs. Mutually exclusive with --from-file.",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        default=None,
        help="Path to a file with one problem ID per line.",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="Which split to load IDs from (default: valid).",
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
        "--enable-mathlib-search",
        action="store_true",
        help="Register mathlib_search as a proposer tool.",
    )
    parser.add_argument(
        "--proposer-llm-timeout",
        type=float,
        default=None,
        help=(
            "Per-LLM-call timeout for the proposer node, in seconds. "
            "Defaults to ProverConfig.proposer_llm_timeout (90.0). "
            "Bump (e.g. 600) for frontier models like Kimi K2.6 1T or "
            "gpt-oss:120b that routinely exceed 90s on a single call."
        ),
    )
    parser.add_argument(
        "--reviewer-llm-timeout",
        type=float,
        default=None,
        help=(
            "Per-LLM-call timeout for the reviewer node, in seconds. "
            "Defaults to ProverConfig.reviewer_llm_timeout (60.0)."
        ),
    )
    parser.add_argument(
        "--pass-n",
        type=int,
        default=1,
        help="Pass@N sampling (default: 1).",
    )
    parser.add_argument(
        "--proposer-models",
        type=str,
        default=None,
        help=(
            "Multi-provider parallel-fanout proposer. Comma- or "
            "pipe-separated list of ``model[@llm_timeout]`` specs. "
            "``@`` separates the (colon-containing) model string from "
            "its per-leg timeout in seconds. Example: "
            "'ollama-cloud:gpt-oss:120b@90,glm:glm-5.1@90,"
            "ollama-cloud:kimi-k2.6@600'. When set, --model is ignored "
            "for the proposer step (reviewer still uses --model)."
        ),
    )
    return parser.parse_args()


def _parse_proposer_models(spec: str | None):
    """Parse ``--proposer-models`` into a list of ``ProposerConfig``.

    Empty / None / whitespace-only input returns ``None`` (single-model
    legacy path). Splits on ``,`` and ``|`` so the flag survives a shell
    that munges either. Inside each entry, ``@`` separates the model
    string from its optional ``llm_timeout`` (seconds) — we can't use
    ``:`` since model strings already contain colons
    (``ollama-cloud:gpt-oss:120b``).
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    from bourbaki.prover.parallel_proposer import ProposerConfig

    # Split on both ``,`` and ``|`` so callers can use whichever their
    # shell tolerates better.
    raw_entries: list[str] = []
    for chunk in spec.split("|"):
        raw_entries.extend(chunk.split(","))
    configs: list[ProposerConfig] = []
    for entry in raw_entries:
        entry = entry.strip()
        if not entry:
            continue
        if "@" in entry:
            model, _, timeout_str = entry.rpartition("@")
            model = model.strip()
            timeout_str = timeout_str.strip()
            try:
                llm_timeout = float(timeout_str)
            except ValueError as e:
                raise ValueError(
                    f"--proposer-models: bad timeout {timeout_str!r} in "
                    f"entry {entry!r}; expected float seconds"
                ) from e
            configs.append(ProposerConfig(model=model, llm_timeout=llm_timeout))
        else:
            configs.append(ProposerConfig(model=entry))
    return configs or None


def _resolve_problem_ids(args: argparse.Namespace, logger) -> list[str]:
    """Resolve the list of problem IDs from --ids, --from-file, or the default subset."""
    if args.ids and args.from_file:
        logger.error("--ids and --from-file are mutually exclusive.")
        sys.exit(2)
    if args.ids:
        return [pid.strip() for pid in args.ids.split(",") if pid.strip()]
    if args.from_file:
        p = Path(args.from_file).expanduser().resolve()
        if not p.is_file():
            logger.error("--from-file %s does not exist.", p)
            sys.exit(2)
        lines = p.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    # Default: 35-problem stratified subset
    try:
        return common.load_default_subset_ids()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(2)


async def _run(args: argparse.Namespace, problem_ids: list[str], logger) -> None:
    from bourbaki.benchmarks.minif2f import run_minif2f

    proposer_configs = _parse_proposer_models(args.proposer_models)
    if proposer_configs:
        logger.info(
            "Parallel proposer fanout: %d legs [%s]",
            len(proposer_configs),
            ", ".join(f"{c.model}@{c.llm_timeout:.0f}s" for c in proposer_configs),
        )

    result = await run_minif2f(
        split=args.split,
        problem_ids=problem_ids,
        timeout=args.timeout,
        use_loop=True,
        loop_max_iterations=args.max_iter,
        loop_model=args.model,
        loop_memory=args.memory,
        loop_memory_k=args.memory_k,
        loop_enable_mathlib_search=args.enable_mathlib_search,
        loop_proposer_llm_timeout=args.proposer_llm_timeout,
        loop_reviewer_llm_timeout=args.reviewer_llm_timeout,
        loop_proposer_configs=proposer_configs,
        pass_n=args.pass_n,
    )

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
        "SUMMARY: miniF2F %s subset verified=%d/%d (%.1f%%) time=%.0fs",
        args.split,
        result.verified,
        result.total,
        100 * result.verified / result.total if result.total else 0.0,
        result.total_time_seconds,
    )


def main() -> None:
    args = parse_args()
    logger, log_path = common.setup_logging("run_minif2f_subset")
    common.require_glm_api_key(logger)

    problem_ids = _resolve_problem_ids(args, logger)
    n = len(problem_ids)
    if n == 0:
        logger.error("No problem IDs resolved — refusing to run an empty subset.")
        sys.exit(2)

    est_minutes = n * args.timeout * args.pass_n / 60.0
    est_str = (
        f"~{est_minutes:.0f}m ({n} problems × {args.timeout}s × pass_n={args.pass_n}, "
        "worst case)"
    )

    cfg = {
        "split": args.split,
        "n_problems": n,
        "max_iter": args.max_iter,
        "timeout_s": args.timeout,
        "model": args.model,
        "memory": args.memory,
        "memory_k": args.memory_k,
        "pass_n": args.pass_n,
        "enable_mathlib_search": args.enable_mathlib_search,
        "proposer_llm_timeout_s": args.proposer_llm_timeout,
        "reviewer_llm_timeout_s": args.reviewer_llm_timeout,
        "proposer_models": args.proposer_models,
        "source": "--ids" if args.ids else ("--from-file " + args.from_file if args.from_file else "default 35-problem subset"),
    }
    common.banner(
        logger,
        "run_minif2f_subset",
        cfg,
        log_path,
        expected_wall_time=est_str,
        expected_result_glob=str(common.RESULTS_DIR / f"YYYY-MM-DD_HHMM_minif2f_{args.split}.json"),
    )

    if not common.MINIF2F_DIR.is_dir():
        logger.error(
            "miniF2F checkout missing at %s — clone it first: "
            "git clone https://github.com/yangky11/miniF2F-lean4 %s",
            common.MINIF2F_DIR, common.MINIF2F_DIR,
        )
        sys.exit(3)

    # Log the first few IDs for context
    head = problem_ids[:5]
    logger.info("Problem IDs (first 5): %s%s", ", ".join(head), "..." if n > 5 else "")

    asyncio.run(_run(args, problem_ids, logger))


if __name__ == "__main__":
    main()
