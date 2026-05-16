#!/usr/bin/env python3
"""Standalone runner for PutnamBench through the proposer-builder-reviewer loop.

Defaults to a 5-problem 2020-2023 dry-run subset suitable for shaking out
infrastructure issues without burning ~3 hours on the full set.  All
answer-sorry problems are excluded by default (the loop can't fill answer
placeholders).

Usage
-----
``python3 backend/scripts/run_putnam_loop.py``                  # 5-problem dry run
``python3 backend/scripts/run_putnam_loop.py --year-range 2020-2023``
``python3 backend/scripts/run_putnam_loop.py --ids putnam_2020_a1,putnam_2020_b1``
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


# Default 5-problem subset: one A and one B from 2020-2023 (skipping
# answer-sorry problems via --exclude-answer at the run_putnam layer).
# Picked to be a representative spread of years and sections, not the
# hardest problems.
DEFAULT_DRY_RUN_IDS = [
    "putnam_2020_a1",
    "putnam_2020_b1",
    "putnam_2022_a1",
    "putnam_2022_b1",
    "putnam_2023_b1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PutnamBench problems through the proposer-builder-reviewer loop.",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help=(
            "Comma-separated list of putnam_YYYY_XN problem IDs. "
            "If unset and no --year-range, defaults to a 5-problem 2020-2023 dry-run subset."
        ),
    )
    parser.add_argument(
        "--year-range",
        type=str,
        default=None,
        help="Year range, inclusive (e.g. '2020-2023').",
    )
    parser.add_argument(
        "--section",
        choices=["a", "b"],
        default=None,
        help="Filter by Putnam section letter.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="Loop max iterations per problem (default: 15).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="Per-problem timeout in seconds (default: 240).",
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
        "--include-answer",
        action="store_true",
        help="Include answer-sorry problems (excluded by default).",
    )
    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=60,
        help="Whole-file lean_prover verify timeout per problem (default: 60).",
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
    parser.add_argument(
        "--race-mode",
        choices=["first_completed", "first_verified"],
        default="first_completed",
        help=(
            "Race semantics for --proposer-models / --smart-fanout. "
            "'first_completed' (default) takes the fastest valid "
            "ProposalMessage. 'first_verified' runs lean_prover on each "
            "leg's proposal and only declares a winner once it compiles "
            "clean — trades latency for compile-correctness."
        ),
    )
    parser.add_argument(
        "--smart-fanout",
        choices=["minimal", "balanced", "max"],
        default=None,
        help=(
            "Auto-construct a multi-provider fanout from whichever API "
            "keys are available in the environment, no --proposer-models "
            "spec required. 'minimal' = baseline + 1 frontier (2 legs); "
            "'balanced' = baseline + 2 frontier + claude-if-available "
            "(3-4 legs); 'max' = every eligible leg (up to 7). Mutually "
            "exclusive with --proposer-models and --auto-route."
        ),
    )
    parser.add_argument(
        "--auto-route",
        action="store_true",
        help=(
            "Per-problem proposer routing via "
            "``fanout_for_problem`` — the fanout legs are picked from "
            "the problem's source prefix (every Putnam problem ID starts "
            "with ``putnam_``, so this always returns the reasoning-heavy "
            "preference: kimi / deepseek / claude). Inherits "
            "--race-mode and --enable-mathlib-search on every leg. "
            "Mutually exclusive with --proposer-models and --smart-fanout."
        ),
    )
    parser.add_argument(
        "--reviewer-models",
        type=str,
        default=None,
        help=(
            "Multi-model reviewer fanout. Comma- or pipe-separated list of "
            "``model[@llm_timeout]`` specs. ``@`` separates the "
            "(colon-containing) model string from its per-leg timeout in "
            "seconds. Example: "
            "'glm:glm-5.1@60,anthropic:claude-sonnet-4-5@90'. When set, "
            "every approval requires consensus per --reviewer-review-mode "
            "and the single-model reviewer path is bypassed for the LLM "
            "decision step. The ``lean_prover`` final gate still runs "
            "exactly once after the ensemble agrees."
        ),
    )
    parser.add_argument(
        "--reviewer-review-mode",
        choices=["first_approved", "all_must_approve"],
        default="first_approved",
        help=(
            "Combination strategy for --reviewer-models. "
            "'first_approved' (default) races the legs and accepts the "
            "first APPROVE (looser, faster); 'all_must_approve' waits "
            "for every leg and accepts only if all approve (stricter, "
            "useful for borderline proofs)."
        ),
    )
    args = parser.parse_args()
    # Three fanout modes are pairwise mutually exclusive. Reject any
    # combination explicitly so a confusing precedence rule never has
    # to be inferred at runtime.
    fanout_flags = [
        ("--smart-fanout", bool(args.smart_fanout)),
        ("--proposer-models", bool(args.proposer_models)),
        ("--auto-route", bool(args.auto_route)),
    ]
    enabled = [name for name, on in fanout_flags if on]
    if len(enabled) > 1:
        parser.error(
            f"{', '.join(enabled)} are mutually exclusive; pick one"
        )
    return args


def _parse_reviewer_models(spec: str | None):
    """Parse ``--reviewer-models`` into a list of ``ReviewerConfig``.

    Empty / None / whitespace-only input returns ``None`` so the loop
    falls through to the single-model reviewer path. Grammar matches
    :func:`_parse_proposer_models`: comma- or pipe-separated entries;
    ``@<timeout>`` after the colon-containing model string sets the
    per-leg ``llm_timeout`` (seconds). No verify flag — the reviewer
    fanout never runs ``lean_prover`` per leg.
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    from bourbaki.prover.reviewer_fanout import ReviewerConfig

    raw_entries: list[str] = []
    for chunk in spec.split("|"):
        raw_entries.extend(chunk.split(","))
    configs: list[ReviewerConfig] = []
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
                    f"--reviewer-models: bad timeout {timeout_str!r} in "
                    f"entry {entry!r}; expected float seconds"
                ) from e
            configs.append(
                ReviewerConfig(model=model, llm_timeout=llm_timeout)
            )
        else:
            configs.append(ReviewerConfig(model=entry))
    return configs or None


def _parse_proposer_models(spec: str | None, race_mode: str = "first_completed"):
    """Parse ``--proposer-models`` into a list of ``ProposerConfig``.

    Empty / None / whitespace-only input returns ``None`` (single-model
    legacy path). Splits on ``,`` and ``|`` so the flag survives a shell
    that munges either. Inside each entry, ``@`` separates the model
    string from its optional ``llm_timeout`` (seconds) — we can't use
    ``:`` since model strings already contain colons
    (``ollama-cloud:gpt-oss:120b``).

    When ``race_mode == "first_verified"``, every parsed leg gets
    ``verify_with_lean=True`` so it's eligible to win the verified race.
    Without this, the CLI would silently produce an all-downgrade fanout
    where no leg can ever win.
    """
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    from bourbaki.prover.parallel_proposer import ProposerConfig

    verify = race_mode == "first_verified"

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
            configs.append(ProposerConfig(
                model=model, llm_timeout=llm_timeout, verify_with_lean=verify,
            ))
        else:
            configs.append(ProposerConfig(model=entry, verify_with_lean=verify))
    return configs or None


def _parse_year_range(raw: str | None, logger) -> tuple[int, int] | None:
    if raw is None:
        return None
    try:
        lo_s, hi_s = raw.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi:
            raise ValueError("year-range start must be <= end")
        return (lo, hi)
    except (ValueError, AttributeError) as e:
        logger.error("Invalid --year-range %r (%s); expected e.g. 2020-2023.", raw, e)
        sys.exit(2)


def _resolve_proposer_configs(args: argparse.Namespace, logger):
    """Resolve proposer fanout configs from CLI args.

    Precedence: ``--auto-route`` is per-problem (no fixed list); when
    set, returns ``None`` here and lets ``run_putnam`` call
    ``fanout_for_problem`` for each problem. ``--smart-fanout``
    (auto-built from env) takes priority when set. ``argparse``
    already rejects pairwise combinations of these flags.  When none
    is set, returns ``None`` so the loop falls back to the legacy
    single-model path.

    ``args.auto_route`` is read via ``getattr`` so older callers that
    build a synthetic ``Namespace`` (the pinning tests written before
    --auto-route landed) keep working without needing to set every
    new field.
    """
    if getattr(args, "auto_route", False):
        # No fixed fanout: per-problem dispatch happens inside run_putnam.
        return None
    if args.smart_fanout:
        from bourbaki.config import export_api_keys
        from bourbaki.prover.default_fanout import smart_default_fanout

        export_api_keys()  # bridge Settings → env before probing keys
        verify = args.race_mode == "first_verified"
        configs = smart_default_fanout(tier=args.smart_fanout, verify_with_lean=verify)
        logger.info(
            "smart_default_fanout: selected %d legs [%s] (tier=%s)",
            len(configs),
            ", ".join(c.label or c.model for c in configs),
            args.smart_fanout,
        )
        return configs
    return _parse_proposer_models(
        args.proposer_models, race_mode=args.race_mode,
    )


def _resolve_reviewer_configs(args: argparse.Namespace, logger):
    """Resolve reviewer fanout configs from CLI args.

    Mirrors :func:`_resolve_proposer_configs` so callers can use the
    same call pattern. Currently a thin wrapper around
    :func:`_parse_reviewer_models`; lifted to a helper for symmetry
    with the miniF2F runner and to give a single place to log the
    resolved leg list.
    """
    configs = _parse_reviewer_models(args.reviewer_models)
    if configs:
        logger.info(
            "Reviewer fanout: %d legs [%s] mode=%s",
            len(configs),
            ", ".join(
                f"{c.model}@{c.llm_timeout:.0f}s" for c in configs
            ),
            args.reviewer_review_mode,
        )
    return configs


async def _run(
    args: argparse.Namespace,
    problem_ids: list[str] | None,
    year_range: tuple[int, int] | None,
    logger,
) -> None:
    from bourbaki.benchmarks.putnam import run_putnam

    proposer_configs = _resolve_proposer_configs(args, logger)
    if proposer_configs:
        logger.info(
            "Parallel proposer fanout: %d legs [%s] race_mode=%s",
            len(proposer_configs),
            ", ".join(f"{c.model}@{c.llm_timeout:.0f}s" for c in proposer_configs),
            args.race_mode,
        )
    elif args.auto_route:
        # ``run_putnam`` will call ``fanout_for_problem`` per problem and
        # log the chosen legs.  Bridge .env -> env vars first so the
        # routing's API-key probes see the same world the dotenv loader
        # would expose at import time.
        from bourbaki.config import export_api_keys

        export_api_keys()
        logger.info(
            "auto-route: enabled, per-problem fanout via "
            "fanout_for_problem (race_mode=%s)",
            args.race_mode,
        )

    reviewer_configs = _resolve_reviewer_configs(args, logger)

    result = await run_putnam(
        problem_ids=problem_ids,
        year_range=year_range,
        section_filter=args.section,
        timeout=args.timeout,
        exclude_answer=not args.include_answer,
        verify_proofs=True,
        verify_timeout=args.verify_timeout,
        use_loop=True,
        loop_max_iterations=args.max_iter,
        loop_model=args.model,
        loop_memory=args.memory,
        loop_memory_k=args.memory_k,
        loop_enable_mathlib_search=args.enable_mathlib_search,
        loop_proposer_llm_timeout=args.proposer_llm_timeout,
        loop_reviewer_llm_timeout=args.reviewer_llm_timeout,
        loop_proposer_configs=proposer_configs,
        loop_proposer_race_mode=args.race_mode,
        loop_reviewer_configs=reviewer_configs,
        loop_reviewer_review_mode=args.reviewer_review_mode,
        loop_auto_route=args.auto_route,
        pass_n=args.pass_n,
    )

    logger.info("")
    logger.info("Final results:")
    logger.info(
        "  verified: %d/%d (%.1f%%)",
        result.verified_count,
        result.total,
        100 * result.verified_count / result.total if result.total else 0.0,
    )
    logger.info(
        "  theorem-only: %d/%d",
        result.theorem_only_solved,
        result.theorem_only_total,
    )
    logger.info("  false positives: %d", result.false_positives)
    logger.info("  total time: %.1fs", result.total_time_seconds)

    # by_decade / by_section breakdown
    if result.by_decade:
        logger.info("Per-decade breakdown:")
        for decade, counts in sorted(result.by_decade.items()):
            total = counts.get("total", 0)
            solved = counts.get("solved", 0)
            pct = 100 * solved / total if total else 0.0
            logger.info("  %-6s %d/%d (%.1f%%)", decade, solved, total, pct)
    if result.by_section:
        logger.info("Per-section breakdown:")
        for sec, counts in sorted(result.by_section.items()):
            total = counts.get("total", 0)
            solved = counts.get("solved", 0)
            pct = 100 * solved / total if total else 0.0
            logger.info("  %-3s %d/%d (%.1f%%)", sec, solved, total, pct)

    logger.info("")
    logger.info(
        "SUMMARY: putnam verified=%d/%d (%.1f%%) time=%.0fs",
        result.verified_count,
        result.total,
        100 * result.verified_count / result.total if result.total else 0.0,
        result.total_time_seconds,
    )


def main() -> None:
    args = parse_args()
    logger, log_path = common.setup_logging("run_putnam_loop")
    common.require_glm_api_key(logger)

    year_range = _parse_year_range(args.year_range, logger)

    # Resolve problem IDs: explicit --ids wins, then --year-range, then default subset
    problem_ids: list[str] | None
    if args.ids:
        problem_ids = [pid.strip() for pid in args.ids.split(",") if pid.strip()]
    elif year_range is not None:
        problem_ids = None  # let run_putnam filter by year_range
    else:
        problem_ids = list(DEFAULT_DRY_RUN_IDS)

    n_est = len(problem_ids) if problem_ids else 10  # rough guess if year-range
    est_minutes = n_est * args.timeout * args.pass_n / 60.0
    est_str = f"~{est_minutes:.0f}m (n≈{n_est} × {args.timeout}s × pass_n={args.pass_n})"

    cfg = {
        "ids": "default 5-problem dry run" if problem_ids == DEFAULT_DRY_RUN_IDS else (args.ids or f"year-range={year_range}"),
        "section": args.section,
        "exclude_answer": not args.include_answer,
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
        "smart_fanout": args.smart_fanout,
        "auto_route": args.auto_route,
        "race_mode": args.race_mode,
        "reviewer_models": args.reviewer_models,
        "reviewer_review_mode": args.reviewer_review_mode,
        "verify_timeout_s": args.verify_timeout,
    }
    common.banner(
        logger,
        "run_putnam_loop",
        cfg,
        log_path,
        expected_wall_time=est_str,
        expected_result_glob=str(common.RESULTS_DIR / "YYYY-MM-DD_HHMM_putnam.json"),
    )

    if not common.PUTNAM_DIR.is_dir():
        logger.error(
            "PutnamBench checkout missing at %s — clone it first: "
            "git clone https://github.com/trishullab/PutnamBench %s",
            common.PUTNAM_DIR, common.PUTNAM_DIR,
        )
        sys.exit(3)

    asyncio.run(_run(args, problem_ids, year_range, logger))


if __name__ == "__main__":
    main()
