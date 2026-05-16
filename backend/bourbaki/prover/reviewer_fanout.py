"""Multi-model reviewer fanout.

Wraps run_reviewer to race N model calls against the same proposal,
combining their decisions via configurable strategy (first_approved
or all_must_approve). Useful when single-model reviewer is too lax or
when borderline proofs deserve stricter gating.

The ``lean_prover`` final gate (the ground-truth standalone compile) is
intentionally NOT run inside the fanout — each leg invokes
:func:`run_reviewer` with ``decide_only=True`` so it returns the LLM's
approve/reject judgement without touching Lean. Running ``lean_prover``
N times in parallel for the same proposal would waste compile time
linearly with the leg count; the coordinator that wires the fanout into
:class:`ProverLoop._reviewer` runs ``lean_prover`` exactly once after
the ensemble agrees.

Conventions:

- An empty ``configs`` list raises ``ValueError`` — there's no sensible
  default behaviour for "review with zero reviewers".
- A single-config fanout is well-defined: it degrades to a single
  ``run_reviewer(..., decide_only=True)`` call, returns its result, and
  the coordinator runs the final ``lean_prover`` gate. Useful when a
  caller wants the fanout API without splitting across models.
- Either mode never raises out of ``fanout_review`` — task exceptions
  are logged and treated as rejections, mirroring the single-model
  reviewer's "errors collapse into review_rejected" contract.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal

from bourbaki.prover import feedback
from bourbaki.prover.reviewer import run_reviewer
from bourbaki.prover.state import FeedbackMessage, ProverState

logger = logging.getLogger(__name__)


ReviewMode = Literal["first_approved", "all_must_approve"]


@dataclass(frozen=True)
class ReviewerConfig:
    """One reviewer leg in a multi-model fanout.

    Attributes:
        model: Pydantic AI model string forwarded to
            :func:`run_reviewer` (e.g. ``"glm:glm-5.1"``,
            ``"anthropic:claude-sonnet-4-5"``).
        llm_timeout: Per-call cap forwarded as :func:`run_reviewer`'s
            ``llm_timeout``. Independent across configs so a slow
            provider can be given more headroom without bumping the
            shared default.
        label: Optional human-readable label used in log lines; defaults
            to the model string at log time.
    """

    model: str
    llm_timeout: float = 60.0
    label: str | None = None


def _label_for(cfg: ReviewerConfig) -> str:
    return cfg.label or cfg.model


def _is_approval(msg: FeedbackMessage) -> bool:
    """A leg approves iff it returned a ``review_approved`` feedback."""
    return msg.kind == "review_approved"


async def fanout_review(
    state: ProverState,
    configs: list[ReviewerConfig],
    *,
    mode: ReviewMode = "first_approved",
    overall_timeout: float | None = None,
) -> FeedbackMessage:
    """Run N reviewers in parallel; combine their decisions via ``mode``.

    Each leg invokes :func:`run_reviewer` with ``decide_only=True`` so
    the LLM structured-output step runs but the per-leg
    :func:`lean_prover` compile is skipped. The caller (typically
    :meth:`ProverLoop._reviewer`) is responsible for running
    :func:`lean_prover` exactly once after this function returns an
    approval — N parallel compiles for the same proposal would burn
    compile time linearly with leg count.

    Args:
        state: Loop state passed through to each ``run_reviewer`` call.
        configs: Non-empty list of reviewer legs to race. With a single
            entry the fanout degrades cleanly to one ``run_reviewer``
            call — both modes return that leg's decision unchanged.
        mode: Combination strategy.

            ``"first_approved"`` races the legs with
            ``asyncio.wait(FIRST_COMPLETED)``. The first leg that
            returns ``review_approved`` wins and pending legs are
            cancelled. If every leg rejects, the first rejection to
            arrive chronologically is returned. Fast, looser.

            ``"all_must_approve"`` waits for ALL legs (via
            ``asyncio.gather``) and returns approval ONLY if every leg
            approved. If any leg rejects, the FIRST non-approval
            encountered is surfaced (preserving the strictest signal
            for the proposer's memory). Slower, stricter.
        overall_timeout: Optional wall-clock cap across the whole
            fanout. If it fires before a decision can be made, pending
            tasks are cancelled and a synthesized ``review_rejected``
            is returned. ``None`` (default) means no outer cap — per-leg
            ``llm_timeout`` governs.

    Returns:
        A :class:`FeedbackMessage` from :mod:`bourbaki.prover.feedback`.
        Approval will be a ``review_approved`` kind; non-approval a
        ``review_rejected``. Never raises — task exceptions are logged
        and treated as rejections.

    Raises:
        ValueError: If ``configs`` is empty.
    """
    if not configs:
        raise ValueError("fanout_review requires at least one ReviewerConfig")

    started = time.monotonic()
    logger.info(
        "fanout_review: launching %d configs [%s] mode=%s",
        len(configs),
        ", ".join(_label_for(c) for c in configs),
        mode,
    )

    async def _run_one(cfg: ReviewerConfig) -> FeedbackMessage:
        """One reviewer leg. Exceptions collapse into a rejection so the
        ensemble fallbacks have a uniform FeedbackMessage to inspect.
        """
        label = _label_for(cfg)
        leg_start = time.monotonic()
        try:
            result = await run_reviewer(
                state,
                model=cfg.model,
                llm_timeout=cfg.llm_timeout,
                decide_only=True,
            )
        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            dur = time.monotonic() - leg_start
            logger.warning(
                "fanout_review: leg %s raised after %.1fs: %s",
                label,
                dur,
                exc,
            )
            return feedback.review_rejected(
                f"reviewer leg {label} raised: {exc}"
            )
        dur = time.monotonic() - leg_start
        verdict = "approved" if _is_approval(result) else "rejected"
        logger.info(
            "fanout_review: leg %s %s after %.1fs",
            label,
            verdict,
            dur,
        )
        return result

    if mode == "first_approved":
        return await _first_approved(
            state, configs, _run_one, started, overall_timeout,
        )
    return await _all_must_approve(
        state, configs, _run_one, started, overall_timeout,
    )


async def _first_approved(
    state: ProverState,
    configs: list[ReviewerConfig],
    run_one,
    started: float,
    overall_timeout: float | None,
) -> FeedbackMessage:
    """FIRST_COMPLETED race: first leg returning approval wins.

    Mirrors :func:`bourbaki.prover.parallel_proposer.parallel_propose`'s
    structure. If every leg rejects, the first rejection chronologically
    is returned so the proposer's memory has the freshest critique.
    """
    tasks: dict[asyncio.Task[FeedbackMessage], ReviewerConfig] = {
        asyncio.create_task(run_one(cfg), name=f"reviewer:{_label_for(cfg)}"): cfg
        for cfg in configs
    }
    start_times: dict[asyncio.Task[FeedbackMessage], float] = {
        t: started for t in tasks
    }

    winner: FeedbackMessage | None = None
    first_rejection: FeedbackMessage | None = None
    pending: set[asyncio.Task[FeedbackMessage]] = set(tasks)

    async def _await_until_approved() -> None:
        nonlocal winner, first_rejection, pending
        while pending and winner is None:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                cfg = tasks[task]
                label = _label_for(cfg)
                duration = time.monotonic() - start_times[task]
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "fanout_review[first_approved]: leg %s raised after "
                        "%.1fs: %s",
                        label, duration, exc,
                    )
                    continue

                if _is_approval(result):
                    winner = result
                    logger.info(
                        "fanout_review[first_approved]: %s approved first "
                        "after %.1fs",
                        label, duration,
                    )
                    break
                if first_rejection is None:
                    first_rejection = result

    try:
        if overall_timeout is not None:
            await asyncio.wait_for(
                _await_until_approved(), timeout=overall_timeout,
            )
        else:
            await _await_until_approved()
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started
        logger.warning(
            "fanout_review[first_approved]: overall_timeout=%.1fs exceeded "
            "(elapsed=%.1fs); cancelling %d pending leg(s)",
            overall_timeout if overall_timeout is not None else -1.0,
            elapsed,
            len(pending),
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return feedback.review_rejected(
            f"fanout_review exceeded overall_timeout of "
            f"{overall_timeout:.0f}s across {len(configs)} configs"
        )

    if winner is not None:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return winner

    if first_rejection is not None:
        logger.info(
            "fanout_review[first_approved]: all %d legs rejected; "
            "surfacing first rejection",
            len(configs),
        )
        return first_rejection

    # Every task raised — synthesize a rejection so the loop can continue.
    logger.warning(
        "fanout_review[first_approved]: every leg raised; synthesizing "
        "review_rejected feedback"
    )
    return feedback.review_rejected(
        f"all {len(configs)} reviewer legs raised exceptions"
    )


async def _all_must_approve(
    state: ProverState,
    configs: list[ReviewerConfig],
    run_one,
    started: float,
    overall_timeout: float | None,
) -> FeedbackMessage:
    """Wait for ALL legs; return approval only if every leg approved.

    If any leg rejects or errors, surface the FIRST non-approval in the
    config list (not in completion order — config order keeps logs
    stable across runs and matches what a human operator would expect
    when listing reviewers). Approval returns the FIRST leg's
    approval message so the loop sees a single coherent ``content``
    payload rather than concatenated reasonings.
    """
    tasks: list[asyncio.Task[FeedbackMessage]] = [
        asyncio.create_task(run_one(cfg), name=f"reviewer:{_label_for(cfg)}")
        for cfg in configs
    ]

    async def _gather_all() -> list[FeedbackMessage]:
        # ``return_exceptions=False`` — ``run_one`` already absorbed
        # leg-level exceptions into FeedbackMessage rejections, so a
        # raise here is genuinely unexpected and should bubble.
        return await asyncio.gather(*tasks)

    try:
        if overall_timeout is not None:
            results = await asyncio.wait_for(
                _gather_all(), timeout=overall_timeout,
            )
        else:
            results = await _gather_all()
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started
        logger.warning(
            "fanout_review[all_must_approve]: overall_timeout=%.1fs exceeded "
            "(elapsed=%.1fs); cancelling %d task(s)",
            overall_timeout if overall_timeout is not None else -1.0,
            elapsed,
            len(tasks),
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return feedback.review_rejected(
            f"fanout_review[all_must_approve] exceeded overall_timeout of "
            f"{overall_timeout:.0f}s across {len(configs)} configs"
        )

    # First non-approval wins (in config-list order). If every leg
    # approved, surface the first leg's approval so the content payload
    # is deterministic.
    for idx, result in enumerate(results):
        if not _is_approval(result):
            label = _label_for(configs[idx])
            logger.info(
                "fanout_review[all_must_approve]: leg %s rejected; "
                "ensemble rejects",
                label,
            )
            return result

    label = _label_for(configs[0])
    logger.info(
        "fanout_review[all_must_approve]: all %d legs approved; surfacing "
        "%s's approval",
        len(configs), label,
    )
    return results[0]
