"""Multi-provider parallel-fanout proposer for the prover loop.

Wraps `run_proposer` to race N model calls against the same ProverState in
parallel; returns the first valid ProposalMessage and cancels the rest.
Same prompt, different models, fastest correct answer wins.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from bourbaki.prover import feedback
from bourbaki.prover.proposer import run_proposer
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
)

logger = logging.getLogger(__name__)


_NATIVE_PREFIX = "ollama-native:"


def _proposer_for(model: str):
    """Return the proposer callable matching a model string's routing prefix.

    The ``ollama-native:`` prefix routes through the native /api/chat
    adapter (see :mod:`bourbaki.prover.native_proposer`); everything else
    flows through :func:`bourbaki.prover.proposer.run_proposer` and the
    standard pydantic_ai providers.

    Imports the native module lazily so its httpx-based code path isn't
    pulled in by callers who never use it.
    """
    if model.startswith(_NATIVE_PREFIX):
        from bourbaki.prover.native_proposer import run_proposer_native
        return run_proposer_native
    return run_proposer


@dataclass(frozen=True)
class ProposerConfig:
    """One proposer leg in a parallel fanout.

    Attributes:
        model: Pydantic AI model string forwarded to ``run_proposer``
            (e.g. ``"glm:glm-5.1"``, ``"ollama-cloud:gpt-oss:120b"``).
        llm_timeout: Per-call cap forwarded as ``run_proposer``'s
            ``llm_timeout``. Independent across configs so slower
            providers can be given more headroom.
        enable_mathlib_search: When True, ``run_proposer`` will register
            the built-in ``mathlib_search`` tool on its agent.
        label: Optional human-readable label used in log lines; defaults
            to the model string at log time.
    """

    model: str
    llm_timeout: float = 90.0
    enable_mathlib_search: bool = False
    label: str | None = None


def _label_for(cfg: ProposerConfig) -> str:
    return cfg.label or cfg.model


def _is_valid_proposal(msg: ProposalMessage | FeedbackMessage) -> bool:
    """A proposal is valid iff it's a ProposalMessage with non-empty code.

    ``run_proposer`` already validates the structured output schema, but
    GLM-5.1 occasionally returns an empty ``updated_theorem`` string that
    parses successfully. Treat that as a loss so a peer with real code
    wins the race.
    """
    return isinstance(msg, ProposalMessage) and bool(msg.code.strip())


async def parallel_propose(
    state: ProverState,
    configs: list[ProposerConfig],
    *,
    overall_timeout: float | None = None,
) -> ProposalMessage | FeedbackMessage:
    """Race multiple proposers; return the first valid ProposalMessage.

    Spawns one task per config, all invoking :func:`run_proposer` on the
    same :class:`ProverState` in parallel. As tasks complete, the first
    valid :class:`ProposalMessage` (parses, code is non-empty) wins: all
    pending tasks are cancelled and that proposal is returned. If every
    config returns a :class:`FeedbackMessage`, the first feedback to
    arrive is returned (richest error signal for the loop's memory).

    Args:
        state: Loop state passed through to each ``run_proposer`` call.
        configs: Non-empty list of proposer legs to race.
        overall_timeout: Optional wall-clock cap across the whole fanout.
            If no winner has emerged by then, all tasks are cancelled and
            a synthesized timeout feedback is returned via
            :func:`feedback.structured_output_parsing_failed` (no
            dedicated factory exists; we follow the same convention the
            single-model proposer uses for its own timeout path).

    Returns:
        A :class:`ProposalMessage` if any leg wins, otherwise a
        :class:`FeedbackMessage`. Never raises — task exceptions are
        logged and treated as failed attempts.

    Raises:
        ValueError: If ``configs`` is empty.
    """

    if not configs:
        raise ValueError("parallel_propose requires at least one ProposerConfig")

    started = time.monotonic()
    logger.info(
        "parallel_propose: launching %d configs [%s]",
        len(configs),
        ", ".join(_label_for(c) for c in configs),
    )

    async def _run_one(cfg: ProposerConfig) -> ProposalMessage | FeedbackMessage:
        proposer_fn = _proposer_for(cfg.model)
        return await proposer_fn(
            state,
            model=cfg.model,
            llm_timeout=cfg.llm_timeout,
            enable_mathlib_search=cfg.enable_mathlib_search,
        )

    tasks: dict[asyncio.Task[ProposalMessage | FeedbackMessage], ProposerConfig] = {
        asyncio.create_task(_run_one(cfg), name=f"proposer:{_label_for(cfg)}"): cfg
        for cfg in configs
    }
    start_times: dict[asyncio.Task[ProposalMessage | FeedbackMessage], float] = {
        t: started for t in tasks
    }

    winner: ProposalMessage | None = None
    winner_label: str | None = None
    first_feedback: FeedbackMessage | None = None
    first_feedback_label: str | None = None

    pending: set[asyncio.Task[ProposalMessage | FeedbackMessage]] = set(tasks)

    async def _await_all() -> None:
        nonlocal winner, winner_label, first_feedback, first_feedback_label, pending
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
                    # Cooperative cancellation — somebody else won.
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "parallel_propose: leg %s raised after %.1fs: %s",
                        label,
                        duration,
                        exc,
                    )
                    continue

                if _is_valid_proposal(result):
                    assert isinstance(result, ProposalMessage)
                    winner = result
                    winner_label = label
                    logger.info(
                        "parallel_propose: %s won after %.1fs",
                        label,
                        duration,
                    )
                    break

                if isinstance(result, FeedbackMessage) and first_feedback is None:
                    first_feedback = result
                    first_feedback_label = label

                snippet = (
                    result.content[:120].replace("\n", " ")
                    if isinstance(result, FeedbackMessage)
                    else "empty-code proposal"
                )
                logger.info(
                    "parallel_propose: leg %s lost after %.1fs (%s)",
                    label,
                    duration,
                    snippet,
                )

    try:
        if overall_timeout is not None:
            await asyncio.wait_for(_await_all(), timeout=overall_timeout)
        else:
            await _await_all()
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started
        logger.warning(
            "parallel_propose: overall_timeout=%.1fs exceeded (elapsed=%.1fs); "
            "cancelling %d pending leg(s)",
            overall_timeout if overall_timeout is not None else -1,
            elapsed,
            len(pending),
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return feedback.structured_output_parsing_failed(
            f"parallel_propose exceeded overall_timeout of "
            f"{overall_timeout:.0f}s across "
            f"{len(configs)} configs"
        )

    if winner is not None:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        logger.info(
            "parallel_propose: returning ProposalMessage from %s "
            "(iteration=%d, cancelled %d peer(s))",
            winner_label,
            winner.iteration,
            len(pending),
        )
        return winner

    if first_feedback is not None:
        logger.info(
            "parallel_propose: all %d legs returned feedback; surfacing first "
            "(%s, kind=%s)",
            len(configs),
            first_feedback_label,
            first_feedback.kind,
        )
        return first_feedback

    # Every task raised — synthesize feedback so the loop can continue.
    logger.warning(
        "parallel_propose: every leg raised; synthesizing parse-failed feedback"
    )
    return feedback.structured_output_parsing_failed(
        f"all {len(configs)} parallel proposer legs raised exceptions"
    )
