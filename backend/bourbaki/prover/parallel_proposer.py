"""Multi-provider parallel-fanout proposer for the prover loop.

Wraps `run_proposer` to race N model calls against the same ProverState in
parallel; returns the first valid ProposalMessage and cancels the rest.
Same prompt, different models, fastest correct answer wins.

Two race modes are supported. ``first_completed`` (default) returns the
fastest leg that emits a structurally valid ``ProposalMessage`` — same as
the original primitive. ``first_verified`` is a stricter variant: each
leg's proposal is fed through ``lean_prover`` against the assembled
standalone source before the leg can win, so a fast-but-broken proof
loses to a slower-but-compiling one. The verify step runs in parallel
with every other leg's proposer call, so the slowest verified winner
still finishes in roughly ``max(prop_s + verify_s)`` across the fanout.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal

from bourbaki.prover import feedback
from bourbaki.prover.proposer import run_proposer
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
)
from bourbaki.tools.lean_prover import lean_prover
from bourbaki.tools.proof_code_builder import assemble_standalone_proof

logger = logging.getLogger(__name__)


_NATIVE_PREFIX = "ollama-native:"

RaceMode = Literal["first_completed", "first_verified"]


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
        verify_with_lean: Only consulted under ``race_mode="first_verified"``.
            When True, the leg's :class:`ProposalMessage` is fed through
            :func:`lean_prover` against
            :func:`assemble_standalone_proof(state.preamble, proposal.code)`;
            only a ``success=True`` return makes the leg eligible to win.
            When False, the leg's proposal is recorded as a fallback but
            it cannot win the verified race — so a fast unverified leg
            never undercuts a slower compiling peer. No effect under
            ``race_mode="first_completed"``.
        verify_timeout: Per-leg cap on the ``lean_prover`` call (seconds).
            Default 180.0 covers cold-cache ``lake env lean`` with
            ``import Mathlib`` (60-180s typical) without burning the
            full single-model proposer budget on a single leg. A verify
            timeout converts the leg into a loss; the fanout continues
            with surviving legs.
    """

    model: str
    llm_timeout: float = 90.0
    enable_mathlib_search: bool = False
    label: str | None = None
    verify_with_lean: bool = False
    verify_timeout: float = 180.0


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


def _summarise_lean_errors(verify_result: dict) -> str:
    """Compact one-line summary of the first lean_prover error, for logs."""
    errors = verify_result.get("errors") or []
    if errors:
        first = errors[0]
        msg = str(first.get("message", "")).replace("\n", " ")
        return f"line {first.get('line', '?')}: {msg[:120]}"
    raw = (verify_result.get("rawOutput") or "").strip().replace("\n", " ")
    return raw[:120] if raw else "no error detail"


async def parallel_propose(
    state: ProverState,
    configs: list[ProposerConfig],
    *,
    overall_timeout: float | None = None,
    race_mode: RaceMode = "first_completed",
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
        race_mode: ``"first_completed"`` (default) keeps the original
            behaviour — fastest valid ProposalMessage wins.
            ``"first_verified"`` gates each leg on a follow-up
            :func:`lean_prover` call against
            :func:`assemble_standalone_proof(state.preamble, proposal.code)`;
            only legs that compile clean are eligible to win.
            Speed-wins-the-race becomes correctness-wins-the-race.

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
        "parallel_propose: launching %d configs [%s] race_mode=%s",
        len(configs),
        ", ".join(_label_for(c) for c in configs),
        race_mode,
    )

    # First-feedback bookkeeping is shared across modes; in verified mode
    # we also use it to surface a failed-verify FeedbackMessage when no
    # leg compiles. The leg pipelines mutate these from inside the task
    # body via closure, so the outer ``_await_all`` only needs the
    # winner / no-winner branch.
    first_feedback_state: dict[str, FeedbackMessage | str | None] = {
        "msg": None,
        "label": None,
    }
    first_feedback_lock = asyncio.Lock()

    async def _maybe_record_first_feedback(
        msg: FeedbackMessage, label: str
    ) -> None:
        # asyncio.Lock guards the check-then-set against two legs racing
        # to record their feedback chronologically. Without it, the
        # ``all-fail`` fallback's "first arrival" guarantee leaks under
        # contention even though the event loop is single-threaded — two
        # tasks can suspend between the check and the assignment.
        async with first_feedback_lock:
            if first_feedback_state["msg"] is None:
                first_feedback_state["msg"] = msg
                first_feedback_state["label"] = label

    async def _verify_proposal(
        proposal: ProposalMessage, cfg: ProposerConfig, label: str, prop_s: float
    ) -> ProposalMessage | FeedbackMessage:
        """Run lean_prover against (preamble, proposal.code). Returns the
        proposal unchanged on success, or a structured FeedbackMessage
        on compile-fail / timeout so the all-fail fallback has signal.
        """
        source = assemble_standalone_proof(state.preamble, proposal.code)
        logger.info(
            "parallel_propose[verified]: leg %s proposed in %.1fs, verifying...",
            label,
            prop_s,
        )
        verify_start = time.monotonic()
        try:
            verify_result = await asyncio.wait_for(
                lean_prover(source, timeout=int(cfg.verify_timeout)),
                timeout=cfg.verify_timeout,
            )
        except asyncio.TimeoutError:
            verify_s = time.monotonic() - verify_start
            logger.info(
                "parallel_propose[verified]: leg %s proposed in %.1fs but "
                "lean_prover timed out after %.1fs",
                label,
                prop_s,
                verify_s,
            )
            return feedback.structured_output_parsing_failed(
                f"leg {label} verify timed out after {verify_s:.0f}s"
            )
        except Exception as exc:  # noqa: BLE001
            verify_s = time.monotonic() - verify_start
            logger.warning(
                "parallel_propose[verified]: leg %s verify raised after %.1fs: %s",
                label,
                verify_s,
                exc,
            )
            return feedback.structured_output_parsing_failed(
                f"leg {label} verify raised: {exc}"
            )

        verify_s = time.monotonic() - verify_start
        if verify_result.get("success"):
            logger.info(
                "parallel_propose[verified]: %s VERIFIED after %.1fs "
                "(proposed in %.1fs, verified in %.1fs)",
                label,
                prop_s + verify_s,
                prop_s,
                verify_s,
            )
            return proposal

        err_summary = _summarise_lean_errors(verify_result)
        logger.info(
            "parallel_propose[verified]: leg %s proposed in %.1fs but "
            "failed lean_prover (%s)",
            label,
            prop_s,
            err_summary,
        )
        return feedback.structured_output_parsing_failed(
            f"leg {label} verify failed: {err_summary}"
        )

    async def _run_one(cfg: ProposerConfig) -> ProposalMessage | FeedbackMessage:
        """Full per-leg pipeline: propose, then (if verified mode + opted in) verify.

        Returns whichever message ultimately disposes of the leg:
        ``ProposalMessage`` if it's a winner-eligible proposal,
        ``FeedbackMessage`` otherwise. The outer ``_await_all`` treats
        Feedback returns as losses (and records the first one) and
        ProposalMessage returns as winners (modulo the empty-code guard
        in ``_is_valid_proposal``).
        """
        label = _label_for(cfg)
        leg_start = time.monotonic()
        proposer_fn = _proposer_for(cfg.model)
        proposal_or_fb = await proposer_fn(
            state,
            model=cfg.model,
            llm_timeout=cfg.llm_timeout,
            enable_mathlib_search=cfg.enable_mathlib_search,
        )
        prop_s = time.monotonic() - leg_start

        if not _is_valid_proposal(proposal_or_fb):
            # Empty-code or FeedbackMessage — record as fallback and end the leg.
            if isinstance(proposal_or_fb, FeedbackMessage):
                await _maybe_record_first_feedback(proposal_or_fb, label)
            return proposal_or_fb

        assert isinstance(proposal_or_fb, ProposalMessage)
        if race_mode != "first_verified" or not cfg.verify_with_lean:
            # Either we're in first_completed mode, or this leg opted
            # out of verification. Under first_verified, a non-verifying
            # leg can't win — we downgrade it to a fallback Feedback so
            # _is_valid_proposal returns False on the outer side.
            # Without the downgrade, a fast unverified leg would beat a
            # slow compiling peer under first_verified — defeating the
            # whole point of the mode.
            if race_mode == "first_verified" and not cfg.verify_with_lean:
                downgrade = feedback.structured_output_parsing_failed(
                    f"leg {label} skipped verification "
                    f"(verify_with_lean=False under first_verified)"
                )
                await _maybe_record_first_feedback(downgrade, label)
                return downgrade
            return proposal_or_fb

        verified = await _verify_proposal(proposal_or_fb, cfg, label, prop_s)
        if isinstance(verified, FeedbackMessage):
            await _maybe_record_first_feedback(verified, label)
        return verified

    tasks: dict[asyncio.Task[ProposalMessage | FeedbackMessage], ProposerConfig] = {
        asyncio.create_task(_run_one(cfg), name=f"proposer:{_label_for(cfg)}"): cfg
        for cfg in configs
    }
    start_times: dict[asyncio.Task[ProposalMessage | FeedbackMessage], float] = {
        t: started for t in tasks
    }

    winner: ProposalMessage | None = None
    winner_label: str | None = None

    pending: set[asyncio.Task[ProposalMessage | FeedbackMessage]] = set(tasks)

    async def _await_all() -> None:
        nonlocal winner, winner_label, pending
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

                # Loss — _run_one already recorded any FeedbackMessage as
                # the first-feedback fallback if it was the first one in.
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

    first_feedback = first_feedback_state["msg"]
    first_feedback_label = first_feedback_state["label"]
    if isinstance(first_feedback, FeedbackMessage):
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
