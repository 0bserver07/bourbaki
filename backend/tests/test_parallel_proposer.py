"""Unit tests for the multi-provider parallel-fanout proposer.

Covers:
- single-config success
- two-config race: first wins, loser is cancelled
- two-config race: first fails, second succeeds
- all-fail: first feedback to arrive is surfaced
- overall timeout: synthesized parse-failed feedback
- per-leg exceptions don't crash the fanout
- per-config llm_timeout is forwarded through run_proposer
- empty config list raises ValueError
- cancellation actually fires on loser tasks
- INFO-level logs include leg labels and winner
- empty-code ProposalMessage is treated as a loss
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from bourbaki.prover import parallel_proposer as pp_mod
from bourbaki.prover.parallel_proposer import (
    ProposerConfig,
    parallel_propose,
)
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
)


def _make_state(iteration: int = 0) -> ProverState:
    return ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=iteration,
        max_iterations=8,
    )


def _make_proposal(code: str = "theorem t : True := trivial") -> ProposalMessage:
    return ProposalMessage(
        reasoning="r",
        imports=[],
        opens=[],
        code=code,
        iteration=1,
    )


def _make_feedback(content: str = "parse failed") -> FeedbackMessage:
    return FeedbackMessage(
        kind="structured_output_parsing_failed",
        content=content,
        is_success=False,
        is_terminal=False,
    )


# ---------------------------------------------------------------------------
# 1. Single config + single success
# ---------------------------------------------------------------------------


async def test_single_config_success_returns_proposal(monkeypatch):
    """One config, one happy proposer call → returns the ProposalMessage."""

    expected = _make_proposal(code="theorem t : True := trivial")
    mock = AsyncMock(return_value=expected)
    monkeypatch.setattr(pp_mod, "run_proposer", mock)

    out = await parallel_propose(
        _make_state(), [ProposerConfig(model="glm:glm-5.1")]
    )

    assert out is expected
    mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# 2. Two configs, first wins → second is cancelled
# ---------------------------------------------------------------------------


async def test_first_wins_cancels_pending(monkeypatch):
    """The fast leg returns immediately; the slow leg should be cancelled
    before it ever returns."""

    fast_proposal = _make_proposal(code="theorem fast : True := trivial")
    slow_started = asyncio.Event()
    slow_was_cancelled = asyncio.Event()

    async def fast_proposer(state, **kwargs):
        return fast_proposal

    async def slow_proposer(state, **kwargs):
        slow_started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            slow_was_cancelled.set()
            raise
        return _make_proposal(code="never")

    call_count = {"n": 0}

    async def dispatch(state, *, model, **kwargs):
        call_count["n"] += 1
        if model == "fast:m":
            return await fast_proposer(state, **kwargs)
        return await slow_proposer(state, **kwargs)

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="fast:m"), ProposerConfig(model="slow:m")],
    )

    assert out is fast_proposal
    # Slow leg must have been cancelled (parallel_propose awaits the cancel).
    assert slow_was_cancelled.is_set()


# ---------------------------------------------------------------------------
# 3. First fails (FeedbackMessage), second succeeds
# ---------------------------------------------------------------------------


async def test_first_fails_second_succeeds_returns_second(monkeypatch):
    """First leg returns FeedbackMessage quickly, second returns a valid
    ProposalMessage after a slight delay → the proposal wins."""

    winner = _make_proposal(code="theorem won : True := trivial")
    loser_feedback = _make_feedback(content="bad json")

    async def dispatch(state, *, model, **kwargs):
        if model == "loser:m":
            return loser_feedback
        await asyncio.sleep(0.02)
        return winner

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="loser:m"), ProposerConfig(model="winner:m")],
    )

    assert out is winner


# ---------------------------------------------------------------------------
# 4. All fail → first FeedbackMessage to arrive is surfaced
# ---------------------------------------------------------------------------


async def test_all_feedback_returns_first(monkeypatch):
    """Every leg returns FeedbackMessage → the first to arrive wins."""

    first_fb = _make_feedback(content="FIRST-ARRIVAL")
    second_fb = _make_feedback(content="SECOND-ARRIVAL")

    async def dispatch(state, *, model, **kwargs):
        if model == "first:m":
            return first_fb
        await asyncio.sleep(0.05)
        return second_fb

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="first:m"), ProposerConfig(model="second:m")],
    )

    assert isinstance(out, FeedbackMessage)
    assert "FIRST-ARRIVAL" in out.content


# ---------------------------------------------------------------------------
# 5. Overall timeout fires → synthesized FeedbackMessage
# ---------------------------------------------------------------------------


async def test_overall_timeout_synthesizes_feedback(monkeypatch):
    """If overall_timeout fires before any leg completes, return a
    structured_output_parsing_failed feedback and cancel all tasks."""

    cancelled = {"count": 0}

    async def slow_proposer(state, **kwargs):
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancelled["count"] += 1
            raise
        return _make_proposal()

    monkeypatch.setattr(pp_mod, "run_proposer", slow_proposer)

    out = await parallel_propose(
        _make_state(),
        [
            ProposerConfig(model="a:m"),
            ProposerConfig(model="b:m"),
        ],
        overall_timeout=0.05,
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "overall_timeout" in out.content
    assert cancelled["count"] == 2


# ---------------------------------------------------------------------------
# 6. A config raises → other configs continue
# ---------------------------------------------------------------------------


async def test_one_leg_raises_others_continue(monkeypatch, caplog):
    """One leg raises an unexpected exception; the fanout logs it and
    returns the surviving leg's proposal."""

    winner = _make_proposal(code="theorem survived : True := trivial")

    async def dispatch(state, *, model, **kwargs):
        if model == "boom:m":
            raise RuntimeError("kaboom")
        await asyncio.sleep(0.01)
        return winner

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)

    with caplog.at_level(logging.WARNING, logger=pp_mod.logger.name):
        out = await parallel_propose(
            _make_state(),
            [ProposerConfig(model="boom:m"), ProposerConfig(model="ok:m")],
        )

    assert out is winner
    assert any("boom:m" in rec.getMessage() and "kaboom" in rec.getMessage()
               for rec in caplog.records)


async def test_all_legs_raise_synthesizes_feedback(monkeypatch):
    """When every leg raises, surface a synthesized parse-failed feedback
    rather than letting the exception bubble up."""

    async def always_raise(state, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(pp_mod, "run_proposer", always_raise)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="x:m"), ProposerConfig(model="y:m")],
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"


# ---------------------------------------------------------------------------
# 7. Per-config llm_timeout is forwarded
# ---------------------------------------------------------------------------


async def test_per_config_llm_timeout_forwarded(monkeypatch):
    """Each config's llm_timeout must reach run_proposer as a kwarg."""

    seen_timeouts: dict[str, float] = {}

    async def dispatch(state, *, model, llm_timeout, enable_mathlib_search, **kwargs):
        seen_timeouts[model] = llm_timeout
        return _make_feedback(content=f"{model}-done")

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)

    await parallel_propose(
        _make_state(),
        [
            ProposerConfig(model="fast:m", llm_timeout=15.0),
            ProposerConfig(model="slow:m", llm_timeout=180.0),
        ],
    )

    assert seen_timeouts == {"fast:m": 15.0, "slow:m": 180.0}


async def test_enable_mathlib_search_forwarded(monkeypatch):
    """``enable_mathlib_search`` must reach run_proposer for each leg."""

    seen_flags: dict[str, bool] = {}

    async def dispatch(state, *, model, enable_mathlib_search, **kwargs):
        seen_flags[model] = enable_mathlib_search
        return _make_proposal(code="theorem t : True := trivial")

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)

    await parallel_propose(
        _make_state(),
        [
            ProposerConfig(model="a:m", enable_mathlib_search=True),
            ProposerConfig(model="b:m", enable_mathlib_search=False),
        ],
    )

    assert seen_flags == {"a:m": True, "b:m": False}


# ---------------------------------------------------------------------------
# 8. Empty config list → ValueError
# ---------------------------------------------------------------------------


async def test_empty_configs_raises_value_error():
    with pytest.raises(ValueError, match="at least one"):
        await parallel_propose(_make_state(), [])


# ---------------------------------------------------------------------------
# 9. Loser tasks are actually cancelled
# ---------------------------------------------------------------------------


async def test_loser_task_observes_cancellation(monkeypatch):
    """Beyond just returning the winner, the parallel_propose function
    must actually cancel pending tasks so CancelledError fires inside
    them. Loser must observe the cancellation before parallel_propose
    returns.
    """

    winner = _make_proposal(code="theorem w : True := trivial")
    loser_cancelled = asyncio.Event()
    loser_started = asyncio.Event()

    async def dispatch(state, *, model, **kwargs):
        if model == "fast:m":
            # Give the slow leg time to actually start before we win.
            await loser_started.wait()
            return winner
        loser_started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            loser_cancelled.set()
            raise
        return _make_proposal()

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="fast:m"), ProposerConfig(model="slow:m")],
    )

    assert out is winner
    assert loser_cancelled.is_set()


# ---------------------------------------------------------------------------
# 10. INFO-level logs include leg labels and winner
# ---------------------------------------------------------------------------


async def test_logs_emit_at_info_with_labels(monkeypatch, caplog):
    """Verify INFO logs include model labels and the winner identity."""

    winner = _make_proposal(code="theorem t : True := trivial")

    async def dispatch(state, *, model, **kwargs):
        if model == "winner:m":
            return winner
        await asyncio.sleep(0.05)
        return _make_feedback()

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)

    with caplog.at_level(logging.INFO, logger=pp_mod.logger.name):
        out = await parallel_propose(
            _make_state(),
            [
                ProposerConfig(model="winner:m", label="alpha"),
                ProposerConfig(model="loser:m", label="beta"),
            ],
        )

    assert out is winner
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("launching 2 configs" in m for m in msgs)
    assert any("alpha won" in m for m in msgs)
    # The label is preferred over the raw model string when both are set.
    assert all("winner:m won" not in m for m in msgs)


# ---------------------------------------------------------------------------
# 11. Bonus: empty-code ProposalMessage is treated as a loss
# ---------------------------------------------------------------------------


async def test_empty_code_proposal_is_loss(monkeypatch):
    """A ProposalMessage with whitespace-only code is not a valid winner;
    a peer with real code should win the race."""

    empty = _make_proposal(code="   \n  ")
    real = _make_proposal(code="theorem t : True := trivial")

    async def dispatch(state, *, model, **kwargs):
        if model == "empty:m":
            return empty
        await asyncio.sleep(0.02)
        return real

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="empty:m"), ProposerConfig(model="real:m")],
    )

    assert out is real


async def test_empty_code_proposal_solo_falls_through(monkeypatch):
    """If the only leg returns an empty-code proposal, parallel_propose
    has no winner; it should surface that proposal as a 'first feedback'
    fallback path. Since an empty proposal is not a FeedbackMessage,
    document the actual behavior: we get a synthesized parse-failed.
    """

    empty = _make_proposal(code="")

    async def returns_empty(state, **kwargs):
        return empty

    monkeypatch.setattr(pp_mod, "run_proposer", returns_empty)

    out = await parallel_propose(
        _make_state(), [ProposerConfig(model="solo:m")]
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
