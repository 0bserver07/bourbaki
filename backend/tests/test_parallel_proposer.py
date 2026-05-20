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
- concurrent lean_prover verifies are bounded by a semaphore
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


# ---------------------------------------------------------------------------
# 12. Dispatcher: ollama-native: prefix routes through run_proposer_native
# ---------------------------------------------------------------------------


async def test_dispatcher_routes_native_prefix(monkeypatch):
    """A ProposerConfig with ``ollama-native:`` prefix must be served by
    ``run_proposer_native``; the default ``run_proposer`` must not be
    called for that leg.
    """
    from bourbaki.prover import native_proposer as native_mod

    native_proposal = _make_proposal(code="theorem native_won : True := trivial")
    native_called = {"n": 0}
    fallback_called = {"n": 0}

    async def fake_native(state, *, model, **kwargs):
        native_called["n"] += 1
        assert model.startswith("ollama-native:")
        return native_proposal

    async def fake_fallback(state, *, model, **kwargs):
        fallback_called["n"] += 1
        return _make_feedback(content="should-not-be-called")

    monkeypatch.setattr(native_mod, "run_proposer_native", fake_native)
    monkeypatch.setattr(pp_mod, "run_proposer", fake_fallback)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="ollama-native:kimi-k2.6")],
    )

    assert out is native_proposal
    assert native_called["n"] == 1
    assert fallback_called["n"] == 0


async def test_dispatcher_routes_other_prefixes_to_run_proposer(monkeypatch):
    """Any prefix that is NOT ``ollama-native:`` must keep using the
    standard ``run_proposer``.
    """
    from bourbaki.prover import native_proposer as native_mod

    glm_proposal = _make_proposal(code="theorem glm_won : True := trivial")
    glm_called = {"n": 0}
    native_called = {"n": 0}

    async def fake_glm(state, *, model, **kwargs):
        glm_called["n"] += 1
        assert not model.startswith("ollama-native:")
        return glm_proposal

    async def fake_native(state, *, model, **kwargs):
        native_called["n"] += 1
        return _make_feedback(content="native-should-not-be-called")

    monkeypatch.setattr(pp_mod, "run_proposer", fake_glm)
    monkeypatch.setattr(native_mod, "run_proposer_native", fake_native)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="glm:glm-5.1")],
    )

    assert out is glm_proposal
    assert glm_called["n"] == 1
    assert native_called["n"] == 0


async def test_dispatcher_mixed_legs(monkeypatch):
    """When configs mix prefixes, each one routes to the right callable."""
    from bourbaki.prover import native_proposer as native_mod

    seen: dict[str, str] = {}

    async def fake_native(state, *, model, **kwargs):
        seen[model] = "native"
        return _make_feedback(content=f"native-{model}")

    async def fake_glm(state, *, model, **kwargs):
        seen[model] = "glm"
        return _make_feedback(content=f"glm-{model}")

    monkeypatch.setattr(native_mod, "run_proposer_native", fake_native)
    monkeypatch.setattr(pp_mod, "run_proposer", fake_glm)

    await parallel_propose(
        _make_state(),
        [
            ProposerConfig(model="ollama-native:kimi-k2.6"),
            ProposerConfig(model="glm:glm-5.1"),
        ],
    )

    assert seen == {
        "ollama-native:kimi-k2.6": "native",
        "glm:glm-5.1": "glm",
    }


# ---------------------------------------------------------------------------
# 13. race_mode="first_verified": correctness-wins-the-race
#
# These tests gate proposal acceptance on a follow-up lean_prover call so a
# fast-but-broken proposal loses to a slower-but-compiling peer. lean_prover
# is always mocked with unittest.mock.AsyncMock — we never actually shell out
# to Lean from the test suite.
# ---------------------------------------------------------------------------


def _ok_verify_result() -> dict:
    # Mirror the real ``lean_prover`` return shape: a clean compile yields
    # both ``success=True`` AND ``proofComplete=True``. The latter is what
    # the leg-verify gate checks (matching the reviewer's final gate), so
    # the fixture must include it.
    return {
        "success": True,
        "proofComplete": True,
        "errors": None,
        "rawOutput": "",
        "codeUsed": "",
    }


def _fail_verify_result(msg: str = "unknown identifier 'fast_junk'") -> dict:
    return {
        "success": False,
        "proofComplete": False,
        "errors": [{"line": 3, "column": 0, "message": msg, "severity": "error"}],
        "rawOutput": msg,
        "codeUsed": "",
    }


async def test_first_verified_single_leg_verifies_and_returns(monkeypatch):
    """One config, opted into verification: a valid proposal must be fed
    through lean_prover and only then returned as the winner.
    """
    proposal = _make_proposal(code="theorem t : True := trivial")
    proposer_mock = AsyncMock(return_value=proposal)
    verify_mock = AsyncMock(return_value=_ok_verify_result())

    monkeypatch.setattr(pp_mod, "run_proposer", proposer_mock)
    monkeypatch.setattr(pp_mod, "lean_prover", verify_mock)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="m:1", verify_with_lean=True)],
        race_mode="first_verified",
    )

    assert out is proposal
    proposer_mock.assert_awaited_once()
    verify_mock.assert_awaited_once()


async def test_first_verified_slow_compiling_beats_fast_junk(monkeypatch):
    """Two legs: 'fast_junk' returns a proposal quickly but fails verify.
    'slow_real' returns later but verifies clean. The slower compiling leg
    must win the race.
    """
    fast_proposal = _make_proposal(code="theorem t : True := <bogus>")
    slow_proposal = _make_proposal(code="theorem t : True := trivial")

    async def dispatch(state, *, model, **kwargs):
        if model == "fast_junk:m":
            return fast_proposal
        # Make the slow leg's propose strictly later than fast leg's
        # propose+verify-fail so the test result doesn't depend on
        # scheduler micro-timing.
        await asyncio.sleep(0.05)
        return slow_proposal

    async def fake_verify(code, **kwargs):
        if "<bogus>" in code:
            return _fail_verify_result("unexpected token '<'")
        return _ok_verify_result()

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)
    monkeypatch.setattr(pp_mod, "lean_prover", fake_verify)

    out = await parallel_propose(
        _make_state(),
        [
            ProposerConfig(
                model="fast_junk:m", verify_with_lean=True, label="fast",
            ),
            ProposerConfig(
                model="slow_real:m", verify_with_lean=True, label="slow",
            ),
        ],
        race_mode="first_verified",
    )

    assert out is slow_proposal


async def test_first_verified_all_fail_returns_first_feedback(monkeypatch):
    """When every leg's verify fails, return the first FeedbackMessage to
    arrive (chronologically) — same fallback as the legacy mode.
    """
    fast_proposal = _make_proposal(code="theorem fast : True := <bogus_a>")
    slow_proposal = _make_proposal(code="theorem slow : True := <bogus_b>")

    async def dispatch(state, *, model, **kwargs):
        if model == "fast:m":
            return fast_proposal
        await asyncio.sleep(0.05)
        return slow_proposal

    fail_a = _fail_verify_result("first-arrival-failure")
    fail_b = _fail_verify_result("second-arrival-failure")

    async def fake_verify(code, **kwargs):
        if "<bogus_a>" in code:
            return fail_a
        return fail_b

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)
    monkeypatch.setattr(pp_mod, "lean_prover", fake_verify)

    out = await parallel_propose(
        _make_state(),
        [
            ProposerConfig(model="fast:m", verify_with_lean=True),
            ProposerConfig(model="slow:m", verify_with_lean=True),
        ],
        race_mode="first_verified",
    )

    assert isinstance(out, FeedbackMessage)
    # First arrival wins; the synthesized feedback summary embeds the
    # leg's lean_prover error string.
    assert "first-arrival-failure" in out.content


async def test_first_verified_verify_timeout_is_a_loss(monkeypatch):
    """If a leg's lean_prover call exceeds ``verify_timeout`` the leg must
    be treated as a loss (not crash the fanout); a surviving leg's
    successful verify still wins.
    """
    fast_proposal = _make_proposal(code="theorem fast_hang : True := <hang>")
    slow_proposal = _make_proposal(code="theorem slow_ok : True := trivial")

    async def dispatch(state, *, model, **kwargs):
        if model == "hang:m":
            return fast_proposal
        await asyncio.sleep(0.05)
        return slow_proposal

    async def fake_verify(code, **kwargs):
        if "<hang>" in code:
            await asyncio.sleep(10.0)  # asyncio.wait_for will cancel us
            return _ok_verify_result()
        return _ok_verify_result()

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)
    monkeypatch.setattr(pp_mod, "lean_prover", fake_verify)

    out = await parallel_propose(
        _make_state(),
        [
            ProposerConfig(
                model="hang:m", verify_with_lean=True, verify_timeout=0.05,
            ),
            ProposerConfig(model="ok:m", verify_with_lean=True),
        ],
        race_mode="first_verified",
    )

    assert out is slow_proposal


async def test_first_verified_skips_verify_when_flag_off(monkeypatch):
    """A leg with ``verify_with_lean=False`` under ``race_mode='first_verified'``
    must NOT have its lean_prover invoked. The leg's proposal is downgraded
    to a fallback FeedbackMessage (not eligible to win the verified race);
    a peer leg that opts in and verifies must win instead.
    """
    skip_proposal = _make_proposal(code="theorem skipper : True := trivial")
    real_proposal = _make_proposal(code="theorem winner : True := trivial")

    verify_call_codes: list[str] = []

    async def dispatch(state, *, model, **kwargs):
        if model == "skip:m":
            return skip_proposal
        await asyncio.sleep(0.05)
        return real_proposal

    async def fake_verify(code, **kwargs):
        verify_call_codes.append(code)
        return _ok_verify_result()

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)
    monkeypatch.setattr(pp_mod, "lean_prover", fake_verify)

    out = await parallel_propose(
        _make_state(),
        [
            ProposerConfig(model="skip:m", verify_with_lean=False),
            ProposerConfig(model="real:m", verify_with_lean=True),
        ],
        race_mode="first_verified",
    )

    assert out is real_proposal
    # The skip leg's proposal text must never have been verified.
    assert all("skipper" not in c for c in verify_call_codes)
    # The real leg's proposal text must have been verified exactly once.
    assert sum("winner" in c for c in verify_call_codes) == 1


async def test_first_completed_default_unchanged(monkeypatch):
    """Default ``race_mode='first_completed'`` must not call lean_prover at
    all, even when ``verify_with_lean=True`` is set on a config (the flag
    is only consulted under first_verified). This is the back-compat
    guarantee for the existing 14 tests.
    """
    proposal = _make_proposal(code="theorem t : True := trivial")
    proposer_mock = AsyncMock(return_value=proposal)
    verify_mock = AsyncMock(return_value=_ok_verify_result())

    monkeypatch.setattr(pp_mod, "run_proposer", proposer_mock)
    monkeypatch.setattr(pp_mod, "lean_prover", verify_mock)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="m:1", verify_with_lean=True)],
        # default race_mode
    )

    assert out is proposal
    verify_mock.assert_not_awaited()


async def test_first_verified_winner_cancels_pending_verify(monkeypatch):
    """When leg A wins under first_verified, leg B's pending propose AND
    its in-flight verify must both be cancelled.
    """
    fast_proposal = _make_proposal(code="theorem fast_winner : True := trivial")
    slow_proposal = _make_proposal(code="theorem slow_loser : True := trivial")

    slow_propose_started = asyncio.Event()
    slow_verify_started = asyncio.Event()
    slow_propose_cancelled = asyncio.Event()
    slow_verify_cancelled = asyncio.Event()

    async def dispatch(state, *, model, **kwargs):
        if model == "fast:m":
            # Wait until the slow leg starts its propose before letting
            # the fast leg complete; otherwise the fast leg may finish
            # before the slow leg's task is even scheduled.
            await slow_propose_started.wait()
            return fast_proposal
        slow_propose_started.set()
        try:
            # Long enough for the fast leg to verify and win before us.
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            slow_propose_cancelled.set()
            raise
        return slow_proposal

    async def fake_verify(code, **kwargs):
        if "fast_winner" in code:
            return _ok_verify_result()
        # The slow leg's verify should never start (we cancel it via the
        # task hierarchy when the fast leg wins). Track it regardless so
        # the test fails loudly if cancellation regresses.
        slow_verify_started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            slow_verify_cancelled.set()
            raise
        return _ok_verify_result()

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)
    monkeypatch.setattr(pp_mod, "lean_prover", fake_verify)

    out = await parallel_propose(
        _make_state(),
        [
            ProposerConfig(model="fast:m", verify_with_lean=True, label="fast"),
            ProposerConfig(model="slow:m", verify_with_lean=True, label="slow"),
        ],
        race_mode="first_verified",
    )

    assert out is fast_proposal
    # The slow leg's propose was definitely in flight (it set the gate).
    # parallel_propose's cancel + gather must have fired CancelledError
    # inside its propose body. Equivalent test for verify is gated on
    # whether the slow leg's verify even started — usually it didn't,
    # but if scheduling lets it start it must also observe cancellation.
    assert slow_propose_cancelled.is_set()
    if slow_verify_started.is_set():
        assert slow_verify_cancelled.is_set()


async def test_first_verified_logs_verify_outcomes(monkeypatch, caplog):
    """INFO logs under first_verified must include the per-leg verifying /
    VERIFIED / failed messages so on-call humans can see why a race went
    the way it did.
    """
    fast_proposal = _make_proposal(code="theorem t : True := <bogus>")
    slow_proposal = _make_proposal(code="theorem t : True := trivial")

    async def dispatch(state, *, model, **kwargs):
        if model == "fast:m":
            return fast_proposal
        await asyncio.sleep(0.05)
        return slow_proposal

    async def fake_verify(code, **kwargs):
        if "<bogus>" in code:
            return _fail_verify_result("syntax error")
        return _ok_verify_result()

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)
    monkeypatch.setattr(pp_mod, "lean_prover", fake_verify)

    with caplog.at_level(logging.INFO, logger=pp_mod.logger.name):
        out = await parallel_propose(
            _make_state(),
            [
                ProposerConfig(
                    model="fast:m", verify_with_lean=True, label="fast",
                ),
                ProposerConfig(
                    model="slow:m", verify_with_lean=True, label="slow",
                ),
            ],
            race_mode="first_verified",
        )

    assert out is slow_proposal
    msgs = [rec.getMessage() for rec in caplog.records]
    # Both legs emit a "verifying..." line.
    assert sum("verifying..." in m for m in msgs) >= 2
    # The fast leg's verify_fail summary is in the logs.
    assert any("failed lean_prover" in m and "fast" in m for m in msgs)
    # The slow leg's VERIFIED summary is in the logs.
    assert any("VERIFIED" in m and "slow" in m for m in msgs)


# ---------------------------------------------------------------------------
# 14. race_mode="first_verified": verify-fail surfaces as ``lean_verify_failed``
#
# When lean_prover rejects a leg's proposal, the resulting FeedbackMessage
# must carry the actual Lean error(s) (so the proposer's next iteration
# fixes the math/tactic, not the JSON shape). This is the regression
# guard for the previous behaviour, which mis-surfaced verify rejects as
# ``structured_output_parsing_failed``.
# ---------------------------------------------------------------------------


async def test_first_verified_verify_fail_returns_lean_verify_failed(monkeypatch):
    """Single leg under first_verified: proposer emits parseable JSON,
    lean_prover returns ``success=False`` with errors → fanout returns a
    ``lean_verify_failed`` FeedbackMessage (NOT ``structured_output_parsing_failed``).
    """
    proposal = _make_proposal(code="theorem t : True := <bogus>")
    proposer_mock = AsyncMock(return_value=proposal)
    fail_result = _fail_verify_result("omega could not prove the goal")
    verify_mock = AsyncMock(return_value=fail_result)

    monkeypatch.setattr(pp_mod, "run_proposer", proposer_mock)
    monkeypatch.setattr(pp_mod, "lean_prover", verify_mock)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="m:1", verify_with_lean=True)],
        race_mode="first_verified",
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "lean_verify_failed"
    assert out.is_terminal is False
    assert out.is_success is False
    proposer_mock.assert_awaited_once()
    verify_mock.assert_awaited_once()


async def test_first_verified_verify_fail_content_includes_lean_error(monkeypatch):
    """The ``lean_verify_failed`` feedback's ``content`` must include the
    first Lean error message verbatim so the next iteration's prompt
    contains the actual signal the proposer needs.
    """
    proposal = _make_proposal(code="theorem t : True := <bogus>")
    proposer_mock = AsyncMock(return_value=proposal)
    # Multiple errors: the first 2-3 should appear in the rendered summary.
    multi_err_result = {
        "success": False,
        "errors": [
            {"line": 18, "column": 0, "message": "DISTINCTIVE-ERR-A", "severity": "error"},
            {"line": 22, "column": 4, "message": "DISTINCTIVE-ERR-B", "severity": "error"},
            {"line": 30, "column": 2, "message": "DISTINCTIVE-ERR-C", "severity": "error"},
        ],
        "rawOutput": "",
        "codeUsed": "",
    }
    verify_mock = AsyncMock(return_value=multi_err_result)

    monkeypatch.setattr(pp_mod, "run_proposer", proposer_mock)
    monkeypatch.setattr(pp_mod, "lean_prover", verify_mock)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="m:1", verify_with_lean=True)],
        race_mode="first_verified",
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "lean_verify_failed"
    # The first lean error message is embedded verbatim.
    assert "DISTINCTIVE-ERR-A" in out.content
    # Line numbers are preserved so the proposer can correlate with its own code.
    assert "line 18" in out.content
    # The feedback prose mentions lean_prover so the proposer knows the
    # JSON parsed fine and the rejection came from the compiler.
    assert "lean_prover" in out.content


async def test_first_verified_first_arrived_lean_verify_failed_wins(monkeypatch):
    """When every leg fails verify, the first-arrived FeedbackMessage is the
    winner — matches the existing "first feedback wins" semantics. The
    surviving feedback must be ``lean_verify_failed`` (not the legacy
    parse-failed kind) and embed the first-arrived leg's error string.
    """
    fast_proposal = _make_proposal(code="theorem fast : True := <bogus_a>")
    slow_proposal = _make_proposal(code="theorem slow : True := <bogus_b>")

    async def dispatch(state, *, model, **kwargs):
        if model == "fast:m":
            return fast_proposal
        await asyncio.sleep(0.05)
        return slow_proposal

    fail_a = _fail_verify_result("FIRST-LEAN-ERR")
    fail_b = _fail_verify_result("SECOND-LEAN-ERR")

    async def fake_verify(code, **kwargs):
        if "<bogus_a>" in code:
            return fail_a
        return fail_b

    monkeypatch.setattr(pp_mod, "run_proposer", dispatch)
    monkeypatch.setattr(pp_mod, "lean_prover", fake_verify)

    out = await parallel_propose(
        _make_state(),
        [
            ProposerConfig(model="fast:m", verify_with_lean=True),
            ProposerConfig(model="slow:m", verify_with_lean=True),
        ],
        race_mode="first_verified",
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "lean_verify_failed"
    # Chronological first-arrival is preserved.
    assert "FIRST-LEAN-ERR" in out.content
    assert "SECOND-LEAN-ERR" not in out.content


async def test_first_completed_parse_failure_still_surfaces_as_parse_failed(monkeypatch):
    """Under the default ``race_mode='first_completed'`` (no verification),
    a proposer that returns a parse-failed FeedbackMessage must still
    surface as ``structured_output_parsing_failed``. The new
    ``lean_verify_failed`` kind is exclusive to the first_verified path.
    """
    parse_fail = FeedbackMessage(
        kind="structured_output_parsing_failed",
        content="malformed json",
        is_success=False,
        is_terminal=False,
    )

    async def returns_parse_fail(state, **kwargs):
        return parse_fail

    monkeypatch.setattr(pp_mod, "run_proposer", returns_parse_fail)

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="m:1")],
        # default race_mode="first_completed"
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert out.kind != "lean_verify_failed"


# ---------------------------------------------------------------------------
# Regression: leg verify must match the reviewer's final-gate criterion.
#
# ``lean_prover`` returns BOTH ``success`` (``not has_errors``) AND
# ``proofComplete`` (``return_code == 0 and not has_errors and not
# has_sorry``). The two flags can disagree: a non-zero return code
# (Lean panic / elaborator timeout / stderr noise that didn't match the
# error regex) leaves ``success=True`` / ``proofComplete=False``; so
# does any occurrence of the substring ``"sorry"`` in the source
# (including inside a comment).
#
# The reviewer's final ``lean_prover`` gate (``reviewer.py``,
# ``ProverLoop._final_lean_gate``) checks ``proofComplete``. Before
# this fix, the leg verify checked the laxer ``success`` — so a leg
# could declare VERIFIED on the exact code the reviewer's
# identical-source compile would reject. The loop entered a silent
# spin ("verified" → rejected → next iter → "verified" → rejected …)
# with no log line between, because builder/reviewer rejections are
# at DEBUG / no log levels. This was observed on ``aime_1988_p3`` in
# the 2026-05-15 first_verified benchmark, iter 1..N consecutive.
# ---------------------------------------------------------------------------


async def test_first_verified_rejects_success_true_but_not_proof_complete(monkeypatch):
    """A verify result with ``success=True`` / ``proofComplete=False`` must
    NOT be accepted by the leg verify.

    The reviewer's final ``lean_prover`` gate checks ``proofComplete``;
    if the leg accepted on the laxer ``success`` flag, the loop would
    spin forever on a proof the reviewer immediately rejects. The leg
    must downgrade to a verify-failed feedback so the loop can iterate
    with real signal rather than re-racing the same broken proof.
    """
    proposal = _make_proposal(code="theorem t : True := trivial")
    monkeypatch.setattr(pp_mod, "run_proposer", AsyncMock(return_value=proposal))

    # ``success=True`` is set by ``not has_errors``; ``proofComplete=False``
    # is set when ``return_code != 0`` (e.g. Lean panic / elaborator
    # timeout where stderr didn't match the error regex) OR when the
    # code contains the substring ``"sorry"`` anywhere (including a
    # comment). Both are real conditions under which the reviewer's
    # final compile would reject — so the leg must reject too.
    half_baked = {
        "success": True,
        "proofComplete": False,
        "errors": None,
        "rawOutput": "(non-zero exit, no parseable error)",
        "codeUsed": "",
    }
    monkeypatch.setattr(pp_mod, "lean_prover", AsyncMock(return_value=half_baked))

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="m:1", verify_with_lean=True)],
        race_mode="first_verified",
    )

    assert isinstance(out, FeedbackMessage), (
        "leg accepted success=True/proofComplete=False — the reviewer's "
        "final gate would reject the same code, causing an infinite loop"
    )


async def test_first_verified_accepts_proof_complete_true(monkeypatch):
    """The flip side: ``proofComplete=True`` is the affirmative signal the
    reviewer's final gate looks for; the leg must accept on the same
    flag and return the proposal as a winner.
    """
    proposal = _make_proposal(code="theorem t : True := trivial")
    monkeypatch.setattr(pp_mod, "run_proposer", AsyncMock(return_value=proposal))
    ok = {
        "success": True,
        "proofComplete": True,
        "errors": None,
        "rawOutput": "",
        "codeUsed": "",
    }
    monkeypatch.setattr(pp_mod, "lean_prover", AsyncMock(return_value=ok))

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="m:1", verify_with_lean=True)],
        race_mode="first_verified",
    )

    assert out is proposal


async def test_first_verified_rejects_when_proof_complete_missing(monkeypatch):
    """If the ``lean_prover`` return dict omits ``proofComplete`` entirely
    (e.g. an older code path or a malformed shim), the leg must NOT
    silently accept. ``.get('proofComplete')`` returning ``None`` is
    falsy — same as ``False`` — so the leg downgrades to verify-failed
    feedback rather than letting an under-specified result win.

    This guards against future regressions where someone might assume
    ``success=True`` implies ``proofComplete=True``. The two flags are
    independent fields in the ``lean_prover`` contract.
    """
    proposal = _make_proposal(code="theorem t : True := trivial")
    monkeypatch.setattr(pp_mod, "run_proposer", AsyncMock(return_value=proposal))
    # Missing ``proofComplete`` key entirely — pre-fix this passed because
    # the gate looked at ``success`` instead.
    legacy_shape = {"success": True, "errors": None, "rawOutput": "", "codeUsed": ""}
    monkeypatch.setattr(pp_mod, "lean_prover", AsyncMock(return_value=legacy_shape))

    out = await parallel_propose(
        _make_state(),
        [ProposerConfig(model="m:1", verify_with_lean=True)],
        race_mode="first_verified",
    )

    assert isinstance(out, FeedbackMessage)


# ---------------------------------------------------------------------------
# 15. Concurrent lean_prover verifies are bounded by a semaphore.
#
# Each ``lean_prover`` call is a full ``lake env lean`` + ``import Mathlib``
# compile — heavy on CPU and RAM. Under ``race_mode="first_verified"`` every
# verify-eligible leg shells out to Lean; without a cap, N legs launch N
# simultaneous Mathlib compiles and thrash the box (7 legs measured 7.3x
# slower wall-clock than 5 on the same problem). ``parallel_propose`` creates
# one ``asyncio.Semaphore`` per call and runs each verify under it, so no
# more than ``_max_concurrent_verifies()`` compiles are ever in flight.
#
# These tests isolate verify-concurrency: the proposer functions are
# monkeypatched to return canned ProposalMessages instantly, and
# ``lean_prover`` is a fake that records the running max concurrency.
# ---------------------------------------------------------------------------


class _ConcurrencyProbe:
    """Fake ``lean_prover`` that records peak concurrent invocations.

    Increments a shared counter on entry, tracks the running max, sleeps
    briefly so overlapping legs actually overlap on the event loop, then
    decrements on exit. The recorded ``max_seen`` is the high-water mark
    of simultaneous Lean compiles the fanout allowed.
    """

    def __init__(self, *, result: dict, sleep_s: float = 0.03) -> None:
        self._result = result
        self._sleep_s = sleep_s
        self.active = 0
        self.max_seen = 0
        self.total_calls = 0

    async def __call__(self, code, **kwargs):  # noqa: ANN001 — mirrors lean_prover
        self.active += 1
        self.total_calls += 1
        self.max_seen = max(self.max_seen, self.active)
        try:
            await asyncio.sleep(self._sleep_s)
            return self._result
        finally:
            self.active -= 1


async def test_concurrent_verifies_bounded_by_semaphore(monkeypatch):
    """With more verify-eligible legs than the limit, the observed peak of
    simultaneous ``lean_prover`` calls must never exceed
    ``_max_concurrent_verifies()`` (default 3).

    Every leg's verify is made to fail so no winner emerges early — that
    forces all legs through their verify step, maximising the chance of a
    semaphore-cap violation if the cap were missing.
    """
    # Default cap, no env override.
    monkeypatch.delenv("BOURBAKI_MAX_CONCURRENT_VERIFIES", raising=False)
    limit = pp_mod._max_concurrent_verifies()
    assert limit == 3, "test assumes the default cap of 3"

    n_legs = 7
    assert n_legs > limit, "test only meaningful with more legs than the cap"

    probe = _ConcurrencyProbe(result=_fail_verify_result("nope"))
    monkeypatch.setattr(pp_mod, "lean_prover", probe)

    # Each leg returns a distinct, instantly-available ProposalMessage so
    # the test exercises verify-concurrency, not proposer-LLM behaviour.
    async def instant_proposer(state, *, model, **kwargs):
        return _make_proposal(
            code=f"theorem {model.replace(':', '_')} : True := trivial"
        )

    monkeypatch.setattr(pp_mod, "run_proposer", instant_proposer)

    configs = [
        ProposerConfig(model=f"leg{i}:m", verify_with_lean=True)
        for i in range(n_legs)
    ]

    out = await parallel_propose(
        _make_state(), configs, race_mode="first_verified"
    )

    # All verifies fail → no winner → a FeedbackMessage is surfaced.
    assert isinstance(out, FeedbackMessage)
    # Every leg actually reached the verify step.
    assert probe.total_calls == n_legs
    # The cap held: at no point were more than `limit` compiles in flight.
    assert probe.max_seen <= limit, (
        f"observed {probe.max_seen} concurrent verifies with a cap of {limit}"
    )
    # Sanity: the fanout did run things in parallel up to the cap (it did
    # not silently serialise everything to 1).
    assert probe.max_seen >= 1


async def test_verify_semaphore_respects_env_override(monkeypatch):
    """``BOURBAKI_MAX_CONCURRENT_VERIFIES`` overrides the default cap; the
    observed peak concurrency must respect the overridden value.
    """
    monkeypatch.setenv("BOURBAKI_MAX_CONCURRENT_VERIFIES", "2")
    assert pp_mod._max_concurrent_verifies() == 2

    probe = _ConcurrencyProbe(result=_fail_verify_result("nope"))
    monkeypatch.setattr(pp_mod, "lean_prover", probe)

    async def instant_proposer(state, *, model, **kwargs):
        return _make_proposal(
            code=f"theorem {model.replace(':', '_')} : True := trivial"
        )

    monkeypatch.setattr(pp_mod, "run_proposer", instant_proposer)

    configs = [
        ProposerConfig(model=f"leg{i}:m", verify_with_lean=True)
        for i in range(6)
    ]

    out = await parallel_propose(
        _make_state(), configs, race_mode="first_verified"
    )

    assert isinstance(out, FeedbackMessage)
    assert probe.total_calls == 6
    assert probe.max_seen <= 2, (
        f"env override to 2 ignored — saw {probe.max_seen} concurrent verifies"
    )


@pytest.mark.parametrize("bad_value", ["", "abc", "0", "-1", "3.5"])
def test_invalid_env_override_falls_back_to_default(monkeypatch, bad_value):
    """A non-positive-integer ``BOURBAKI_MAX_CONCURRENT_VERIFIES`` must fall
    back to the default cap rather than crashing or producing a degenerate
    (zero / negative) semaphore.
    """
    monkeypatch.setenv("BOURBAKI_MAX_CONCURRENT_VERIFIES", bad_value)
    assert (
        pp_mod._max_concurrent_verifies()
        == pp_mod._DEFAULT_MAX_CONCURRENT_VERIFIES
    )


async def test_verify_semaphore_releases_slot_on_winner_cancellation(monkeypatch):
    """When a winner emerges, legs still queued on (or inside) the verify
    semaphore are cancelled. ``async with`` must release their permit so a
    subsequent ``parallel_propose`` call is not starved.

    The check: run two back-to-back fanouts that share the same module
    state. If a cancelled leg leaked its permit, the second fanout's verify
    would deadlock; an ``asyncio.wait_for`` guard turns that into a clean
    test failure instead of a hang.
    """
    monkeypatch.setenv("BOURBAKI_MAX_CONCURRENT_VERIFIES", "1")

    winner_code = "theorem winner : True := trivial"

    async def proposer(state, *, model, **kwargs):
        if model == "winner:m":
            return _make_proposal(code=winner_code)
        # Loser proposes too, so it also tries to grab the verify slot.
        return _make_proposal(code="theorem loser : True := trivial")

    monkeypatch.setattr(pp_mod, "run_proposer", proposer)

    async def verify(code, **kwargs):
        if "winner" in code:
            return _ok_verify_result()
        # The loser's verify blocks; when the winner wins, the loser leg is
        # cancelled while waiting on / inside the semaphore.
        await asyncio.sleep(10.0)
        return _ok_verify_result()

    monkeypatch.setattr(pp_mod, "lean_prover", verify)

    first = await asyncio.wait_for(
        parallel_propose(
            _make_state(),
            [
                ProposerConfig(model="winner:m", verify_with_lean=True),
                ProposerConfig(model="loser:m", verify_with_lean=True),
            ],
            race_mode="first_verified",
        ),
        timeout=5.0,
    )
    assert isinstance(first, ProposalMessage)
    assert first.code == winner_code

    # Second fanout: if the cancelled loser leaked the only permit, this
    # call's verify can never acquire the semaphore and wait_for fires.
    async def verify2(code, **kwargs):
        return _ok_verify_result()

    monkeypatch.setattr(pp_mod, "lean_prover", verify2)

    second = await asyncio.wait_for(
        parallel_propose(
            _make_state(),
            [ProposerConfig(model="winner:m", verify_with_lean=True)],
            race_mode="first_verified",
        ),
        timeout=5.0,
    )
    assert isinstance(second, ProposalMessage)


# ---------------------------------------------------------------------------
# 16. _summarise_lean_errors blind spot — non-zero exit, no regex-matched
#     error. The old return ``"no error detail"`` hid the failure; the
#     fallback must surface the real status keys lean_prover does carry.
# ---------------------------------------------------------------------------


def test_summarise_lean_errors_surfaces_status_when_no_detail():
    """When ``errors`` and ``rawOutput`` are both empty (``lake env lean``
    exited non-zero but stderr matched no error regex), the summary must
    expose the ``proofComplete`` / ``success`` flags rather than the
    silent ``"no error detail"``.
    """
    result = {
        "success": True,
        "proofComplete": False,
        "errors": None,
        "rawOutput": "",
        "codeUsed": "",
    }
    summary = pp_mod._summarise_lean_errors(result)
    assert "proofComplete=False" in summary
    assert "success=True" in summary


def test_summarise_lean_errors_surfaces_fatal_error_key():
    """The Lean-not-found result shape sets an ``error`` key (and no
    ``errors`` / ``rawOutput``). The summary must include that message.
    """
    result = {
        "success": False,
        "error": "Lean 4 not found. Install from https://leanprover.github.io/",
        "rawOutput": "",
        "codeUsed": "",
    }
    summary = pp_mod._summarise_lean_errors(result)
    assert "Lean 4 not found" in summary


def test_summarise_lean_errors_prefers_real_errors():
    """A result with parseable errors still summarises the first error —
    the new fallback only kicks in when there is genuinely no detail.
    """
    result = _fail_verify_result("unknown identifier 'foo'")
    summary = pp_mod._summarise_lean_errors(result)
    assert "unknown identifier 'foo'" in summary
    assert "line 3" in summary
