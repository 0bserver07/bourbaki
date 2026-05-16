"""Unit tests for the multi-model reviewer fanout.

Covers:
- single config + first_approved + leg approves → review_approved
- single config + leg rejects → review_rejected
- two configs first_approved, first approves → approved (second cancelled)
- two configs first_approved, both reject → first rejection chronologically
- two configs all_must_approve, both approve → approved
- two configs all_must_approve, one rejects → rejected
- per-leg exception → treated as rejection, fanout continues
- overall_timeout fires → synthesized rejection feedback
- empty config list → ValueError
- ProverLoop integration: reviewer_configs set → fanout_review called and
  lean_prover gate runs ONCE after approval (not per leg)

``run_reviewer`` is monkey-patched on the fanout module so no LLM call
or Lean shell-out happens. ``lean_prover`` is also patched on the
``prover`` module for the integration test.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from bourbaki.prover import feedback as fb
from bourbaki.prover import prover as prover_mod
from bourbaki.prover import reviewer_fanout as rf_mod
from bourbaki.prover.prover import ProverConfig, ProverLoop
from bourbaki.prover.reviewer_fanout import (
    ReviewerConfig,
    fanout_review,
)
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_state(code: str = "theorem t : 1 = 1 := rfl") -> ProverState:
    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : 1 = 1 := sorry",
        preamble="import Mathlib\nopen Nat",
    )
    state.last_proposal = ProposalMessage(
        reasoning="rfl",
        code=code,
        iteration=1,
    )
    return state


def _approved(reasoning: str = "clean") -> FeedbackMessage:
    return fb.review_approved(reasoning)


def _rejected(reasoning: str = "bad") -> FeedbackMessage:
    return fb.review_rejected(reasoning)


# ---------------------------------------------------------------------------
# 1. Single config, leg approves → review_approved
# ---------------------------------------------------------------------------


async def test_single_config_first_approved_leg_approves(monkeypatch):
    """One config in ``first_approved`` mode whose leg approves must
    yield the approval message verbatim — no lean_prover, no rejection.
    """
    ok = _approved("looks great")
    run_mock = AsyncMock(return_value=ok)
    monkeypatch.setattr(rf_mod, "run_reviewer", run_mock)

    out = await fanout_review(
        _make_state(),
        [ReviewerConfig(model="glm:glm-5.1")],
        mode="first_approved",
    )

    assert out.kind == "review_approved"
    assert "looks great" in out.content
    run_mock.assert_awaited_once()
    # The fanout must always pass ``decide_only=True`` so the per-leg
    # lean_prover call is skipped.
    _, kwargs = run_mock.await_args
    assert kwargs.get("decide_only") is True


# ---------------------------------------------------------------------------
# 2. Single config, leg rejects → review_rejected
# ---------------------------------------------------------------------------


async def test_single_config_leg_rejects(monkeypatch):
    """A single leg that returns ``review_rejected`` propagates through."""
    bad = _rejected("signature changed")
    monkeypatch.setattr(rf_mod, "run_reviewer", AsyncMock(return_value=bad))

    out = await fanout_review(
        _make_state(),
        [ReviewerConfig(model="glm:glm-5.1")],
        mode="first_approved",
    )

    assert out.kind == "review_rejected"
    assert "signature changed" in out.content


# ---------------------------------------------------------------------------
# 3. Two configs first_approved, first approves → second cancelled
# ---------------------------------------------------------------------------


async def test_first_approved_two_configs_first_wins_cancels_second(monkeypatch):
    """Race semantics: the fast approving leg wins, the slow leg is
    cancelled before it can return."""
    approval = _approved("fast yes")
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()

    async def dispatch(state, *, model, llm_timeout, decide_only):
        if model == "fast:m":
            return approval
        slow_started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            slow_cancelled.set()
            raise
        return _approved("never gets here")

    monkeypatch.setattr(rf_mod, "run_reviewer", dispatch)

    out = await fanout_review(
        _make_state(),
        [
            ReviewerConfig(model="fast:m", label="fast"),
            ReviewerConfig(model="slow:m", label="slow"),
        ],
        mode="first_approved",
    )

    assert out.kind == "review_approved"
    assert "fast yes" in out.content
    assert slow_started.is_set()
    assert slow_cancelled.is_set()


# ---------------------------------------------------------------------------
# 4. Two configs first_approved, both reject → first chronologically wins
# ---------------------------------------------------------------------------


async def test_first_approved_both_reject_returns_first_chronologically(
    monkeypatch,
):
    """When every leg rejects, the FIRST rejection to arrive wins —
    fresher critique reaches the proposer's memory."""
    fast_reject = _rejected("FAST-REJECT")
    slow_reject = _rejected("SLOW-REJECT")

    async def dispatch(state, *, model, llm_timeout, decide_only):
        if model == "fast:m":
            return fast_reject
        await asyncio.sleep(0.05)
        return slow_reject

    monkeypatch.setattr(rf_mod, "run_reviewer", dispatch)

    out = await fanout_review(
        _make_state(),
        [
            ReviewerConfig(model="fast:m"),
            ReviewerConfig(model="slow:m"),
        ],
        mode="first_approved",
    )

    assert out.kind == "review_rejected"
    assert "FAST-REJECT" in out.content
    assert "SLOW-REJECT" not in out.content


# ---------------------------------------------------------------------------
# 5. Two configs all_must_approve, both approve → approved
# ---------------------------------------------------------------------------


async def test_all_must_approve_both_approve_returns_first_approval(
    monkeypatch,
):
    """When every leg approves, the first leg (config-list order) wins."""
    a_approve = _approved("a says ok")
    b_approve = _approved("b says ok")

    async def dispatch(state, *, model, llm_timeout, decide_only):
        if model == "a:m":
            return a_approve
        return b_approve

    monkeypatch.setattr(rf_mod, "run_reviewer", dispatch)

    out = await fanout_review(
        _make_state(),
        [
            ReviewerConfig(model="a:m", label="A"),
            ReviewerConfig(model="b:m", label="B"),
        ],
        mode="all_must_approve",
    )

    assert out.kind == "review_approved"
    # First leg in config-list order wins the approval surface so logs
    # stay deterministic regardless of completion order.
    assert "a says ok" in out.content


# ---------------------------------------------------------------------------
# 6. Two configs all_must_approve, one rejects → rejected
# ---------------------------------------------------------------------------


async def test_all_must_approve_one_rejects_returns_rejection(monkeypatch):
    """If any single leg rejects under ``all_must_approve``, the ensemble
    rejects. Surfaces the rejecting leg's payload (first non-approval in
    config-list order) so the loop's memory sees actionable critique."""
    a_approve = _approved("a is ok")
    b_reject = _rejected("b found a sorry")

    async def dispatch(state, *, model, llm_timeout, decide_only):
        if model == "a:m":
            return a_approve
        return b_reject

    monkeypatch.setattr(rf_mod, "run_reviewer", dispatch)

    out = await fanout_review(
        _make_state(),
        [
            ReviewerConfig(model="a:m"),
            ReviewerConfig(model="b:m"),
        ],
        mode="all_must_approve",
    )

    assert out.kind == "review_rejected"
    assert "b found a sorry" in out.content


# ---------------------------------------------------------------------------
# 7. Per-leg exception → treated as rejection, fanout continues
# ---------------------------------------------------------------------------


async def test_one_leg_raises_is_treated_as_rejection(monkeypatch, caplog):
    """If a leg raises an unexpected exception, the fanout logs it and
    treats the leg as a rejection. Surviving legs still drive the
    outcome."""
    survivor = _approved("survived")

    async def dispatch(state, *, model, llm_timeout, decide_only):
        if model == "boom:m":
            raise RuntimeError("boom")
        await asyncio.sleep(0.01)
        return survivor

    monkeypatch.setattr(rf_mod, "run_reviewer", dispatch)

    with caplog.at_level(logging.WARNING, logger=rf_mod.logger.name):
        out = await fanout_review(
            _make_state(),
            [
                ReviewerConfig(model="boom:m"),
                ReviewerConfig(model="ok:m"),
            ],
            mode="first_approved",
        )

    # Survivor approves → fanout returns approval.
    assert out.kind == "review_approved"
    assert "survived" in out.content
    # The leg-raise was logged so on-call humans see why the ensemble
    # disagreed in production.
    assert any(
        "boom:m" in rec.getMessage() and "boom" in rec.getMessage()
        for rec in caplog.records
    )


async def test_all_must_approve_one_leg_raises_returns_rejection(monkeypatch):
    """Under ``all_must_approve``, a raised exception is converted to a
    rejection by the per-leg shim and surfaces as the ensemble's
    rejection."""
    survivor = _approved("ok")

    async def dispatch(state, *, model, llm_timeout, decide_only):
        if model == "boom:m":
            raise RuntimeError("kaboom")
        return survivor

    monkeypatch.setattr(rf_mod, "run_reviewer", dispatch)

    out = await fanout_review(
        _make_state(),
        [
            ReviewerConfig(model="boom:m"),
            ReviewerConfig(model="ok:m"),
        ],
        mode="all_must_approve",
    )

    assert out.kind == "review_rejected"
    assert "kaboom" in out.content


# ---------------------------------------------------------------------------
# 8. Overall timeout fires → synthesized rejection feedback
# ---------------------------------------------------------------------------


async def test_overall_timeout_fires_synthesizes_rejection(monkeypatch):
    """If ``overall_timeout`` expires before any decision, the fanout
    cancels every pending leg and returns a synthesized rejection."""
    cancelled = {"count": 0}

    async def slow(state, *, model, llm_timeout, decide_only):
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancelled["count"] += 1
            raise
        return _approved()

    monkeypatch.setattr(rf_mod, "run_reviewer", slow)

    out = await fanout_review(
        _make_state(),
        [
            ReviewerConfig(model="a:m"),
            ReviewerConfig(model="b:m"),
        ],
        mode="first_approved",
        overall_timeout=0.05,
    )

    assert out.kind == "review_rejected"
    assert "overall_timeout" in out.content
    assert cancelled["count"] == 2


async def test_overall_timeout_all_must_approve_fires(monkeypatch):
    """``all_must_approve`` honours ``overall_timeout`` too."""
    async def slow(state, *, model, llm_timeout, decide_only):
        await asyncio.sleep(10.0)
        return _approved()

    monkeypatch.setattr(rf_mod, "run_reviewer", slow)

    out = await fanout_review(
        _make_state(),
        [ReviewerConfig(model="a:m"), ReviewerConfig(model="b:m")],
        mode="all_must_approve",
        overall_timeout=0.05,
    )

    assert out.kind == "review_rejected"
    assert "overall_timeout" in out.content


# ---------------------------------------------------------------------------
# 9. Empty config list → ValueError
# ---------------------------------------------------------------------------


async def test_empty_configs_raises_value_error():
    """Zero reviewers means the caller has misconfigured — refuse to
    pretend approval or rejection."""
    with pytest.raises(ValueError, match="at least one"):
        await fanout_review(_make_state(), [])


# ---------------------------------------------------------------------------
# 10. ProverLoop integration: reviewer_configs set → fanout_review called
#     and lean_prover runs ONCE after approval, not per leg
# ---------------------------------------------------------------------------


async def test_prover_loop_uses_fanout_review_when_reviewer_configs_set(
    monkeypatch,
):
    """Integration: when ``ProverConfig.reviewer_configs`` is non-empty,
    :meth:`ProverLoop._reviewer` must call :func:`fanout_review` once
    and, on approval, run :func:`lean_prover` exactly ONCE (not once
    per leg). The single-model :func:`run_reviewer` path must NOT fire.
    """
    state = _make_state()

    fanout_mock = AsyncMock(return_value=_approved("ensemble approved"))
    lean_mock = AsyncMock(return_value={
        "success": True, "proofComplete": True, "errors": None,
    })
    # The single-model path's run_reviewer must not be called.
    single_mock = AsyncMock(
        side_effect=AssertionError(
            "single-model run_reviewer must not run when reviewer_configs set"
        ),
    )

    monkeypatch.setattr(prover_mod, "fanout_review", fanout_mock)
    monkeypatch.setattr(prover_mod, "lean_prover", lean_mock)
    monkeypatch.setattr(prover_mod, "run_reviewer", single_mock)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        reviewer_configs=[
            ReviewerConfig(model="glm:glm-5.1"),
            ReviewerConfig(model="anthropic:claude-sonnet-4-5"),
        ],
        reviewer_review_mode="first_approved",
    )
    loop = ProverLoop(config=cfg, session=AsyncMock())

    out = await loop._reviewer(state)

    assert out.kind == "review_approved"
    fanout_mock.assert_awaited_once()
    lean_mock.assert_awaited_once()  # one compile total, not N
    # Sanity: the fanout call was forwarded the configured mode.
    _, fanout_kwargs = fanout_mock.await_args
    assert fanout_kwargs.get("mode") == "first_approved"


async def test_prover_loop_fanout_rejection_skips_lean_gate(monkeypatch):
    """When ``fanout_review`` rejects, the loop must NOT invoke
    :func:`lean_prover` at all — the rejection surfaces verbatim."""
    state = _make_state()

    fanout_mock = AsyncMock(return_value=_rejected("ensemble disagreed"))
    lean_mock = AsyncMock(
        side_effect=AssertionError(
            "lean_prover must not run when ensemble rejects"
        ),
    )

    monkeypatch.setattr(prover_mod, "fanout_review", fanout_mock)
    monkeypatch.setattr(prover_mod, "lean_prover", lean_mock)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        reviewer_configs=[ReviewerConfig(model="glm:glm-5.1")],
        reviewer_review_mode="all_must_approve",
    )
    loop = ProverLoop(config=cfg, session=AsyncMock())

    out = await loop._reviewer(state)

    assert out.kind == "review_rejected"
    assert "ensemble disagreed" in out.content
    fanout_mock.assert_awaited_once()
    # Critical: the lean_prover gate must not have run.
    lean_mock.assert_not_awaited()


async def test_prover_loop_default_path_unchanged(monkeypatch):
    """When ``reviewer_configs`` is None (default), the legacy
    single-model path must run — ``fanout_review`` must NOT be called.
    This is the back-compat guarantee for every existing reviewer test.
    """
    state = _make_state()

    single_mock = AsyncMock(return_value=_approved("single ok"))
    fanout_mock = AsyncMock(
        side_effect=AssertionError(
            "fanout_review must not run when reviewer_configs is None"
        ),
    )

    monkeypatch.setattr(prover_mod, "run_reviewer", single_mock)
    monkeypatch.setattr(prover_mod, "fanout_review", fanout_mock)

    cfg = ProverConfig(memory_cls="MemorylessMemory")  # no reviewer_configs
    loop = ProverLoop(config=cfg, session=AsyncMock())

    out = await loop._reviewer(state)

    assert out.kind == "review_approved"
    single_mock.assert_awaited_once()
    fanout_mock.assert_not_awaited()


async def test_prover_loop_fanout_final_gate_failure_returns_rejection(
    monkeypatch,
):
    """If ``fanout_review`` approves but the final ``lean_prover`` compile
    rejects, the loop must downgrade to a rejection so the proposer can
    retry. The compile error summary must be embedded in the feedback.
    """
    state = _make_state()

    fanout_mock = AsyncMock(return_value=_approved("ensemble says yes"))
    lean_mock = AsyncMock(return_value={
        "success": True,
        "proofComplete": False,
        "errors": [{"line": 3, "column": 0, "message": "type mismatch",
                    "severity": "error"}],
        "rawOutput": "(deterministic) timeout at 'elaborator'",
    })

    monkeypatch.setattr(prover_mod, "fanout_review", fanout_mock)
    monkeypatch.setattr(prover_mod, "lean_prover", lean_mock)

    cfg = ProverConfig(
        memory_cls="MemorylessMemory",
        reviewer_configs=[ReviewerConfig(model="glm:glm-5.1")],
    )
    loop = ProverLoop(config=cfg, session=AsyncMock())

    out = await loop._reviewer(state)

    assert out.kind == "review_rejected"
    assert "final lean_prover verification failed" in out.content
    assert "type mismatch" in out.content
