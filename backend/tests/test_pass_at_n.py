"""Tests for Pass@N sampling on the proposer-builder-reviewer loop.

These tests mock :func:`bourbaki.benchmarks.minif2f.attempt_proof_loop`
directly, so no LLM, no Lean REPL, and no network are required.  The
Pass@N wrapper's only job is to call the loop up to N times, stop early
on first verified success, and tag each result with ``attempts_pass_n``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from bourbaki.benchmarks import minif2f as minif2f_mod
from bourbaki.benchmarks.loader import MiniF2FProblem
from bourbaki.benchmarks.minif2f import (
    ProblemResult,
    attempt_proof_pass_at_n,
)


def _make_problem() -> MiniF2FProblem:
    return MiniF2FProblem(
        id="mathd_test_1",
        source="mathd",
        split="valid",
        statement="theorem mathd_test_1 : 1 = 1 := sorry",
        imports=["Mathlib"],
        file_path="/dev/null",
        full_lean_code=(
            "import Mathlib\n\ntheorem mathd_test_1 : 1 = 1 := sorry\n"
        ),
    )


def _failed_result(problem_id: str = "mathd_test_1") -> ProblemResult:
    return ProblemResult(
        problem_id=problem_id,
        source="mathd",
        solved=False,
        repl_reported=False,
        verified=False,
        proof_code=None,
        error="loop timeout",
        tactics_used=0,
        duration_seconds=1.0,
        attempts=0,
    )


def _success_result(problem_id: str = "mathd_test_1") -> ProblemResult:
    return ProblemResult(
        problem_id=problem_id,
        source="mathd",
        solved=True,
        repl_reported=True,
        verified=True,
        proof_code="import Mathlib\n\ntheorem mathd_test_1 : 1 = 1 := rfl\n",
        error=None,
        tactics_used=1,
        duration_seconds=1.0,
        attempts=1,
    )


@pytest.mark.asyncio
async def test_pass_n_returns_first_success(monkeypatch):
    """Pass@N stops at the first verified success and tags attempts_pass_n
    with the 1-indexed attempt number that succeeded."""
    side_effect = [_failed_result(), _failed_result(), _success_result()]
    mock_loop = AsyncMock(side_effect=side_effect)
    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", mock_loop)

    session = AsyncMock()  # never touched; loop is mocked
    result = await attempt_proof_pass_at_n(
        _make_problem(), session, n=4, timeout_per_attempt=10,
    )

    # Stopped early at attempt 3 — the 4th call must NOT happen.
    assert mock_loop.await_count == 3
    assert result.solved is True
    assert result.verified is True
    assert result.attempts_pass_n == 3


@pytest.mark.asyncio
async def test_pass_n_returns_last_failure_when_all_fail(monkeypatch):
    """When every attempt fails, return the LAST attempt's result with
    attempts_pass_n=N (so error/proof_code from the final attempt is
    captured)."""
    n = 4
    # Distinguish the failures so we can check we return the *last* one.
    failures = [
        ProblemResult(
            problem_id="mathd_test_1",
            source="mathd",
            solved=False,
            error=f"failure-{i}",
        )
        for i in range(1, n + 1)
    ]
    mock_loop = AsyncMock(side_effect=failures)
    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", mock_loop)

    session = AsyncMock()
    result = await attempt_proof_pass_at_n(
        _make_problem(), session, n=n, timeout_per_attempt=10,
    )

    assert mock_loop.await_count == n
    assert result.solved is False
    assert result.verified is False
    # The returned result is the LAST failure, not the first.
    assert result.error == f"failure-{n}"
    assert result.attempts_pass_n == n


@pytest.mark.asyncio
async def test_pass_n_n_equals_1_no_op(monkeypatch):
    """n=1 degenerates to a single attempt_proof_loop call."""
    mock_loop = AsyncMock(side_effect=[_success_result()])
    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", mock_loop)

    session = AsyncMock()
    result = await attempt_proof_pass_at_n(
        _make_problem(), session, n=1, timeout_per_attempt=10,
    )

    assert mock_loop.await_count == 1
    assert result.solved is True
    assert result.attempts_pass_n == 1


@pytest.mark.asyncio
async def test_pass_n_n_equals_1_failure(monkeypatch):
    """n=1 returns the single failing result with attempts_pass_n=1."""
    mock_loop = AsyncMock(side_effect=[_failed_result()])
    monkeypatch.setattr(minif2f_mod, "attempt_proof_loop", mock_loop)

    session = AsyncMock()
    result = await attempt_proof_pass_at_n(
        _make_problem(), session, n=1, timeout_per_attempt=10,
    )

    assert mock_loop.await_count == 1
    assert result.solved is False
    assert result.attempts_pass_n == 1


@pytest.mark.asyncio
async def test_pass_n_invalid_n_raises():
    """n < 1 is a programming error — raise loudly."""
    session = AsyncMock()
    with pytest.raises(ValueError):
        await attempt_proof_pass_at_n(
            _make_problem(), session, n=0, timeout_per_attempt=10,
        )


def test_problem_result_dict_includes_attempts_pass_n():
    """to_dict must surface attempts_pass_n so it lands in the saved JSON."""
    r = ProblemResult(
        problem_id="x", source="mathd", solved=False, attempts_pass_n=3,
    )
    d = r.to_dict()
    assert "attempts_pass_n" in d
    assert d["attempts_pass_n"] == 3


def test_problem_result_attempts_pass_n_default_is_one():
    """Default attempts_pass_n=1 keeps existing single-attempt code paths
    correct (callers that don't run Pass@N see ``attempts_pass_n=1``)."""
    r = ProblemResult(problem_id="x", source="mathd", solved=False)
    assert r.attempts_pass_n == 1
