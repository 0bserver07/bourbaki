"""Tests for the proposer-builder-reviewer loop's builder node.

The builder runs each proposal through a warm :class:`LeanREPLSession` and
emits a typed :class:`FeedbackMessage`. These tests mock the session to
exercise each branch of the dispatcher without needing a live Lean REPL.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from bourbaki.prover.builder import (
    _count_pattern,
    _extract_sorry_goals,
    _find_declared_names,
    _render_repl_errors,
    run_builder,
)
from bourbaki.prover.state import ProposalMessage, ProverState
from bourbaki.tools.lean_repl import LeanREPLSession


def _make_state(code: str, *, problem_id: str = "test_thm",
                preamble: str = "") -> ProverState:
    """Helper: build a ProverState with a single proposal."""
    proposal = ProposalMessage(
        reasoning="test",
        imports=[],
        opens=[],
        code=code,
        iteration=0,
    )
    return ProverState(
        problem_id=problem_id,
        target_theorem=f"theorem {problem_id} : True := sorry",
        preamble=preamble,
        full_file="",
        last_proposal=proposal,
    )


# ---------------------------------------------------------------------------
# Branch tests for run_builder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_target_returns_terminal_feedback() -> None:
    """If the proposal does not declare the target theorem, builder emits
    a terminal ``missing_target_theorem`` feedback and does not call the REPL.
    """
    state = _make_state(
        code="theorem some_other_name : True := trivial",
        problem_id="my_target",
    )
    mock_session = AsyncMock(spec=LeanREPLSession)

    fb = await run_builder(state, mock_session)

    assert fb.kind == "missing_target_theorem"
    assert fb.is_terminal is True
    assert "my_target" in fb.content
    mock_session.send_cmd.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_failed_on_severity_error() -> None:
    """Any REPL message with severity=='error' produces ``build_failed``."""
    state = _make_state(
        code="theorem my_target : True := wrong_tactic",
        problem_id="my_target",
    )
    mock_session = AsyncMock(spec=LeanREPLSession)
    mock_session.send_cmd.return_value = {
        "messages": [
            {
                "severity": "error",
                "pos": {"line": 1, "column": 30},
                "data": "unknown identifier 'wrong_tactic'",
            },
            {
                "severity": "warning",
                "data": "deprecated lemma",
            },
        ],
        "sorries": [],
        "env": 1,
    }

    fb = await run_builder(state, mock_session)

    assert fb.kind == "build_failed"
    assert fb.is_terminal is False
    assert "unknown identifier 'wrong_tactic'" in fb.content
    # Warnings should not appear in the rendered output.
    assert "deprecated lemma" not in fb.content
    mock_session.send_cmd.assert_awaited_once()


@pytest.mark.asyncio
async def test_sorries_goal_state_when_sorries_present() -> None:
    """Compiles cleanly but contains sorries → ``sorries_goal_state`` with
    the goal text from each sorry entry.
    """
    state = _make_state(
        code=(
            "theorem my_target (n : Nat) : n + 0 = n := by\n"
            "  have h : n = n := sorry\n"
            "  sorry"
        ),
        problem_id="my_target",
    )
    mock_session = AsyncMock(spec=LeanREPLSession)
    mock_session.send_cmd.return_value = {
        "messages": [],
        "sorries": [
            {
                "goal": "n : Nat\n⊢ n = n",
                "proofState": 0,
            },
            {
                "goal": "n : Nat\n⊢ n + 0 = n",
                "proofState": 1,
            },
        ],
        "env": 1,
    }

    fb = await run_builder(state, mock_session)

    assert fb.kind == "sorries_goal_state"
    assert "2 remaining `sorry`" in fb.content
    assert "n + 0 = n" in fb.content
    assert "n = n" in fb.content


@pytest.mark.asyncio
async def test_axiom_detected_when_proposal_contains_axiom() -> None:
    """A proposal with ``axiom foo : True`` → ``axiom_detected``."""
    state = _make_state(
        code=(
            "axiom my_axiom : True\n"
            "theorem my_target : True := my_axiom"
        ),
        problem_id="my_target",
    )
    mock_session = AsyncMock(spec=LeanREPLSession)
    mock_session.send_cmd.return_value = {
        "messages": [],
        "sorries": [],
        "env": 1,
    }

    fb = await run_builder(state, mock_session)

    assert fb.kind == "axiom_detected"
    assert "1" in fb.content
    assert "line 1" in fb.content


@pytest.mark.asyncio
async def test_search_tactics_detected_for_apply_question_mark() -> None:
    """Proposals using ``apply?`` or ``exact?`` → ``search_tactics_detected``."""
    state = _make_state(
        code=(
            "theorem my_target : True := by\n"
            "  apply?"
        ),
        problem_id="my_target",
    )
    mock_session = AsyncMock(spec=LeanREPLSession)
    mock_session.send_cmd.return_value = {
        "messages": [],
        "sorries": [],
        "env": 1,
    }

    fb = await run_builder(state, mock_session)

    assert fb.kind == "search_tactics_detected"
    assert "line 2" in fb.content


@pytest.mark.asyncio
async def test_build_success_on_clean_proposal() -> None:
    """A proposal with no sorries, no errors, no banned tactics → success."""
    state = _make_state(
        code="theorem my_target : True := trivial",
        problem_id="my_target",
    )
    mock_session = AsyncMock(spec=LeanREPLSession)
    mock_session.send_cmd.return_value = {
        "messages": [],
        "sorries": [],
        "env": 1,
    }

    fb = await run_builder(state, mock_session)

    assert fb.kind == "build_success"
    assert fb.is_success is True


@pytest.mark.asyncio
async def test_imports_are_stripped_before_send_cmd() -> None:
    """``import ...`` lines are removed from the proposal because the REPL
    has Mathlib pre-loaded.
    """
    state = _make_state(
        code=(
            "import Mathlib\n"
            "import Mathlib.Tactic\n"
            "theorem my_target : True := trivial"
        ),
        problem_id="my_target",
    )
    mock_session = AsyncMock(spec=LeanREPLSession)
    mock_session.send_cmd.return_value = {
        "messages": [],
        "sorries": [],
        "env": 1,
    }

    await run_builder(state, mock_session)

    sent = mock_session.send_cmd.await_args.args[0]
    assert "import Mathlib" not in sent
    assert "theorem my_target : True := trivial" in sent


@pytest.mark.asyncio
async def test_preamble_is_prepended_when_non_empty() -> None:
    """Non-empty preamble is prepended to the stripped code."""
    state = _make_state(
        code="theorem my_target : True := trivial",
        problem_id="my_target",
        preamble="open BigOperators Nat",
    )
    mock_session = AsyncMock(spec=LeanREPLSession)
    mock_session.send_cmd.return_value = {
        "messages": [],
        "sorries": [],
        "env": 1,
    }

    await run_builder(state, mock_session)

    sent = mock_session.send_cmd.await_args.args[0]
    assert sent.startswith("open BigOperators Nat")
    assert "theorem my_target" in sent


@pytest.mark.asyncio
async def test_repl_level_error_is_reported_as_build_failed() -> None:
    """A ``send_cmd`` returning ``{"error": ...}`` (timeout / pipe) is
    reported as ``build_failed`` so the loop can retry rather than crash.
    """
    state = _make_state(
        code="theorem my_target : True := trivial",
        problem_id="my_target",
    )
    mock_session = AsyncMock(spec=LeanREPLSession)
    mock_session.send_cmd.return_value = {"error": "REPL timed out"}

    fb = await run_builder(state, mock_session)

    assert fb.kind == "build_failed"
    assert "REPL timed out" in fb.content


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_find_declared_names_handles_lemma_and_theorem() -> None:
    code = (
        "theorem foo (n : Nat) : n = n := rfl\n"
        "lemma bar : True := trivial\n"
        "theorem Namespace.baz : 1 = 1 := rfl\n"
    )
    names = _find_declared_names(code)
    assert "foo" in names
    assert "bar" in names
    assert "Namespace.baz" in names


def test_count_pattern_returns_locations() -> None:
    code = "line one\napply?\nline three\nexact?\n"
    count, locations = _count_pattern(code, r"\b(?:apply|exact)\?")
    assert count == 2
    assert "line 2" in locations
    assert "line 4" in locations


def test_render_repl_errors_includes_position() -> None:
    rendered = _render_repl_errors([
        {"severity": "error", "pos": {"line": 3, "column": 5}, "data": "boom"},
        {"severity": "error", "data": "no pos"},
    ])
    assert "line 3" in rendered
    assert "column 5" in rendered
    assert "boom" in rendered
    assert "no pos" in rendered


def test_extract_sorry_goals_handles_singular_and_plural() -> None:
    sorries = [
        {"goal": "⊢ True", "proofState": 0},
        {"goals": ["⊢ A", "⊢ B"], "proofState": 1},
        {"proofState": 2},  # No goal info
    ]
    goals = _extract_sorry_goals(sorries)
    assert goals[0] == "⊢ True"
    assert "⊢ A" in goals[1] and "⊢ B" in goals[1]
    assert "no goal information" in goals[2]
