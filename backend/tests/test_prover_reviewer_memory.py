"""Phase 2 tests: reviewer node + memory strategies.

Both the reviewer and the LLM-backed memory call out to Pydantic AI; we
mock `Agent.run` directly via monkeypatch and (for the reviewer) `lean_prover`
via patch so no Lean binary is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bourbaki.prover import memory as memory_mod
from bourbaki.prover import reviewer as reviewer_mod
from bourbaki.prover.memory import (
    ExperienceMemory,
    MemorylessMemory,
    PreviousKMemory,
)
from bourbaki.prover.reviewer import run_reviewer
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
    ReviewDecision,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _state_with_proposal(code: str = "theorem t : 1 = 1 := rfl") -> ProverState:
    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : 1 = 1 := sorry",
        preamble="open Nat",
    )
    state.last_proposal = ProposalMessage(
        reasoning="rfl closes it",
        code=code,
        iteration=0,
    )
    return state


def _agent_run_returning(output):
    """Return an AsyncMock suitable as a stand-in for `Agent.run`."""
    mock_result = MagicMock()
    mock_result.output = output
    return AsyncMock(return_value=mock_result)


# ---------------------------------------------------------------------------
# Reviewer tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_rejects_when_check_1_false(monkeypatch):
    state = _state_with_proposal()
    decision = ReviewDecision(
        reasoning="signature was renamed",
        check_1=False,
        check_2=True,
        check_3=True,
        approved=True,  # honeypot
    )
    monkeypatch.setattr(
        "pydantic_ai.Agent.run", _agent_run_returning(decision)
    )
    monkeypatch.setattr(
        reviewer_mod, "lean_prover",
        AsyncMock(side_effect=AssertionError("lean_prover must not be called when rejected")),
    )

    result = await run_reviewer(state, model="glm:glm-5.1")

    assert result.kind == "review_rejected"
    assert result.is_success is False
    assert "signature was renamed" in result.content


@pytest.mark.asyncio
async def test_reviewer_rejects_when_check_2_false(monkeypatch):
    state = _state_with_proposal()
    decision = ReviewDecision(
        reasoning="found a sorry inside a have block",
        check_1=True,
        check_2=False,
        check_3=True,
        approved=True,
    )
    monkeypatch.setattr(
        "pydantic_ai.Agent.run", _agent_run_returning(decision)
    )
    monkeypatch.setattr(
        reviewer_mod, "lean_prover",
        AsyncMock(side_effect=AssertionError("lean_prover must not run when rejected")),
    )

    result = await run_reviewer(state, model="glm:glm-5.1")

    assert result.kind == "review_rejected"
    assert "sorry inside a have block" in result.content


@pytest.mark.asyncio
async def test_reviewer_approves_and_lean_prover_succeeds(monkeypatch):
    state = _state_with_proposal()
    decision = ReviewDecision(
        reasoning="clean proof",
        check_1=True,
        check_2=True,
        check_3=True,
        approved=False,  # honeypot — caller must ignore this
    )
    monkeypatch.setattr(
        "pydantic_ai.Agent.run", _agent_run_returning(decision)
    )
    lean_mock = AsyncMock(return_value={"success": True, "errors": None})
    monkeypatch.setattr(reviewer_mod, "lean_prover", lean_mock)

    result = await run_reviewer(state, model="glm:glm-5.1")

    assert result.kind == "review_approved"
    assert result.is_success is True
    assert "clean proof" in result.content
    lean_mock.assert_awaited_once()
    # confirm the assembled source got both an import and the preamble
    call_kwargs = lean_mock.await_args.kwargs
    src = call_kwargs.get("code") or lean_mock.await_args.args[0]
    assert "import Mathlib" in src
    assert "open Nat" in src
    assert "theorem t : 1 = 1 := rfl" in src


@pytest.mark.asyncio
async def test_reviewer_rejects_when_lean_prover_fails(monkeypatch):
    state = _state_with_proposal(code="theorem t : 1 = 1 := by rfl")
    decision = ReviewDecision(
        reasoning="looks good to me",
        check_1=True,
        check_2=True,
        check_3=True,
        approved=True,
    )
    monkeypatch.setattr(
        "pydantic_ai.Agent.run", _agent_run_returning(decision)
    )
    lean_mock = AsyncMock(return_value={
        "success": False,
        "errors": [
            {"line": 3, "column": 0, "message": "unknown identifier 'foo'", "severity": "error"}
        ],
    })
    monkeypatch.setattr(reviewer_mod, "lean_prover", lean_mock)

    result = await run_reviewer(state, model="glm:glm-5.1")

    assert result.kind == "review_rejected"
    assert "final lean_prover verification failed" in result.content
    assert "unknown identifier 'foo'" in result.content


@pytest.mark.asyncio
async def test_reviewer_handles_llm_exception(monkeypatch):
    state = _state_with_proposal()

    async def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("pydantic_ai.Agent.run", boom)
    monkeypatch.setattr(
        reviewer_mod, "lean_prover",
        AsyncMock(side_effect=AssertionError("lean_prover must not run when LLM errored")),
    )

    result = await run_reviewer(state, model="glm:glm-5.1")

    assert result.kind == "review_rejected"
    assert "reviewer LLM error" in result.content
    assert "network down" in result.content


@pytest.mark.asyncio
async def test_reviewer_skips_import_when_already_present(monkeypatch):
    state = _state_with_proposal(
        code="import Mathlib\n\ntheorem t : 1 = 1 := rfl"
    )
    decision = ReviewDecision(
        reasoning="ok",
        check_1=True,
        check_2=True,
        check_3=True,
        approved=True,
    )
    monkeypatch.setattr(
        "pydantic_ai.Agent.run", _agent_run_returning(decision)
    )
    lean_mock = AsyncMock(return_value={"success": True, "errors": None})
    monkeypatch.setattr(reviewer_mod, "lean_prover", lean_mock)

    result = await run_reviewer(state, model="glm:glm-5.1")

    assert result.kind == "review_approved"
    src = lean_mock.await_args.kwargs.get("code") or lean_mock.await_args.args[0]
    # only one `import Mathlib` line should be present
    assert src.count("import Mathlib") == 1


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memoryless_returns_empty():
    state = _state_with_proposal()
    state.last_feedback = FeedbackMessage(kind="build_failed", content="nope")

    result = await MemorylessMemory().process(state)
    assert result == ""


@pytest.mark.asyncio
async def test_previous_k_renders_last_k_pairs():
    state = ProverState(problem_id="p", target_theorem="theorem t : True := sorry")
    for i in range(3):
        state.messages.append(
            ProposalMessage(reasoning=f"reason-{i}", code=f"code-{i}", iteration=i)
        )
        state.messages.append(
            FeedbackMessage(kind="build_failed", content=f"feedback-{i}")
        )

    result = await PreviousKMemory(k=2).process(state)

    assert result.startswith("<previous-attempts>")
    assert result.rstrip().endswith("</previous-attempts>")
    # last 2 pairs in chronological order: indices 1 and 2
    assert "reason-1" in result
    assert "reason-2" in result
    assert "feedback-1" in result
    assert "feedback-2" in result
    # oldest pair (index 0) must NOT appear
    assert "reason-0" not in result
    assert "feedback-0" not in result
    # chronological order: reason-1 appears before reason-2
    assert result.index("reason-1") < result.index("reason-2")


@pytest.mark.asyncio
async def test_previous_k_empty_messages():
    state = ProverState(problem_id="p", target_theorem="theorem t : True := sorry")
    result = await PreviousKMemory(k=2).process(state)
    assert result == ""


@pytest.mark.asyncio
async def test_previous_k_handles_unpaired_proposal():
    """A trailing ProposalMessage without matching FeedbackMessage is ignored."""
    state = ProverState(problem_id="p", target_theorem="theorem t : True := sorry")
    state.messages.append(ProposalMessage(reasoning="r0", code="c0", iteration=0))
    state.messages.append(FeedbackMessage(kind="build_failed", content="fb0"))
    state.messages.append(ProposalMessage(reasoning="r1", code="c1", iteration=1))

    result = await PreviousKMemory(k=5).process(state)
    assert "r0" in result
    assert "fb0" in result
    # r1 has no feedback yet — should not appear
    assert "r1" not in result


@pytest.mark.asyncio
async def test_experience_calls_llm_and_wraps_output(monkeypatch):
    state = _state_with_proposal()
    state.last_feedback = FeedbackMessage(kind="build_failed", content="rfl failed: type mismatch")
    state.experience = ""

    monkeypatch.setattr(
        "pydantic_ai.Agent.run",
        _agent_run_returning("- rfl failed; try `decide`"),
    )

    result = await ExperienceMemory(model="glm:glm-5.1").process(state)

    assert result.startswith("<experience>")
    assert result.rstrip().endswith("</experience>")
    assert "rfl failed; try `decide`" in result


@pytest.mark.asyncio
async def test_experience_returns_existing_when_no_proposal():
    state = ProverState(
        problem_id="p",
        target_theorem="theorem t : True := sorry",
        experience="<experience>old wisdom</experience>",
    )
    state.last_proposal = None
    state.last_feedback = None

    result = await ExperienceMemory(model="glm:glm-5.1").process(state)
    assert result == "<experience>old wisdom</experience>"


@pytest.mark.asyncio
async def test_experience_falls_back_when_llm_errors(monkeypatch):
    state = _state_with_proposal()
    state.last_feedback = FeedbackMessage(kind="build_failed", content="oops")
    state.experience = "<experience>prior</experience>"

    async def boom(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr("pydantic_ai.Agent.run", boom)

    result = await ExperienceMemory(model="glm:glm-5.1").process(state)
    # On error we keep prior experience so context isn't lost.
    assert result == "<experience>prior</experience>"
