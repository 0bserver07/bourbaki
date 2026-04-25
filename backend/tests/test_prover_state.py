"""Unit tests for the prover/ Pydantic models and feedback factories.

Covers Phase 1 scaffold: model construction, defaults, validation, and the
typed feedback factories. The actual loop logic is stubbed (raises
``NotImplementedError``); tests for proposer/builder/reviewer behaviour
land in Phase 2.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bourbaki.prover import feedback as fb
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverResult,
    ProverState,
    ReviewDecision,
)


# ---------- ProposalMessage ----------


def test_proposal_message_basic():
    msg = ProposalMessage(
        reasoning="try rfl",
        code="theorem foo : 1 = 1 := rfl",
        iteration=0,
    )
    assert msg.reasoning == "try rfl"
    assert msg.imports == []
    assert msg.opens == []
    assert msg.iteration == 0


def test_proposal_message_requires_reasoning_and_code():
    with pytest.raises(ValidationError):
        ProposalMessage(code="x", iteration=0)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ProposalMessage(reasoning="x", iteration=0)  # type: ignore[call-arg]


# ---------- FeedbackMessage ----------


def test_feedback_message_defaults():
    msg = FeedbackMessage(kind="build_failed", content="oops")
    assert msg.is_success is False
    assert msg.is_terminal is False


def test_feedback_message_terminal_flag():
    msg = FeedbackMessage(
        kind="max_iterations", content="done", is_terminal=True
    )
    assert msg.is_terminal is True


# ---------- ProverResult (structured output schema) ----------


def test_prover_result_minimal():
    r = ProverResult(reasoning="r", updated_theorem="theorem t : True := trivial")
    assert r.imports == []
    assert r.opens == []
    assert r.updated_theorem.startswith("theorem")


def test_prover_result_with_extras():
    r = ProverResult(
        reasoning="r",
        imports=["Mathlib.Tactic"],
        opens=["Nat"],
        updated_theorem="theorem t : True := trivial",
    )
    assert r.imports == ["Mathlib.Tactic"]
    assert r.opens == ["Nat"]


# ---------- ReviewDecision ----------


def test_review_decision_all_fields_required():
    rd = ReviewDecision(
        reasoning="ok", check_1=True, check_2=True, check_3=True, approved=True
    )
    assert rd.check_1 and rd.check_2


def test_review_decision_missing_field_raises():
    with pytest.raises(ValidationError):
        ReviewDecision(  # type: ignore[call-arg]
            reasoning="ok", check_1=True, check_2=True, check_3=True
        )


# ---------- ProverState ----------


def test_prover_state_defaults():
    s = ProverState(problem_id="p1", target_theorem="theorem t : 1 = 1 := sorry")
    assert s.iteration == 0
    assert s.max_iterations == 50
    assert s.experience == ""
    assert s.last_proposal is None
    assert s.last_feedback is None
    assert s.messages == []
    assert s.approved is False
    assert s.verified is False
    assert s.final_proof_code is None


def test_prover_state_messages_accept_both_types():
    s = ProverState(problem_id="p1", target_theorem="theorem t : True := sorry")
    proposal = ProposalMessage(reasoning="x", code="y", iteration=0)
    feedback = FeedbackMessage(kind="build_success", content="ok", is_success=True)
    s.messages.append(proposal)
    s.messages.append(feedback)
    assert isinstance(s.messages[0], ProposalMessage)
    assert isinstance(s.messages[1], FeedbackMessage)


# ---------- Feedback factories ----------


def test_build_success_factory():
    m = fb.build_success()
    assert m.kind == "build_success"
    assert m.is_success is True
    assert m.is_terminal is False
    assert "successfully" in m.content


def test_build_failed_truncates_long_output():
    long_err = "X" * 10_000
    m = fb.build_failed(long_err, max_tokens=1000)
    assert m.kind == "build_failed"
    assert m.is_success is False
    assert "truncated" in m.content
    assert len(m.content) < len(long_err)


def test_build_failed_short_output_passes_through():
    short = "type mismatch at line 4"
    m = fb.build_failed(short)
    assert short in m.content
    assert "truncated" not in m.content


def test_sorries_goal_state_factory():
    m = fb.sorries_goal_state(2, ["⊢ p", "⊢ q"])
    assert m.kind == "sorries_goal_state"
    assert "Goal 1" in m.content and "Goal 2" in m.content
    assert "⊢ p" in m.content


def test_axiom_detected_factory():
    m = fb.axiom_detected(1, "line 7")
    assert m.kind == "axiom_detected"
    assert "axiom" in m.content
    assert "line 7" in m.content


def test_search_tactics_detected_factory():
    m = fb.search_tactics_detected(2, "lines 4, 9")
    assert m.kind == "search_tactics_detected"
    assert "exact?" in m.content or "apply?" in m.content


def test_missing_target_theorem_is_retry_not_terminal():
    """missing_target_theorem must NOT be terminal — a single character
    typo in a long theorem name shouldn't kill the loop. The proposer
    sees the feedback and corrects on the next iteration.
    """
    m = fb.missing_target_theorem("mathd_algebra_116")
    assert m.kind == "missing_target_theorem"
    assert m.is_terminal is False
    assert "mathd_algebra_116" in m.content


def test_review_approved_is_success():
    m = fb.review_approved("looks clean")
    assert m.kind == "review_approved"
    assert m.is_success is True
    assert "looks clean" in m.content


def test_review_approved_no_comments():
    m = fb.review_approved()
    assert m.is_success is True


def test_review_rejected_factory():
    m = fb.review_rejected("statement modified")
    assert m.kind == "review_rejected"
    assert m.is_success is False
    assert "statement modified" in m.content


def test_max_iterations_is_terminal():
    m = fb.max_iterations(50)
    assert m.kind == "max_iterations"
    assert m.is_terminal is True
    assert "50" in m.content


def test_structured_output_parsing_failed_factory():
    m = fb.structured_output_parsing_failed("expected JSON, got prose")
    assert m.kind == "structured_output_parsing_failed"
    assert m.is_success is False
    assert "JSON" in m.content


# ---------- Cross-check: factory kinds are unique ----------


def test_factory_kinds_are_unique():
    factories = [
        fb.build_success(),
        fb.build_failed("e"),
        fb.sorries_goal_state(1, ["g"]),
        fb.axiom_detected(1, "l"),
        fb.search_tactics_detected(1, "l"),
        fb.missing_target_theorem("t"),
        fb.review_approved(),
        fb.review_rejected("r"),
        fb.max_iterations(1),
        fb.structured_output_parsing_failed("e"),
    ]
    kinds = [f.kind for f in factories]
    assert len(kinds) == len(set(kinds))


def test_only_terminal_factories_are_terminal():
    terminal_kinds = {"max_iterations"}
    factories = {
        "build_success": fb.build_success(),
        "build_failed": fb.build_failed("e"),
        "sorries_goal_state": fb.sorries_goal_state(1, ["g"]),
        "axiom_detected": fb.axiom_detected(1, "l"),
        "search_tactics_detected": fb.search_tactics_detected(1, "l"),
        "missing_target_theorem": fb.missing_target_theorem("t"),
        "review_approved": fb.review_approved(),
        "review_rejected": fb.review_rejected("r"),
        "max_iterations": fb.max_iterations(1),
        "structured_output_parsing_failed": fb.structured_output_parsing_failed("e"),
    }
    for kind, msg in factories.items():
        assert msg.is_terminal == (kind in terminal_kinds), kind
