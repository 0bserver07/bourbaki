"""Unit tests for the proposer node (Phase 2).

Covers:
- iteration-limit guard returns terminal feedback without LLM call
- happy-path returns ProposalMessage with the right fields
- previous-attempt rendering is included in the user message
- ValidationError / UnexpectedModelBehavior surfaces as
  ``structured_output_parsing_failed`` feedback (no exception bubbles up)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from bourbaki.prover import proposer as proposer_mod
from bourbaki.prover.proposer import _build_user_message, run_proposer
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverResult,
    ProverState,
)


# ---------------------------------------------------------------------------
# Iteration-limit guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposer_returns_max_iterations_when_budget_exceeded(monkeypatch):
    """When state.iteration >= max_iterations, return terminal feedback
    WITHOUT calling the LLM. We verify by patching model resolution to
    raise — if we got there, the test fails.
    """

    def _explode(_model_str: str):
        raise AssertionError("Model resolution must NOT be called when budget exceeded")

    monkeypatch.setattr(proposer_mod, "_resolve_model_object", _explode)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : 1 = 1 := sorry",
        iteration=8,
        max_iterations=8,
    )

    out = await run_proposer(state)

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "max_iterations"
    assert out.is_terminal is True


# ---------------------------------------------------------------------------
# Happy path: TestModel auto-generates a structured output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposer_returns_proposal_on_clean_run(monkeypatch):
    """Use pydantic_ai's TestModel — it auto-generates a plausible
    ``ProverResult``. We just check that we wrap it into a ProposalMessage
    with iteration = state.iteration + 1 and the right fields plumbed
    through.
    """

    # Patch _resolve_model_object so any model string yields a TestModel.
    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : 1 = 1 := sorry",
        iteration=0,
        max_iterations=8,
    )

    out = await run_proposer(state)

    assert isinstance(out, ProposalMessage)
    assert out.iteration == 1
    # TestModel will populate reasoning + updated_theorem with non-empty strings.
    assert isinstance(out.reasoning, str)
    assert isinstance(out.code, str)
    assert isinstance(out.imports, list)
    assert isinstance(out.opens, list)


@pytest.mark.asyncio
async def test_proposer_returns_proposal_with_explicit_stub(monkeypatch):
    """Belt-and-suspenders: also verify against a hand-rolled stub so the
    exact field plumbing is pinned regardless of TestModel internals.
    """

    monkeypatch.setattr(proposer_mod, "_resolve_model_object", lambda _m: TestModel())

    fake_output = ProverResult(
        reasoning="r",
        imports=["Mathlib.Tactic"],
        opens=["Nat"],
        updated_theorem="theorem t : True := trivial",
    )

    async def fake_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(output=fake_output)

    monkeypatch.setattr(Agent, "run", fake_run)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=3,
        max_iterations=8,
    )

    out = await run_proposer(state)

    assert isinstance(out, ProposalMessage)
    assert out.reasoning == "r"
    assert out.imports == ["Mathlib.Tactic"]
    assert out.opens == ["Nat"]
    assert out.code == "theorem t : True := trivial"
    assert out.iteration == 4  # state.iteration + 1


# ---------------------------------------------------------------------------
# Previous-attempt rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposer_renders_previous_attempt(monkeypatch):
    """When ``state.last_proposal`` and ``state.last_feedback`` are set,
    the rendered user message must include the prior reasoning, code, and
    feedback text. We capture the prompt by patching ``Agent.run``.
    """

    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    captured: dict[str, str] = {}

    async def capture_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        captured["prompt"] = user_prompt
        return SimpleNamespace(
            output=ProverResult(
                reasoning="r2",
                imports=[],
                opens=[],
                updated_theorem="theorem t : True := trivial",
            )
        )

    monkeypatch.setattr(Agent, "run", capture_run)

    last_proposal = ProposalMessage(
        reasoning="UNIQUE-PRIOR-REASONING",
        code="UNIQUE-PRIOR-CODE",
        iteration=1,
    )
    last_feedback = FeedbackMessage(
        kind="build_failed",
        content="UNIQUE-PRIOR-FEEDBACK",
    )

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        full_file="-- complete file --",
        iteration=1,
        max_iterations=8,
        last_proposal=last_proposal,
        last_feedback=last_feedback,
        experience="<experience>UNIQUE-EXPERIENCE</experience>",
    )

    out = await run_proposer(state)
    assert isinstance(out, ProposalMessage)

    prompt = captured["prompt"]
    assert "UNIQUE-PRIOR-REASONING" in prompt
    assert "UNIQUE-PRIOR-CODE" in prompt
    assert "UNIQUE-PRIOR-FEEDBACK" in prompt
    assert "UNIQUE-EXPERIENCE" in prompt
    # Base prompt is still there too:
    assert "theorem t : True := sorry" in prompt


def test_build_user_message_omits_previous_attempt_when_absent():
    """First iteration: no last_proposal/last_feedback → no previous-attempt
    block, no experience block.
    """

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        full_file="-- file --",
    )
    msg = _build_user_message(state)
    assert "Your previous attempt" not in msg
    assert "<attempt>" not in msg


def test_build_user_message_appends_experience():
    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        full_file="-- file --",
        experience="<experience>EXP</experience>",
    )
    msg = _build_user_message(state)
    assert msg.endswith("<experience>EXP</experience>")


# ---------------------------------------------------------------------------
# Error handling: ValidationError / UnexpectedModelBehavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposer_returns_parsing_failed_on_validation_error(monkeypatch):
    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    async def fake_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        # Trigger a real ValidationError by validating a missing field.
        try:
            ProverResult.model_validate({"reasoning": "r"})  # missing updated_theorem
        except ValidationError as e:
            raise e
        raise AssertionError("expected ValidationError")

    monkeypatch.setattr(Agent, "run", fake_run)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=0,
        max_iterations=8,
    )

    out = await run_proposer(state)

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert out.is_success is False
    assert out.is_terminal is False


@pytest.mark.asyncio
async def test_proposer_returns_parsing_failed_on_unexpected_model_behavior(monkeypatch):
    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    async def fake_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        raise UnexpectedModelBehavior("model produced garbage")

    monkeypatch.setattr(Agent, "run", fake_run)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=0,
        max_iterations=8,
    )

    out = await run_proposer(state)

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "garbage" in out.content


@pytest.mark.asyncio
async def test_proposer_swallows_generic_exceptions(monkeypatch):
    """Hard constraint: do NOT let exceptions bubble up. A generic Exception
    must also surface as parsing_failed feedback.
    """

    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    async def fake_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("network exploded")

    monkeypatch.setattr(Agent, "run", fake_run)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=0,
        max_iterations=8,
    )

    out = await run_proposer(state)

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "network exploded" in out.content


# ---------------------------------------------------------------------------
# Single-shot vs iterative system prompt selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposer_uses_single_shot_prompt_when_max_iterations_is_one(monkeypatch):
    """When max_iterations == 1, the proposer should use the single-shot
    system prompt. We verify by capturing the Agent constructor.
    """

    from bourbaki.prover import prompts as prompts_mod

    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    captured: dict[str, str] = {}
    original_init = Agent.__init__

    def capture_init(self, *args, **kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", capture_init)

    async def fake_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            output=ProverResult(
                reasoning="r",
                imports=[],
                opens=[],
                updated_theorem="theorem t : True := trivial",
            )
        )

    monkeypatch.setattr(Agent, "run", fake_run)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=0,
        max_iterations=1,
    )

    out = await run_proposer(state)
    assert isinstance(out, ProposalMessage)
    assert captured["system_prompt"] == prompts_mod.PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT


@pytest.mark.asyncio
async def test_proposer_uses_iterative_prompt_by_default(monkeypatch):
    from bourbaki.prover import prompts as prompts_mod

    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    captured: dict[str, str] = {}
    original_init = Agent.__init__

    def capture_init(self, *args, **kwargs):
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", capture_init)

    async def fake_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            output=ProverResult(
                reasoning="r",
                imports=[],
                opens=[],
                updated_theorem="theorem t : True := trivial",
            )
        )

    monkeypatch.setattr(Agent, "run", fake_run)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=0,
        max_iterations=8,
    )

    out = await run_proposer(state)
    assert isinstance(out, ProposalMessage)
    assert captured["system_prompt"] == prompts_mod.PROPOSER_SYSTEM_PROMPT
