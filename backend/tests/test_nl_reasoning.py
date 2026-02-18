"""Tests for Aletheia-style NL reasoning pre-pass in sketch generation and coordinator.

Validates:
- NL reasoning prompt construction
- NL reasoning integration into sketch prompts
- parse_sketch_response attaches nl_reasoning to ProofSketch objects
- LLMSketchGenerator two-phase generation (with mocked LLM)
- Coordinator NL reasoning integration (with mocked LLM)
- DecompositionConfig.use_nl_reasoning flag
- Backward compatibility when use_nl_reasoning=False
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bourbaki.autonomous.sketch import (
    LLMSketchGenerator,
    NL_REASONING_MAX_CHARS,
    NL_REASONING_PROMPT,
    ProofSketch,
    SketchContext,
    SketchStep,
    build_nl_reasoning_prompt,
    build_sketch_prompt,
    parse_sketch_response,
)
from bourbaki.autonomous.decomposer import DecompositionConfig


# ---------------------------------------------------------------------------
# NL reasoning prompt construction
# ---------------------------------------------------------------------------

def test_nl_reasoning_prompt_contains_theorem():
    """NL reasoning prompt should include the theorem text."""
    prompt = build_nl_reasoning_prompt("theorem foo : 1 + 1 = 2")
    assert "theorem foo : 1 + 1 = 2" in prompt


def test_nl_reasoning_prompt_has_all_sections():
    """NL reasoning prompt should ask for all five analysis sections."""
    prompt = build_nl_reasoning_prompt("theorem x : True")
    assert "PROBLEM TYPE" in prompt
    assert "KEY OBSERVATIONS" in prompt
    assert "PROOF STRATEGY" in prompt
    assert "INTERMEDIATE STEPS" in prompt
    assert "POTENTIAL PITFALLS" in prompt


def test_nl_reasoning_prompt_template_format():
    """The NL_REASONING_PROMPT template should have a {theorem} placeholder."""
    assert "{theorem}" in NL_REASONING_PROMPT


# ---------------------------------------------------------------------------
# NL reasoning in sketch prompt injection
# ---------------------------------------------------------------------------

def test_build_sketch_prompt_with_nl_reasoning():
    """When NL reasoning is provided, it should appear in the sketch prompt."""
    ctx = SketchContext(theorem="theorem foo : True", depth=0)
    reasoning = "This is a trivial proposition. Direct proof via trivial."
    prompt = build_sketch_prompt(ctx, nl_reasoning=reasoning)
    assert "MATHEMATICAL ANALYSIS" in prompt
    assert reasoning in prompt


def test_build_sketch_prompt_without_nl_reasoning():
    """Without NL reasoning, the prompt should not contain the analysis header."""
    ctx = SketchContext(theorem="theorem foo : True", depth=0)
    prompt = build_sketch_prompt(ctx, nl_reasoning=None)
    assert "MATHEMATICAL ANALYSIS" not in prompt


def test_build_sketch_prompt_nl_reasoning_with_other_context():
    """NL reasoning should coexist with mathlib results and previous attempts."""
    ctx = SketchContext(
        theorem="theorem foo : True",
        mathlib_results=[{"name": "True.intro"}],
        previous_attempts=["Failed with ring"],
        depth=0,
    )
    reasoning = "This is trivially true."
    prompt = build_sketch_prompt(ctx, nl_reasoning=reasoning)
    assert "MATHEMATICAL ANALYSIS" in prompt
    assert "True.intro" in prompt
    assert "Failed with ring" in prompt


def test_build_sketch_prompt_nl_reasoning_at_depth_2():
    """NL reasoning should still be injected for subgoal prompts at depth >= 2."""
    ctx = SketchContext(theorem="theorem sub : 1 = 1", depth=2)
    reasoning = "Identity equality, trivial by refl."
    prompt = build_sketch_prompt(ctx, nl_reasoning=reasoning)
    assert "MATHEMATICAL ANALYSIS" in prompt
    assert reasoning in prompt
    assert "SUBGOAL" in prompt  # Should still use SUBGOAL_SKETCH_PROMPT


# ---------------------------------------------------------------------------
# parse_sketch_response with nl_reasoning
# ---------------------------------------------------------------------------

def test_parse_sketch_response_attaches_nl_reasoning():
    """NL reasoning should be attached to each parsed ProofSketch."""
    response = json.dumps({
        "sketches": [{
            "strategy": "direct",
            "steps": [{"statement": "Done", "formal_type": "True"}],
            "key_lemmas": [],
        }]
    })
    reasoning = "This is a trivial proposition."
    sketches = parse_sketch_response(response, nl_reasoning=reasoning)
    assert len(sketches) == 1
    assert sketches[0].nl_reasoning == reasoning


def test_parse_sketch_response_nl_reasoning_none_by_default():
    """Without nl_reasoning, the field should be None."""
    response = json.dumps({
        "sketches": [{
            "strategy": "direct",
            "steps": [{"statement": "Done", "formal_type": "True"}],
            "key_lemmas": [],
        }]
    })
    sketches = parse_sketch_response(response)
    assert len(sketches) == 1
    assert sketches[0].nl_reasoning is None


def test_parse_sketch_response_multiple_sketches_all_get_reasoning():
    """All sketches in a response should receive the same nl_reasoning."""
    response = json.dumps({
        "sketches": [
            {"strategy": "induction", "steps": [{"statement": "A", "formal_type": "X"}], "key_lemmas": []},
            {"strategy": "direct", "steps": [{"statement": "B", "formal_type": "Y"}], "key_lemmas": []},
        ]
    })
    reasoning = "Try induction or direct proof."
    sketches = parse_sketch_response(response, nl_reasoning=reasoning)
    assert len(sketches) == 2
    assert all(s.nl_reasoning == reasoning for s in sketches)


# ---------------------------------------------------------------------------
# ProofSketch nl_reasoning field
# ---------------------------------------------------------------------------

def test_proof_sketch_nl_reasoning_field():
    """ProofSketch should support the nl_reasoning field."""
    sketch = ProofSketch(
        strategy="induction",
        steps=[SketchStep(statement="Base", formal_type="P 0")],
        nl_reasoning="Use strong induction on n.",
    )
    assert sketch.nl_reasoning == "Use strong induction on n."


def test_proof_sketch_nl_reasoning_defaults_to_none():
    """ProofSketch.nl_reasoning should default to None."""
    sketch = ProofSketch(strategy="direct", steps=[])
    assert sketch.nl_reasoning is None


# ---------------------------------------------------------------------------
# DecompositionConfig
# ---------------------------------------------------------------------------

def test_decomposition_config_use_nl_reasoning_default():
    """use_nl_reasoning should default to True."""
    config = DecompositionConfig()
    assert config.use_nl_reasoning is True


def test_decomposition_config_use_nl_reasoning_false():
    """use_nl_reasoning can be set to False."""
    config = DecompositionConfig(use_nl_reasoning=False)
    assert config.use_nl_reasoning is False


# ---------------------------------------------------------------------------
# LLMSketchGenerator: NL reasoning flag
# ---------------------------------------------------------------------------

def test_llm_sketch_generator_default_nl_reasoning():
    """LLMSketchGenerator should enable NL reasoning by default."""
    gen = LLMSketchGenerator(model="openai:gpt-4o")
    assert gen.use_nl_reasoning is True


def test_llm_sketch_generator_disable_nl_reasoning():
    """LLMSketchGenerator can disable NL reasoning."""
    gen = LLMSketchGenerator(model="openai:gpt-4o", use_nl_reasoning=False)
    assert gen.use_nl_reasoning is False


# ---------------------------------------------------------------------------
# LLMSketchGenerator: two-phase generation (mocked)
# ---------------------------------------------------------------------------

def _mock_agent_run(output: str):
    """Create a mock pydantic_ai Agent.run that returns the given output."""
    mock_result = MagicMock()
    mock_result.output = output
    return AsyncMock(return_value=mock_result)


@pytest.mark.asyncio
async def test_llm_sketch_generator_with_nl_reasoning():
    """With NL reasoning enabled, generator should call NL reasoning then sketch."""
    gen = LLMSketchGenerator(model="openai:gpt-4o", use_nl_reasoning=True)

    nl_output = "This is a number theory problem. Use induction."
    sketch_output = json.dumps({
        "sketches": [{
            "strategy": "induction",
            "steps": [{"statement": "Base", "formal_type": "P 0"}],
            "key_lemmas": [],
        }]
    })

    # Mock _generate_nl_reasoning directly
    gen._generate_nl_reasoning = AsyncMock(return_value=nl_output)

    # Mock the Agent used in generate() for the sketch phase
    mock_sketch_result = MagicMock()
    mock_sketch_result.output = sketch_output

    with patch("bourbaki.agent.core._resolve_model_object", return_value="mock-model"):
        with patch("pydantic_ai.Agent") as MockAgent:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_sketch_result)
            MockAgent.return_value = instance

            context = SketchContext(theorem="theorem foo : P 0", depth=0)
            sketches = await gen.generate(context)

    gen._generate_nl_reasoning.assert_awaited_once_with("theorem foo : P 0")
    assert len(sketches) == 1
    assert sketches[0].nl_reasoning == nl_output
    assert sketches[0].strategy == "induction"


@pytest.mark.asyncio
async def test_llm_sketch_generator_without_nl_reasoning():
    """With NL reasoning disabled, generator should skip the NL phase."""
    gen = LLMSketchGenerator(model="openai:gpt-4o", use_nl_reasoning=False)

    sketch_output = json.dumps({
        "sketches": [{
            "strategy": "direct",
            "steps": [{"statement": "Done", "formal_type": "True"}],
            "key_lemmas": [],
        }]
    })

    # Mock _generate_nl_reasoning to track calls
    gen._generate_nl_reasoning = AsyncMock(return_value="should not be called")

    mock_sketch_result = MagicMock()
    mock_sketch_result.output = sketch_output

    with patch("bourbaki.agent.core._resolve_model_object", return_value="mock-model"):
        with patch("pydantic_ai.Agent") as MockAgent:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_sketch_result)
            MockAgent.return_value = instance

            context = SketchContext(theorem="theorem foo : True", depth=0)
            sketches = await gen.generate(context)

    gen._generate_nl_reasoning.assert_not_awaited()
    assert len(sketches) == 1
    assert sketches[0].nl_reasoning is None


@pytest.mark.asyncio
async def test_llm_sketch_generator_skips_nl_for_deep_subgoals():
    """NL reasoning should be skipped for depth > 0 even when enabled."""
    gen = LLMSketchGenerator(model="openai:gpt-4o", use_nl_reasoning=True)

    sketch_output = json.dumps({
        "sketches": [{
            "strategy": "direct",
            "steps": [{"statement": "Done", "formal_type": "True"}],
            "key_lemmas": [],
        }]
    })

    gen._generate_nl_reasoning = AsyncMock(return_value="should not be called")

    mock_sketch_result = MagicMock()
    mock_sketch_result.output = sketch_output

    with patch("bourbaki.agent.core._resolve_model_object", return_value="mock-model"):
        with patch("pydantic_ai.Agent") as MockAgent:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_sketch_result)
            MockAgent.return_value = instance

            context = SketchContext(theorem="theorem sub : True", depth=1)
            sketches = await gen.generate(context)

    gen._generate_nl_reasoning.assert_not_awaited()
    assert sketches[0].nl_reasoning is None


@pytest.mark.asyncio
async def test_llm_sketch_generator_nl_reasoning_failure_graceful():
    """If NL reasoning fails, sketch generation should still proceed."""
    gen = LLMSketchGenerator(model="openai:gpt-4o", use_nl_reasoning=True)

    sketch_output = json.dumps({
        "sketches": [{
            "strategy": "direct",
            "steps": [{"statement": "Done", "formal_type": "True"}],
            "key_lemmas": [],
        }]
    })

    # Simulate NL reasoning returning None (failure)
    gen._generate_nl_reasoning = AsyncMock(return_value=None)

    mock_sketch_result = MagicMock()
    mock_sketch_result.output = sketch_output

    with patch("bourbaki.agent.core._resolve_model_object", return_value="mock-model"):
        with patch("pydantic_ai.Agent") as MockAgent:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_sketch_result)
            MockAgent.return_value = instance

            context = SketchContext(theorem="theorem foo : True", depth=0)
            sketches = await gen.generate(context)

    assert len(sketches) == 1
    assert sketches[0].nl_reasoning is None  # Gracefully None


# ---------------------------------------------------------------------------
# NL reasoning character cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_sketch_generator_nl_reasoning_capped():
    """NL reasoning output should be capped at NL_REASONING_MAX_CHARS."""
    gen = LLMSketchGenerator(model="openai:gpt-4o", use_nl_reasoning=True)

    long_reasoning = "x" * (NL_REASONING_MAX_CHARS + 500)

    # Test the capping via _generate_nl_reasoning directly
    mock_result = MagicMock()
    mock_result.output = long_reasoning

    with patch("bourbaki.agent.core._resolve_model_object", return_value="mock-model"):
        with patch("pydantic_ai.Agent") as MockAgent:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_result)
            MockAgent.return_value = instance

            reasoning = await gen._generate_nl_reasoning("theorem foo : True")

    assert reasoning is not None
    assert len(reasoning) == NL_REASONING_MAX_CHARS + 3  # +3 for "..."
    assert reasoning.endswith("...")


# ---------------------------------------------------------------------------
# Coordinator NL reasoning integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coordinator_with_nl_reasoning():
    """Coordinator should generate NL reasoning and pass it to strategist and prover."""
    from bourbaki.agent.coordinator import ProofCoordinator

    coord = ProofCoordinator(model="openai:gpt-4o", use_nl_reasoning=True)

    # Track what gets passed to strategist and prover
    strategist_kwargs = {}
    prover_kwargs = {}

    async def mock_nl_reasoning(theorem):
        return "This is an algebraic identity. Use ring tactic."

    async def mock_strategist(theorem, errors, nl_reasoning=None):
        strategist_kwargs["nl_reasoning"] = nl_reasoning
        return {"sketch": ["ring"], "subgoals": []}

    async def mock_searcher(theorem, subgoals):
        return []

    async def mock_prover(theorem, strategy, lemmas, nl_reasoning=None):
        prover_kwargs["nl_reasoning"] = nl_reasoning
        return "theorem foo : 1 + 1 = 2 := by ring"

    async def mock_verifier(proof_code):
        return True

    with patch.object(coord, "_generate_nl_reasoning", side_effect=mock_nl_reasoning):
        with patch.object(coord, "_run_strategist", side_effect=mock_strategist):
            with patch.object(coord, "_run_searcher", side_effect=mock_searcher):
                with patch.object(coord, "_run_prover", side_effect=mock_prover):
                    with patch.object(coord, "_run_verifier", side_effect=mock_verifier):
                        result = await coord.prove("theorem foo : 1 + 1 = 2")

    assert result.success is True
    assert strategist_kwargs["nl_reasoning"] == "This is an algebraic identity. Use ring tactic."
    assert prover_kwargs["nl_reasoning"] == "This is an algebraic identity. Use ring tactic."


@pytest.mark.asyncio
async def test_coordinator_without_nl_reasoning():
    """When use_nl_reasoning=False, coordinator should not generate NL reasoning."""
    from bourbaki.agent.coordinator import ProofCoordinator

    coord = ProofCoordinator(model="openai:gpt-4o", use_nl_reasoning=False)

    strategist_kwargs = {}

    async def mock_strategist(theorem, errors, nl_reasoning=None):
        strategist_kwargs["nl_reasoning"] = nl_reasoning
        return {"sketch": ["trivial"], "subgoals": []}

    async def mock_prover(theorem, strategy, lemmas, nl_reasoning=None):
        return "theorem foo : True := trivial"

    with patch.object(coord, "_run_strategist", side_effect=mock_strategist):
        with patch.object(coord, "_run_searcher", new_callable=AsyncMock, return_value=[]):
            with patch.object(coord, "_run_prover", side_effect=mock_prover):
                with patch.object(coord, "_run_verifier", new_callable=AsyncMock, return_value=True):
                    result = await coord.prove("theorem foo : True")

    assert result.success is True
    assert strategist_kwargs["nl_reasoning"] is None


@pytest.mark.asyncio
async def test_coordinator_nl_reasoning_failure_graceful():
    """If NL reasoning fails in the coordinator, proving should still work."""
    from bourbaki.agent.coordinator import ProofCoordinator

    coord = ProofCoordinator(model="openai:gpt-4o", use_nl_reasoning=True)

    async def mock_nl_reasoning_fails(theorem):
        return None  # Simulates failure

    async def mock_strategist(theorem, errors, nl_reasoning=None):
        return {"sketch": ["trivial"], "subgoals": []}

    async def mock_prover(theorem, strategy, lemmas, nl_reasoning=None):
        return "theorem foo : True := trivial"

    with patch.object(coord, "_generate_nl_reasoning", side_effect=mock_nl_reasoning_fails):
        with patch.object(coord, "_run_strategist", side_effect=mock_strategist):
            with patch.object(coord, "_run_searcher", new_callable=AsyncMock, return_value=[]):
                with patch.object(coord, "_run_prover", side_effect=mock_prover):
                    with patch.object(coord, "_run_verifier", new_callable=AsyncMock, return_value=True):
                        result = await coord.prove("theorem foo : True")

    assert result.success is True


# ---------------------------------------------------------------------------
# NL_REASONING_MAX_CHARS constant
# ---------------------------------------------------------------------------

def test_nl_reasoning_max_chars_is_reasonable():
    """NL_REASONING_MAX_CHARS should be around 2000 (~500 tokens)."""
    assert NL_REASONING_MAX_CHARS == 2000
