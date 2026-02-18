"""Tests for proof sketch generator."""

import json
from bourbaki.autonomous.sketch import (
    ProofSketch,
    SketchStep,
    SketchContext,
    parse_sketch_response,
    build_sketch_prompt,
    SKETCH_PROMPT,
    SUBGOAL_SKETCH_PROMPT,
)


def test_sketch_step_creation():
    step = SketchStep(
        statement="Show base case n = 0",
        formal_type="0 * (0 + 1) / 2 = 0",
    )
    assert step.statement == "Show base case n = 0"
    assert step.depends_on == []


def test_proof_sketch_creation():
    sketch = ProofSketch(
        strategy="induction",
        steps=[
            SketchStep(statement="Base case", formal_type="P 0"),
            SketchStep(statement="Inductive step", formal_type="P n → P (n+1)",
                       depends_on=[0]),
        ],
        key_lemmas=["Nat.add_comm"],
    )
    assert sketch.strategy == "induction"
    assert len(sketch.steps) == 2
    assert sketch.steps[1].depends_on == [0]


def test_proof_sketch_with_final_step():
    sketch = ProofSketch(
        strategy="have_chain",
        steps=[
            SketchStep(statement="Step A", formal_type="1 + 1 = 2"),
        ],
        key_lemmas=[],
        final_step="linarith [step_0]",
    )
    assert sketch.final_step == "linarith [step_0]"


def test_proof_sketch_default_final_step():
    sketch = ProofSketch(
        strategy="direct",
        steps=[],
        key_lemmas=[],
    )
    assert sketch.final_step is None


def test_parse_sketch_response_valid():
    response = json.dumps({
        "sketches": [{
            "strategy": "direct",
            "steps": [
                {"statement": "Simplify LHS", "formal_type": "a + 0 = a"},
                {"statement": "Apply commutativity", "formal_type": "a = a"},
            ],
            "key_lemmas": ["Nat.add_zero"],
        }]
    })
    sketches = parse_sketch_response(response)
    assert len(sketches) == 1
    assert sketches[0].strategy == "direct"
    assert len(sketches[0].steps) == 2


def test_parse_sketch_response_with_final_step():
    response = json.dumps({
        "sketches": [{
            "strategy": "have_chain",
            "steps": [
                {"statement": "Step", "formal_type": "A"},
            ],
            "key_lemmas": [],
            "final_step": "exact step_0",
        }]
    })
    sketches = parse_sketch_response(response)
    assert len(sketches) == 1
    assert sketches[0].final_step == "exact step_0"


def test_parse_sketch_response_filters_steps_without_formal_type():
    """Steps without formal_type should be filtered out."""
    response = json.dumps({
        "sketches": [{
            "strategy": "direct",
            "steps": [
                {"statement": "Has type", "formal_type": "1 + 1 = 2"},
                {"statement": "No type"},  # Missing formal_type
                {"statement": "Also has type", "formal_type": "True"},
            ],
            "key_lemmas": [],
        }]
    })
    sketches = parse_sketch_response(response)
    assert len(sketches) == 1
    assert len(sketches[0].steps) == 2  # Only steps with formal_type


def test_parse_sketch_response_multiple():
    response = json.dumps({
        "sketches": [
            {"strategy": "induction", "steps": [{"statement": "Induct", "formal_type": "A"}], "key_lemmas": []},
            {"strategy": "direct", "steps": [{"statement": "Simplify", "formal_type": "B"}], "key_lemmas": []},
        ]
    })
    sketches = parse_sketch_response(response)
    assert len(sketches) == 2


def test_parse_sketch_response_from_markdown():
    """Handle LLM responses wrapped in markdown code blocks."""
    response = '```json\n{"sketches": [{"strategy": "direct", "steps": [{"statement": "Done", "formal_type": "True"}], "key_lemmas": []}]}\n```'
    sketches = parse_sketch_response(response)
    assert len(sketches) == 1


def test_parse_sketch_response_invalid_returns_empty():
    sketches = parse_sketch_response("this is not json at all")
    assert sketches == []


def test_sketch_context():
    ctx = SketchContext(
        theorem="theorem foo : 1 + 1 = 2",
        mathlib_results=[{"name": "Nat.add_comm"}],
    )
    assert "foo" in ctx.theorem


def test_build_sketch_prompt_depth_0():
    """At depth 0, should use the main SKETCH_PROMPT."""
    ctx = SketchContext(
        theorem="theorem foo : True",
        depth=0,
    )
    prompt = build_sketch_prompt(ctx)
    assert "INDEPENDENTLY PROVABLE" in prompt
    assert "SUBGOAL" not in prompt


def test_build_sketch_prompt_depth_1():
    """At depth 1, should use main prompt with subgoal note."""
    ctx = SketchContext(
        theorem="theorem foo : True",
        depth=1,
    )
    prompt = build_sketch_prompt(ctx)
    assert "subgoal" in prompt.lower()


def test_build_sketch_prompt_depth_2():
    """At depth 2+, should use the SUBGOAL_SKETCH_PROMPT."""
    ctx = SketchContext(
        theorem="theorem foo : True",
        depth=2,
    )
    prompt = build_sketch_prompt(ctx)
    assert "SUBGOAL" in prompt
    assert "recursion depth 2" in prompt


def test_build_sketch_prompt_previous_attempts():
    """Previous attempts should be included in the prompt."""
    ctx = SketchContext(
        theorem="theorem foo : True",
        previous_attempts=["Failed with induction", "Failed with contradiction"],
        depth=0,
    )
    prompt = build_sketch_prompt(ctx)
    assert "Failed with induction" in prompt
    assert "Failed with contradiction" in prompt


def test_build_sketch_prompt_mathlib_results():
    """Mathlib results should appear in context."""
    ctx = SketchContext(
        theorem="theorem foo : True",
        mathlib_results=[{"name": "Nat.add_zero"}, {"name": "Nat.mul_comm"}],
        depth=0,
    )
    prompt = build_sketch_prompt(ctx)
    assert "Nat.add_zero" in prompt
    assert "Nat.mul_comm" in prompt
