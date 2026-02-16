"""Tests for proof sketch generator."""

import json
from bourbaki.autonomous.sketch import (
    ProofSketch,
    SketchStep,
    SketchContext,
    parse_sketch_response,
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


def test_parse_sketch_response_multiple():
    response = json.dumps({
        "sketches": [
            {"strategy": "induction", "steps": [{"statement": "Induct"}], "key_lemmas": []},
            {"strategy": "direct", "steps": [{"statement": "Simplify"}], "key_lemmas": []},
        ]
    })
    sketches = parse_sketch_response(response)
    assert len(sketches) == 2


def test_parse_sketch_response_from_markdown():
    """Handle LLM responses wrapped in markdown code blocks."""
    response = '```json\n{"sketches": [{"strategy": "direct", "steps": [{"statement": "Done"}], "key_lemmas": []}]}\n```'
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
