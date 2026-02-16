"""Tests for sketch-to-Lean formalizer."""

import re
from bourbaki.autonomous.formalizer import (
    FormalizedSkeleton,
    Subgoal,
    build_skeleton_code,
    extract_subgoals_from_code,
    stitch_proofs,
)
from bourbaki.autonomous.sketch import ProofSketch, SketchStep


def test_build_skeleton_simple():
    sketch = ProofSketch(
        strategy="direct",
        steps=[
            SketchStep(statement="Simplify", formal_type="a + 0 = a"),
        ],
        key_lemmas=[],
    )
    theorem = "theorem foo (a : Nat) : a + 0 = a"
    code = build_skeleton_code(theorem, sketch)
    assert "have" in code
    assert "sorry" in code
    assert theorem in code or "foo" in code


def test_build_skeleton_multi_step():
    sketch = ProofSketch(
        strategy="induction",
        steps=[
            SketchStep(statement="Base case", formal_type="P 0"),
            SketchStep(statement="Inductive step", formal_type="∀ n, P n → P (n+1)"),
        ],
        key_lemmas=[],
    )
    theorem = "theorem bar : ∀ n, P n"
    code = build_skeleton_code(theorem, sketch)
    sorry_count = code.count("sorry")
    assert sorry_count >= 2  # At least one sorry per step


def test_extract_subgoals():
    code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  have step_1 : 2 + 2 = 4 := by sorry
  trivial"""
    subgoals = extract_subgoals_from_code(code)
    assert len(subgoals) == 2
    assert subgoals[0].label == "step_0"
    assert subgoals[0].lean_type == "1 + 1 = 2"
    assert subgoals[1].label == "step_1"


def test_stitch_proofs_replaces_sorry():
    skeleton_code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  trivial"""
    subgoal_proofs = {"step_0": ["norm_num"]}
    result = stitch_proofs(skeleton_code, subgoal_proofs)
    assert "sorry" not in result
    assert "norm_num" in result


def test_stitch_proofs_multi_tactic():
    skeleton_code = """\
theorem foo : True := by
  have step_0 : P := by sorry
  trivial"""
    subgoal_proofs = {"step_0": ["simp", "ring"]}
    result = stitch_proofs(skeleton_code, subgoal_proofs)
    assert "sorry" not in result
    assert "simp" in result
    assert "ring" in result
