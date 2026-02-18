"""Tests for sketch-to-Lean formalizer."""

import re
from bourbaki.autonomous.formalizer import (
    FormalizedSkeleton,
    Subgoal,
    build_skeleton_code,
    extract_subgoals_from_code,
    extract_final_step,
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


def test_build_skeleton_with_final_step():
    """Test that final_step from sketch is used in the skeleton."""
    sketch = ProofSketch(
        strategy="have_chain",
        steps=[
            SketchStep(statement="Step A", formal_type="1 + 1 = 2"),
        ],
        key_lemmas=[],
        final_step="linarith [step_0]",
    )
    theorem = "theorem foo : 2 > 0"
    code = build_skeleton_code(theorem, sketch)
    assert "linarith [step_0]" in code
    assert "exact?" not in code


def test_build_skeleton_without_final_step():
    """Test that exact? is used when no final_step."""
    sketch = ProofSketch(
        strategy="direct",
        steps=[
            SketchStep(statement="Step A", formal_type="1 + 1 = 2"),
        ],
        key_lemmas=[],
    )
    theorem = "theorem foo : True"
    code = build_skeleton_code(theorem, sketch)
    assert "exact?" in code


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


def test_extract_subgoals_with_dependencies():
    """Test that inter-subgoal dependencies are detected."""
    code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  have step_1 : step_0 → True := by sorry
  trivial"""
    subgoals = extract_subgoals_from_code(code)
    assert len(subgoals) == 2
    assert subgoals[0].depends_on == []
    assert "step_0" in subgoals[1].depends_on


def test_extract_subgoals_no_deps():
    """Test that independent subgoals have no dependencies."""
    code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  have step_1 : 3 + 3 = 6 := by sorry
  trivial"""
    subgoals = extract_subgoals_from_code(code)
    assert len(subgoals) == 2
    assert subgoals[0].depends_on == []
    assert subgoals[1].depends_on == []


def test_extract_final_step():
    code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  linarith [step_0]"""
    final = extract_final_step(code)
    assert final == "linarith [step_0]"


def test_extract_final_step_sorry():
    """sorry and exact? should return None."""
    code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  sorry"""
    final = extract_final_step(code)
    assert final is None


def test_extract_final_step_exact_question():
    code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  exact?"""
    final = extract_final_step(code)
    assert final is None


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


def test_stitch_proofs_multiple_subgoals():
    """Test stitching with multiple subgoals."""
    skeleton_code = """\
theorem foo : True := by
  have step_0 : 1 + 1 = 2 := by sorry
  have step_1 : 2 + 2 = 4 := by sorry
  trivial"""
    subgoal_proofs = {
        "step_0": ["norm_num"],
        "step_1": ["norm_num"],
    }
    result = stitch_proofs(skeleton_code, subgoal_proofs)
    assert "sorry" not in result
    assert result.count("norm_num") == 2


def test_subgoal_has_depends_on():
    """Test the new depends_on field on Subgoal."""
    sg = Subgoal(
        index=0,
        label="step_0",
        lean_type="1 + 1 = 2",
        depends_on=["step_prev"],
    )
    assert sg.depends_on == ["step_prev"]


def test_formalized_skeleton_has_final_step():
    """Test the new final_step field on FormalizedSkeleton."""
    skel = FormalizedSkeleton(
        code="theorem foo : True := by sorry",
        subgoals=[],
        final_step="trivial",
    )
    assert skel.final_step == "trivial"
