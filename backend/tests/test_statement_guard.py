"""Tests for the anti-trivialization guard in :mod:`bourbaki.tools.statement_guard`.

The guard blocks runs, so its false-reject behaviour matters more than its
recall: a wrongly rejected formalization silently lowers a benchmark pass rate,
which is the failure mode this work exists to prevent. The "must pass" cases
below are therefore the load-bearing half of this file.
"""

from __future__ import annotations

from bourbaki.tools.statement_guard import check_trivialization

# ---------------------------------------------------------------------------
# Rejects — targets that assert nothing.
# ---------------------------------------------------------------------------


def test_rejects_target_true():
    v = check_trivialization("theorem foo : True := trivial")
    assert v.is_trivial
    assert v.findings[0].decl_name == "foo"
    assert v.findings[0].target == "True"


def test_rejects_target_false():
    assert check_trivialization("theorem bad : False := by sorry").is_trivial


def test_rejects_parenthesized_target():
    assert check_trivialization("theorem foo : (True) := trivial").is_trivial
    assert check_trivialization("theorem foo : ((True)) := trivial").is_trivial


def test_rejects_with_binders_present():
    """Binders contain `:` — the guard must find the depth-zero one."""
    code = "theorem foo (n : ℕ) (h : n > 0) : True := trivial"
    v = check_trivialization(code)
    assert v.is_trivial
    assert v.findings[0].target == "True"


def test_rejects_with_instance_and_implicit_binders():
    code = "theorem foo {α : Type*} [Fintype α] (s : Finset α) : True := trivial"
    assert check_trivialization(code).is_trivial


def test_rejects_lemma_and_example_forms():
    assert check_trivialization("lemma foo : True := trivial").is_trivial
    assert check_trivialization("example : True := trivial").is_trivial


def test_rejects_multiline_declaration():
    code = """theorem foo
    (n : ℕ)
    (h : 0 < n) :
    True := by
  trivial"""
    assert check_trivialization(code).is_trivial


def test_finds_vacuous_decl_among_real_ones():
    code = """theorem real_one (n : ℕ) : n + 0 = n := by simp

theorem sneaky : True := trivial

theorem another (a b : ℕ) : a + b = b + a := by ring"""
    v = check_trivialization(code)
    assert v.is_trivial
    assert len(v.findings) == 1
    assert v.findings[0].decl_name == "sneaky"
    assert v.findings[0].line == 3


def test_trivial_body_is_recorded_as_supporting_evidence():
    v = check_trivialization("theorem foo : True := trivial")
    assert "closed by `trivial`" in v.findings[0].reason


def test_summary_names_the_offending_declaration():
    v = check_trivialization("theorem sneaky : True := trivial")
    assert "sneaky" in v.summary()


# ---------------------------------------------------------------------------
# Must pass — a false reject here silently lowers a pass rate.
# ---------------------------------------------------------------------------


def test_accepts_ordinary_theorem():
    assert not check_trivialization("theorem foo (n : ℕ) : n + 0 = n := by simp").is_trivial


def test_accepts_real_statement_closed_by_rfl():
    """A trivial body never rejects on its own — only a vacuous target does."""
    assert not check_trivialization("theorem foo : 1 + 1 = 2 := rfl").is_trivial
    assert not check_trivialization("theorem foo : 2 = 2 := by rfl").is_trivial


def test_accepts_target_mentioning_true_as_a_subterm():
    """`True` inside a larger proposition is not a vacuous target."""
    cases = [
        "theorem foo (p : Prop) : p ∨ True → True ∨ p := by tauto",
        "theorem foo (p : Prop) (h : p ↔ True) : p := by simp_all",
        "theorem foo : ¬False := by simp",
    ]
    for code in cases:
        assert not check_trivialization(code).is_trivial, code


def test_accepts_decidable_true_style_identifiers():
    """Identifiers that merely start with True must not match."""
    code = "theorem foo (h : TrueSet α) : Nonempty (TrueSet α) := ⟨h⟩"
    assert not check_trivialization(code).is_trivial


def test_accepts_statement_with_sorry_body():
    """A `sorry` body is the reviewer's business, not the guard's."""
    assert not check_trivialization("theorem foo (n : ℕ) : n = n := by sorry").is_trivial


def test_accepts_definitions_and_non_theorem_decls():
    code = """def myTrue : Prop := True

instance : Inhabited ℕ := ⟨0⟩"""
    assert not check_trivialization(code).is_trivial


# ---------------------------------------------------------------------------
# Degenerate input — must never raise, must default to passing.
# ---------------------------------------------------------------------------


def test_empty_and_whitespace_input_passes():
    for code in ["", "   ", "\n\n"]:
        assert not check_trivialization(code).is_trivial


def test_unparseable_input_passes_rather_than_rejecting():
    for code in ["theorem", "theorem foo", "not lean at all", "((("]:
        assert not check_trivialization(code).is_trivial, code


def test_comments_cannot_trigger_a_reject():
    code = """-- theorem fake : True := trivial
/- theorem alsoFake : True := trivial -/
theorem real (n : ℕ) : n = n := rfl"""
    assert not check_trivialization(code).is_trivial


def test_nested_block_comments_are_handled():
    code = """/- outer /- inner : True -/ still outer -/
theorem real (n : ℕ) : n = n := rfl"""
    assert not check_trivialization(code).is_trivial


def test_comment_stripping_preserves_line_numbers():
    code = """-- padding
-- padding
theorem sneaky : True := trivial"""
    v = check_trivialization(code)
    assert v.is_trivial
    assert v.findings[0].line == 3


def test_verdict_summary_is_stable_when_clean():
    assert check_trivialization("theorem foo : 1 = 1 := rfl").summary() == (
        "no vacuous statement detected"
    )
