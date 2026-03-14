"""Tests for tactic candidate generation — including semantic query generation."""

from bourbaki.autonomous.tactics import (
    filter_blocked_tactics,
    generate_candidates,
    generate_mathlib_queries,
    is_blocked_tactic,
)


def test_generate_mathlib_queries_includes_semantic():
    """Semantic queries should be generated alongside type and natural."""
    goals = ["⊢ 0 < m → 0 < n → 0 < m * n"]
    queries = generate_mathlib_queries(goals)
    modes = [mode for _, mode in queries]
    assert "semantic" in modes
    assert "type" in modes


def test_generate_mathlib_queries_semantic_content():
    """Semantic query should contain meaningful goal-derived text."""
    goals = ["⊢ ∀ n : ℕ, n + 0 = n"]
    queries = generate_mathlib_queries(goals)
    semantic_queries = [(q, m) for q, m in queries if m == "semantic"]
    assert len(semantic_queries) >= 1
    assert len(semantic_queries[0][0]) > 5  # Should be meaningful


def test_generate_mathlib_queries_empty_goals():
    """No queries generated for empty goals."""
    assert generate_mathlib_queries([]) == []


def test_generate_candidates_with_mathlib():
    """Candidates should include exact/apply for Mathlib results."""
    goals = ["⊢ 1 + 1 = 2"]
    mathlib = [{"name": "Nat.add_comm", "type": "∀ m n, m + n = n + m"}]
    candidates = generate_candidates(goals, mathlib_results=mathlib)
    assert "exact Nat.add_comm" in candidates
    assert "apply Nat.add_comm" in candidates


# ---------------------------------------------------------------------------
# Blocklist tests (issue #3: exact ⟨_, _⟩ false positives)
# ---------------------------------------------------------------------------


def test_blocked_anonymous_constructors():
    """Anonymous constructor tactics with only wildcards should be blocked."""
    assert is_blocked_tactic("exact ⟨_, _⟩")
    assert is_blocked_tactic("exact ⟨_⟩")
    assert is_blocked_tactic("exact ⟨_, _, _⟩")
    assert is_blocked_tactic("exact ⟨_, _, _, _⟩")
    assert is_blocked_tactic("exact ⟨_, _, _, _, _⟩")  # regex catches any count
    assert is_blocked_tactic("refine ⟨?_, ?_⟩")
    assert is_blocked_tactic("refine ⟨?_, ?_, ?_⟩")
    assert is_blocked_tactic("refine ⟨?_, ?_, ?_, ?_⟩")


def test_blocked_bogus_lemmas():
    """Known false-positive lemma applications should be blocked."""
    assert is_blocked_tactic("exact mem_of")
    assert is_blocked_tactic("apply Set.mem_of_mem_filter")


def test_blocked_typeclass_instances():
    """Lean internals and typeclass instances should be blocked."""
    assert is_blocked_tactic("exact instDecidableEqRat")
    assert is_blocked_tactic("exact Lean.defaultMaxRecDepth")
    assert is_blocked_tactic("exact Float.toRatParts")
    assert is_blocked_tactic("exact Real.commRing")
    assert is_blocked_tactic("exact Real.instAdd")


def test_legitimate_tactics_not_blocked():
    """Normal mathematical tactics should NOT be blocked."""
    assert not is_blocked_tactic("exact Nat.succ_pos n")
    assert not is_blocked_tactic("exact ⟨n, rfl⟩")  # real arguments, not just wildcards
    assert not is_blocked_tactic("exact ⟨h, fun x => by simp⟩")
    assert not is_blocked_tactic("constructor")
    assert not is_blocked_tactic("simp")
    assert not is_blocked_tactic("ring")
    assert not is_blocked_tactic("exact h")
    assert not is_blocked_tactic("apply Nat.Prime.eq_one_or_self_of_dvd")


def test_filter_blocked_tactics():
    """filter_blocked_tactics should remove only blocked entries."""
    input_tactics = ["simp", "exact ⟨_, _⟩", "ring", "refine ⟨?_, ?_⟩", "omega"]
    result = filter_blocked_tactics(input_tactics)
    assert result == ["simp", "ring", "omega"]


def test_generate_candidates_excludes_blocked():
    """generate_candidates output should never contain blocked tactics."""
    # Use a goal pattern that might trigger constructor-like suggestions
    goals = ["⊢ ∃ n : ℕ, n = 0"]
    candidates = generate_candidates(goals)
    for c in candidates:
        assert not is_blocked_tactic(c), f"Blocked tactic in candidates: {c!r}"
