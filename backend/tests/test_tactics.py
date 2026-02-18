"""Tests for tactic candidate generation — including semantic query generation."""

from bourbaki.autonomous.tactics import generate_candidates, generate_mathlib_queries


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
