"""Proof state scoring for best-first search.

v1 uses heuristic scoring based on goal count and complexity.
Future versions can use LLM-based or trained value functions.
"""

from __future__ import annotations


def score_proof_state(goals: list[str], depth: int) -> float:
    """Score a proof state — lower is more promising (for min-heap).

    Heuristic: fewer remaining goals + simpler goals + shallower depth = better.

    Args:
        goals: List of remaining goal strings from Lean.
        depth: Current depth in the search tree.

    Returns:
        Float score (lower = more promising).
    """
    if not goals:
        return 0.0  # No goals = proof complete, best possible score

    # Goal count: each remaining goal adds 10 points
    goal_count_score = len(goals) * 10.0

    # Goal complexity: use goal string length as a rough proxy
    # Shorter goals tend to be simpler (e.g., "⊢ True" vs "⊢ ∀ x, ...")
    avg_complexity = sum(len(g) for g in goals) / len(goals)
    complexity_score = min(avg_complexity / 20.0, 10.0)  # Cap at 10

    # Depth penalty: slight preference for shallower proofs
    depth_score = depth * 0.5

    return goal_count_score + complexity_score + depth_score


def goal_matches_pattern(goal: str, pattern: str) -> bool:
    """Check if a goal roughly matches a pattern for tactic selection."""
    return pattern in goal
