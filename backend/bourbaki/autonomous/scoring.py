"""Proof state scoring for best-first search.

v2 adds novelty bonus (intrinsic reward) inspired by DeepSeek-Prover V1.5:
states that reach previously-unseen goal configurations get a score bonus.
"""

from __future__ import annotations


class NoveltyTracker:
    """Tracks seen goal states for novelty-based exploration bonus."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def _key(self, goals: list[str]) -> str:
        return "|".join(sorted(goals))

    def has_seen(self, goals: list[str]) -> bool:
        return self._key(goals) in self._seen

    def mark_seen(self, goals: list[str]) -> None:
        self._seen.add(self._key(goals))

    @property
    def seen_count(self) -> int:
        return len(self._seen)


def score_proof_state(
    goals: list[str],
    depth: int,
    novelty_tracker: NoveltyTracker | None = None,
    novelty_bonus: float = 3.0,
) -> float:
    """Score a proof state — lower is more promising (for min-heap).

    Heuristic: fewer remaining goals + simpler goals + shallower depth = better.
    Novel states (first time seeing this goal set) get a bonus (lower score).

    Args:
        goals: List of remaining goal strings from Lean.
        depth: Current depth in the search tree.
        novelty_tracker: Optional tracker for novelty bonus.
        novelty_bonus: Score reduction for novel states (default 3.0).

    Returns:
        Float score (lower = more promising).
    """
    if not goals:
        return 0.0  # No goals = proof complete, best possible score

    # Goal count: each remaining goal adds 10 points
    goal_count_score = len(goals) * 10.0

    # Goal complexity: use goal string length as a rough proxy
    avg_complexity = sum(len(g) for g in goals) / len(goals)
    complexity_score = min(avg_complexity / 20.0, 10.0)  # Cap at 10

    # Depth penalty: slight preference for shallower proofs
    depth_score = depth * 0.5

    base_score = goal_count_score + complexity_score + depth_score

    # Novelty bonus: reduce score for never-before-seen goal states
    if novelty_tracker is not None:
        if not novelty_tracker.has_seen(goals):
            novelty_tracker.mark_seen(goals)
            base_score -= novelty_bonus

    return base_score


def goal_matches_pattern(goal: str, pattern: str) -> bool:
    """Check if a goal roughly matches a pattern for tactic selection."""
    return pattern in goal
