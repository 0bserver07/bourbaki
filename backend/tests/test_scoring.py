"""Tests for proof state scoring with novelty bonus."""

from bourbaki.autonomous.scoring import score_proof_state, NoveltyTracker


def test_completed_proof_scores_zero():
    tracker = NoveltyTracker()
    assert score_proof_state([], 0, tracker) == 0.0


def test_fewer_goals_scores_lower():
    tracker = NoveltyTracker()
    one_goal = score_proof_state(["⊢ True"], 0, tracker)
    two_goals = score_proof_state(["⊢ True", "⊢ False"], 0, tracker)
    assert one_goal < two_goals


def test_novelty_bonus_for_unseen_state():
    tracker = NoveltyTracker()
    first = score_proof_state(["⊢ a + b = b + a"], 1, tracker)
    # Same goals again — no novelty bonus
    second = score_proof_state(["⊢ a + b = b + a"], 1, tracker)
    assert first < second  # First visit gets bonus (lower = better)


def test_novelty_tracker_tracks_seen():
    tracker = NoveltyTracker()
    assert not tracker.has_seen(["⊢ True"])
    tracker.mark_seen(["⊢ True"])
    assert tracker.has_seen(["⊢ True"])


def test_novelty_tracker_order_independent():
    tracker = NoveltyTracker()
    tracker.mark_seen(["⊢ A", "⊢ B"])
    assert tracker.has_seen(["⊢ B", "⊢ A"])
