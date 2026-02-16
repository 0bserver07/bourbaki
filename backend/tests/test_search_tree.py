"""Tests for proof search tree UCB exploration."""

import math
from bourbaki.autonomous.search_tree import ProofNode, ucb_adjusted_score


def test_ucb_no_parent_returns_base_score():
    node = ProofNode(proof_state=0, goals=["⊢ True"], tactic_history=[], score=10.0)
    assert ucb_adjusted_score(node) == 10.0


def test_ucb_unvisited_child_gets_bonus():
    parent = ProofNode(proof_state=0, goals=["⊢ True"], tactic_history=[], visits=10)
    child = ProofNode(proof_state=1, goals=["⊢ True"], tactic_history=["simp"],
                      parent=parent, score=10.0, visits=0)
    adjusted = ucb_adjusted_score(child)
    assert adjusted < 10.0  # Should get exploration bonus (lower = better)


def test_ucb_visited_child_less_bonus():
    parent = ProofNode(proof_state=0, goals=["⊢ True"], tactic_history=[], visits=10)
    child_new = ProofNode(proof_state=1, goals=["⊢ True"], tactic_history=["simp"],
                          parent=parent, score=10.0, visits=0)
    child_old = ProofNode(proof_state=2, goals=["⊢ True"], tactic_history=["ring"],
                          parent=parent, score=10.0, visits=5)
    assert ucb_adjusted_score(child_new) < ucb_adjusted_score(child_old)


def test_proof_node_is_complete():
    complete = ProofNode(proof_state=0, goals=[], tactic_history=["ring"])
    incomplete = ProofNode(proof_state=0, goals=["⊢ True"], tactic_history=[])
    assert complete.is_complete
    assert not incomplete.is_complete
