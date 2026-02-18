"""Tests for proof search tree UCB exploration and LSP integration."""

import asyncio
import math
from unittest.mock import AsyncMock, patch

import pytest

from bourbaki.autonomous.search_tree import ProofNode, ProofSearchTree, ucb_adjusted_score


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


# ---------------------------------------------------------------------------
# LSP tactic integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_lsp_tactics_returns_new_tactics():
    """LSP suggestions that are not already in candidates should be returned."""
    tree = ProofSearchTree("theorem foo : 1 + 1 = 2")
    node = ProofNode(
        proof_state=0,
        goals=["⊢ 1 + 1 = 2"],
        tactic_history=[],
        depth=0,
    )
    existing = {"simp", "ring", "omega"}

    with patch(
        "bourbaki.autonomous.search_tree.lsp_suggest_tactics",
        new_callable=AsyncMock,
        return_value=["ring", "norm_num", "decide", "simp"],
    ):
        result = await tree._fetch_lsp_tactics(node, existing, timeout=5.0)

    # "ring" and "simp" already exist — only "norm_num" and "decide" are new
    assert "norm_num" in result
    assert "decide" in result
    assert "ring" not in result
    assert "simp" not in result


@pytest.mark.asyncio
async def test_fetch_lsp_tactics_handles_timeout():
    """LSP timeout should return an empty list, not raise."""
    tree = ProofSearchTree("theorem foo : 1 + 1 = 2")
    node = ProofNode(
        proof_state=0,
        goals=["⊢ 1 + 1 = 2"],
        tactic_history=[],
        depth=0,
    )

    async def slow_lsp(**kwargs):
        await asyncio.sleep(10)  # Will exceed the 0.01s timeout
        return ["ring"]

    with patch(
        "bourbaki.autonomous.search_tree.lsp_suggest_tactics",
        side_effect=slow_lsp,
    ):
        result = await tree._fetch_lsp_tactics(node, set(), timeout=0.01)

    assert result == []


@pytest.mark.asyncio
async def test_fetch_lsp_tactics_handles_exception():
    """LSP errors should be swallowed and return an empty list."""
    tree = ProofSearchTree("theorem foo : 1 + 1 = 2")
    node = ProofNode(
        proof_state=0,
        goals=["⊢ 1 + 1 = 2"],
        tactic_history=[],
        depth=0,
    )

    with patch(
        "bourbaki.autonomous.search_tree.lsp_suggest_tactics",
        new_callable=AsyncMock,
        side_effect=ConnectionError("LSP server not running"),
    ):
        result = await tree._fetch_lsp_tactics(node, set(), timeout=5.0)

    assert result == []


@pytest.mark.asyncio
async def test_fetch_lsp_tactics_builds_sorry_code_with_history():
    """LSP call should receive code with the current tactic history + sorry."""
    tree = ProofSearchTree("theorem foo : 1 + 1 = 2")
    node = ProofNode(
        proof_state=0,
        goals=["⊢ 1 + 1 = 2"],
        tactic_history=["simp", "ring_nf"],
        depth=2,
    )

    captured_code = None

    async def capture_lsp(*, theorem, **kwargs):
        nonlocal captured_code
        captured_code = theorem
        return ["omega"]

    with patch(
        "bourbaki.autonomous.search_tree.lsp_suggest_tactics",
        side_effect=capture_lsp,
    ):
        await tree._fetch_lsp_tactics(node, set(), timeout=5.0)

    assert captured_code is not None
    assert "simp" in captured_code
    assert "ring_nf" in captured_code
    assert "sorry" in captured_code


@pytest.mark.asyncio
async def test_fetch_lsp_tactics_strips_whitespace():
    """Whitespace-only and padded labels should be handled correctly."""
    tree = ProofSearchTree("theorem foo : 1 + 1 = 2")
    node = ProofNode(
        proof_state=0,
        goals=["⊢ 1 + 1 = 2"],
        tactic_history=[],
        depth=0,
    )

    with patch(
        "bourbaki.autonomous.search_tree.lsp_suggest_tactics",
        new_callable=AsyncMock,
        return_value=["  ring  ", "", "  ", "omega"],
    ):
        result = await tree._fetch_lsp_tactics(node, set(), timeout=5.0)

    # Empty / whitespace-only strings should be excluded
    assert "" not in result
    assert "ring" in result
    assert "omega" in result
