"""Tests for multi-agent proof coordinator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from bourbaki.agent.coordinator import ProofCoordinator, CoordinatorResult


@pytest.mark.asyncio
async def test_coordinator_prove_success():
    """Coordinator should orchestrate roles to produce a proof."""
    coord = ProofCoordinator(model="openai:gpt-4o")

    # Mock the strategy phase
    mock_strategy = {
        "sketch": ["intro n", "induction n", "ring"],
        "subgoals": ["base case: 0 + 0 = 0", "inductive step"],
    }

    # Mock the search phase
    mock_lemmas = [
        {"name": "Nat.add_comm", "type": "forall m n, m + n = n + m"},
    ]

    # Mock the prover phase
    mock_proof = "theorem foo : 1 + 1 = 2 := by norm_num"

    with patch.object(coord, "_run_strategist", new_callable=AsyncMock, return_value=mock_strategy):
        with patch.object(coord, "_run_searcher", new_callable=AsyncMock, return_value=mock_lemmas):
            with patch.object(coord, "_run_prover", new_callable=AsyncMock, return_value=mock_proof):
                with patch.object(coord, "_run_verifier", new_callable=AsyncMock, return_value=True):
                    result = await coord.prove("theorem foo : 1 + 1 = 2")

    assert result.success is True
    assert result.proof_code == mock_proof


@pytest.mark.asyncio
async def test_coordinator_prove_retry_on_failure():
    """Coordinator should retry with new strategy on prover failure."""
    coord = ProofCoordinator(model="openai:gpt-4o")

    call_count = 0

    async def mock_prover(theorem, strategy, lemmas, nl_reasoning=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None  # First attempt fails
        return "theorem foo : True := trivial"

    with patch.object(coord, "_run_strategist", new_callable=AsyncMock, return_value={"sketch": ["trivial"], "subgoals": []}):
        with patch.object(coord, "_run_searcher", new_callable=AsyncMock, return_value=[]):
            with patch.object(coord, "_run_prover", side_effect=mock_prover):
                with patch.object(coord, "_run_verifier", new_callable=AsyncMock, return_value=True):
                    result = await coord.prove("theorem foo : True", max_retries=3)

    assert result.success is True
    assert call_count == 2


@pytest.mark.asyncio
async def test_coordinator_prove_exhausted():
    """Coordinator should return failure after exhausting retries."""
    coord = ProofCoordinator(model="openai:gpt-4o")

    with patch.object(coord, "_run_strategist", new_callable=AsyncMock, return_value={"sketch": [], "subgoals": []}):
        with patch.object(coord, "_run_searcher", new_callable=AsyncMock, return_value=[]):
            with patch.object(coord, "_run_prover", new_callable=AsyncMock, return_value=None):
                result = await coord.prove("theorem hard : False", max_retries=2)

    assert result.success is False
    assert "exhausted" in result.error.lower() or "failed" in result.error.lower()


@pytest.mark.asyncio
async def test_coordinator_result_dataclass():
    result = CoordinatorResult(
        success=True,
        proof_code="theorem x : True := trivial",
        agent_stats={"strategist": 1, "prover": 2},
    )
    assert result.success
    assert result.proof_code is not None
    assert result.agent_stats["prover"] == 2
