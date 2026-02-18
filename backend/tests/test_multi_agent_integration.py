"""Integration test for multi-agent proof pipeline.

Tests the full flow: Coordinator -> Strategist -> Searcher -> Prover -> Verifier,
including the retry path on prover failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from bourbaki.agent.coordinator import ProofCoordinator, CoordinatorResult
from bourbaki.agent.messages import MessageBus
from bourbaki.agent.roles import ALL_ROLES, STRATEGIST, PROVER, VERIFIER


@pytest.mark.asyncio
async def test_full_pipeline_success():
    """Full pipeline: strategy -> search -> prove -> verify -> success."""
    coord = ProofCoordinator(model="openai:gpt-4o")

    # Mock strategy generation
    async def mock_strategist(theorem, previous_errors, nl_reasoning=None):
        return {
            "sketch": ["intro n", "induction n", "simp", "ring"],
            "subgoals": ["base case: P(0)", "inductive step: P(n) -> P(n+1)"],
        }

    # Mock lemma search
    async def mock_searcher(theorem, subgoals):
        return [
            {"name": "Nat.add_comm", "type": "forall m n, m + n = n + m", "module": "Mathlib"},
            {"name": "Nat.succ_pos", "type": "forall n, 0 < n + 1", "module": "Mathlib"},
        ]

    # Mock proof construction
    async def mock_prover(theorem, strategy, lemmas, nl_reasoning=None):
        return (
            "import Mathlib.Tactic\n\n"
            "theorem test_sum : forall n : Nat, 2 * (Finset.range (n + 1)).sum id = n * (n + 1) := by\n"
            "  intro n\n"
            "  induction n with\n"
            "  | zero => simp\n"
            "  | succ n ih => simp [Finset.sum_range_succ]; omega\n"
        )

    # Mock verification
    async def mock_verifier(proof_code):
        return True

    with patch.object(coord, "_run_strategist", side_effect=mock_strategist):
        with patch.object(coord, "_run_searcher", side_effect=mock_searcher):
            with patch.object(coord, "_run_prover", side_effect=mock_prover):
                with patch.object(coord, "_run_verifier", side_effect=mock_verifier):
                    result = await coord.prove(
                        "theorem test_sum : forall n : Nat, 2 * (Finset.range (n + 1)).sum id = n * (n + 1)",
                        max_retries=3,
                    )

    assert result.success is True
    assert result.proof_code is not None
    assert "theorem" in result.proof_code
    assert result.agent_stats["strategist"] == 1
    assert result.agent_stats["prover"] == 1
    assert result.agent_stats["verifier"] == 1


@pytest.mark.asyncio
async def test_pipeline_retry_after_prover_failure():
    """Pipeline retry: prover fails -> back to strategist -> second attempt succeeds."""
    coord = ProofCoordinator(model="openai:gpt-4o")

    strategy_calls = 0

    async def mock_strategist(theorem, previous_errors, nl_reasoning=None):
        nonlocal strategy_calls
        strategy_calls += 1
        if strategy_calls == 1:
            return {"sketch": ["ring"], "subgoals": []}
        return {"sketch": ["norm_num"], "subgoals": []}

    async def mock_searcher(theorem, subgoals):
        return []

    prover_calls = 0

    async def mock_prover(theorem, strategy, lemmas, nl_reasoning=None):
        nonlocal prover_calls
        prover_calls += 1
        if prover_calls == 1:
            return None  # First attempt fails
        return "theorem t : 1 + 1 = 2 := by norm_num"

    async def mock_verifier(proof_code):
        return True

    with patch.object(coord, "_run_strategist", side_effect=mock_strategist):
        with patch.object(coord, "_run_searcher", side_effect=mock_searcher):
            with patch.object(coord, "_run_prover", side_effect=mock_prover):
                with patch.object(coord, "_run_verifier", side_effect=mock_verifier):
                    result = await coord.prove("theorem t : 1 + 1 = 2", max_retries=3)

    assert result.success is True
    assert strategy_calls == 2  # Called twice (retry after failure)
    assert prover_calls == 2
    assert result.agent_stats["strategist"] == 2
    assert result.agent_stats["prover"] == 2


@pytest.mark.asyncio
async def test_pipeline_verification_failure_triggers_retry():
    """Verifier rejects proof -> back to strategist -> retry."""
    coord = ProofCoordinator(model="openai:gpt-4o")

    verifier_calls = 0

    async def mock_strategist(theorem, previous_errors, nl_reasoning=None):
        return {"sketch": ["simp"], "subgoals": []}

    async def mock_searcher(theorem, subgoals):
        return []

    async def mock_prover(theorem, strategy, lemmas, nl_reasoning=None):
        return "theorem t : True := by simp"

    async def mock_verifier(proof_code):
        nonlocal verifier_calls
        verifier_calls += 1
        return verifier_calls >= 2  # Fail first, succeed second

    with patch.object(coord, "_run_strategist", side_effect=mock_strategist):
        with patch.object(coord, "_run_searcher", side_effect=mock_searcher):
            with patch.object(coord, "_run_prover", side_effect=mock_prover):
                with patch.object(coord, "_run_verifier", side_effect=mock_verifier):
                    result = await coord.prove("theorem t : True", max_retries=3)

    assert result.success is True
    assert verifier_calls == 2


@pytest.mark.asyncio
async def test_message_bus_integration():
    """MessageBus routes messages between agent roles correctly."""
    bus = MessageBus()

    from bourbaki.agent.messages import AgentMessage

    # Strategist sends to Prover
    strategy_msg = AgentMessage(
        from_agent="strategist",
        to_agent="prover",
        msg_type="strategy",
        content={"sketch": ["ring", "simp"]},
    )
    await bus.send(strategy_msg)

    # Prover receives
    received = await bus.receive("prover", timeout=1.0)
    assert received is not None
    assert received.msg_type == "strategy"

    # Prover sends result to Verifier
    proof_msg = AgentMessage(
        from_agent="prover",
        to_agent="verifier",
        msg_type="proof_state",
        content={"code": "theorem t : True := trivial"},
    )
    await bus.send(proof_msg)

    # Verifier receives
    received = await bus.receive("verifier", timeout=1.0)
    assert received is not None
    assert received.msg_type == "proof_state"

    # Strategist should NOT receive the proof message
    not_received = await bus.receive("strategist", timeout=0.1)
    assert not_received is None


@pytest.mark.asyncio
async def test_roles_tool_isolation():
    """Each role should only have access to its designated tools."""
    from bourbaki.agent.roles import STRATEGIST, SEARCHER, PROVER, VERIFIER

    # No overlap between prover and verifier tools
    prover_tools = set(PROVER.tools)
    verifier_tools = set(VERIFIER.tools)
    assert prover_tools & verifier_tools == set()  # No overlap

    # Strategist shouldn't have lean tools
    assert "lean_prover" not in STRATEGIST.tools
    assert "lean_tactic" not in STRATEGIST.tools
