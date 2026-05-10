"""Tests for tactic replay through the REPL session.

Phase 3 deprecation:
    The original module also covered ``ProofSearchTree.expand_parallel``,
    ``best_first_search(parallel=N)``, ``prove_with_search``, and
    ``AutonomousSearchConfig`` — all of which lived in
    ``bourbaki.autonomous.search_tree`` / ``bourbaki.autonomous.search`` and
    were removed when the legacy autonomous pipeline was deleted.  The
    parallel-expansion tests went with them; what remains here is the
    coverage for ``LeanREPLSession.replay_tactics``, which is still in use
    inside ``bourbaki.tools.lean_repl`` and is independent of the deleted
    modules.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bourbaki.tools.lean_repl import LeanREPLSession


# ---------------------------------------------------------------------------
# replay_tactics() tests
# ---------------------------------------------------------------------------


class TestReplayTactics:
    """Tests for LeanREPLSession.replay_tactics()."""

    @pytest.mark.asyncio
    async def test_replay_empty_history(self):
        """Replaying an empty tactic history returns the initial proof state."""
        session = LeanREPLSession()
        session._initialized = True

        # Mock send_cmd to return the sorry response
        session.send_cmd = AsyncMock(return_value={
            "env": 1,
            "sorries": [{"proofState": 0, "goals": ["a : Nat\n⊢ a = a"]}],
        })
        # Mock start and ensure_initialized as no-ops
        session.proc = MagicMock()
        session.proc.returncode = None
        session.start = AsyncMock()
        session.ensure_initialized = AsyncMock()

        result = await session.replay_tactics("theorem foo : a = a", [])

        assert result["success"] is True
        assert result["proofState"] == 0
        assert len(result["goals"]) == 1
        # send_cmd called once (for sorry init), send_tactic never called
        session.send_cmd.assert_called_once()

    @pytest.mark.asyncio
    async def test_replay_single_tactic(self):
        """Replaying one tactic applies it to the initial proof state."""
        session = LeanREPLSession()
        session._initialized = True
        session.proc = MagicMock()
        session.proc.returncode = None
        session.start = AsyncMock()
        session.ensure_initialized = AsyncMock()

        session.send_cmd = AsyncMock(return_value={
            "env": 1,
            "sorries": [{"proofState": 0, "goals": ["⊢ 1 + 1 = 2"]}],
        })
        session.send_tactic = AsyncMock(return_value={
            "proofState": 1,
            "goals": [],
        })

        result = await session.replay_tactics(
            "theorem foo : 1 + 1 = 2", ["ring"],
        )

        assert result["success"] is True
        assert result["proofState"] == 1
        assert result["goals"] == []
        session.send_tactic.assert_called_once_with("ring", 0, timeout=120)

    @pytest.mark.asyncio
    async def test_replay_multiple_tactics(self):
        """Replaying multiple tactics chains proof states correctly."""
        session = LeanREPLSession()
        session._initialized = True
        session.proc = MagicMock()
        session.proc.returncode = None
        session.start = AsyncMock()
        session.ensure_initialized = AsyncMock()

        session.send_cmd = AsyncMock(return_value={
            "env": 1,
            "sorries": [{"proofState": 0, "goals": ["n : Nat\n⊢ n + 0 = n"]}],
        })

        call_count = 0

        async def mock_send_tactic(tactic, proof_state, timeout=60):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                assert tactic == "induction n"
                assert proof_state == 0
                return {"proofState": 1, "goals": ["⊢ 0 + 0 = 0", "n : Nat\nih : n + 0 = n\n⊢ n + 1 + 0 = n + 1"]}
            elif call_count == 2:
                assert tactic == "simp"
                assert proof_state == 1
                return {"proofState": 2, "goals": ["n : Nat\nih : n + 0 = n\n⊢ n + 1 + 0 = n + 1"]}
            else:
                assert tactic == "simp_all"
                assert proof_state == 2
                return {"proofState": 3, "goals": []}

        session.send_tactic = AsyncMock(side_effect=mock_send_tactic)

        result = await session.replay_tactics(
            "theorem foo : ∀ n : Nat, n + 0 = n",
            ["induction n", "simp", "simp_all"],
        )

        assert result["success"] is True
        assert result["proofState"] == 3
        assert result["goals"] == []
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_replay_fails_on_initialization_error(self):
        """If the theorem cannot be initialized, return an error."""
        session = LeanREPLSession()
        session._initialized = True
        session.proc = MagicMock()
        session.proc.returncode = None
        session.start = AsyncMock()
        session.ensure_initialized = AsyncMock()

        session.send_cmd = AsyncMock(return_value={
            "env": 1,
            "messages": [{"severity": "error", "data": "unknown identifier 'bad'"}],
        })

        result = await session.replay_tactics("theorem bad : bad", ["ring"])

        assert result["success"] is False
        assert "Failed to initialize theorem" in result["error"]

    @pytest.mark.asyncio
    async def test_replay_fails_mid_sequence(self):
        """If a tactic fails during replay, return error with failed_at index."""
        session = LeanREPLSession()
        session._initialized = True
        session.proc = MagicMock()
        session.proc.returncode = None
        session.start = AsyncMock()
        session.ensure_initialized = AsyncMock()

        session.send_cmd = AsyncMock(return_value={
            "env": 1,
            "sorries": [{"proofState": 0, "goals": ["⊢ True"]}],
        })

        async def mock_send_tactic(tactic, proof_state, timeout=60):
            if tactic == "bad_tactic":
                return {"error": "unknown tactic 'bad_tactic'"}
            return {"proofState": 1, "goals": []}

        session.send_tactic = AsyncMock(side_effect=mock_send_tactic)

        result = await session.replay_tactics(
            "theorem foo : True", ["intro", "bad_tactic", "trivial"],
        )

        assert result["success"] is False
        assert result["failed_at"] == 1
        assert "bad_tactic" in result["error"]
