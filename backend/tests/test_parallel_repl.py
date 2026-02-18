"""Tests for parallel tactic expansion using the REPL session pool.

Covers:
- LeanREPLSession.replay_tactics()
- ProofSearchTree.expand_parallel()
- ProofSearchTree._expand_node_in_session()
- best_first_search(parallel=N) behavior
- Backward compatibility: parallel=1 matches sequential behavior
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bourbaki.autonomous.search_tree import (
    ProofNode,
    ProofSearchTree,
    SearchResult,
    prove_with_search,
)
from bourbaki.tools.lean_repl import LeanREPLSession, REPLSessionPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_mock(
    *,
    replay_result: dict | None = None,
    tactic_results: list[dict] | None = None,
) -> AsyncMock:
    """Create a mock LeanREPLSession with configurable replay/tactic results."""
    session = AsyncMock(spec=LeanREPLSession)
    session.is_running = True
    session.env_id = 0
    session._initialized = True

    if replay_result is not None:
        session.replay_tactics = AsyncMock(return_value=replay_result)

    if tactic_results is not None:
        # Return results in order, cycling if needed
        _iter = iter(tactic_results)
        session.send_tactic = AsyncMock(side_effect=lambda *a, **kw: next(_iter, {"error": "exhausted"}))

    return session


def _make_pool_mock(sessions: list[AsyncMock] | None = None) -> AsyncMock:
    """Create a mock REPLSessionPool that returns pre-built sessions."""
    pool = AsyncMock(spec=REPLSessionPool)
    _idx = 0

    if sessions:
        async def _acquire() -> AsyncMock:
            nonlocal _idx
            s = sessions[_idx % len(sessions)]
            _idx += 1
            return s
        pool.acquire = AsyncMock(side_effect=_acquire)
    else:
        pool.acquire = AsyncMock(return_value=_make_session_mock())

    pool.release = AsyncMock()
    return pool


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
            "sorries": [{"proofState": 0, "goals": ["a : Nat\n\u22a2 a = a"]}],
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


# ---------------------------------------------------------------------------
# expand_parallel() tests
# ---------------------------------------------------------------------------


class TestExpandParallel:
    """Tests for ProofSearchTree.expand_parallel()."""

    @pytest.mark.asyncio
    async def test_parallel_expansion_basic(self):
        """Two nodes expanded in parallel should each get their own session."""
        # Create two sessions that will handle the two nodes
        session1 = _make_session_mock(
            replay_result={"success": True, "proofState": 10, "goals": ["⊢ 1 + 1 = 2"]},
        )
        session2 = _make_session_mock(
            replay_result={"success": True, "proofState": 20, "goals": ["⊢ True"]},
        )

        pool = _make_pool_mock(sessions=[session1, session2])

        tree = ProofSearchTree("theorem foo : 1 + 1 = 2", pool=pool)
        tree._novelty_tracker = MagicMock()
        tree._novelty_tracker.has_seen = MagicMock(return_value=False)

        node1 = ProofNode(
            proof_state=0, goals=["⊢ 1 + 1 = 2"],
            tactic_history=[], depth=0,
        )
        node2 = ProofNode(
            proof_state=1, goals=["⊢ True"],
            tactic_history=["intro"], depth=1,
        )

        # Mock lean_tactic for the expansion phase
        with patch(
            "bourbaki.autonomous.search_tree.lean_tactic",
            new_callable=AsyncMock,
            return_value={
                "success": True, "goals": ["⊢ remaining"],
                "proofState": 99,
            },
        ):
            results = await tree.expand_parallel(
                [node1, node2], pool,
                [["ring"], ["trivial"]],
            )

        assert len(results) == 2
        # Both sessions should have been acquired and released
        assert pool.acquire.call_count == 2
        assert pool.release.call_count == 2
        # Both sessions should have had replay_tactics called
        session1.replay_tactics.assert_called_once()
        session2.replay_tactics.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_expansion_skips_deep_nodes(self):
        """Nodes deeper than MAX_REPLAY_DEPTH should return empty results."""
        session = _make_session_mock(
            replay_result={"success": True, "proofState": 10, "goals": ["⊢ x"]},
        )
        pool = _make_pool_mock(sessions=[session])

        tree = ProofSearchTree("theorem foo : True", pool=pool)

        deep_node = ProofNode(
            proof_state=0, goals=["⊢ True"],
            tactic_history=["step"] * 25,  # exceeds MAX_REPLAY_DEPTH=20
            depth=25,
        )

        with patch(
            "bourbaki.autonomous.search_tree.lean_tactic",
            new_callable=AsyncMock,
        ):
            results = await tree.expand_parallel(
                [deep_node], pool, [["trivial"]],
            )

        assert len(results) == 1
        assert results[0] == []  # Deep node was skipped

    @pytest.mark.asyncio
    async def test_parallel_expansion_handles_replay_failure(self):
        """If replay fails for a node, that node returns no children."""
        session = _make_session_mock(
            replay_result={"success": False, "error": "session crashed"},
        )
        pool = _make_pool_mock(sessions=[session])

        tree = ProofSearchTree("theorem foo : True", pool=pool)

        node = ProofNode(
            proof_state=0, goals=["⊢ True"],
            tactic_history=["intro"], depth=1,
        )

        results = await tree.expand_parallel(
            [node], pool, [["trivial"]],
        )

        assert len(results) == 1
        assert results[0] == []  # Replay failed → no children
        # Session should still be released
        pool.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_expansion_finds_proof(self):
        """If a parallel expansion finds a complete proof, it returns it."""
        session = _make_session_mock(
            replay_result={"success": True, "proofState": 5, "goals": ["⊢ 1 + 1 = 2"]},
        )
        pool = _make_pool_mock(sessions=[session])

        tree = ProofSearchTree("theorem foo : 1 + 1 = 2", pool=pool)
        tree._novelty_tracker = MagicMock()
        tree._novelty_tracker.has_seen = MagicMock(return_value=False)

        node = ProofNode(
            proof_state=0, goals=["⊢ 1 + 1 = 2"],
            tactic_history=[], depth=0,
        )

        # lean_tactic returns a complete proof (empty goals)
        with patch(
            "bourbaki.autonomous.search_tree.lean_tactic",
            new_callable=AsyncMock,
            return_value={"success": True, "goals": [], "proofState": 6},
        ):
            results = await tree.expand_parallel(
                [node], pool, [["ring"]],
            )

        assert len(results) == 1
        children = results[0]
        assert len(children) == 1
        assert children[0].is_complete
        assert children[0].tactic == "ring"

    @pytest.mark.asyncio
    async def test_parallel_expansion_concurrent_execution(self):
        """Verify that parallel nodes are actually expanded concurrently."""
        execution_log: list[tuple[str, float]] = []
        start_time = asyncio.get_event_loop().time()

        async def slow_replay(theorem, tactics, timeout=120):
            execution_log.append(("replay_start", asyncio.get_event_loop().time() - start_time))
            await asyncio.sleep(0.05)  # Simulate replay time
            execution_log.append(("replay_end", asyncio.get_event_loop().time() - start_time))
            return {"success": True, "proofState": 0, "goals": ["⊢ x"]}

        session1 = _make_session_mock()
        session1.replay_tactics = AsyncMock(side_effect=slow_replay)
        session2 = _make_session_mock()
        session2.replay_tactics = AsyncMock(side_effect=slow_replay)

        pool = _make_pool_mock(sessions=[session1, session2])

        tree = ProofSearchTree("theorem foo : True", pool=pool)
        tree._novelty_tracker = MagicMock()
        tree._novelty_tracker.has_seen = MagicMock(return_value=False)

        node1 = ProofNode(proof_state=0, goals=["⊢ x"], tactic_history=[], depth=0)
        node2 = ProofNode(proof_state=1, goals=["⊢ y"], tactic_history=["intro"], depth=1)

        with patch(
            "bourbaki.autonomous.search_tree.lean_tactic",
            new_callable=AsyncMock,
            return_value={"success": True, "goals": ["⊢ z"], "proofState": 99},
        ):
            await tree.expand_parallel(
                [node1, node2], pool,
                [["ring"], ["trivial"]],
            )

        # Both replays should have started before either finished
        # (i.e., they ran concurrently, not sequentially)
        replay_starts = [t for label, t in execution_log if label == "replay_start"]
        replay_ends = [t for label, t in execution_log if label == "replay_end"]
        assert len(replay_starts) == 2
        # The second replay should start before the first replay ends
        assert replay_starts[1] < replay_ends[0]


# ---------------------------------------------------------------------------
# best_first_search(parallel=N) tests
# ---------------------------------------------------------------------------


class TestBestFirstSearchParallel:
    """Tests for best_first_search with parallel > 1."""

    @pytest.mark.asyncio
    async def test_parallel_1_is_sequential(self):
        """parallel=1 should use the original sequential code path."""
        import heapq
        tree = ProofSearchTree("theorem foo : 1 + 1 = 2")

        root = ProofNode(
            proof_state=0, goals=["⊢ 1 + 1 = 2"], tactic_history=[],
            score=10.0,
        )

        # Mock initialize to set up root and push to frontier
        async def mock_init():
            tree.root = root
            heapq.heappush(tree._frontier, root)
            return root

        with patch.object(tree, "initialize", new_callable=AsyncMock, side_effect=mock_init):
            # Mock expand to return a complete proof on first expansion
            with patch.object(tree, "expand", new_callable=AsyncMock) as mock_expand:
                complete_node = ProofNode(
                    proof_state=1, goals=[], tactic_history=["ring"],
                    depth=1, tactic="ring",
                )
                mock_expand.return_value = [complete_node]

                with patch(
                    "bourbaki.autonomous.search_tree.generate_candidates",
                    return_value=["ring"],
                ):
                    result = await tree.best_first_search(
                        budget=10, timeout=60, use_mathlib=False, parallel=1,
                    )

            assert result.success is True
            assert result.proof_tactics == ["ring"]
            # expand() should have been called (sequential path)
            mock_expand.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_uses_expand_parallel(self):
        """parallel > 1 should use expand_parallel instead of expand."""
        import heapq
        pool = _make_pool_mock()
        tree = ProofSearchTree("theorem foo : 1 + 1 = 2", pool=pool)

        root = ProofNode(
            proof_state=0, goals=["⊢ 1 + 1 = 2"],
            tactic_history=[], score=10.0,
        )

        async def mock_init():
            tree.root = root
            heapq.heappush(tree._frontier, root)
            return root

        with patch.object(tree, "initialize", new_callable=AsyncMock, side_effect=mock_init):
            complete_node = ProofNode(
                proof_state=1, goals=[], tactic_history=["ring"],
                depth=1, tactic="ring",
            )

            with patch.object(
                tree, "expand_parallel", new_callable=AsyncMock,
            ) as mock_par:
                mock_par.return_value = [[complete_node]]

                with patch(
                    "bourbaki.autonomous.search_tree.generate_candidates",
                    return_value=["ring"],
                ):
                    result = await tree.best_first_search(
                        budget=10, timeout=60, use_mathlib=False, parallel=4,
                    )

            assert result.success is True
            assert result.proof_tactics == ["ring"]
            # expand_parallel should have been called, not expand
            mock_par.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_exhaustion(self):
        """When the frontier is empty with parallel > 1, search should stop."""
        import heapq
        pool = _make_pool_mock()
        tree = ProofSearchTree("theorem foo : True", pool=pool)

        root = ProofNode(
            proof_state=0, goals=["⊢ True"],
            tactic_history=[], score=10.0,
        )

        async def mock_init():
            tree.root = root
            heapq.heappush(tree._frontier, root)
            return root

        with patch.object(tree, "initialize", new_callable=AsyncMock, side_effect=mock_init):
            with patch.object(
                tree, "expand_parallel", new_callable=AsyncMock,
            ) as mock_par:
                # Return no children — frontier will be empty after first round
                mock_par.return_value = [[]]

                with patch(
                    "bourbaki.autonomous.search_tree.generate_candidates",
                    return_value=["trivial"],
                ):
                    result = await tree.best_first_search(
                        budget=100, timeout=60, use_mathlib=False, parallel=4,
                    )

            assert result.success is False
            assert "exhausted" in result.error.lower()


# ---------------------------------------------------------------------------
# prove_with_search() parallel wiring
# ---------------------------------------------------------------------------


class TestProveWithSearchParallel:
    """Tests for prove_with_search() with parallel parameter."""

    @pytest.mark.asyncio
    async def test_parallel_kwarg_passed_through(self):
        """prove_with_search(parallel=4) should pass parallel to best_first_search."""
        with patch(
            "bourbaki.autonomous.search_tree.ProofSearchTree",
        ) as MockTree:
            mock_tree_instance = AsyncMock()
            mock_tree_instance.best_first_search = AsyncMock(
                return_value=SearchResult(success=False, error="test"),
            )
            MockTree.return_value = mock_tree_instance

            with patch(
                "bourbaki.autonomous.search_tree.stop_session",
                new_callable=AsyncMock,
            ):
                await prove_with_search(
                    "theorem foo : True",
                    parallel=4,
                    budget=10,
                    timeout=5,
                )

            # Verify parallel was passed through
            call_kwargs = mock_tree_instance.best_first_search.call_args
            assert call_kwargs.kwargs.get("parallel") == 4 or call_kwargs[1].get("parallel") == 4

    @pytest.mark.asyncio
    async def test_default_parallel_is_1(self):
        """prove_with_search() without parallel should default to 1."""
        with patch(
            "bourbaki.autonomous.search_tree.ProofSearchTree",
        ) as MockTree:
            mock_tree_instance = AsyncMock()
            mock_tree_instance.best_first_search = AsyncMock(
                return_value=SearchResult(success=False, error="test"),
            )
            MockTree.return_value = mock_tree_instance

            with patch(
                "bourbaki.autonomous.search_tree.stop_session",
                new_callable=AsyncMock,
            ):
                await prove_with_search(
                    "theorem foo : True",
                    budget=10,
                    timeout=5,
                )

            call_kwargs = mock_tree_instance.best_first_search.call_args
            assert call_kwargs.kwargs.get("parallel") == 1 or call_kwargs[1].get("parallel") == 1


# ---------------------------------------------------------------------------
# AutonomousSearchConfig wiring
# ---------------------------------------------------------------------------


class TestAutonomousSearchConfig:
    """Tests for AutonomousSearchConfig.parallel_sessions parameter."""

    def test_default_parallel_sessions_is_1(self):
        from bourbaki.autonomous.search import AutonomousSearchConfig
        config = AutonomousSearchConfig()
        assert config.parallel_sessions == 1

    def test_custom_parallel_sessions(self):
        from bourbaki.autonomous.search import AutonomousSearchConfig
        config = AutonomousSearchConfig(parallel_sessions=8)
        assert config.parallel_sessions == 8
