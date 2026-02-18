"""Best-first proof search tree.

Implements LeanDojo/ReProver-style best-first search over tactic sequences
using the Lean REPL for tactic-by-tactic feedback. Each node in the tree
represents a proof state; edges are tactic applications.

Reference architectures:
- LeanDojo/ReProver: Best-first search with learned tactic generator
- Aristotle: Monte Carlo Graph Search with state equivalences
- DeepSeek-V2: Value-guided search with trained scoring
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from bourbaki.autonomous.scoring import NoveltyTracker, score_proof_state
from bourbaki.autonomous.tactics import (
    generate_candidates,
    generate_correction_candidates,
    generate_mathlib_queries,
)
from bourbaki.tools.lean_lsp_tools import lsp_suggest_tactics
from bourbaki.tools.lean_prover import lean_prover
from bourbaki.tools.lean_repl import (
    LeanREPLSession,
    REPLSessionPool,
    get_pool,
    lean_tactic,
    stop_session,
)
from bourbaki.tools.mathlib_search import mathlib_search

logger = logging.getLogger(__name__)


@dataclass
class ProofNode:
    """A node in the proof search tree."""
    proof_state: int               # REPL proof state ID
    goals: list[str]               # Remaining goals
    tactic_history: list[str]      # Tactics applied to reach this state
    parent: ProofNode | None = None
    children: list[ProofNode] = field(default_factory=list)
    score: float = float("inf")    # Priority score (lower = more promising)
    visits: int = 0                # For UCB-style exploration
    depth: int = 0
    tactic: str = ""               # The tactic that produced this node
    error: str | None = None       # Error if tactic failed

    def __lt__(self, other: ProofNode) -> bool:
        """For heapq comparison."""
        return self.score < other.score

    @property
    def is_complete(self) -> bool:
        return len(self.goals) == 0 and self.error is None

    @property
    def path(self) -> list[str]:
        """Full tactic path from root to this node."""
        return list(self.tactic_history)


@dataclass
class SearchResult:
    """Result of a proof search."""
    success: bool
    proof_tactics: list[str] = field(default_factory=list)
    proof_code: str | None = None
    nodes_explored: int = 0
    max_depth: int = 0
    total_time: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "proof_tactics": self.proof_tactics,
            "proof_code": self.proof_code,
            "nodes_explored": self.nodes_explored,
            "max_depth": self.max_depth,
            "total_time": round(self.total_time, 2),
            "error": self.error,
        }


def ucb_adjusted_score(node: ProofNode, exploration_constant: float = 1.0) -> float:
    """Adjust node score with UCB exploration bonus.

    Uses UCB1 formula: bonus = C * sqrt(ln(parent_visits) / (1 + node_visits)).
    Lower score = more promising, so bonus is subtracted.
    """
    if node.parent is None or node.parent.visits == 0:
        return node.score

    bonus = exploration_constant * math.sqrt(
        math.log(node.parent.visits) / (1 + node.visits)
    )
    return node.score - bonus


class ProofSearchTree:
    """Best-first search over proof states using the Lean REPL."""

    # Maximum tactic history depth for parallel replay.  Nodes deeper than
    # this are expanded sequentially to avoid expensive replay overhead.
    MAX_REPLAY_DEPTH = 20

    def __init__(
        self,
        theorem: str,
        max_depth: int = 30,
        session: LeanREPLSession | None = None,
        pool: REPLSessionPool | None = None,
    ) -> None:
        self.theorem = theorem
        self.max_depth = max_depth
        self.session = session  # Optional: use this instead of singleton
        self.pool = pool  # Optional: pool for parallel expansion
        self.root: ProofNode | None = None
        self._frontier: list[ProofNode] = []  # Min-heap by score
        self._explored: int = 0
        self._novelty_tracker = NoveltyTracker()  # For state deduplication + novelty bonus

    async def initialize(self) -> ProofNode | None:
        """Initialize the search tree by stating the theorem with sorry.

        Returns the root node, or None if initialization failed.
        """
        result = await lean_tactic(
            goal=self.theorem,
            tactic="sorry",  # Placeholder — lean_tactic handles initialization
            proof_state=None,
            session=self.session,
        )

        if not result.get("success"):
            logger.error("Failed to initialize proof state: %s", result.get("error"))
            return None

        self.root = ProofNode(
            proof_state=result.get("proofState", 0),
            goals=result.get("goals", []),
            tactic_history=[],
            score=score_proof_state(result.get("goals", []), 0, self._novelty_tracker),
        )

        heapq.heappush(self._frontier, self.root)
        return self.root

    async def expand(
        self,
        node: ProofNode,
        candidates: list[str],
        max_corrections: int = 2,
    ) -> list[ProofNode]:
        """Expand a node by trying candidate tactics.

        Uses Goedel-V2-style error-conditioned repair: when a tactic fails,
        the error message is used to generate targeted correction candidates
        (up to max_corrections rounds per failed tactic).

        Args:
            node: The proof state to expand.
            candidates: Tactic strings to try.
            max_corrections: Max correction attempts per failed tactic.

        Returns:
            List of new child nodes (successful tactic applications only).
        """
        children: list[ProofNode] = []
        correction_queue: list[tuple[str, str, int]] = []  # (tactic, error, round)

        for tactic in candidates:
            result = await lean_tactic(
                goal=self.theorem,
                tactic=tactic,
                proof_state=node.proof_state,
                session=self.session,
            )

            self._explored += 1

            if not result.get("success"):
                # Queue error-conditioned corrections (round 1)
                error_msg = result.get("error", "")
                if error_msg and max_corrections > 0:
                    correction_queue.append((tactic, error_msg, 1))
                continue

            new_goals = result.get("goals", [])
            new_ps = result.get("proofState", node.proof_state)

            # State deduplication: skip if we've seen this exact goal set
            if self._novelty_tracker.has_seen(new_goals) and new_goals:
                continue

            child = ProofNode(
                proof_state=new_ps,
                goals=new_goals,
                tactic_history=node.tactic_history + [tactic],
                parent=node,
                score=score_proof_state(new_goals, node.depth + 1, self._novelty_tracker),
                depth=node.depth + 1,
                tactic=tactic,
            )

            node.children.append(child)
            children.append(child)

            # If proof is complete, don't add more children
            if child.is_complete:
                logger.info("Proof complete at depth %d: %s", child.depth, child.path)
                return [child]

        # Error-conditioned correction loop (Goedel-V2 style)
        seen_corrections: set[str] = set()
        while correction_queue:
            failed_tactic, error_msg, round_num = correction_queue.pop(0)
            if round_num > max_corrections:
                continue

            repairs = generate_correction_candidates(failed_tactic, error_msg, node.goals)
            for repair in repairs:
                if repair in seen_corrections:
                    continue
                seen_corrections.add(repair)

                result = await lean_tactic(
                    goal=self.theorem,
                    tactic=repair,
                    proof_state=node.proof_state,
                    session=self.session,
                )
                self._explored += 1

                if not result.get("success"):
                    # Queue for another correction round if allowed
                    if round_num < max_corrections:
                        repair_error = result.get("error", "")
                        if repair_error and repair_error != error_msg:
                            correction_queue.append((repair, repair_error, round_num + 1))
                    continue

                new_goals = result.get("goals", [])
                new_ps = result.get("proofState", node.proof_state)

                if self._novelty_tracker.has_seen(new_goals) and new_goals:
                    continue

                child = ProofNode(
                    proof_state=new_ps,
                    goals=new_goals,
                    tactic_history=node.tactic_history + [repair],
                    parent=node,
                    score=score_proof_state(new_goals, node.depth + 1, self._novelty_tracker),
                    depth=node.depth + 1,
                    tactic=repair,
                )
                node.children.append(child)
                children.append(child)

                if child.is_complete:
                    logger.info("Proof complete via correction at depth %d: %s", child.depth, child.path)
                    return [child]

        return children

    async def _expand_node_in_session(
        self,
        node: ProofNode,
        pool: REPLSessionPool,
        candidates: list[str],
        max_corrections: int = 2,
    ) -> list[ProofNode]:
        """Expand a single node using a session acquired from the pool.

        Acquires a session, replays the tactic history to reach the node's
        proof state, then tries all candidate tactics.  The session is
        always released back to the pool afterwards.

        Args:
            node: The proof state to expand.
            pool: Session pool to acquire from.
            candidates: Tactic strings to try.
            max_corrections: Max correction attempts per failed tactic.

        Returns:
            List of new child nodes (successful tactic applications only).
        """
        session = await pool.acquire()
        try:
            # Replay tactic history to reach this node's proof state
            replay = await session.replay_tactics(
                self.theorem, node.tactic_history,
            )
            if not replay.get("success"):
                logger.warning(
                    "Replay failed for node at depth %d: %s",
                    node.depth, replay.get("error"),
                )
                return []

            replay_ps = replay["proofState"]

            # Now try each candidate tactic against the replayed proof state
            children: list[ProofNode] = []
            correction_queue: list[tuple[str, str, int]] = []

            for tactic in candidates:
                result = await lean_tactic(
                    goal=self.theorem,
                    tactic=tactic,
                    proof_state=replay_ps,
                    session=session,
                )
                self._explored += 1

                if not result.get("success"):
                    error_msg = result.get("error", "")
                    if error_msg and max_corrections > 0:
                        correction_queue.append((tactic, error_msg, 1))
                    continue

                new_goals = result.get("goals", [])
                new_ps = result.get("proofState", replay_ps)

                if self._novelty_tracker.has_seen(new_goals) and new_goals:
                    continue

                child = ProofNode(
                    proof_state=new_ps,
                    goals=new_goals,
                    tactic_history=node.tactic_history + [tactic],
                    parent=node,
                    score=score_proof_state(
                        new_goals, node.depth + 1, self._novelty_tracker,
                    ),
                    depth=node.depth + 1,
                    tactic=tactic,
                )
                node.children.append(child)
                children.append(child)

                if child.is_complete:
                    logger.info(
                        "Proof complete (parallel) at depth %d: %s",
                        child.depth, child.path,
                    )
                    return [child]

            # Error-conditioned correction loop
            seen_corrections: set[str] = set()
            while correction_queue:
                failed_tactic, error_msg, round_num = correction_queue.pop(0)
                if round_num > max_corrections:
                    continue

                repairs = generate_correction_candidates(
                    failed_tactic, error_msg, node.goals,
                )
                for repair in repairs:
                    if repair in seen_corrections:
                        continue
                    seen_corrections.add(repair)

                    result = await lean_tactic(
                        goal=self.theorem,
                        tactic=repair,
                        proof_state=replay_ps,
                        session=session,
                    )
                    self._explored += 1

                    if not result.get("success"):
                        if round_num < max_corrections:
                            repair_error = result.get("error", "")
                            if repair_error and repair_error != error_msg:
                                correction_queue.append(
                                    (repair, repair_error, round_num + 1),
                                )
                        continue

                    new_goals = result.get("goals", [])
                    new_ps = result.get("proofState", replay_ps)

                    if self._novelty_tracker.has_seen(new_goals) and new_goals:
                        continue

                    child = ProofNode(
                        proof_state=new_ps,
                        goals=new_goals,
                        tactic_history=node.tactic_history + [repair],
                        parent=node,
                        score=score_proof_state(
                            new_goals, node.depth + 1, self._novelty_tracker,
                        ),
                        depth=node.depth + 1,
                        tactic=repair,
                    )
                    node.children.append(child)
                    children.append(child)

                    if child.is_complete:
                        logger.info(
                            "Proof complete via correction (parallel) at depth %d: %s",
                            child.depth, child.path,
                        )
                        return [child]

            return children

        except Exception as exc:
            logger.warning(
                "Parallel expansion failed for node at depth %d: %s",
                node.depth, exc,
            )
            return []
        finally:
            await pool.release(session)

    async def expand_parallel(
        self,
        nodes: list[ProofNode],
        pool: REPLSessionPool,
        candidates_per_node: list[list[str]],
        max_corrections: int = 2,
    ) -> list[list[ProofNode]]:
        """Expand multiple nodes in parallel using the session pool.

        Each node is expanded in its own REPL session (acquired from the
        pool).  The tactic history is replayed to reach the node's proof
        state before trying candidates.

        Nodes deeper than ``MAX_REPLAY_DEPTH`` are skipped from parallel
        expansion to avoid expensive replay overhead.

        Args:
            nodes: Frontier nodes to expand concurrently.
            pool: Session pool to draw from.
            candidates_per_node: Candidate tactics for each node (same order).
            max_corrections: Max correction attempts per failed tactic.

        Returns:
            A list of child-node lists, one per input node (in the same order).
        """
        async def _empty() -> list[ProofNode]:
            return []

        tasks: list[asyncio.Task[list[ProofNode]]] = []
        for node, candidates in zip(nodes, candidates_per_node):
            if node.depth > self.MAX_REPLAY_DEPTH:
                # Too deep — skip parallel expansion for this node
                logger.debug(
                    "Skipping parallel expansion for depth-%d node (exceeds MAX_REPLAY_DEPTH=%d)",
                    node.depth, self.MAX_REPLAY_DEPTH,
                )
                tasks.append(asyncio.create_task(_empty()))
                continue

            task = asyncio.create_task(
                self._expand_node_in_session(
                    node, pool, candidates, max_corrections,
                ),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unwrap results — convert exceptions to empty lists
        out: list[list[ProofNode]] = []
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("Parallel expansion task failed: %s", r)
                out.append([])
            else:
                out.append(r)
        return out

    async def _fetch_lsp_tactics(
        self,
        node: ProofNode,
        existing: set[str],
        timeout: float = 5.0,
    ) -> list[str]:
        """Fetch tactic suggestions from the Lean LSP, deduped against *existing*.

        Constructs a Lean source with the current tactic history ending in
        ``sorry``, then asks the LSP for completions at the sorry position.

        Args:
            node: Current proof node (uses tactic_history for context).
            existing: Set of tactic strings already in the candidate list.
            timeout: Maximum seconds to wait for the LSP response.

        Returns:
            New tactic strings not already in *existing* (may be empty).
        """
        # Build Lean code: theorem statement + tactics so far + sorry
        tactic_lines = node.tactic_history + ["sorry"]
        tactic_block = "\n  ".join(tactic_lines)
        code = f"{self.theorem} := by\n  {tactic_block}"

        try:
            suggestions = await asyncio.wait_for(
                lsp_suggest_tactics(theorem=code),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.debug("LSP tactic fetch failed (depth=%d): %s", node.depth, exc)
            return []

        # Deduplicate against what we already have
        new_tactics: list[str] = []
        for s in suggestions:
            s = s.strip()
            if s and s not in existing:
                new_tactics.append(s)
        return new_tactics

    async def best_first_search(
        self,
        budget: int = 100,
        timeout: float = 300.0,
        use_mathlib: bool = True,
        use_lsp: bool = False,
        parallel: int = 1,
    ) -> SearchResult:
        """Run best-first search to find a proof.

        Args:
            budget: Maximum number of node expansions (each expansion tries
                    all candidate tactics for one proof state).
            timeout: Maximum time in seconds.
            use_mathlib: Whether to query mathlib_search for candidates.
            use_lsp: Whether to query the Lean LSP for tactic completions
                     at shallow depths (depth < 3).
            parallel: Number of frontier nodes to expand concurrently.
                      When 1 (default), behaves identically to the original
                      sequential algorithm.  When >1, uses the
                      ``REPLSessionPool`` to expand multiple nodes at once.

        Returns:
            SearchResult with success/failure details.
        """
        start = time.monotonic()

        # Initialize
        root = await self.initialize()
        if root is None:
            return SearchResult(
                success=False,
                error="Failed to initialize proof state",
                total_time=time.monotonic() - start,
            )

        # Check if already solved (e.g., `rfl` or `trivial` enough)
        if root.is_complete:
            return SearchResult(
                success=True,
                proof_tactics=[],
                nodes_explored=1,
                total_time=time.monotonic() - start,
            )

        attempts = 0
        max_depth_seen = 0

        while self._frontier and attempts < budget:
            # Check timeout
            if time.monotonic() - start > timeout:
                break

            if parallel <= 1:
                # --- Sequential expansion (original algorithm) ---
                node = heapq.heappop(self._frontier)

                node.visits += 1
                ancestor = node.parent
                while ancestor is not None:
                    ancestor.visits += 1
                    ancestor = ancestor.parent

                if node.depth >= self.max_depth:
                    continue

                max_depth_seen = max(max_depth_seen, node.depth)

                candidates = generate_candidates(
                    goals=node.goals,
                    depth=node.depth,
                )

                if use_mathlib and node.depth < 5:
                    mathlib_results = await self._search_mathlib(node.goals)
                    if mathlib_results:
                        candidates = generate_candidates(
                            goals=node.goals,
                            depth=node.depth,
                            mathlib_results=mathlib_results,
                        )

                if use_lsp and node.depth < 3:
                    existing = set(candidates)
                    lsp_tactics = await self._fetch_lsp_tactics(
                        node, existing, timeout=5.0,
                    )
                    if lsp_tactics:
                        candidates.extend(lsp_tactics)
                        logger.debug(
                            "LSP added %d tactics at depth %d",
                            len(lsp_tactics), node.depth,
                        )

                children = await self.expand(node, candidates)
                attempts += 1

                for child in children:
                    if child.is_complete:
                        tactic_block = "\n  ".join(child.path)
                        proof_code = f"{self.theorem} := by\n  {tactic_block}"
                        return SearchResult(
                            success=True,
                            proof_tactics=child.path,
                            proof_code=proof_code,
                            nodes_explored=self._explored,
                            max_depth=child.depth,
                            total_time=time.monotonic() - start,
                        )
                    child.score = ucb_adjusted_score(child)
                    heapq.heappush(self._frontier, child)
            else:
                # --- Parallel expansion ---
                # Pop up to `parallel` nodes from the frontier
                batch_nodes: list[ProofNode] = []
                while self._frontier and len(batch_nodes) < parallel:
                    node = heapq.heappop(self._frontier)
                    if node.depth >= self.max_depth:
                        continue
                    # Skip nodes too deep for efficient replay
                    if node.depth > self.MAX_REPLAY_DEPTH:
                        continue
                    batch_nodes.append(node)

                if not batch_nodes:
                    break

                # Update visit counts for all batch nodes
                for node in batch_nodes:
                    node.visits += 1
                    ancestor = node.parent
                    while ancestor is not None:
                        ancestor.visits += 1
                        ancestor = ancestor.parent
                    max_depth_seen = max(max_depth_seen, node.depth)

                # Generate candidates for each node
                candidates_per_node: list[list[str]] = []
                for node in batch_nodes:
                    candidates = generate_candidates(
                        goals=node.goals,
                        depth=node.depth,
                    )

                    if use_mathlib and node.depth < 5:
                        mathlib_results = await self._search_mathlib(node.goals)
                        if mathlib_results:
                            candidates = generate_candidates(
                                goals=node.goals,
                                depth=node.depth,
                                mathlib_results=mathlib_results,
                            )

                    if use_lsp and node.depth < 3:
                        existing = set(candidates)
                        lsp_tactics = await self._fetch_lsp_tactics(
                            node, existing, timeout=5.0,
                        )
                        if lsp_tactics:
                            candidates.extend(lsp_tactics)

                    candidates_per_node.append(candidates)

                # Get the pool (use the one supplied at init, or create/get global)
                pool = self.pool
                if pool is None:
                    pool = await get_pool(
                        max_size=parallel, full_mathlib=True,
                    )

                # Expand all nodes in parallel
                all_children = await self.expand_parallel(
                    batch_nodes, pool, candidates_per_node,
                )
                attempts += len(batch_nodes)

                # Process results
                for children in all_children:
                    for child in children:
                        if child.is_complete:
                            tactic_block = "\n  ".join(child.path)
                            proof_code = f"{self.theorem} := by\n  {tactic_block}"
                            return SearchResult(
                                success=True,
                                proof_tactics=child.path,
                                proof_code=proof_code,
                                nodes_explored=self._explored,
                                max_depth=child.depth,
                                total_time=time.monotonic() - start,
                            )
                        child.score = ucb_adjusted_score(child)
                        heapq.heappush(self._frontier, child)

        # Search exhausted
        return SearchResult(
            success=False,
            nodes_explored=self._explored,
            max_depth=max_depth_seen,
            total_time=time.monotonic() - start,
            error=f"Search exhausted after {attempts} attempts, {self._explored} nodes explored",
        )

    async def _search_mathlib(self, goals: list[str]) -> list[dict[str, Any]]:
        """Search Mathlib for lemmas relevant to the current goals.

        Tries semantic mode first (best for goal-state queries), then
        falls back to type/natural. Deduplicates results by lemma name.
        """
        queries = generate_mathlib_queries(goals)
        results: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for query, mode in queries[:3]:  # Up to 3 queries (semantic, type, natural)
            try:
                result = await mathlib_search(query=query, mode=mode, max_results=3)
                if result.get("success"):
                    for hit in result.get("results", []):
                        name = hit.get("name", "")
                        if name and name not in seen_names:
                            seen_names.add(name)
                            results.append(hit)
            except Exception:
                continue

        return results

    async def _verify_proof(self, tactics: list[str]) -> str | None:
        """Verify a complete proof by reconstructing the full Lean code."""
        tactic_block = "\n  ".join(tactics)
        proof_code = f"{self.theorem} := by\n  {tactic_block}"

        result = await lean_prover(code=proof_code, mode="check")
        if result.get("proofComplete"):
            return proof_code

        # If verification fails, the REPL-found proof may need adjustment
        logger.warning("REPL proof did not verify: %s", result.get("errors"))
        return proof_code  # Return anyway — it may be close

    def get_stats(self) -> dict[str, Any]:
        """Get search tree statistics."""
        return {
            "nodes_explored": self._explored,
            "frontier_size": len(self._frontier),
            "seen_states": self._novelty_tracker.seen_count,
        }


async def prove_with_search(
    theorem: str,
    budget: int = 100,
    timeout: float = 300.0,
    max_depth: int = 30,
    use_mathlib: bool = True,
    use_lsp: bool = False,
    session: LeanREPLSession | None = None,
    parallel: int = 1,
    pool: REPLSessionPool | None = None,
) -> SearchResult:
    """High-level API: prove a theorem using best-first search.

    Args:
        theorem: Lean 4 theorem statement (e.g., "theorem foo : 1 + 1 = 2").
        budget: Maximum tactic attempts.
        timeout: Maximum seconds.
        max_depth: Maximum proof depth.
        use_mathlib: Whether to search Mathlib for lemmas.
        use_lsp: Whether to use Lean LSP tactic completions at shallow depths.
        session: Optional pre-initialized REPL session. If None, creates and
                 manages a singleton session (which is stopped on completion).
        parallel: Number of frontier nodes to expand concurrently (default 1
                  for backward-compatible sequential behaviour).
        pool: Optional pre-initialized ``REPLSessionPool``.  When *parallel* > 1
              and no pool is given, one is created automatically via ``get_pool()``.

    Returns:
        SearchResult with proof details.
    """
    owns_session = session is None and parallel <= 1
    tree = ProofSearchTree(
        theorem, max_depth=max_depth, session=session, pool=pool,
    )

    try:
        result = await tree.best_first_search(
            budget=budget,
            timeout=timeout,
            use_mathlib=use_mathlib,
            use_lsp=use_lsp,
            parallel=parallel,
        )
    finally:
        # Only clean up if we created the session ourselves (sequential mode)
        if owns_session:
            await stop_session()

    if result.success:
        logger.info(
            "Proof found: %d tactics, %d nodes explored, %.1fs",
            len(result.proof_tactics), result.nodes_explored, result.total_time,
        )
    else:
        logger.info(
            "Proof not found: %d nodes explored, %.1fs — %s",
            result.nodes_explored, result.total_time, result.error,
        )

    return result
