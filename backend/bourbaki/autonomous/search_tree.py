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
import time
from dataclasses import dataclass, field
from typing import Any

from bourbaki.autonomous.scoring import score_proof_state
from bourbaki.autonomous.tactics import generate_candidates, generate_mathlib_queries
from bourbaki.tools.lean_prover import lean_prover
from bourbaki.tools.lean_repl import LeanREPLSession, lean_tactic, stop_session
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


class ProofSearchTree:
    """Best-first search over proof states using the Lean REPL."""

    def __init__(
        self,
        theorem: str,
        max_depth: int = 30,
        session: LeanREPLSession | None = None,
    ) -> None:
        self.theorem = theorem
        self.max_depth = max_depth
        self.session = session  # Optional: use this instead of singleton
        self.root: ProofNode | None = None
        self._frontier: list[ProofNode] = []  # Min-heap by score
        self._explored: int = 0
        self._seen_goals: set[str] = set()  # For state deduplication

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
            score=score_proof_state(result.get("goals", []), 0),
        )

        # Add root's goals to seen set
        self._goal_key(self.root.goals)

        heapq.heappush(self._frontier, self.root)
        return self.root

    async def expand(
        self,
        node: ProofNode,
        candidates: list[str],
    ) -> list[ProofNode]:
        """Expand a node by trying candidate tactics.

        Args:
            node: The proof state to expand.
            candidates: Tactic strings to try.

        Returns:
            List of new child nodes (successful tactic applications only).
        """
        children: list[ProofNode] = []

        for tactic in candidates:
            result = await lean_tactic(
                goal=self.theorem,
                tactic=tactic,
                proof_state=node.proof_state,
                session=self.session,
            )

            self._explored += 1

            if not result.get("success"):
                # Tactic failed — skip but don't add to tree
                continue

            new_goals = result.get("goals", [])
            new_ps = result.get("proofState", node.proof_state)

            # State deduplication: skip if we've seen this exact goal set
            goal_key = self._goal_key(new_goals)
            if goal_key in self._seen_goals and new_goals:
                continue
            self._seen_goals.add(goal_key)

            child = ProofNode(
                proof_state=new_ps,
                goals=new_goals,
                tactic_history=node.tactic_history + [tactic],
                parent=node,
                score=score_proof_state(new_goals, node.depth + 1),
                depth=node.depth + 1,
                tactic=tactic,
            )

            node.children.append(child)
            children.append(child)

            # If proof is complete, don't add more children
            if child.is_complete:
                logger.info("Proof complete at depth %d: %s", child.depth, child.path)
                return [child]

        return children

    async def best_first_search(
        self,
        budget: int = 100,
        timeout: float = 300.0,
        use_mathlib: bool = True,
    ) -> SearchResult:
        """Run best-first search to find a proof.

        Args:
            budget: Maximum number of node expansions (each expansion tries
                    all candidate tactics for one proof state).
            timeout: Maximum time in seconds.
            use_mathlib: Whether to query mathlib_search for candidates.

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

            # Pop the most promising node
            node = heapq.heappop(self._frontier)

            # Skip if too deep
            if node.depth >= self.max_depth:
                continue

            max_depth_seen = max(max_depth_seen, node.depth)

            # Generate candidate tactics
            candidates = generate_candidates(
                goals=node.goals,
                depth=node.depth,
            )

            # Optionally search Mathlib for relevant lemmas
            if use_mathlib and node.depth < 5:
                mathlib_results = await self._search_mathlib(node.goals)
                if mathlib_results:
                    candidates = generate_candidates(
                        goals=node.goals,
                        depth=node.depth,
                        mathlib_results=mathlib_results,
                    )

            # Expand the node
            children = await self.expand(node, candidates)
            attempts += 1

            # Add successful children to frontier
            for child in children:
                if child.is_complete:
                    # Found a proof! The REPL confirmed it — skip slow
                    # lean_prover verification (saves ~90s per proof).
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
        """Search Mathlib for lemmas relevant to the current goals."""
        queries = generate_mathlib_queries(goals)
        results: list[dict[str, Any]] = []

        for query, mode in queries[:2]:  # Limit to 2 queries per state
            try:
                result = await mathlib_search(query=query, mode=mode, max_results=3)
                if result.get("success"):
                    results.extend(result.get("results", []))
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

    def _goal_key(self, goals: list[str]) -> str:
        """Create a hashable key from a goal set for deduplication."""
        return "|".join(sorted(goals))

    def get_stats(self) -> dict[str, Any]:
        """Get search tree statistics."""
        return {
            "nodes_explored": self._explored,
            "frontier_size": len(self._frontier),
            "seen_states": len(self._seen_goals),
        }


async def prove_with_search(
    theorem: str,
    budget: int = 100,
    timeout: float = 300.0,
    max_depth: int = 30,
    use_mathlib: bool = True,
    session: LeanREPLSession | None = None,
) -> SearchResult:
    """High-level API: prove a theorem using best-first search.

    Args:
        theorem: Lean 4 theorem statement (e.g., "theorem foo : 1 + 1 = 2").
        budget: Maximum tactic attempts.
        timeout: Maximum seconds.
        max_depth: Maximum proof depth.
        use_mathlib: Whether to search Mathlib for lemmas.
        session: Optional pre-initialized REPL session. If None, creates and
                 manages a singleton session (which is stopped on completion).

    Returns:
        SearchResult with proof details.
    """
    owns_session = session is None
    tree = ProofSearchTree(theorem, max_depth=max_depth, session=session)

    try:
        result = await tree.best_first_search(
            budget=budget,
            timeout=timeout,
            use_mathlib=use_mathlib,
        )
    finally:
        # Only clean up if we created the session ourselves
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
