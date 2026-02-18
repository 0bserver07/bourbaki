"""Multi-agent proof coordinator.

Orchestrates Strategist, Searcher, Prover, and Verifier roles
to collaboratively construct and verify Lean 4 proofs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from bourbaki.agent.messages import AgentMessage, MessageBus
from bourbaki.agent.roles import STRATEGIST, SEARCHER, PROVER, VERIFIER
from bourbaki.tools.lean_prover import lean_prover
from bourbaki.tools.mathlib_search import mathlib_search

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorResult:
    """Result from the multi-agent proof coordinator."""
    success: bool
    proof_code: str | None = None
    error: str | None = None
    agent_stats: dict[str, int] = field(default_factory=dict)
    total_time: float = 0.0


class ProofCoordinator:
    """Orchestrates multi-agent proof construction.

    Workflow:
    1. Strategist generates proof sketch + subgoals
    2. Searcher finds relevant Mathlib lemmas for each subgoal
    3. Prover constructs proof using tactics guided by strategy + lemmas
    4. Verifier confirms with lean_prover
    5. On failure: loop back to Strategist with insights
    """

    def __init__(
        self,
        model: str,
        pool: Any = None,  # Optional REPLSessionPool
    ) -> None:
        self.model = model
        self.pool = pool
        self.bus = MessageBus()
        self._stats: dict[str, int] = {
            "strategist": 0,
            "searcher": 0,
            "prover": 0,
            "verifier": 0,
        }

    async def prove(
        self,
        theorem: str,
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> CoordinatorResult:
        """Run the coordinated proof pipeline.

        Args:
            theorem: Lean 4 theorem statement.
            timeout: Maximum time in seconds.
            max_retries: Maximum retry cycles (strategist → prover).

        Returns:
            CoordinatorResult with proof details.
        """
        start = time.monotonic()
        previous_errors: list[str] = []

        for attempt in range(max_retries):
            if time.monotonic() - start > timeout:
                break

            logger.info("Coordinator attempt %d/%d for: %s", attempt + 1, max_retries, theorem[:80])

            # Phase 1: Strategist generates proof sketch
            strategy = await self._run_strategist(theorem, previous_errors)
            self._stats["strategist"] += 1

            # Phase 2: Searcher finds relevant lemmas
            lemmas = await self._run_searcher(
                theorem, strategy.get("subgoals", []),
            )
            self._stats["searcher"] += 1

            # Phase 3: Prover attempts proof
            proof_code = await self._run_prover(theorem, strategy, lemmas)
            self._stats["prover"] += 1

            if proof_code is None:
                previous_errors.append(f"Attempt {attempt + 1}: prover failed to construct proof")
                continue

            # Phase 4: Verifier confirms
            verified = await self._run_verifier(proof_code)
            self._stats["verifier"] += 1

            if verified:
                elapsed = time.monotonic() - start
                return CoordinatorResult(
                    success=True,
                    proof_code=proof_code,
                    agent_stats=dict(self._stats),
                    total_time=elapsed,
                )
            else:
                previous_errors.append(
                    f"Attempt {attempt + 1}: proof failed verification"
                )

        elapsed = time.monotonic() - start
        return CoordinatorResult(
            success=False,
            error=f"Proof failed after {max_retries} attempts exhausted",
            agent_stats=dict(self._stats),
            total_time=elapsed,
        )

    async def _run_strategist(
        self,
        theorem: str,
        previous_errors: list[str],
    ) -> dict[str, Any]:
        """Generate a proof strategy using LLM.

        Returns dict with 'sketch' (list of steps) and 'subgoals' (list of goals).
        """
        try:
            from pydantic_ai import Agent
            import json

            error_context = ""
            if previous_errors:
                error_context = (
                    "\n\nPrevious attempts failed:\n"
                    + "\n".join(f"- {e}" for e in previous_errors[-3:])
                    + "\n\nTry a fundamentally different approach."
                )

            prompt = (
                f"{STRATEGIST.system_prompt_addendum}\n\n"
                f"Theorem: {theorem}\n"
                f"{error_context}\n\n"
                "Output a JSON object with:\n"
                '- "sketch": list of proof steps (tactics or strategies)\n'
                '- "subgoals": list of intermediate goals to prove\n\n'
                "Output ONLY the JSON:"
            )

            agent: Agent[None, str] = Agent(self.model)
            result = await agent.run(prompt)
            text = result.output

            # Parse JSON from response
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"sketch": [text], "subgoals": []}

        except Exception as e:
            logger.error("Strategist failed: %s", e)
            return {"sketch": [], "subgoals": []}

    async def _run_searcher(
        self,
        theorem: str,
        subgoals: list[str],
    ) -> list[dict[str, Any]]:
        """Search Mathlib for lemmas relevant to the theorem and subgoals."""
        all_lemmas: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        # Search for the main theorem type
        queries = [theorem]
        queries.extend(subgoals[:5])

        for query in queries[:6]:
            for mode in ("semantic", "type", "natural"):
                try:
                    result = await mathlib_search(
                        query=query[:200], mode=mode, max_results=3,
                    )
                    if result.get("success"):
                        for hit in result.get("results", []):
                            name = hit.get("name", "")
                            if name and name not in seen_names:
                                seen_names.add(name)
                                all_lemmas.append(hit)
                except Exception:
                    continue

        return all_lemmas[:15]  # Cap at 15 lemmas

    async def _run_prover(
        self,
        theorem: str,
        strategy: dict[str, Any],
        lemmas: list[dict[str, Any]],
    ) -> str | None:
        """Attempt to construct a proof using LLM guided by strategy and lemmas.

        Returns the proof code string if successful, None otherwise.
        """
        try:
            from pydantic_ai import Agent

            sketch = strategy.get("sketch", [])
            lemma_info = "\n".join(
                f"- {l.get('name', '')}: {l.get('type', '')}"
                for l in lemmas[:10]
            )

            prompt = (
                f"{PROVER.system_prompt_addendum}\n\n"
                f"Theorem: {theorem}\n\n"
                f"Proof strategy:\n"
                + "\n".join(f"- {s}" for s in sketch)
                + f"\n\nAvailable Mathlib lemmas:\n{lemma_info}\n\n"
                "Write a complete Lean 4 proof. Output ONLY the Lean code:"
            )

            agent: Agent[None, str] = Agent(self.model)
            result = await agent.run(prompt)
            text = result.output

            # Extract Lean code
            import re
            match = re.search(r"```(?:lean4?|lean)\n(.*?)```", text, re.DOTALL)
            if match:
                return match.group(1).strip()

            match = re.search(r"```\n(.*?)```", text, re.DOTALL)
            if match:
                return match.group(1).strip()

            # Return raw if it looks like Lean code
            if "theorem" in text or "lemma" in text or ":= by" in text:
                return text.strip()

            return None

        except Exception as e:
            logger.error("Prover failed: %s", e)
            return None

    async def _run_verifier(self, proof_code: str) -> bool:
        """Verify a proof with lean_prover."""
        try:
            result = await lean_prover(code=proof_code, mode="check")
            return bool(result.get("proofComplete") or result.get("success"))
        except Exception as e:
            logger.error("Verifier failed: %s", e)
            return False

    async def ensemble_prove(
        self,
        theorem: str,
        strategies: list[dict[str, Any]] | None = None,
        lemmas: list[dict[str, Any]] | None = None,
        timeout: float = 120.0,
    ) -> CoordinatorResult:
        """Launch multiple provers in parallel with different strategies.

        First successful proof wins; others are cancelled.
        """
        start = time.monotonic()

        if strategies is None:
            strategies = [
                {"sketch": ["simp", "ring", "norm_num", "omega"], "focus": "automation"},
                {"sketch": ["induction", "cases", "simp_all"], "focus": "induction"},
                {"sketch": ["exact", "apply", "rw"], "focus": "mathlib"},
            ]

        lemmas = lemmas or []

        async def _try_strategy(strategy: dict[str, Any]) -> str | None:
            return await self._run_prover(theorem, strategy, lemmas)

        tasks = [
            asyncio.create_task(_try_strategy(s))
            for s in strategies
        ]

        try:
            done, pending = await asyncio.wait(
                tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                proof_code = task.result()
                if proof_code is not None:
                    # Verify before declaring success
                    verified = await self._run_verifier(proof_code)
                    if verified:
                        # Cancel remaining
                        for p in pending:
                            p.cancel()
                        elapsed = time.monotonic() - start
                        return CoordinatorResult(
                            success=True,
                            proof_code=proof_code,
                            agent_stats=dict(self._stats),
                            total_time=elapsed,
                        )

            # No immediate winner — wait for remaining
            if pending:
                remaining_timeout = max(0, timeout - (time.monotonic() - start))
                done2, still_pending = await asyncio.wait(
                    pending, timeout=remaining_timeout,
                )
                for task in done2:
                    proof_code = task.result()
                    if proof_code is not None:
                        verified = await self._run_verifier(proof_code)
                        if verified:
                            for p in still_pending:
                                p.cancel()
                            elapsed = time.monotonic() - start
                            return CoordinatorResult(
                                success=True,
                                proof_code=proof_code,
                                agent_stats=dict(self._stats),
                                total_time=elapsed,
                            )
                # Cancel any still pending
                for p in still_pending:
                    p.cancel()

        except Exception as e:
            logger.error("Ensemble prove failed: %s", e)
            for t in tasks:
                t.cancel()

        elapsed = time.monotonic() - start
        return CoordinatorResult(
            success=False,
            error="Ensemble: no strategy produced a verified proof",
            agent_stats=dict(self._stats),
            total_time=elapsed,
        )
