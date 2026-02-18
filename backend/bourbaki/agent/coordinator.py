"""Multi-agent proof coordinator.

Orchestrates Strategist, Searcher, Prover, and Verifier roles
to collaboratively construct and verify Lean 4 proofs.

Includes Aletheia-style NL reasoning pre-pass: before strategizing,
the LLM reasons freely in natural language about the theorem, and
those insights guide the strategist and prover phases.
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
from bourbaki.tools.lemma_library import LemmaEntry, get_lemma_library
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
    1. (Optional) NL reasoning: free-form analysis of the theorem
    2. Strategist generates proof sketch + subgoals (guided by NL reasoning)
    3. Searcher finds relevant Mathlib lemmas for each subgoal
    4. Prover constructs proof using tactics guided by strategy + lemmas + NL reasoning
    5. Verifier confirms with lean_prover
    6. On failure: loop back to Strategist with insights
    """

    def __init__(
        self,
        model: str,
        pool: Any = None,  # Optional REPLSessionPool
        use_nl_reasoning: bool = True,
    ) -> None:
        from bourbaki.agent.core import _resolve_model_object
        self.model = _resolve_model_object(model)
        self.pool = pool
        self.use_nl_reasoning = use_nl_reasoning
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
            max_retries: Maximum retry cycles (strategist -> prover).

        Returns:
            CoordinatorResult with proof details.
        """
        start = time.monotonic()
        previous_errors: list[str] = []

        # Pre-pass: Generate NL reasoning once (shared across retries)
        nl_reasoning: str | None = None
        if self.use_nl_reasoning:
            nl_reasoning = await self._generate_nl_reasoning(theorem)

        for attempt in range(max_retries):
            if time.monotonic() - start > timeout:
                break

            logger.info("Coordinator attempt %d/%d for: %s", attempt + 1, max_retries, theorem[:80])

            # Phase 1: Strategist generates proof sketch (guided by NL reasoning)
            strategy = await self._run_strategist(
                theorem, previous_errors, nl_reasoning=nl_reasoning,
            )
            self._stats["strategist"] += 1

            # Phase 2: Searcher finds relevant lemmas
            lemmas = await self._run_searcher(
                theorem, strategy.get("subgoals", []),
            )
            self._stats["searcher"] += 1

            # Phase 3: Prover attempts proof (with NL reasoning context)
            proof_code = await self._run_prover(
                theorem, strategy, lemmas, nl_reasoning=nl_reasoning,
            )
            self._stats["prover"] += 1

            if proof_code is None:
                previous_errors.append(f"Attempt {attempt + 1}: prover failed to construct proof")
                continue

            # Phase 4: Verifier confirms
            verified = await self._run_verifier(proof_code)
            self._stats["verifier"] += 1

            if verified:
                elapsed = time.monotonic() - start

                # Save verified proof tactics to the persistent lemma library
                try:
                    library = get_lemma_library()
                    tactics = _extract_coordinator_tactics(proof_code)
                    if tactics:
                        library.add(LemmaEntry(
                            goal_pattern=theorem,
                            tactics=tactics,
                            source="coordinator",
                            theorem_context=theorem,
                        ))
                        library.save_if_dirty()
                except Exception as exc:
                    logger.debug("Failed to save coordinator lemma: %s", exc)

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

    async def _generate_nl_reasoning(self, theorem: str) -> str | None:
        """Generate free-form NL reasoning about the theorem (Aletheia-style).

        This pre-pass lets the LLM reason freely before any formal planning,
        producing insights that guide the strategist and prover phases.

        Returns the NL analysis string, or None on failure.
        """
        from bourbaki.autonomous.sketch import (
            build_nl_reasoning_prompt,
            NL_REASONING_MAX_CHARS,
        )

        try:
            from pydantic_ai import Agent

            prompt = build_nl_reasoning_prompt(theorem)
            agent: Agent[None, str] = Agent(self.model, system_prompt=(
                "You are a mathematician. Provide a concise but insightful "
                "analysis. Do NOT write any Lean code or formal proofs."
            ))
            result = await agent.run(prompt)
            reasoning = result.output.strip()
            if len(reasoning) > NL_REASONING_MAX_CHARS:
                reasoning = reasoning[:NL_REASONING_MAX_CHARS] + "..."
            logger.info(
                "Coordinator NL reasoning generated (%d chars)", len(reasoning),
            )
            return reasoning
        except Exception as e:
            logger.warning("Coordinator NL reasoning failed: %s", e)
            return None

    async def _run_strategist(
        self,
        theorem: str,
        previous_errors: list[str],
        nl_reasoning: str | None = None,
    ) -> dict[str, Any]:
        """Generate a proof strategy using LLM.

        Returns dict with 'sketch' (list of steps) and 'subgoals' (list of goals).
        When NL reasoning is provided, it is included as context so the
        strategy benefits from the free-form mathematical analysis.
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

            reasoning_context = ""
            if nl_reasoning:
                reasoning_context = (
                    "\n\nMATHEMATICAL ANALYSIS (use these insights to guide your strategy):\n"
                    + nl_reasoning
                    + "\n"
                )

            prompt = (
                f"{STRATEGIST.system_prompt_addendum}\n\n"
                f"Theorem: {theorem}\n"
                f"{reasoning_context}"
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
        nl_reasoning: str | None = None,
    ) -> str | None:
        """Attempt to construct a proof using LLM guided by strategy and lemmas.

        Returns the proof code string if successful, None otherwise.
        When NL reasoning is provided, it gives the prover additional
        mathematical context for constructing the proof.
        """
        try:
            from pydantic_ai import Agent

            sketch = strategy.get("sketch", [])
            lemma_info = "\n".join(
                f"- {l.get('name', '')}: {l.get('type', '')}"
                for l in lemmas[:10]
            )

            reasoning_section = ""
            if nl_reasoning:
                reasoning_section = (
                    f"\nMathematical analysis:\n{nl_reasoning}\n"
                )

            prompt = (
                f"{PROVER.system_prompt_addendum}\n\n"
                f"Theorem: {theorem}\n"
                f"{reasoning_section}\n"
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

            # No immediate winner -- wait for remaining
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


def _extract_coordinator_tactics(proof_code: str) -> list[str]:
    """Extract tactic lines from a coordinator-produced proof."""
    import re

    by_match = re.search(r":=\s*by\b", proof_code)
    if not by_match:
        return []

    tactic_block = proof_code[by_match.end():].strip()
    tactics = [
        line.strip()
        for line in tactic_block.split("\n")
        if line.strip() and not line.strip().startswith("--")
    ]
    return tactics
