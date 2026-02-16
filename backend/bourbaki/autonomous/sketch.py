"""Proof sketch generator for recursive subgoal decomposition.

Generates informal proof plans that guide the formalizer in creating
Lean have/sorry skeletons. The SketchGenerator protocol is pluggable —
the default LLMSketchGenerator uses the configured model, but a
Numina-style generator-verifier can be swapped in.

Reference: HILBERT (Apple), Aristotle lemma-based reasoning, Goedel-V2 proof planning.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class SketchStep:
    """A single step in an informal proof sketch."""
    statement: str              # NL description: "Show base case n=0"
    formal_type: str | None = None  # Optional Lean type hint: "P 0"
    depends_on: list[int] = field(default_factory=list)  # Indices of prior steps


@dataclass
class ProofSketch:
    """An informal proof plan with ordered steps."""
    strategy: str               # e.g., "induction", "contradiction", "direct"
    steps: list[SketchStep]     # Ordered informal steps
    key_lemmas: list[str] = field(default_factory=list)  # Expected Mathlib lemmas


@dataclass
class SketchContext:
    """Context provided to the sketch generator."""
    theorem: str                # Lean theorem statement
    mathlib_results: list[dict[str, Any]] = field(default_factory=list)
    previous_attempts: list[str] = field(default_factory=list)  # Failed approaches
    depth: int = 0              # Recursion depth (0 = top-level theorem)


SKETCH_PROMPT = """\
You are a mathematical proof planner. Given a Lean 4 theorem statement, produce
an informal proof sketch broken into small, verifiable steps.

Theorem:
{theorem}

{context_section}

Respond with a JSON object (no markdown wrapping) containing a "sketches" array.
Each sketch has:
- "strategy": proof technique name (e.g., "induction", "contradiction", "direct", "cases")
- "steps": array of steps, each with:
  - "statement": natural language description of what this step proves
  - "formal_type": (optional) the Lean 4 type this step would have as a `have` statement
  - "depends_on": (optional) array of 0-indexed step indices this depends on
- "key_lemmas": array of Mathlib lemma names expected to be useful

Generate 1-3 sketches with different strategies when possible.
Keep each sketch to 2-6 steps. Prefer smaller decompositions.

Example response:
{{"sketches": [{{
  "strategy": "induction",
  "steps": [
    {{"statement": "Base case: show P(0)", "formal_type": "P 0"}},
    {{"statement": "Inductive step: assuming P(n), show P(n+1)", "formal_type": "P n → P (n + 1)", "depends_on": [0]}}
  ],
  "key_lemmas": ["Nat.succ_eq_add_one"]
}}]}}
"""


def parse_sketch_response(response: str) -> list[ProofSketch]:
    """Parse LLM response into ProofSketch objects.

    Handles:
    - Raw JSON
    - JSON wrapped in markdown code blocks
    - Invalid JSON (returns empty list)
    """
    # Strip markdown code blocks if present
    text = response.strip()
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse sketch response as JSON")
        return []

    sketches: list[ProofSketch] = []
    for s in data.get("sketches", []):
        steps = []
        for step_data in s.get("steps", []):
            steps.append(SketchStep(
                statement=step_data.get("statement", ""),
                formal_type=step_data.get("formal_type"),
                depends_on=step_data.get("depends_on", []),
            ))
        sketches.append(ProofSketch(
            strategy=s.get("strategy", "unknown"),
            steps=steps,
            key_lemmas=s.get("key_lemmas", []),
        ))

    return sketches


def build_sketch_prompt(context: SketchContext) -> str:
    """Build the prompt for sketch generation."""
    context_parts = []
    if context.mathlib_results:
        lemma_names = [r.get("name", "") for r in context.mathlib_results[:5]]
        context_parts.append(f"Potentially relevant Mathlib lemmas: {', '.join(lemma_names)}")
    if context.previous_attempts:
        context_parts.append(
            "Previous failed approaches (avoid these):\n"
            + "\n".join(f"- {a}" for a in context.previous_attempts)
        )
    if context.depth > 0:
        context_parts.append(
            f"This is a subgoal at recursion depth {context.depth}. "
            "Keep decomposition minimal (1-3 steps)."
        )

    context_section = "\n".join(context_parts) if context_parts else ""
    return SKETCH_PROMPT.format(theorem=context.theorem, context_section=context_section)


@runtime_checkable
class SketchGenerator(Protocol):
    """Protocol for sketch generators (pluggable)."""

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
        """Generate proof sketches for a theorem."""
        ...


class LLMSketchGenerator:
    """Default sketch generator using the configured LLM."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
        """Generate proof sketches by prompting the LLM."""
        from pydantic_ai import Agent

        prompt = build_sketch_prompt(context)
        agent: Agent[None, str] = Agent(self.model, system_prompt=(
            "You are a proof planning assistant. Output valid JSON only."
        ))

        try:
            result = await agent.run(prompt)
            return parse_sketch_response(result.output)
        except Exception as e:
            logger.error("Sketch generation failed: %s", e)
            return []
