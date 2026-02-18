"""Proof sketch generator for recursive subgoal decomposition.

Generates informal proof plans that guide the formalizer in creating
Lean have/sorry skeletons. The SketchGenerator protocol is pluggable —
the default LLMSketchGenerator uses the configured model, but a
Numina-style generator-verifier can be swapped in.

Includes an Aletheia-style NL reasoning pre-pass: before generating
formal proof steps, the LLM reasons freely in natural language about
the problem, then uses those insights to guide the formal decomposition.

Reference: HILBERT (Apple), Aristotle lemma-based reasoning, Goedel-V2 proof planning,
DeepMind Aletheia (NL reasoning → formal verification).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Maximum character budget for NL reasoning output to avoid prompt bloat (~500 tokens)
NL_REASONING_MAX_CHARS = 2000


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
    final_step: str | None = None  # Optional tactic to close after all haves
    nl_reasoning: str | None = None  # NL reasoning from Aletheia-style pre-analysis


@dataclass
class SketchContext:
    """Context provided to the sketch generator."""
    theorem: str                # Lean theorem statement
    mathlib_results: list[dict[str, Any]] = field(default_factory=list)
    previous_attempts: list[str] = field(default_factory=list)  # Failed approaches
    depth: int = 0              # Recursion depth (0 = top-level theorem)


SKETCH_PROMPT = """\
You are a mathematical proof planner specializing in Lean 4 formalization.
Given a theorem statement, decompose the proof into small, independently provable steps.

Theorem:
{theorem}

{context_section}

CRITICAL REQUIREMENTS for each step:
1. Every step MUST have a "formal_type" — the exact Lean 4 type for a `have` statement.
   Use fully explicit types with all necessary type annotations.
2. Each step should be INDEPENDENTLY PROVABLE — it should be possible to prove it
   using only standard Mathlib tactics (simp, ring, omega, norm_num, linarith, etc.)
   or a few lines of tactic proof, without needing the proofs of other steps.
3. Steps should build INTERMEDIATE LEMMAS toward the conclusion. The final step
   should follow easily from the intermediate results.
4. Use EXPLICIT type annotations — prefer `(n : Nat)` over implicit arguments.
5. Each formal_type must be a well-formed Lean 4 proposition that type-checks.

Respond with a JSON object (no markdown wrapping) containing a "sketches" array.
Each sketch has:
- "strategy": proof technique name (e.g., "induction", "contradiction", "direct",
  "cases", "calc", "have_chain")
- "steps": array of steps, each with:
  - "statement": natural language description of what this step proves
  - "formal_type": REQUIRED — the Lean 4 type for `have step_N : <formal_type>`
  - "depends_on": (optional) array of 0-indexed step indices this depends on
- "key_lemmas": array of Mathlib lemma names expected to be useful
- "final_step": (optional) the Lean tactic to close the proof after all `have` steps
  (e.g., "exact step_0.trans step_1", "linarith [step_0, step_1]")

Generate 2-3 diverse sketches using different proof strategies.
Keep each sketch to 2-5 steps. Fewer, cleaner steps are better than many.

GOOD decomposition example — each step is self-contained:
{{"sketches": [{{
  "strategy": "have_chain",
  "steps": [
    {{"statement": "Establish base inequality", "formal_type": "0 ≤ n * n"}},
    {{"statement": "Use base to derive main result", "formal_type": "n * n + 1 > 0", "depends_on": [0]}}
  ],
  "key_lemmas": ["Nat.zero_le", "Nat.succ_pos"],
  "final_step": "linarith [step_0, step_1]"
}}]}}

BAD decomposition (avoid): steps that restate the goal, steps without formal types,
steps that are not independently provable, overly complex single steps.
"""


NL_REASONING_PROMPT = """\
You are a mathematician analyzing a theorem to prove.

THEOREM:
{theorem}

Before writing any formal proof, reason through the following in natural language:

1. PROBLEM TYPE: What area of mathematics is this? What structures are involved?
2. KEY OBSERVATIONS: What patterns, symmetries, or special cases do you notice?
3. PROOF STRATEGY: What approach is most likely to work? Why?
4. INTERMEDIATE STEPS: What intermediate results or lemmas would you need?
5. POTENTIAL PITFALLS: What could go wrong with this approach?

Be specific and mathematical. Reference known theorems when applicable.
Keep your analysis concise — focus on the most important insights.
"""


SUBGOAL_SKETCH_PROMPT = """\
You are decomposing a SUBGOAL that arose from a larger proof decomposition.
The original proof was split into parts, and this particular subgoal could not
be proved directly. Decompose it into even simpler pieces.

Subgoal to decompose:
{theorem}

{context_section}

Since this is a subgoal (recursion depth {depth}), keep the decomposition MINIMAL:
- Use 1-3 steps at most
- Each step must have an explicit formal_type
- Prefer using automation-friendly lemmas (things provable by simp, omega, norm_num, ring)
- If the subgoal looks atomic (directly provable by a single tactic), return an empty
  sketches array: {{"sketches": []}}

Respond with a JSON object (no markdown wrapping) containing a "sketches" array.
Each sketch has:
- "strategy": proof technique name
- "steps": array with "statement", "formal_type" (REQUIRED), "depends_on"
- "key_lemmas": Mathlib lemma names
- "final_step": tactic to close proof after have steps
"""


def parse_sketch_response(
    response: str,
    nl_reasoning: str | None = None,
) -> list[ProofSketch]:
    """Parse LLM response into ProofSketch objects.

    Handles:
    - Raw JSON
    - JSON wrapped in markdown code blocks
    - Invalid JSON (returns empty list)

    Args:
        response: Raw LLM response containing JSON sketch data.
        nl_reasoning: Optional NL reasoning to attach to each sketch.
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
            formal_type = step_data.get("formal_type")
            # Skip steps without formal types — they can't be formalized
            if not formal_type:
                logger.debug(
                    "Skipping step without formal_type: %s",
                    step_data.get("statement", ""),
                )
                continue
            steps.append(SketchStep(
                statement=step_data.get("statement", ""),
                formal_type=formal_type,
                depends_on=step_data.get("depends_on", []),
            ))
        sketches.append(ProofSketch(
            strategy=s.get("strategy", "unknown"),
            steps=steps,
            key_lemmas=s.get("key_lemmas", []),
            final_step=s.get("final_step"),
            nl_reasoning=nl_reasoning,
        ))

    return sketches


def build_nl_reasoning_prompt(theorem: str) -> str:
    """Build the prompt for the NL reasoning pre-analysis phase.

    This is the first phase of Aletheia-style generation: let the LLM
    reason freely about the problem before formal decomposition.
    """
    return NL_REASONING_PROMPT.format(theorem=theorem)


def build_sketch_prompt(
    context: SketchContext,
    nl_reasoning: str | None = None,
) -> str:
    """Build the prompt for sketch generation.

    Uses the subgoal-specific prompt at depth > 0, which emphasizes
    minimal decomposition and atomic steps.

    Args:
        context: Sketch generation context.
        nl_reasoning: Optional NL reasoning from the pre-analysis phase.
            When provided, it is prepended to the context section so the
            formal sketch is guided by the free-form analysis.
    """
    context_parts = []

    # Inject NL reasoning as context for the formal sketch
    if nl_reasoning:
        context_parts.append(
            "MATHEMATICAL ANALYSIS (use this to guide your decomposition):\n"
            + nl_reasoning
        )

    if context.mathlib_results:
        lemma_names = [r.get("name", "") for r in context.mathlib_results[:5]]
        context_parts.append(f"Potentially relevant Mathlib lemmas: {', '.join(lemma_names)}")
    if context.previous_attempts:
        context_parts.append(
            "Previous failed approaches (avoid these):\n"
            + "\n".join(f"- {a}" for a in context.previous_attempts[-5:])
        )
    if context.depth > 0 and context.depth <= 1:
        context_parts.append(
            f"This is a subgoal at recursion depth {context.depth}. "
            "Prefer 1-3 simple steps. If the goal looks directly provable, "
            "return empty sketches."
        )

    context_section = "\n".join(context_parts) if context_parts else ""

    # Use the subgoal-specific prompt for deeper recursion
    if context.depth >= 2:
        return SUBGOAL_SKETCH_PROMPT.format(
            theorem=context.theorem,
            context_section=context_section,
            depth=context.depth,
        )

    return SKETCH_PROMPT.format(theorem=context.theorem, context_section=context_section)


@runtime_checkable
class SketchGenerator(Protocol):
    """Protocol for sketch generators (pluggable)."""

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
        """Generate proof sketches for a theorem."""
        ...


class LLMSketchGenerator:
    """Default sketch generator using the configured LLM.

    Supports an optional Aletheia-style NL reasoning pre-pass: when
    ``use_nl_reasoning`` is True, the generator first asks the LLM to
    reason freely about the theorem in natural language, then feeds
    those insights into the formal sketch generation prompt.
    """

    def __init__(
        self,
        model: str,
        use_nl_reasoning: bool = True,
    ) -> None:
        self.model = model
        self.use_nl_reasoning = use_nl_reasoning

    async def _generate_nl_reasoning(self, theorem: str) -> str | None:
        """Phase 1: Generate free-form NL reasoning about the theorem.

        Returns the NL analysis string, or None on failure.
        The output is capped at NL_REASONING_MAX_CHARS to keep the
        downstream sketch prompt concise.
        """
        from pydantic_ai import Agent
        from bourbaki.agent.core import _resolve_model_object

        prompt = build_nl_reasoning_prompt(theorem)
        resolved_model = _resolve_model_object(self.model)
        agent: Agent[None, str] = Agent(resolved_model, system_prompt=(
            "You are a mathematician. Provide a concise but insightful "
            "analysis. Do NOT write any Lean code or formal proofs."
        ))

        try:
            result = await agent.run(prompt)
            reasoning = result.output.strip()
            # Cap length to avoid bloating the sketch prompt
            if len(reasoning) > NL_REASONING_MAX_CHARS:
                reasoning = reasoning[:NL_REASONING_MAX_CHARS] + "..."
            logger.info(
                "NL reasoning generated (%d chars) for: %s",
                len(reasoning), theorem[:60],
            )
            return reasoning
        except Exception as e:
            logger.warning("NL reasoning generation failed: %s", e)
            return None

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
        """Generate proof sketches by prompting the LLM.

        When ``use_nl_reasoning`` is enabled:
        - Phase 1: Ask the LLM to analyze the theorem in natural language.
        - Phase 2: Feed the NL analysis into the sketch generation prompt
          so the formal decomposition is guided by free-form reasoning.

        The NL reasoning is stored on each ``ProofSketch.nl_reasoning``
        field so it can be passed downstream to the decomposer, coordinator,
        and prover.

        Uses _resolve_model_object for custom provider support.
        """
        from pydantic_ai import Agent
        from bourbaki.agent.core import _resolve_model_object

        # Phase 1: NL reasoning (skip for deep subgoals — they are simple)
        nl_reasoning: str | None = None
        if self.use_nl_reasoning and context.depth == 0:
            nl_reasoning = await self._generate_nl_reasoning(context.theorem)

        # Phase 2: Formal sketch generation (with NL reasoning as context)
        prompt = build_sketch_prompt(context, nl_reasoning=nl_reasoning)
        resolved_model = _resolve_model_object(self.model)
        agent: Agent[None, str] = Agent(resolved_model, system_prompt=(
            "You are a proof planning assistant for Lean 4 formalization. "
            "Output valid JSON only. Every step MUST have a formal_type field."
        ))

        try:
            result = await agent.run(prompt)
            sketches = parse_sketch_response(
                result.output, nl_reasoning=nl_reasoning,
            )
            # Filter out sketches with no usable steps
            return [s for s in sketches if s.steps]
        except Exception as e:
            logger.error("Sketch generation failed: %s", e)
            return []
