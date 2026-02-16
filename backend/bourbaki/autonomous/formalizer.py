"""Formalizer: converts proof sketches to Lean have/sorry skeletons.

Takes a ProofSketch (informal plan) and a theorem statement, produces
a Lean code skeleton where each step is a `have` with `sorry`, ready
for the subgoal solver to fill in.

Reference: HILBERT skeleton generation, DeepSeek-V2 have/sorry decomposition.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bourbaki.autonomous.sketch import ProofSketch

logger = logging.getLogger(__name__)


@dataclass
class Subgoal:
    """A single subgoal extracted from a formalized skeleton."""
    index: int                      # Position in the skeleton
    label: str                      # e.g., "step_0", "step_1"
    lean_type: str                  # The type of the have statement
    lean_context: str = ""          # Lean code preceding this subgoal
    proof_state_id: int | None = None  # REPL proof state ID (set later)


@dataclass
class FormalizedSkeleton:
    """Result of formalizing a proof sketch."""
    code: str                       # Full Lean code with sorry placeholders
    subgoals: list[Subgoal]         # Extracted subgoals to solve
    compilation_ok: bool = False    # Whether the skeleton type-checks in Lean
    errors: list[str] = field(default_factory=list)


def build_skeleton_code(theorem: str, sketch: ProofSketch) -> str:
    """Build Lean code with have/sorry from a proof sketch.

    Args:
        theorem: The Lean theorem statement (e.g., "theorem foo : T").
        sketch: The informal proof sketch.

    Returns:
        Lean code string with sorry placeholders.
    """
    lines = [f"{theorem} := by"]

    for i, step in enumerate(sketch.steps):
        label = f"step_{i}"
        if step.formal_type:
            lines.append(f"  have {label} : {step.formal_type} := by sorry")
        else:
            # No formal type hint — use a placeholder that the LLM correction
            # loop will need to fix
            lines.append(f"  -- Step {i}: {step.statement}")
            lines.append(f"  have {label} : sorry := by sorry")

    # Final step: try to close the proof using the intermediate steps
    step_labels = [f"step_{i}" for i in range(len(sketch.steps))]
    if step_labels:
        lines.append(f"  exact?")
    else:
        lines.append("  sorry")

    return "\n".join(lines)


def extract_subgoals_from_code(code: str) -> list[Subgoal]:
    """Extract subgoals (have ... := by sorry) from skeleton code.

    Returns:
        List of Subgoal objects with label and lean_type populated.
    """
    subgoals: list[Subgoal] = []

    # Match: have <label> : <type> := by sorry
    pattern = re.compile(
        r"have\s+(\w+)\s*:\s*(.+?)\s*:=\s*by\s+sorry"
    )

    lines = code.split("\n")
    context_lines: list[str] = []

    for line in lines:
        match = pattern.search(line)
        if match:
            label = match.group(1)
            lean_type = match.group(2).strip()
            subgoals.append(Subgoal(
                index=len(subgoals),
                label=label,
                lean_type=lean_type,
                lean_context="\n".join(context_lines),
            ))
        context_lines.append(line)

    return subgoals


def stitch_proofs(
    skeleton_code: str,
    subgoal_proofs: dict[str, list[str]],
) -> str:
    """Replace sorry placeholders with actual tactic proofs.

    Args:
        skeleton_code: The Lean skeleton with sorry placeholders.
        subgoal_proofs: Map of subgoal label -> list of tactic strings.

    Returns:
        Complete Lean code with sorry replaced by actual proofs.
    """
    result = skeleton_code

    for label, tactics in subgoal_proofs.items():
        if not tactics:
            continue

        # Pattern: have <label> : <type> := by sorry
        pattern = re.compile(
            rf"(have\s+{re.escape(label)}\s*:\s*.+?)\s*:=\s*by\s+sorry"
        )

        if len(tactics) == 1:
            replacement = rf"\1 := by {tactics[0]}"
        else:
            # Multi-tactic: indent under `by`
            tactic_block = "\n    ".join(tactics)
            replacement = rf"\1 := by\n    {tactic_block}"

        result = pattern.sub(replacement, result)

    return result


FORMALIZE_PROMPT = """\
Convert this proof sketch into a Lean 4 skeleton with `have` statements.
Each intermediate step should be a `have step_N : <type> := by sorry`.
The final line should close the proof using the intermediate steps.

Theorem: {theorem}

Proof sketch:
{sketch_text}

Key Mathlib lemmas that may be useful: {lemmas}

Output ONLY the Lean 4 code, no explanation. The code should type-check
except for the `sorry` placeholders. Use `have step_0`, `step_1`, etc.
as labels.
"""


async def formalize_sketch(
    theorem: str,
    sketch: ProofSketch,
    model: str,
    max_retries: int = 2,
) -> FormalizedSkeleton:
    """Convert a proof sketch to a Lean skeleton using LLM + Lean feedback.

    Tries to formalize, checks compilation, and retries with error
    feedback up to max_retries times (per Goedel-V2: 2 rounds optimal).

    Args:
        theorem: Lean theorem statement.
        sketch: The informal proof sketch.
        model: LLM model identifier.
        max_retries: Max error-correction rounds.

    Returns:
        FormalizedSkeleton with code, subgoals, and compilation status.
    """
    from pydantic_ai import Agent

    # First attempt: build from sketch directly
    code = build_skeleton_code(theorem, sketch)
    subgoals = extract_subgoals_from_code(code)

    # Try to compile and get REPL feedback
    skeleton = FormalizedSkeleton(code=code, subgoals=subgoals)

    try:
        from bourbaki.tools.lean_repl import lean_tactic
        result = await lean_tactic(goal=theorem, tactic="sorry", proof_state=None)
        if result.get("success"):
            skeleton.compilation_ok = True
            return skeleton
    except Exception:
        pass

    # If direct build fails, ask LLM to formalize with error feedback
    sketch_text = "\n".join(
        f"{i+1}. {s.statement}" + (f" [{s.formal_type}]" if s.formal_type else "")
        for i, s in enumerate(sketch.steps)
    )
    lemmas = ", ".join(sketch.key_lemmas) if sketch.key_lemmas else "none specified"
    prompt = FORMALIZE_PROMPT.format(
        theorem=theorem, sketch_text=sketch_text, lemmas=lemmas,
    )

    errors: list[str] = []
    for attempt in range(max_retries + 1):
        try:
            agent: Agent[None, str] = Agent(model, system_prompt=(
                "You are a Lean 4 formalization assistant. Output only valid Lean 4 code."
            ))

            if attempt > 0 and errors:
                prompt += f"\n\nPrevious attempt failed with errors:\n" + "\n".join(errors[-3:])
                prompt += "\nFix the errors and output corrected code."

            result = await agent.run(prompt)
            generated_code = _extract_lean_code(result.output)
            if generated_code:
                code = generated_code

            subgoals = extract_subgoals_from_code(code)
            skeleton = FormalizedSkeleton(code=code, subgoals=subgoals)

            # Try to compile
            try:
                from bourbaki.tools.lean_repl import lean_tactic
                check = await lean_tactic(goal=theorem, tactic="sorry", proof_state=None)
                if check.get("success"):
                    skeleton.compilation_ok = True
                    return skeleton
                else:
                    errors.append(check.get("error", "unknown compilation error"))
            except Exception as e:
                errors.append(str(e))

        except Exception as e:
            errors.append(str(e))

    skeleton.errors = errors
    return skeleton


def _extract_lean_code(text: str) -> str | None:
    """Extract Lean code from an LLM response (may be wrapped in markdown)."""
    # Try markdown code block
    match = re.search(r"```(?:lean4?)\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block, check if the whole response looks like Lean
    if "theorem" in text or "lemma" in text or ":= by" in text:
        return text.strip()

    return None
