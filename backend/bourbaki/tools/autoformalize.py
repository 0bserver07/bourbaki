"""Autoformalize: convert natural language math to Lean 4 code.

Supports two modes:
- statement: Convert NL theorem statement → Lean 4 type signature
- proof_step: Convert NL proof step → Lean 4 tactic
"""

from __future__ import annotations

import logging
import time
from typing import Any

from bourbaki.tools.lean_prover import lean_prover

logger = logging.getLogger(__name__)


# Prompt templates for autoformalization
_STATEMENT_PROMPT = """Convert the following mathematical statement to a Lean 4 theorem declaration.

**Rules:**
- Use `theorem` keyword with a descriptive name
- Import Mathlib.Tactic if needed
- Use standard Mathlib types and notation
- End with `:= by sorry` (we'll fill in the proof later)

**Input statement:**
{input_text}

{context_section}

**Output ONLY the Lean 4 code, no explanation:**"""

_PROOF_STEP_PROMPT = """Convert the following informal proof step to a Lean 4 tactic.

**Current proof context:**
{context_section}

**Informal proof step:**
{input_text}

**Output ONLY the Lean 4 tactic (a single line or tactic block), no explanation:**"""


async def autoformalize(
    input_text: str,
    mode: str = "statement",
    context: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Convert natural language math to Lean 4 code.

    Args:
        input_text: Natural language mathematical statement or proof step.
        mode: "statement" to convert NL → Lean theorem declaration,
              "proof_step" to convert NL → Lean tactic.
        context: Optional proof context (e.g., current goal state, hypotheses).
        model: Optional model override for LLM call.

    Returns:
        Dict with success, lean_code, verified (for statements), confidence, duration.
    """
    start = time.monotonic()

    if mode not in ("statement", "proof_step"):
        return {
            "success": False,
            "error": f"Unknown mode: {mode!r}. Use 'statement' or 'proof_step'.",
            "duration": int((time.monotonic() - start) * 1000),
        }

    context_section = f"**Context:**\n{context}" if context else ""

    if mode == "statement":
        return await _formalize_statement(input_text, context_section, model, start)
    else:
        return await _formalize_proof_step(input_text, context_section, start)


async def _formalize_statement(
    input_text: str,
    context_section: str,
    model: str | None,
    start: float,
) -> dict[str, Any]:
    """Convert NL statement to Lean 4 theorem declaration with verification."""
    prompt = _STATEMENT_PROMPT.format(
        input_text=input_text,
        context_section=context_section,
    )

    # Use pydantic-ai to call LLM
    lean_code = await _call_llm(prompt, model)
    if lean_code is None:
        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": "LLM failed to generate Lean code",
            "duration": elapsed,
        }

    # Clean up the response
    lean_code = _extract_lean_code(lean_code)

    # Verify with lean_prover
    verification = await lean_prover(code=lean_code, mode="check")
    verified = bool(verification.get("success") or verification.get("env"))

    if not verified:
        # Retry once with error feedback
        errors = verification.get("errors", [])
        error_text = "; ".join(
            e.get("message", str(e)) if isinstance(e, dict) else str(e)
            for e in errors[:3]
        ) if errors else "type check failed"

        retry_prompt = (
            f"{prompt}\n\n"
            f"**Previous attempt failed with errors:**\n{error_text}\n\n"
            f"**Previous attempt:**\n```lean\n{lean_code}\n```\n\n"
            f"**Fix the errors and output corrected Lean 4 code:**"
        )
        retry_code = await _call_llm(retry_prompt, model)
        if retry_code is not None:
            retry_code = _extract_lean_code(retry_code)
            retry_verification = await lean_prover(code=retry_code, mode="check")
            if retry_verification.get("success") or retry_verification.get("env"):
                lean_code = retry_code
                verified = True

    elapsed = int((time.monotonic() - start) * 1000)
    return {
        "success": True,
        "lean_code": lean_code,
        "verified": verified,
        "mode": "statement",
        "duration": elapsed,
    }


async def _formalize_proof_step(
    input_text: str,
    context_section: str,
    start: float,
) -> dict[str, Any]:
    """Convert NL proof step to Lean 4 tactic."""
    prompt = _PROOF_STEP_PROMPT.format(
        input_text=input_text,
        context_section=context_section,
    )

    lean_code = await _call_llm(prompt)
    if lean_code is None:
        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": "LLM failed to generate tactic",
            "duration": elapsed,
        }

    # Clean up — extract just the tactic
    lean_code = _extract_lean_code(lean_code).strip()

    elapsed = int((time.monotonic() - start) * 1000)
    return {
        "success": True,
        "lean_code": lean_code,
        "mode": "proof_step",
        "duration": elapsed,
    }


async def _call_llm(prompt: str, model: str | None = None) -> str | None:
    """Call LLM to generate Lean code from a prompt.

    Uses pydantic-ai Agent for the LLM call.
    """
    try:
        from pydantic_ai import Agent
        from bourbaki.config import settings

        model_str = model or settings.default_model
        agent: Agent[None, str] = Agent(model_str)
        result = await agent.run(prompt)
        return result.output
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None


def _extract_lean_code(text: str) -> str:
    """Extract Lean code from LLM response, handling markdown fences."""
    # Try to extract from ```lean ... ``` blocks
    import re
    match = re.search(r"```(?:lean4?|lean)\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code blocks
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No code blocks — return as-is, stripping leading/trailing whitespace
    return text.strip()
