"""Answer generation pipeline for PutnamBench answer-type problems.

PutnamBench has ~224 "answer problems" with the shape:

    abbrev putnam_YYYY_XN_solution : TYPE := sorry
    -- REFERENCE_ANSWER
    theorem putnam_YYYY_XN ... : ... putnam_YYYY_XN_solution ... := by sorry

The answer placeholder is ``sorry``.  To solve these properly we need to:

1. Extract or generate a candidate answer (Lean expression)
2. Substitute it into the ``abbrev`` (replacing ``sorry``)
3. Attempt to prove the theorem with the filled-in answer
4. Verify the whole file with ``lean_prover``

The pipeline tries the reference answer (commented out below the abbrev) first,
then falls back to LLM-generated answers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from bourbaki.benchmarks.putnam_loader import PutnamProblem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for answer extraction
# ---------------------------------------------------------------------------

# Match the abbrev line including optional "noncomputable" prefix.
# Captures: (1) prefix "noncomputable " or "", (2) answer name, (3) type annotation
_ABBREV_LINE_RE = re.compile(
    r"^(noncomputable\s+)?abbrev\s+(putnam_\w+_solution)\s*:\s*(.+?)\s*:=\s*sorry\s*$",
    re.MULTILINE,
)

# Match a comment line immediately after the abbrev (the reference answer).
# PutnamBench uses ``-- ANSWER`` or ``--- ANSWER`` (with 2 or 3 dashes).
_REFERENCE_ANSWER_RE = re.compile(
    r"^---?\s*(.+)$",
    re.MULTILINE,
)


@dataclass
class AnswerAttempt:
    """Result of an answer generation attempt."""

    answer_code: str  # The Lean expression to fill in
    theorem_solved: bool  # Did the theorem prove with this answer?
    verified: bool  # Did lean_prover verify the whole file?
    proof_tactics: list[str] = field(default_factory=list)
    error: str | None = None
    source: str = "unknown"  # "reference" or "llm"


# ---------------------------------------------------------------------------
# Reference answer extraction
# ---------------------------------------------------------------------------

def extract_reference_answer(lean_source: str, answer_name: str) -> str | None:
    """Extract the reference answer comment from a PutnamBench Lean file.

    PutnamBench files typically have the reference answer as a comment
    immediately following the ``abbrev ... := sorry`` line::

        abbrev putnam_1962_a5_solution : ℕ → ℕ := sorry
        -- fun n : ℕ => n * (n + 1) * 2^(n - 2)

    The comment may use ``--`` or ``---`` (2 or 3 dashes).

    Args:
        lean_source: The full Lean source code of the file.
        answer_name: The expected answer name (e.g. "putnam_1962_a5_solution").

    Returns:
        The reference answer expression (without the ``--`` prefix), or None
        if no reference answer comment is found.
    """
    # Find the abbrev line with sorry
    pattern = re.compile(
        r"(?:noncomputable\s+)?abbrev\s+"
        + re.escape(answer_name)
        + r"\s*:.*?:=\s*sorry\s*\n"
        + r"---?\s*(.+)",
    )
    m = pattern.search(lean_source)
    if m:
        return m.group(1).strip()
    return None


def extract_answer_type(lean_source: str, answer_name: str) -> str | None:
    """Extract the type annotation from the abbrev declaration.

    Given::

        noncomputable abbrev putnam_2024_a1_solution : Set ℕ := sorry

    Returns ``"Set ℕ"``.

    Args:
        lean_source: The full Lean source code.
        answer_name: The answer name to find.

    Returns:
        The Lean type annotation string, or None if not found.
    """
    pattern = re.compile(
        r"(?:noncomputable\s+)?abbrev\s+"
        + re.escape(answer_name)
        + r"\s*:\s*(.+?)\s*:=",
    )
    m = pattern.search(lean_source)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Answer substitution
# ---------------------------------------------------------------------------

def insert_answer(lean_source: str, answer_name: str, answer_expr: str) -> str:
    """Replace ``sorry`` in the answer abbrev with the given expression.

    Transforms::

        abbrev putnam_XXXX_solution : TYPE := sorry

    into::

        abbrev putnam_XXXX_solution : TYPE := ANSWER_EXPR

    The comment line (reference answer) is preserved as-is -- it doesn't
    affect compilation.

    Args:
        lean_source: Full Lean source code.
        answer_name: The abbrev name to patch.
        answer_expr: The Lean expression to insert.

    Returns:
        Modified Lean source with the answer filled in.
    """
    pattern = re.compile(
        r"((?:noncomputable\s+)?abbrev\s+"
        + re.escape(answer_name)
        + r"\s*:.*?:=\s*)sorry",
    )
    replaced, count = pattern.subn(r"\g<1>" + answer_expr, lean_source, count=1)
    if count == 0:
        logger.warning(
            "Could not substitute answer for %s -- pattern not found", answer_name,
        )
    return replaced


# ---------------------------------------------------------------------------
# Build verification code
# ---------------------------------------------------------------------------

def build_answer_verification_code(
    problem: PutnamProblem,
    answer_expr: str,
) -> str:
    """Build the full Lean code for verifying an answer + theorem proof.

    Constructs: imports + preamble + (setup_block with answer filled in) +
    theorem := by sorry.

    The answer is substituted into the setup_block (which contains the abbrev).

    Args:
        problem: The PutnamProblem to build code for.
        answer_expr: The Lean expression to fill into the answer abbrev.

    Returns:
        Full Lean source code string.
    """
    # Start with the setup block and substitute the answer
    setup = problem.setup_block
    if problem.answer_name:
        setup = insert_answer(setup, problem.answer_name, answer_expr)

    parts: list[str] = ["import Mathlib"]
    if problem.preamble:
        parts.append(problem.preamble)
    if setup:
        parts.append(setup)
    parts.append(f"{problem.statement} :=\n  sorry")

    return "\n\n".join(parts)


def build_full_proof_code(
    problem: PutnamProblem,
    answer_expr: str,
    proof_code: str,
) -> str:
    """Build the full Lean code with both answer and proof filled in.

    Args:
        problem: The PutnamProblem.
        answer_expr: The answer expression.
        proof_code: The full theorem proof code (e.g. ``theorem ... := by tactic``).

    Returns:
        Full Lean source code string.
    """
    setup = problem.setup_block
    if problem.answer_name:
        setup = insert_answer(setup, problem.answer_name, answer_expr)

    parts: list[str] = ["import Mathlib"]
    if problem.preamble:
        parts.append(problem.preamble)
    if setup:
        parts.append(setup)
    parts.append(proof_code)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

_ANSWER_PROMPT_TEMPLATE = """\
You are a mathematician solving a Putnam competition problem formalized in Lean 4.

The problem has an answer placeholder that you need to fill in. Given the problem
statement and the answer type signature, provide the mathematical answer as a
valid Lean 4 expression.

## Answer Type Signature

```lean
abbrev {answer_name} : {answer_type} := ???
```

## Theorem Statement

```lean
{theorem_statement}
```

{docstring_section}
{reference_section}

## Instructions

Provide ONLY the Lean 4 expression that should replace ``sorry`` in the abbrev.
The expression must have type ``{answer_type}``.

- Do NOT include the ``abbrev`` keyword or ``:=``
- Do NOT include any explanation or comments
- The expression must be a valid Lean 4 term of the given type
- Use Mathlib notation where appropriate (e.g., ``Finset.Icc``, ``Set.Icc``,
  ``Real.sqrt``, etc.)

Your answer (Lean expression only):
"""


def build_answer_prompt(
    problem: PutnamProblem,
    answer_type: str,
    reference_answer: str | None = None,
) -> str:
    """Build the LLM prompt for answer generation.

    Args:
        problem: The PutnamProblem.
        answer_type: The Lean type of the answer.
        reference_answer: Optional commented-out reference answer.

    Returns:
        The prompt string.
    """
    docstring_section = ""
    if problem.docstring:
        docstring_section = f"## Problem Description\n\n{problem.docstring}\n"

    reference_section = ""
    if reference_answer:
        reference_section = (
            f"## Reference Answer (from comments, may need adjustment)\n\n"
            f"```lean\n{reference_answer}\n```\n"
        )

    return _ANSWER_PROMPT_TEMPLATE.format(
        answer_name=problem.answer_name or "solution",
        answer_type=answer_type,
        theorem_statement=problem.statement,
        docstring_section=docstring_section,
        reference_section=reference_section,
    )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def generate_answer(
    problem: PutnamProblem,
    model: str = "glm:glm-5",
    max_attempts: int = 3,
    prove_timeout: int = 60,
    search_budget: int = 64,
    verify_timeout: int = 30,
) -> AnswerAttempt | None:
    """Generate a candidate answer for an answer-type problem.

    Strategy:
    1. Read the source file and look for a reference answer comment
    2. If a reference answer exists, try it first
    3. If no reference or it fails, ask the LLM for an answer
    4. For each candidate answer:
       a. Insert it into the abbrev
       b. Try to prove the theorem with prove_with_search
       c. Verify with lean_prover

    Args:
        problem: The PutnamProblem (must have answer_is_sorry=True).
        model: Model string for the LLM (e.g. "glm:glm-5").
        max_attempts: Maximum LLM answer attempts.
        prove_timeout: Timeout for proof search per answer attempt.
        search_budget: Tactic budget for proof search.
        verify_timeout: Timeout for whole-file verification.

    Returns:
        An AnswerAttempt if any candidate succeeded, or the best failed
        attempt, or None if the problem is not an answer-type problem.
    """
    if not problem.has_answer or not problem.answer_is_sorry or not problem.answer_name:
        return None

    # Read the source file for reference answer extraction
    source_code = _read_source_file(problem)
    if source_code is None:
        return AnswerAttempt(
            answer_code="",
            theorem_solved=False,
            verified=False,
            error="Could not read source file",
        )

    answer_type = extract_answer_type(source_code, problem.answer_name)
    if answer_type is None:
        answer_type = "unknown"

    reference_answer = extract_reference_answer(source_code, problem.answer_name)

    # Collect candidate answers: reference first, then LLM
    candidates: list[tuple[str, str]] = []  # (answer_expr, source)

    if reference_answer:
        candidates.append((reference_answer, "reference"))

    # Generate LLM candidates
    llm_answers = await _generate_llm_answers(
        problem, answer_type, reference_answer, model, max_attempts,
    )
    for ans in llm_answers:
        candidates.append((ans, "llm"))

    if not candidates:
        return AnswerAttempt(
            answer_code="",
            theorem_solved=False,
            verified=False,
            error="No candidate answers generated",
        )

    # Try each candidate
    best_attempt: AnswerAttempt | None = None
    for answer_expr, source in candidates:
        attempt = await _try_answer(
            problem=problem,
            answer_expr=answer_expr,
            source=source,
            prove_timeout=prove_timeout,
            search_budget=search_budget,
            verify_timeout=verify_timeout,
        )

        if attempt.verified:
            return attempt  # Full success

        if attempt.theorem_solved and (
            best_attempt is None or not best_attempt.theorem_solved
        ):
            best_attempt = attempt
        elif best_attempt is None:
            best_attempt = attempt

    return best_attempt


async def _try_answer(
    problem: PutnamProblem,
    answer_expr: str,
    source: str,
    prove_timeout: int = 60,
    search_budget: int = 64,
    verify_timeout: int = 30,
) -> AnswerAttempt:
    """Try a single answer candidate: insert, prove, verify.

    Args:
        problem: The PutnamProblem.
        answer_expr: The Lean expression to try.
        source: Where the answer came from ("reference" or "llm").
        prove_timeout: Timeout for proof search.
        search_budget: Tactic budget.
        verify_timeout: Timeout for verification.

    Returns:
        An AnswerAttempt with the results.
    """
    # Lazy imports to avoid circular dependencies and startup cost
    import asyncio
    from bourbaki.autonomous.search_tree import prove_with_search
    from bourbaki.tools.lean_prover import lean_prover

    logger.info(
        "  Trying answer (%s): %s",
        source,
        answer_expr[:80] + ("..." if len(answer_expr) > 80 else ""),
    )

    # Build the theorem statement with the answer substituted
    parts: list[str] = []
    setup = problem.setup_block
    if problem.answer_name:
        setup = insert_answer(setup, problem.answer_name, answer_expr)
    if problem.preamble:
        parts.append(problem.preamble)
    if setup:
        parts.append(setup)
    parts.append(problem.statement)
    theorem = "\n".join(parts)

    # Try proof search
    proof_tactics: list[str] = []
    theorem_solved = False
    proof_code: str | None = None

    try:
        search_result = await asyncio.wait_for(
            prove_with_search(
                theorem=theorem,
                budget=search_budget,
                timeout=prove_timeout,
                max_depth=15,
                use_mathlib=True,
            ),
            timeout=prove_timeout + 5,
        )

        if search_result.success:
            theorem_solved = True
            proof_tactics = search_result.proof_tactics
            proof_code = search_result.proof_code
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("Proof search failed for answer: %s", e)

    if not theorem_solved:
        return AnswerAttempt(
            answer_code=answer_expr,
            theorem_solved=False,
            verified=False,
            proof_tactics=proof_tactics,
            error="Proof search failed",
            source=source,
        )

    # Whole-file verification
    verified = False
    verify_error: str | None = None

    if proof_code:
        full_code = build_full_proof_code(problem, answer_expr, proof_code)
        try:
            result = await asyncio.wait_for(
                lean_prover(code=full_code, mode="check", timeout=verify_timeout),
                timeout=verify_timeout + 5,
            )
            verified = result.get("proofComplete", False)
            if not verified:
                errors = result.get("errors") or []
                err_msgs = [
                    e.get("message", "") for e in errors if isinstance(e, dict)
                ]
                verify_error = "; ".join(err_msgs) if err_msgs else "Verification failed"
        except (asyncio.TimeoutError, Exception) as e:
            verify_error = f"Verification error: {e}"

    return AnswerAttempt(
        answer_code=answer_expr,
        theorem_solved=theorem_solved,
        verified=verified,
        proof_tactics=proof_tactics,
        error=verify_error,
        source=source,
    )


async def _generate_llm_answers(
    problem: PutnamProblem,
    answer_type: str,
    reference_answer: str | None,
    model: str,
    max_attempts: int,
) -> list[str]:
    """Generate answer candidates using an LLM.

    Uses pydantic-ai Agent for the LLM call, same pattern as other
    autonomous components.

    Args:
        problem: The PutnamProblem.
        answer_type: The Lean type of the answer.
        reference_answer: Optional reference answer hint.
        model: The model string.
        max_attempts: Number of attempts.

    Returns:
        List of candidate answer expressions (may be empty on failure).
    """
    try:
        from pydantic_ai import Agent
        from bourbaki.agent.core import _resolve_model_object
    except ImportError:
        logger.warning("pydantic-ai not available for LLM answer generation")
        return []

    prompt = build_answer_prompt(problem, answer_type, reference_answer)
    resolved_model = _resolve_model_object(model)

    answers: list[str] = []
    seen: set[str] = set()

    for attempt_num in range(max_attempts):
        try:
            agent: Agent[None, str] = Agent(resolved_model)
            result = await agent.run(prompt)
            answer = result.output.strip()

            # Clean up: remove markdown fencing if present
            answer = _clean_llm_answer(answer)

            if answer and answer not in seen:
                seen.add(answer)
                answers.append(answer)
        except Exception as e:
            logger.debug(
                "LLM answer generation attempt %d failed: %s", attempt_num + 1, e,
            )
            continue

    return answers


def _clean_llm_answer(raw: str) -> str:
    """Clean up an LLM-generated answer expression.

    Removes markdown fencing, leading/trailing whitespace, and common
    preamble/suffix artifacts.
    """
    # Remove markdown code blocks
    raw = re.sub(r"^```(?:lean4?|)\s*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```\s*$", "", raw, flags=re.MULTILINE)

    # Remove leading "abbrev ... :=" if the LLM included it
    raw = re.sub(
        r"^(?:noncomputable\s+)?abbrev\s+\w+\s*:.*?:=\s*",
        "",
        raw,
        flags=re.DOTALL,
    )

    return raw.strip()


def _read_source_file(problem: PutnamProblem) -> str | None:
    """Read the source Lean file for a problem."""
    try:
        from pathlib import Path
        path = Path(problem.file_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("Could not read source file %s: %s", problem.file_path, e)
    return None
