"""Lean 4 error classifier with structured recovery hints.

Parses Lean error messages into categories and provides actionable
recovery suggestions so the agent can self-correct instead of retrying
blindly. Inspired by Goedel-Prover V2's self-correction loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ClassifiedError:
    """A parsed and classified Lean 4 error."""
    category: str
    message: str
    recovery: str
    details: dict[str, str]


# --- Patterns for error classification ---

# "unknown identifier 'Foo.bar'"
_UNKNOWN_ID_RE = re.compile(
    r"unknown (?:identifier|constant)\s+'([^']+)'",
)

# "type mismatch\n  ...\nhas type\n  ...\nbut is expected to have type\n  ..."
_TYPE_MISMATCH_RE = re.compile(
    r"type mismatch",
)
_TYPE_EXPECTED_RE = re.compile(
    r"expected to have type\s+(.+?)(?:\n|$)",
)
_TYPE_ACTUAL_RE = re.compile(
    r"has type\s+(.+?)(?:\nbut|\n|$)",
)

# "tactic 'simp' failed" / "tactic 'ring' failed" / etc.
_TACTIC_FAILED_RE = re.compile(
    r"tactic '(\w+)' failed",
)

# "unsolved goals\n..."
_UNSOLVED_GOALS_RE = re.compile(
    r"unsolved goals",
)

# "expected token" / "unexpected token" / parse errors
_SYNTAX_RE = re.compile(
    r"(?:expected|unexpected) (?:token|command|identifier)|unknown command|parse error",
)

# "unknown tactic" / "unknown attribute"
_UNKNOWN_TACTIC_RE = re.compile(
    r"unknown tactic '(\w+)'",
)

# "maximum recursion depth" / "deterministic timeout"
_TIMEOUT_RE = re.compile(
    r"(?:deterministic timeout|maximum (?:recursion|heartbeat))",
)

# "unknown package" / "unknown module" / import failures
_IMPORT_RE = re.compile(
    r"(?:unknown package|unknown module|could not find module|import)\s+'?([^']*)'?",
)

# "application type mismatch" — wrong number of args to a function
_APP_MISMATCH_RE = re.compile(
    r"application type mismatch",
)

# "don't know how to synthesize placeholder" / "failed to synthesize instance"
_SYNTH_RE = re.compile(
    r"(?:failed to synthesize|don't know how to synthesize)",
)

# Alternative tactic suggestions by goal shape
_TACTIC_ALTERNATIVES: dict[str, list[str]] = {
    "simp": ["simp only [...]", "simp_all", "norm_num", "ring", "omega", "decide"],
    "ring": ["ring_nf", "norm_num", "linarith", "omega"],
    "omega": ["linarith", "norm_num", "decide", "simp [Nat.add_comm]"],
    "linarith": ["omega", "nlinarith", "norm_num", "simp"],
    "norm_num": ["decide", "native_decide", "simp", "ring"],
    "exact": ["apply", "refine", "use"],
    "apply": ["exact", "refine", "have"],
    "aesop": ["simp_all", "tauto", "omega", "decide"],
    "decide": ["native_decide", "norm_num", "simp"],
}


def classify_lean_error(message: str) -> ClassifiedError:
    """Classify a single Lean 4 error message and suggest recovery.

    Args:
        message: The error message text from Lean output.

    Returns:
        ClassifiedError with category, recovery hint, and extracted details.
    """
    # Unknown identifier — most common failure mode (hallucinated lemma names)
    m = _UNKNOWN_ID_RE.search(message)
    if m:
        identifier = m.group(1)
        # Extract the likely namespace prefix for search
        parts = identifier.rsplit(".", 1)
        search_hint = identifier if len(parts) == 1 else parts[-1]
        return ClassifiedError(
            category="unknown_identifier",
            message=message,
            recovery=(
                f"The identifier '{identifier}' does not exist in Mathlib. "
                f"Search for the correct name: use mathlib_search(query='{identifier}', mode='name') "
                f"or mathlib_search(query='{search_hint}', mode='natural') to find alternatives."
            ),
            details={"identifier": identifier},
        )

    # Unknown tactic
    m = _UNKNOWN_TACTIC_RE.search(message)
    if m:
        tactic = m.group(1)
        return ClassifiedError(
            category="unknown_tactic",
            message=message,
            recovery=(
                f"The tactic '{tactic}' is not available. "
                f"Check if a Mathlib import is needed, or use an alternative tactic. "
                f"Common tactics: simp, ring, omega, linarith, norm_num, decide, aesop."
            ),
            details={"tactic": tactic},
        )

    # Tactic failed
    m = _TACTIC_FAILED_RE.search(message)
    if m:
        tactic = m.group(1)
        alternatives = _TACTIC_ALTERNATIVES.get(tactic, ["simp", "omega", "ring", "linarith", "norm_num"])
        alt_str = ", ".join(alternatives[:4])
        return ClassifiedError(
            category="tactic_failed",
            message=message,
            recovery=(
                f"Tactic '{tactic}' failed on this goal. Try alternatives: {alt_str}. "
                f"If the goal involves a specific Mathlib lemma, search for it with "
                f"mathlib_search and use 'exact' or 'apply' instead."
            ),
            details={"tactic": tactic, "alternatives": alt_str},
        )

    # Unsolved goals
    if _UNSOLVED_GOALS_RE.search(message):
        return ClassifiedError(
            category="unsolved_goals",
            message=message,
            recovery=(
                "There are unsolved goals remaining. Use lean_tactic to address each "
                "remaining goal one at a time. Read the goal statement carefully — "
                "it tells you exactly what needs to be proved."
            ),
            details={},
        )

    # Type mismatch
    if _TYPE_MISMATCH_RE.search(message):
        expected = ""
        actual = ""
        m_exp = _TYPE_EXPECTED_RE.search(message)
        m_act = _TYPE_ACTUAL_RE.search(message)
        if m_exp:
            expected = m_exp.group(1).strip()
        if m_act:
            actual = m_act.group(1).strip()
        return ClassifiedError(
            category="type_mismatch",
            message=message,
            recovery=(
                f"Type mismatch: expected '{expected}' but got '{actual}'. "
                f"Check that the lemma/theorem is applied to arguments of the correct type. "
                f"Use mathlib_search with mode='type' to find lemmas with the expected type signature."
            ),
            details={"expected": expected, "actual": actual},
        )

    # Application type mismatch (wrong arg count)
    if _APP_MISMATCH_RE.search(message):
        return ClassifiedError(
            category="application_mismatch",
            message=message,
            recovery=(
                "A function or lemma was applied to the wrong number or type of arguments. "
                "Check the lemma's type signature with mathlib_search(mode='name') "
                "and adjust the arguments accordingly."
            ),
            details={},
        )

    # Synthesis failure
    if _SYNTH_RE.search(message):
        return ClassifiedError(
            category="synthesis_failed",
            message=message,
            recovery=(
                "Lean could not synthesize a required typeclass instance. "
                "This usually means a needed instance or import is missing. "
                "Check if the type has the required algebraic structure "
                "(e.g., CommRing, LinearOrder) and add the appropriate import."
            ),
            details={},
        )

    # Timeout / heartbeat
    if _TIMEOUT_RE.search(message):
        return ClassifiedError(
            category="timeout",
            message=message,
            recovery=(
                "Lean's proof search timed out. The approach may be too complex. "
                "Try breaking the proof into smaller lemmas, using more specific "
                "simp lemmas instead of plain 'simp', or a completely different tactic."
            ),
            details={},
        )

    # Syntax error
    if _SYNTAX_RE.search(message):
        return ClassifiedError(
            category="syntax_error",
            message=message,
            recovery=(
                "Syntax error in the Lean code. Check for: missing commas, "
                "unclosed parentheses/brackets, incorrect keyword usage, "
                "or Lean 3 syntax used in Lean 4 code."
            ),
            details={},
        )

    # Import error
    m = _IMPORT_RE.search(message)
    if m:
        module = m.group(1)
        return ClassifiedError(
            category="import_error",
            message=message,
            recovery=(
                f"Module '{module}' could not be found. Check the import path — "
                f"use mathlib_search to find the correct module name, or use "
                f"'import Mathlib.Tactic' for general tactic access."
            ),
            details={"module": module},
        )

    # Fallback — unrecognized error
    return ClassifiedError(
        category="other",
        message=message,
        recovery=(
            "Read the error message carefully and adjust the proof. "
            "If a lemma name is wrong, search Mathlib. If a tactic failed, "
            "try an alternative approach."
        ),
        details={},
    )


def classify_lean_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify a list of Lean errors (from lean_prover output).

    Args:
        errors: List of error dicts with 'message' and 'severity' fields.

    Returns:
        List of classified error dicts with 'category', 'recovery', and 'details' added.
    """
    classified = []
    for error in errors:
        if error.get("severity") != "error":
            continue
        msg = error.get("message", "")
        result = classify_lean_error(msg)
        classified.append({
            "category": result.category,
            "message": result.message,
            "recovery": result.recovery,
            "details": result.details,
            # Preserve original location info
            "line": error.get("line"),
            "column": error.get("column"),
        })
    return classified


def format_error_guidance(classified_errors: list[dict[str, Any]]) -> str:
    """Format classified errors into a concise guidance string for the agent.

    This is appended to tool results so the LLM gets structured recovery hints
    alongside the raw error output.
    """
    if not classified_errors:
        return ""

    lines = ["\n--- Error Analysis ---"]
    for i, err in enumerate(classified_errors, 1):
        lines.append(f"\nError {i} [{err['category']}]: {err['message'][:150]}")
        lines.append(f"Recovery: {err['recovery']}")

    # Deduplicate categories for a summary
    categories = list(dict.fromkeys(e["category"] for e in classified_errors))
    if "unknown_identifier" in categories:
        lines.append("\nCRITICAL: Do NOT guess lemma names. Use mathlib_search to find the correct name before retrying.")

    return "\n".join(lines)
