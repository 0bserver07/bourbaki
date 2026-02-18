"""High-level tool functions built on the Lean LSP session.

These wrap :class:`LeanLSPSession` into simple async functions suitable for
use by the agent, the proof search tree, or any other consumer that needs
intelligent Lean assistance (diagnostics, completions, hover, goal state).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from bourbaki.tools.lean_lsp import LeanLSPSession, get_lsp_session

logger = logging.getLogger(__name__)

# Lean 4 Mathlib tactic import — prepended when the user code has no imports
_DEFAULT_HEADER = "import Mathlib.Tactic\n\n"


def _ensure_imports(code: str) -> str:
    """Prepend a Mathlib import if the code has none."""
    if "import " in code:
        return code
    return _DEFAULT_HEADER + code


def _severity_label(sev: int) -> str:
    """Map LSP DiagnosticSeverity int to a human label."""
    return {1: "error", 2: "warning", 3: "info", 4: "hint"}.get(sev, "unknown")


def _format_diagnostic(diag: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw LSP diagnostic into a flat dict."""
    rng = diag.get("range", {})
    start = rng.get("start", {})
    return {
        "line": start.get("line", 0),
        "column": start.get("character", 0),
        "severity": _severity_label(diag.get("severity", 4)),
        "message": diag.get("message", ""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def lsp_check(
    code: str,
    timeout: float = 60.0,
    session: LeanLSPSession | None = None,
) -> dict[str, Any]:
    """Check Lean code via the LSP and return structured diagnostics.

    Opens a scratch file, waits for diagnostics, then closes the file.

    Returns a dict with:
      - ``success`` (bool): True when there are no errors.
      - ``errors``, ``warnings``, ``infos``: lists of diagnostic dicts.
      - ``all_diagnostics``: the full list.
      - ``duration`` (int): milliseconds elapsed.
    """
    start = time.monotonic()

    if session is None:
        session = await get_lsp_session()

    code = _ensure_imports(code)
    uri = session.make_uri()

    try:
        await session.open_file(uri, code)
        raw_diags = await session.get_diagnostics(uri, timeout=timeout)
    finally:
        try:
            await session.close_file(uri)
        except Exception:
            pass

    formatted = [_format_diagnostic(d) for d in raw_diags]
    errors = [d for d in formatted if d["severity"] == "error"]
    warnings = [d for d in formatted if d["severity"] == "warning"]
    infos = [d for d in formatted if d["severity"] == "info"]

    elapsed = int((time.monotonic() - start) * 1000)
    return {
        "success": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "all_diagnostics": formatted,
        "duration": elapsed,
    }


async def lsp_completions(
    code: str,
    line: int,
    col: int,
    session: LeanLSPSession | None = None,
) -> list[str]:
    """Get tactic/term completions at a position in Lean code.

    Opens a scratch file, requests completions at (*line*, *col*), then
    returns a list of completion labels (strings).

    Args:
        code: Full Lean source (imports will be auto-prepended if missing).
        line: 0-indexed line number in the *original* (user-supplied) code.
              If imports are prepended, the offset is adjusted automatically.
        col:  0-indexed column number.
        session: Optional pre-existing LSP session.

    Returns:
        A list of completion label strings.
    """
    if session is None:
        session = await get_lsp_session()

    original_code = code
    code = _ensure_imports(code)

    # Adjust line offset if we prepended an import header
    offset = 0
    if code != original_code:
        offset = code.count("\n", 0, code.index(original_code))
    adjusted_line = line + offset

    uri = session.make_uri()

    try:
        await session.open_file(uri, code)
        # Give the server a moment to elaborate (completions may fail
        # if requested before elaboration starts)
        items = await session.get_completions(uri, adjusted_line, col)
    finally:
        try:
            await session.close_file(uri)
        except Exception:
            pass

    return [item.get("label", "") for item in items if item.get("label")]


async def lsp_goal(
    code: str,
    line: int,
    col: int,
    session: LeanLSPSession | None = None,
) -> str | None:
    """Get the proof goal at a position in Lean code.

    Uses Lean's ``$/lean/plainGoal`` LSP extension.

    Args:
        code: Full Lean source.
        line: 0-indexed line in the original code.
        col:  0-indexed column.
        session: Optional LSP session.

    Returns:
        The goal string, or ``None`` if no goal at that position.
    """
    if session is None:
        session = await get_lsp_session()

    original_code = code
    code = _ensure_imports(code)

    offset = 0
    if code != original_code:
        offset = code.count("\n", 0, code.index(original_code))
    adjusted_line = line + offset

    uri = session.make_uri()

    try:
        await session.open_file(uri, code)
        # Wait briefly for elaboration so goal state is available
        await session.get_diagnostics(uri, timeout=30)
        goals = await session.get_goal(uri, adjusted_line, col)
    finally:
        try:
            await session.close_file(uri)
        except Exception:
            pass

    if not goals:
        return None

    return "\n".join(goals)


async def lsp_hover(
    code: str,
    line: int,
    col: int,
    session: LeanLSPSession | None = None,
) -> str | None:
    """Get type / hover info at a position in Lean code.

    Args:
        code: Full Lean source.
        line: 0-indexed line in the original code.
        col:  0-indexed column.
        session: Optional LSP session.

    Returns:
        The hover info string, or ``None``.
    """
    if session is None:
        session = await get_lsp_session()

    original_code = code
    code = _ensure_imports(code)

    offset = 0
    if code != original_code:
        offset = code.count("\n", 0, code.index(original_code))
    adjusted_line = line + offset

    uri = session.make_uri()

    try:
        await session.open_file(uri, code)
        await session.get_diagnostics(uri, timeout=30)
        info = await session.get_hover(uri, adjusted_line, col)
    finally:
        try:
            await session.close_file(uri)
        except Exception:
            pass

    return info


async def lsp_suggest_tactics(
    theorem: str,
    session: LeanLSPSession | None = None,
) -> list[str]:
    """Get LSP tactic suggestions for a theorem with ``sorry``.

    Constructs a Lean file with the theorem (appending ``by sorry`` if
    needed), then queries completions at the ``sorry`` position.

    Args:
        theorem: A Lean theorem statement, e.g.
                 ``theorem foo : 1 + 1 = 2 := by sorry``
        session: Optional LSP session.

    Returns:
        A list of tactic labels from the LSP completion engine.
    """
    if session is None:
        session = await get_lsp_session()

    # Normalise: ensure the theorem ends with ``by sorry``
    stmt = theorem.strip()
    if "sorry" not in stmt:
        stmt = re.sub(r"\s*:=\s*$", "", stmt)
        stmt = stmt.rstrip()
        if not stmt.endswith("by"):
            stmt += " := by"
        stmt += "\n  sorry"

    code = _ensure_imports(stmt)

    # Find the position of ``sorry`` — completions just *before* it
    # give us tactic suggestions
    lines = code.split("\n")
    sorry_line: int | None = None
    sorry_col: int | None = None
    for i, ln in enumerate(lines):
        idx = ln.find("sorry")
        if idx != -1:
            sorry_line = i
            sorry_col = idx
            break

    if sorry_line is None or sorry_col is None:
        return []

    uri = session.make_uri()

    try:
        await session.open_file(uri, code)
        # Wait for elaboration to finish (diagnostics include the sorry warning)
        await session.get_diagnostics(uri, timeout=45)
        items = await session.get_completions(uri, sorry_line, sorry_col)
    finally:
        try:
            await session.close_file(uri)
        except Exception:
            pass

    return [item.get("label", "") for item in items if item.get("label")]
