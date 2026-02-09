"""Lean 4 proof verification via asyncio subprocess."""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from pathlib import Path
from typing import Any


MATHLIB_IMPORTS = "import Mathlib\nimport Mathlib.Tactic\n"
LEAN_TEMP_DIR = Path(".bourbaki/lean-temp")

# Regex for Lean error output: filename:line:col: severity: message
ERROR_RE = re.compile(r"^(.+?):(\d+):(\d+):\s+(error|warning|info):\s+(.+)$", re.MULTILINE)
GOAL_RE = re.compile(r"⊢\s+(.+)")


async def lean_prover(
    code: str,
    mode: str = "check",
    timeout: int = 30,
) -> dict[str, Any]:
    """Verify Lean 4 code.

    Args:
        code: Lean 4 code (theorem, tactic proof, or expression).
        mode: One of 'check', 'elaborate', 'tactic'.
        timeout: Timeout in seconds (default 30).

    Returns:
        Dict with success, goals, proofComplete, errors, rawOutput, codeUsed.
    """
    start = time.monotonic()

    # Only add Mathlib imports if the code explicitly uses Mathlib features
    # (e.g. references Mathlib tactics, types, or theorems).
    # Skip for pure Lean 4 code that doesn't need Mathlib.
    if "import Mathlib" not in code and "import " not in code:
        # No imports at all — try without Mathlib first (faster).
        # If it fails, we could retry with Mathlib, but for now keep it simple.
        pass

    # Write to temp file
    LEAN_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"bourbaki_{secrets.token_hex(8)}.lean"
    filepath = LEAN_TEMP_DIR / filename

    try:
        filepath.write_text(code, encoding="utf-8")

        # Try `lean` first, then `lake env lean`
        raw_output, return_code = await _run_lean(filepath, timeout)
        if raw_output is None:
            # lean not found, try lake
            raw_output, return_code = await _run_lake_lean(filepath, timeout)

        if raw_output is None:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return {
                "success": False,
                "error": "Lean 4 not found. Install from https://leanprover.github.io/",
                "rawOutput": "",
                "codeUsed": code,
                "duration": elapsed_ms,
            }

        # Parse output
        errors = _parse_errors(raw_output, filename)
        goals = _parse_goals(raw_output)
        has_errors = any(e["severity"] == "error" for e in errors)
        has_sorry = "sorry" in code
        proof_complete = return_code == 0 and not has_errors and not has_sorry

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {
            "success": not has_errors,
            "goals": goals,
            "proofComplete": proof_complete,
            "errors": errors if errors else None,
            "rawOutput": raw_output,
            "codeUsed": code,
            "duration": elapsed_ms,
        }
    finally:
        filepath.unlink(missing_ok=True)


async def _run_lean(filepath: Path, timeout: int) -> tuple[str | None, int]:
    """Run `lean filepath` and return (output, returncode) or (None, -1) if not found."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lean", str(filepath),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = (stdout or b"").decode() + (stderr or b"").decode()
        return output, proc.returncode or 0
    except FileNotFoundError:
        return None, -1
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[union-attr]
        return "Lean timed out", 1


async def _run_lake_lean(filepath: Path, timeout: int) -> tuple[str | None, int]:
    """Fallback: run `lake env lean filepath`."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lake", "env", "lean", str(filepath),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = (stdout or b"").decode() + (stderr or b"").decode()
        return output, proc.returncode or 0
    except (FileNotFoundError, asyncio.TimeoutError):
        return None, -1


def _parse_errors(output: str, filename: str) -> list[dict[str, Any]]:
    """Parse Lean error output into structured error list."""
    errors = []
    for match in ERROR_RE.finditer(output):
        errors.append({
            "line": int(match.group(2)),
            "column": int(match.group(3)),
            "message": match.group(5).strip(),
            "severity": match.group(4),
        })
    return errors


def _parse_goals(output: str) -> list[str]:
    """Extract goal states from Lean output."""
    return GOAL_RE.findall(output)
