"""Lean 4 proof verification via asyncio subprocess."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

MATHLIB_IMPORTS = "import Mathlib\nimport Mathlib.Tactic\n"

# Resolve .bourbaki paths relative to the project root (parent of backend/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LEAN_TEMP_DIR = _PROJECT_ROOT / ".bourbaki" / "lean-temp"
LEAN_PROJECT_DIR = _PROJECT_ROOT / ".bourbaki" / "lean-project"

# Regex for Lean error output: filename:line:col: severity: message
ERROR_RE = re.compile(r"^(.+?):(\d+):(\d+):\s+(error|warning|info):\s+(.+)$", re.MULTILINE)
GOAL_RE = re.compile(r"⊢\s+(.+)")

# Cached Lean capabilities (detected once at startup)
_lean_capabilities: dict[str, Any] | None = None


def detect_lean_capabilities() -> dict[str, Any]:
    """Detect Lean 4 installation and Mathlib availability (cached).

    Returns dict with:
        installed: bool
        version: str | None
        mathlib: bool
    """
    global _lean_capabilities
    if _lean_capabilities is not None:
        return _lean_capabilities

    caps: dict[str, Any] = {"installed": False, "version": None, "mathlib": False}

    lean_bin = shutil.which("lean")
    if not lean_bin:
        _lean_capabilities = caps
        return caps

    # Get version
    try:
        result = subprocess.run(
            ["lean", "--version"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            caps["installed"] = True
            # Parse "Lean (version 4.22.0, ...)"
            m = re.search(r"version\s+([\d.]+)", result.stdout)
            caps["version"] = m.group(1) if m else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        _lean_capabilities = caps
        return caps

    # Detect Mathlib — check if lean-project exists and try importing Mathlib
    if LEAN_PROJECT_DIR.is_dir() and (LEAN_PROJECT_DIR / "lean-toolchain").exists():
        test_file = LEAN_PROJECT_DIR / f"_mathlib_check_{secrets.token_hex(4)}.lean"
        try:
            test_file.write_text("import Mathlib.Tactic\n#check Nat.add_comm\n", encoding="utf-8")
            result = subprocess.run(
                ["lake", "env", "lean", str(test_file)],
                capture_output=True, text=True, timeout=120,
                cwd=str(LEAN_PROJECT_DIR),
            )
            caps["mathlib"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            caps["mathlib"] = False
        finally:
            test_file.unlink(missing_ok=True)
    else:
        caps["mathlib"] = False

    logger.info("Lean capabilities: %s", caps)
    _lean_capabilities = caps
    return caps


def get_lean_prompt_section() -> str:
    """Return a system prompt section describing available Lean 4 capabilities."""
    caps = detect_lean_capabilities()

    if not caps["installed"]:
        return """## Lean 4 Environment

Lean 4 is NOT installed. Do not attempt to use the lean_prover tool.
Write informal proofs only."""

    if caps["mathlib"]:
        return f"""## Lean 4 Environment

Lean 4 v{caps['version']} is installed WITH Mathlib.

Available tactics include: simp, norm_num, ring, omega, linarith, nlinarith,
field_simp, push_neg, contrapose, decide, native_decide, ext, funext,
apply, exact, intro, cases, induction, rw, rfl, constructor, use,
have, let, show, calc, conv, gcongr, positivity, polyrith, aesop.

Available Mathlib imports: Mathlib.Tactic, Mathlib.Data.*, Mathlib.Algebra.*,
Mathlib.Analysis.*, Mathlib.Topology.*, Mathlib.NumberTheory.*, etc.

When writing Lean proofs:
- Start with `import Mathlib` or specific module imports
- Use `import Mathlib.Tactic` for tactic access without importing everything
- The lean_prover tool allows multiple attempts — read error output carefully and fix issues iteratively"""

    return f"""## Lean 4 Environment

Lean 4 v{caps['version']} is installed WITHOUT Mathlib (vanilla Lean 4 only).

Available tactics (built-in only): simp, decide, native_decide, rfl,
apply, exact, intro, cases, induction, rw, constructor, use,
have, let, show, calc, omega, trivial, assumption, contradiction.

DO NOT use Mathlib tactics: norm_num, ring, linarith, nlinarith, field_simp,
push_neg, polyrith, aesop, positivity, gcongr — these will fail.

DO NOT write `import Mathlib` — it is not installed.

When writing Lean proofs:
- Use only core Lean 4 tactics listed above
- For arithmetic equalities, prefer `native_decide` or `decide` over `norm_num`
- For simple equalities, `rfl` often works
- The lean_prover tool allows multiple attempts — read error output carefully and fix issues iteratively"""


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

    caps = detect_lean_capabilities()
    use_mathlib = caps.get("mathlib", False)

    # Auto-prepend Mathlib import when Mathlib is available and code
    # doesn't already have its own imports
    if use_mathlib and "import " not in code:
        code = "import Mathlib.Tactic\n\n" + code

    # Write to temp file — use the Lake project dir when Mathlib is available
    # so `lake env lean` can resolve Mathlib imports
    if use_mathlib and LEAN_PROJECT_DIR.is_dir():
        temp_dir = LEAN_PROJECT_DIR
    else:
        temp_dir = LEAN_TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)
    filename = f"bourbaki_{secrets.token_hex(8)}.lean"
    filepath = temp_dir / filename

    try:
        filepath.write_text(code, encoding="utf-8")

        if use_mathlib and LEAN_PROJECT_DIR.is_dir():
            # Run via lake env lean from the Mathlib project directory
            raw_output, return_code = await _run_lake_lean(
                filepath, timeout, cwd=LEAN_PROJECT_DIR,
            )
        else:
            # Try bare `lean` first, then `lake env lean`
            raw_output, return_code = await _run_lean(filepath, timeout)
            if raw_output is None:
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


async def _run_lake_lean(filepath: Path, timeout: int, cwd: Path | None = None) -> tuple[str | None, int]:
    """Run `lake env lean filepath`, optionally within a Lake project directory."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "lake", "env", "lean", str(filepath),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
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
