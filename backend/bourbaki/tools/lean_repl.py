"""Lean 4 REPL session manager for tactic-by-tactic interaction.

Uses lean4-repl (https://github.com/leanprover-community/repl) for a
persistent subprocess that keeps Mathlib loaded, eliminating 20-30s
import cost per call and enabling incremental proof construction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Resolve paths relative to the project root (parent of backend/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LEAN_PROJECT_DIR = _PROJECT_ROOT / ".bourbaki" / "lean-project"
REPL_BINARY = LEAN_PROJECT_DIR / ".lake" / "repl" / ".lake" / "build" / "bin" / "repl"

# Singleton REPL session (one per backend process)
_active_session: LeanREPLSession | None = None


class LeanREPLSession:
    """Manages a persistent lean4-repl subprocess."""

    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.env_id: int = 0  # Current environment ID for chaining commands
        self._initialized: bool = False
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> None:
        """Start the lean4-repl subprocess."""
        if self.is_running:
            return

        repl_path = _find_repl_binary()
        if repl_path is None:
            raise RuntimeError(
                "lean4-repl not found. Run scripts/setup-lean.sh to build it."
            )

        self.proc = await asyncio.create_subprocess_exec(
            str(repl_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(LEAN_PROJECT_DIR),
        )
        self.env_id = 0
        self._initialized = False
        logger.info("Started lean4-repl (pid=%s)", self.proc.pid)

    async def stop(self) -> None:
        """Stop the REPL subprocess."""
        if self.proc is not None:
            try:
                self.proc.terminate()
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                self.proc.kill()
            logger.info("Stopped lean4-repl")
            self.proc = None
            self._initialized = False
            self.env_id = 0

    async def ensure_initialized(self) -> None:
        """Import Mathlib.Tactic on first use (pays cost once)."""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            result = await self.send_cmd("import Mathlib.Tactic")
            if result.get("env") is not None:
                self.env_id = result["env"]
                self._initialized = True
                logger.info("Mathlib.Tactic loaded in REPL (env=%d)", self.env_id)
            else:
                logger.warning("Failed to load Mathlib.Tactic: %s", result)

    async def send_cmd(self, cmd: str, env: int | None = None) -> dict[str, Any]:
        """Send a command to the REPL and return the parsed JSON response.

        The lean4-repl protocol: send a JSON object per line, read a JSON response.
        Command format: {"cmd": "...", "env": N}
        """
        if not self.is_running:
            await self.start()

        assert self.proc is not None
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        request: dict[str, Any] = {"cmd": cmd}
        if env is not None:
            request["env"] = env
        elif self._initialized:
            request["env"] = self.env_id

        line = json.dumps(request) + "\n"
        self.proc.stdin.write(line.encode())
        await self.proc.stdin.drain()

        # Read response — lean4-repl outputs one JSON object per line
        response_line = await asyncio.wait_for(
            self.proc.stdout.readline(), timeout=120,
        )
        if not response_line:
            return {"error": "REPL returned empty response (process may have crashed)"}

        try:
            return json.loads(response_line.decode())
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON from REPL: {response_line.decode()[:200]}"}

    async def send_tactic(self, tactic: str, proof_state: int) -> dict[str, Any]:
        """Send a tactic to apply to a given proof state.

        Tactic format: {"tactic": "...", "proofState": N}
        """
        if not self.is_running:
            await self.start()
            await self.ensure_initialized()

        assert self.proc is not None
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        request = {"tactic": tactic, "proofState": proof_state}
        line = json.dumps(request) + "\n"
        self.proc.stdin.write(line.encode())
        await self.proc.stdin.drain()

        response_line = await asyncio.wait_for(
            self.proc.stdout.readline(), timeout=60,
        )
        if not response_line:
            return {"error": "REPL returned empty response"}

        try:
            return json.loads(response_line.decode())
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON from REPL: {response_line.decode()[:200]}"}


def _find_repl_binary() -> Path | None:
    """Find the lean4-repl binary."""
    # Primary location: built inside .lake/repl
    if REPL_BINARY.is_file():
        return REPL_BINARY

    # Alternative: repl in PATH or .lake/build/bin
    alt = LEAN_PROJECT_DIR / ".lake" / "build" / "bin" / "repl"
    if alt.is_file():
        return alt

    return None


async def get_session() -> LeanREPLSession:
    """Get or create the singleton REPL session."""
    global _active_session
    if _active_session is None or not _active_session.is_running:
        _active_session = LeanREPLSession()
        await _active_session.start()
        await _active_session.ensure_initialized()
    return _active_session


async def stop_session() -> None:
    """Stop the singleton REPL session."""
    global _active_session
    if _active_session is not None:
        await _active_session.stop()
        _active_session = None


async def lean_tactic(
    goal: str,
    tactic: str,
    proof_state: int | None = None,
) -> dict[str, Any]:
    """Apply a single Lean 4 tactic to a proof state.

    Args:
        goal: The theorem statement (used to initialize on first call, e.g.
              "theorem foo : 1 + 1 = 2 := by sorry").
        tactic: The tactic to apply (e.g. "ring", "intro n", "induction n").
        proof_state: Proof state ID from a previous lean_tactic call. If None,
                     initializes a new proof from the goal statement.

    Returns:
        Dict with success, goals, proofState, proofComplete, duration fields.
    """
    start = time.monotonic()

    repl_available = _find_repl_binary() is not None
    if not repl_available:
        return {
            "success": False,
            "error": "lean4-repl not found. Run scripts/setup-lean.sh to build it. "
                     "Use lean_prover for whole-file verification instead.",
            "duration": int((time.monotonic() - start) * 1000),
        }

    try:
        session = await get_session()

        if proof_state is None:
            # Initialize: send the goal statement with `sorry` to get the proof state
            # The goal should be a theorem/lemma statement ending in `by sorry`
            stmt = goal if "sorry" in goal else goal.rstrip().rstrip(":=") + " := by sorry"
            result = await session.send_cmd(stmt)

            if "env" in result:
                session.env_id = result["env"]

            # Extract proof state from sorries
            sorries = result.get("sorries", [])
            if sorries:
                ps = sorries[0]
                goals = ps.get("goals", []) if isinstance(ps, dict) else []
                ps_id = ps.get("proofState", 0) if isinstance(ps, dict) else 0

                elapsed = int((time.monotonic() - start) * 1000)
                return {
                    "success": True,
                    "goals": goals,
                    "proofState": ps_id,
                    "proofComplete": False,
                    "duration": elapsed,
                }
            elif result.get("messages"):
                # Might have errors
                messages = result["messages"]
                error_msgs = [
                    m.get("data", "") for m in messages
                    if m.get("severity") == "error"
                ]
                if error_msgs:
                    elapsed = int((time.monotonic() - start) * 1000)
                    return {
                        "success": False,
                        "error": "; ".join(error_msgs),
                        "duration": elapsed,
                    }

            # If no sorries and no errors, the proof might be complete already
            elapsed = int((time.monotonic() - start) * 1000)
            return {
                "success": True,
                "goals": [],
                "proofState": 0,
                "proofComplete": True,
                "duration": elapsed,
            }
        else:
            # Apply tactic to existing proof state
            result = await session.send_tactic(tactic, proof_state)

            goals = result.get("goals", [])
            new_ps = result.get("proofState")
            proof_complete = isinstance(goals, list) and len(goals) == 0 and "error" not in result

            if "error" in result:
                elapsed = int((time.monotonic() - start) * 1000)
                return {
                    "success": False,
                    "error": result["error"],
                    "goals": goals if goals else None,
                    "proofState": proof_state,  # Keep the old state for retry
                    "proofComplete": False,
                    "duration": elapsed,
                }

            elapsed = int((time.monotonic() - start) * 1000)
            return {
                "success": True,
                "goals": goals,
                "proofState": new_ps if new_ps is not None else proof_state,
                "proofComplete": proof_complete,
                "duration": elapsed,
            }

    except asyncio.TimeoutError:
        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": "REPL timed out",
            "duration": elapsed,
        }
    except RuntimeError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": str(e),
            "duration": elapsed,
        }
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        logger.exception("lean_tactic error")
        # Reset session on unexpected errors
        await stop_session()
        return {
            "success": False,
            "error": f"REPL error: {e}",
            "duration": elapsed,
        }
