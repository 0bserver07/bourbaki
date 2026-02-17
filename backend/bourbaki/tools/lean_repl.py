"""Lean 4 REPL session manager for tactic-by-tactic interaction.

Uses lean4-repl (https://github.com/leanprover-community/repl) for a
persistent subprocess that keeps Mathlib loaded, eliminating ~90s
import cost per call and enabling incremental proof construction.

Protocol notes (from lean4-repl README):
- Commands are JSON objects separated by BLANK LINES (\\n\\n)
- Responses are multi-line JSON separated by blank lines
- Must run via ``lake env <repl-binary>`` for proper Lean environment
- Imports can only be used without an ``env`` field
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
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
    """Manages a persistent lean4-repl subprocess.

    The REPL must be started via ``lake env <repl>`` so that the Lake
    project environment (and pre-built Mathlib oleans) are available.
    Commands are separated by blank lines; responses are multi-line JSON
    also separated by blank lines.
    """

    _STDERR_BUFFER_SIZE = 20

    def __init__(self, import_full_mathlib: bool = False) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.env_id: int = 0  # Current environment ID for chaining commands
        self._initialized: bool = False
        self._lock = asyncio.Lock()
        self._import_full_mathlib = import_full_mathlib
        self._stderr_buffer: list[str] = []
        self._stderr_task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> None:
        """Start the lean4-repl subprocess via ``lake env``."""
        if self.is_running:
            return

        repl_path = _find_repl_binary()
        if repl_path is None:
            raise RuntimeError(
                "lean4-repl not found. Run scripts/setup-lean.sh to build it."
            )

        lake_bin = shutil.which("lake")
        if lake_bin is None:
            raise RuntimeError("lake not found in PATH")

        # Must run via `lake env` so Lean can resolve Mathlib imports
        self.proc = await asyncio.create_subprocess_exec(
            lake_bin, "env", str(repl_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(LEAN_PROJECT_DIR),
        )
        self.env_id = 0
        self._initialized = False
        self._stderr_buffer.clear()
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        logger.info("Started lean4-repl via lake env (pid=%s)", self.proc.pid)

    async def _drain_stderr(self) -> None:
        """Background task: read stderr and buffer recent lines."""
        assert self.proc is not None and self.proc.stderr is not None
        try:
            while True:
                raw = await self.proc.stderr.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip()
                if line:
                    logger.debug("lean4-repl stderr: %s", line)
                    self._stderr_buffer.append(line)
                    if len(self._stderr_buffer) > self._STDERR_BUFFER_SIZE:
                        self._stderr_buffer.pop(0)
        except (asyncio.CancelledError, Exception):
            pass

    def get_stderr_recent(self) -> list[str]:
        """Return the most recent stderr lines."""
        return list(self._stderr_buffer)

    async def stop(self) -> None:
        """Stop the REPL subprocess."""
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
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
        """Import Mathlib on first use (pays cost once, ~16-37s)."""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            import_cmd = "import Mathlib" if self._import_full_mathlib else "import Mathlib.Tactic"
            result = await self.send_cmd(import_cmd, env=None, timeout=300)
            if result.get("env") is not None:
                self.env_id = result["env"]
                self._initialized = True
                logger.info("%s loaded in REPL (env=%d)", import_cmd, self.env_id)
            else:
                logger.warning("Failed to load %s: %s", import_cmd, result)

    async def _read_response(self, timeout: float = 120) -> dict[str, Any]:
        """Read a multi-line JSON response from the REPL.

        The lean4-repl outputs pretty-printed JSON separated by blank lines.
        We read lines until we hit a blank line after accumulating content,
        then parse the concatenated result.
        """
        assert self.proc is not None
        assert self.proc.stdout is not None

        lines: list[str] = []

        async def _read_until_blank() -> None:
            while True:
                raw = await self.proc.stdout.readline()  # type: ignore[union-attr]
                if not raw:
                    # EOF — process may have crashed
                    if not lines:
                        lines.append('{"error": "REPL process ended unexpectedly"}')
                    return
                text = raw.decode()
                if text.strip() == "":
                    if lines:
                        return  # End of this response
                    continue  # Skip leading blank lines
                lines.append(text)

        await asyncio.wait_for(_read_until_blank(), timeout=timeout)

        joined = "".join(lines)
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON from REPL: {joined[:300]}"}

    async def send_cmd(
        self, cmd: str, env: int | None = None, timeout: float = 120,
    ) -> dict[str, Any]:
        """Send a command to the REPL and return the parsed JSON response.

        Commands are JSON objects followed by a blank line (the REPL protocol
        uses blank lines as delimiters between commands/responses).
        """
        if not self.is_running:
            await self.start()

        assert self.proc is not None
        assert self.proc.stdin is not None

        request: dict[str, Any] = {"cmd": cmd}
        if env is not None:
            request["env"] = env
        elif self._initialized:
            request["env"] = self.env_id

        # Protocol: JSON + blank line separator
        payload = json.dumps(request) + "\n\n"
        self.proc.stdin.write(payload.encode())
        await self.proc.stdin.drain()

        return await self._read_response(timeout=timeout)

    async def send_tactic(
        self, tactic: str, proof_state: int, timeout: float = 60,
    ) -> dict[str, Any]:
        """Send a tactic to apply to a given proof state.

        Tactic format: {"tactic": "...", "proofState": N}
        """
        if not self.is_running:
            await self.start()
            await self.ensure_initialized()

        assert self.proc is not None
        assert self.proc.stdin is not None

        request = {"tactic": tactic, "proofState": proof_state}
        payload = json.dumps(request) + "\n\n"
        self.proc.stdin.write(payload.encode())
        await self.proc.stdin.drain()

        return await self._read_response(timeout=timeout)


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


async def get_session(full_mathlib: bool = False) -> LeanREPLSession:
    """Get or create the singleton REPL session.

    Args:
        full_mathlib: If True, import full Mathlib (slower init, needed for
                      problems using Mathlib types). Default imports only
                      Mathlib.Tactic (faster, sufficient for most proofs).
    """
    global _active_session
    if _active_session is None or not _active_session.is_running:
        _active_session = LeanREPLSession(import_full_mathlib=full_mathlib)
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
    session: LeanREPLSession | None = None,
) -> dict[str, Any]:
    """Apply a single Lean 4 tactic to a proof state.

    Args:
        goal: The theorem statement (used to initialize on first call, e.g.
              "theorem foo : 1 + 1 = 2 := by sorry").
        tactic: The tactic to apply (e.g. "ring", "intro n", "induction n").
        proof_state: Proof state ID from a previous lean_tactic call. If None,
                     initializes a new proof from the goal statement.
        session: Optional pre-initialized REPL session. If None, uses the
                 singleton session.

    Returns:
        Dict with success, goals, proofState, proofComplete, duration fields.
    """
    start = time.monotonic()

    repl_available = _find_repl_binary() is not None
    if not repl_available and session is None:
        return {
            "success": False,
            "error": "lean4-repl not found. Run scripts/setup-lean.sh to build it. "
                     "Use lean_prover for whole-file verification instead.",
            "duration": int((time.monotonic() - start) * 1000),
        }

    try:
        if session is None:
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
                # lean4-repl uses "goal" (singular string) in sorry responses,
                # but "goals" (list) in tactic responses
                if isinstance(ps, dict):
                    if "goals" in ps:
                        goals = ps["goals"]
                    elif "goal" in ps:
                        goals = [ps["goal"]] if ps["goal"] else []
                    else:
                        goals = []
                    ps_id = ps.get("proofState", 0)
                else:
                    goals = []
                    ps_id = 0

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
            # lean4-repl signals completion via proofStatus or empty goals
            proof_complete = (
                result.get("proofStatus") == "Completed"
                or (isinstance(goals, list) and len(goals) == 0
                    and "error" not in result and "message" not in result)
            )

            if "error" in result or ("message" in result and not proof_complete):
                error_msg = result.get("error") or result.get("message", "Unknown error")
                elapsed = int((time.monotonic() - start) * 1000)
                return {
                    "success": False,
                    "error": error_msg,
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
