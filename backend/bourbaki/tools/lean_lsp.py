"""Lean 4 Language Server Protocol (LSP) session manager.

Communicates directly with ``lean --server`` (or ``lake env lean --server``)
via JSON-RPC 2.0 over stdin/stdout using standard LSP framing
(``Content-Length: N\\r\\n\\r\\n{json}``).

This complements the REPL (tactic-by-tactic proving) and the one-shot prover
(whole-file verification) by providing *intelligent assistance*:

- Diagnostics (errors, warnings, info) after elaboration
- Completions at a cursor position (including tactic completions at ``sorry``)
- Hover / type info at a position
- Goal state at a position via Lean's ``$/lean/plainGoal`` extension

Architecture notes:

- The LSP subprocess is heavier than the REPL because each ``didOpen`` triggers
  full elaboration with Mathlib.  Sessions are lazily initialized and reused.
- Diagnostics arrive as *notifications* from the server, not as responses.
  A background reader task collects them into per-URI buffers.
- Virtual file URIs of the form ``file:///tmp/bourbaki-lsp/scratch_N.lean``
  are used; no actual files are written to disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Re-use the same Lean project directory as the REPL / prover
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LEAN_PROJECT_DIR = _PROJECT_ROOT / ".bourbaki" / "lean-project"

# Virtual URI prefix for scratch files
_LSP_SCRATCH_PREFIX = "file:///tmp/bourbaki-lsp"

# Singleton session
_active_lsp_session: LeanLSPSession | None = None


# ---------------------------------------------------------------------------
# JSON-RPC framing helpers
# ---------------------------------------------------------------------------

def encode_lsp_message(obj: dict[str, Any]) -> bytes:
    """Encode a JSON-RPC object into an LSP wire-format message.

    Format: ``Content-Length: <len>\\r\\n\\r\\n<json-utf8>``
    """
    body = json.dumps(obj).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


async def read_lsp_message(reader: asyncio.StreamReader) -> dict[str, Any]:
    """Read one LSP message from *reader*.

    Parses the ``Content-Length`` header, then reads exactly that many bytes
    of JSON body.  Raises ``EOFError`` when the stream ends.
    """
    # Read headers until blank line (\r\n\r\n)
    content_length: int | None = None
    while True:
        raw_line = await reader.readline()
        if not raw_line:
            raise EOFError("LSP stream ended (no more headers)")
        line = raw_line.decode("ascii", errors="replace").strip()
        if line == "":
            # End of headers
            break
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())

    if content_length is None:
        raise ValueError("LSP message missing Content-Length header")

    body = await reader.readexactly(content_length)
    return json.loads(body)


# ---------------------------------------------------------------------------
# LeanLSPSession
# ---------------------------------------------------------------------------

class LeanLSPSession:
    """Manages a persistent ``lean --server`` subprocess speaking LSP.

    Lifecycle:
    1. ``await session.start()``        — spawns subprocess
    2. ``await session.initialize()``   — LSP handshake
    3. Use ``open_file`` / ``update_file`` / ``get_*`` methods
    4. ``await session.shutdown()``     — clean exit
    """

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_id: int = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}  # uri -> diagnostics list
        self._diag_events: dict[str, asyncio.Event] = {}  # uri -> "diagnostics arrived" flag
        self._initialized: bool = False
        self._lock = asyncio.Lock()
        self._file_counter: int = 0
        self._file_versions: dict[str, int] = {}  # uri -> version

    # ---- Properties ----

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    # ---- Lifecycle ----

    async def start(self) -> None:
        """Spawn the ``lean --server`` subprocess."""
        if self.is_running:
            return

        lake_bin = shutil.which("lake")
        if lake_bin is None:
            raise RuntimeError("lake not found in PATH — cannot start Lean LSP server")

        if not LEAN_PROJECT_DIR.is_dir():
            raise RuntimeError(
                f"Lean project dir not found at {LEAN_PROJECT_DIR}. "
                "Run scripts/setup-lean.sh to create it."
            )

        self._proc = await asyncio.create_subprocess_exec(
            lake_bin, "env", "lean", "--server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(LEAN_PROJECT_DIR),
            limit=4 * 1024 * 1024,
        )
        logger.info("Started lean --server (pid=%s)", self._proc.pid)

        # Background task: continuously read messages from stdout
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def initialize(self) -> dict[str, Any]:
        """Perform the LSP initialize / initialized handshake.

        Returns the server capabilities.
        """
        if self._initialized:
            return {}

        async with self._lock:
            if self._initialized:
                return {}

            if not self.is_running:
                await self.start()

            result = await self._request("initialize", {
                "processId": None,
                "rootUri": f"file://{LEAN_PROJECT_DIR}",
                "capabilities": {
                    "textDocument": {
                        "completion": {
                            "completionItem": {
                                "snippetSupport": False,
                            },
                        },
                        "hover": {"contentFormat": ["plaintext", "markdown"]},
                        "publishDiagnostics": {},
                    },
                },
                "initializationOptions": {},
            })

            # Send the mandatory "initialized" notification
            await self._notify("initialized", {})
            self._initialized = True
            logger.info("LSP initialized — server capabilities received")
            return result

    async def shutdown(self) -> None:
        """Cleanly shut down the LSP server and subprocess."""
        if not self.is_running:
            return

        try:
            # LSP shutdown request
            await self._request("shutdown", None, timeout=10)
            # LSP exit notification
            await self._notify("exit", None)
        except Exception:
            pass

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        if self._proc is not None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._proc.kill()
            self._proc = None

        # Reset state
        self._initialized = False
        self._pending.clear()
        self._diagnostics.clear()
        self._diag_events.clear()
        self._file_versions.clear()
        logger.info("LSP session shut down")

    # ---- File operations ----

    def make_uri(self, tag: str | None = None) -> str:
        """Generate a unique virtual file URI."""
        self._file_counter += 1
        name = tag or f"scratch_{self._file_counter}"
        return f"{_LSP_SCRATCH_PREFIX}/{name}.lean"

    async def open_file(self, uri: str, content: str) -> None:
        """Send ``textDocument/didOpen`` and reset diagnostics for *uri*."""
        self._diagnostics[uri] = []
        self._diag_events[uri] = asyncio.Event()
        self._file_versions[uri] = 1

        await self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "lean4",
                "version": 1,
                "text": content,
            },
        })

    async def update_file(self, uri: str, content: str) -> None:
        """Send ``textDocument/didChange`` with full-document sync."""
        version = self._file_versions.get(uri, 0) + 1
        self._file_versions[uri] = version
        # Reset diagnostics — new ones will arrive
        self._diagnostics[uri] = []
        self._diag_events[uri] = asyncio.Event()

        await self._notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": content}],
        })

    async def close_file(self, uri: str) -> None:
        """Send ``textDocument/didClose``."""
        await self._notify("textDocument/didClose", {
            "textDocument": {"uri": uri},
        })
        self._diagnostics.pop(uri, None)
        self._diag_events.pop(uri, None)
        self._file_versions.pop(uri, None)

    # ---- Query methods ----

    async def get_diagnostics(self, uri: str, timeout: float = 60.0) -> list[dict[str, Any]]:
        """Wait for diagnostics on *uri* and return them.

        Diagnostics arrive asynchronously via ``textDocument/publishDiagnostics``
        notifications.  This method waits up to *timeout* seconds for at least
        one batch to arrive after the most recent ``didOpen``/``didChange``.
        """
        event = self._diag_events.get(uri)
        if event is None:
            return []

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug("Timed out waiting for diagnostics on %s", uri)

        return list(self._diagnostics.get(uri, []))

    async def get_completions(
        self, uri: str, line: int, col: int,
    ) -> list[dict[str, Any]]:
        """Request completions at (*line*, *col*) in *uri*.

        Returns a list of completion items (each with at least ``label``).
        Lines and columns are 0-indexed.
        """
        await self._ensure_ready()
        result = await self._request("textDocument/completion", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": col},
        })

        # The result can be a CompletionList or a plain list
        if isinstance(result, dict):
            items = result.get("items", [])
        elif isinstance(result, list):
            items = result
        else:
            items = []

        return items

    async def get_hover(
        self, uri: str, line: int, col: int,
    ) -> str | None:
        """Request hover info at (*line*, *col*) in *uri*.

        Returns the hover content as a string, or ``None``.
        """
        await self._ensure_ready()
        result = await self._request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": col},
        })

        if result is None:
            return None

        contents = result.get("contents")
        if contents is None:
            return None

        # MarkupContent: {"kind": "...", "value": "..."}
        if isinstance(contents, dict):
            return contents.get("value")
        # Plain string
        if isinstance(contents, str):
            return contents
        # Array of MarkedString
        if isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("value", ""))
            return "\n".join(parts)

        return None

    async def get_goal(
        self, uri: str, line: int, col: int,
    ) -> list[str]:
        """Request the proof goal at (*line*, *col*) via ``$/lean/plainGoal``.

        This is a Lean-specific LSP extension.  Returns a list of goal strings,
        or an empty list if no goal is available.
        """
        await self._ensure_ready()
        try:
            result = await self._request("$/lean/plainGoal", {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": col},
            })
        except Exception:
            return []

        if result is None:
            return []

        goals = result.get("goals", [])
        rendered = result.get("rendered")
        if rendered and not goals:
            return [rendered]
        return goals

    # ---- JSON-RPC transport ----

    async def _ensure_ready(self) -> None:
        """Ensure the LSP is started and initialized."""
        if not self.is_running:
            await self.start()
        if not self._initialized:
            await self.initialize()

    async def _request(
        self,
        method: str,
        params: Any,
        timeout: float = 120,
    ) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        if not self.is_running:
            raise RuntimeError("LSP process is not running")

        msg_id = self._next_id
        self._next_id += 1

        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = future

        await self._send(msg)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise asyncio.TimeoutError(
                f"LSP request {method} (id={msg_id}) timed out after {timeout}s"
            )

        if "error" in response:
            err = response["error"]
            code = err.get("code", -1)
            message = err.get("message", "Unknown LSP error")
            raise RuntimeError(f"LSP error {code}: {message}")

        return response.get("result")

    async def _notify(self, method: str, params: Any) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.is_running:
            raise RuntimeError("LSP process is not running")

        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        await self._send(msg)

    async def _send(self, msg: dict[str, Any]) -> None:
        """Write an LSP message to the subprocess stdin."""
        assert self._proc is not None and self._proc.stdin is not None
        data = encode_lsp_message(msg)
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def _reader_loop(self) -> None:
        """Background task: read LSP messages from stdout and dispatch."""
        assert self._proc is not None and self._proc.stdout is not None
        reader = self._proc.stdout

        try:
            while True:
                try:
                    msg = await read_lsp_message(reader)
                except EOFError:
                    logger.debug("LSP reader: stream ended")
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    logger.warning("LSP reader: malformed message: %s", exc)
                    continue

                self._dispatch(msg)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("LSP reader loop crashed")

    def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route an incoming LSP message to the right handler."""
        msg_id = msg.get("id")

        if msg_id is not None and msg_id in self._pending:
            # This is a response to one of our requests
            future = self._pending.pop(msg_id)
            if not future.done():
                future.set_result(msg)
            return

        # Otherwise it's a server-initiated notification
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "textDocument/publishDiagnostics":
            self._handle_diagnostics(params)
        elif method == "window/logMessage":
            level = params.get("type", 4)  # 1=Error, 2=Warning, 3=Info, 4=Log
            text = params.get("message", "")
            if level <= 2:
                logger.warning("LSP server: %s", text)
            else:
                logger.debug("LSP server: %s", text)
        # Ignore other notifications (progress, etc.)

    def _handle_diagnostics(self, params: dict[str, Any]) -> None:
        """Store published diagnostics and signal waiters."""
        uri = params.get("uri", "")
        diags = params.get("diagnostics", [])

        self._diagnostics[uri] = diags

        event = self._diag_events.get(uri)
        if event is not None:
            event.set()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

async def get_lsp_session() -> LeanLSPSession:
    """Get or create the singleton LSP session (lazy init)."""
    global _active_lsp_session
    if _active_lsp_session is None or not _active_lsp_session.is_running:
        _active_lsp_session = LeanLSPSession()
        await _active_lsp_session.start()
        await _active_lsp_session.initialize()
    return _active_lsp_session


async def stop_lsp_session() -> None:
    """Shut down the singleton LSP session."""
    global _active_lsp_session
    if _active_lsp_session is not None:
        await _active_lsp_session.shutdown()
        _active_lsp_session = None
