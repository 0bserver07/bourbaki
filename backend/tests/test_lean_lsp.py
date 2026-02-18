"""Tests for Lean LSP session manager and tool functions.

Tests cover:
- JSON-RPC message framing (encode / decode)
- LSP session lifecycle with mocked subprocess
- Diagnostic parsing and routing
- Completion parsing
- High-level tool functions with mocked session
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bourbaki.tools.lean_lsp import (
    LeanLSPSession,
    encode_lsp_message,
    read_lsp_message,
)
from bourbaki.tools.lean_lsp_tools import (
    _ensure_imports,
    _format_diagnostic,
    _severity_label,
    lsp_check,
    lsp_completions,
    lsp_goal,
    lsp_hover,
    lsp_suggest_tactics,
)


# ---------------------------------------------------------------------------
# JSON-RPC framing tests
# ---------------------------------------------------------------------------

class TestEncoding:
    """Test the LSP Content-Length framing encoder."""

    def test_encode_simple_message(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        encoded = encode_lsp_message(msg)
        body = json.dumps(msg).encode("utf-8")
        expected_header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        assert encoded == expected_header + body

    def test_encode_notification_no_id(self):
        msg = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        encoded = encode_lsp_message(msg)
        assert b"Content-Length:" in encoded
        # Should not have an id field in the body
        body_start = encoded.index(b"\r\n\r\n") + 4
        body_json = json.loads(encoded[body_start:])
        assert "id" not in body_json

    def test_encode_unicode_content(self):
        msg = {"jsonrpc": "2.0", "method": "test", "params": {"text": "theorem \u22a2 True"}}
        encoded = encode_lsp_message(msg)
        body = json.dumps(msg).encode("utf-8")
        # Content-Length should be byte length, not character length
        header_line = encoded.split(b"\r\n")[0].decode("ascii")
        assert header_line == f"Content-Length: {len(body)}"


class TestDecoding:
    """Test the LSP Content-Length framing decoder."""

    @pytest.mark.asyncio
    async def test_decode_simple_message(self):
        msg = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        raw = encode_lsp_message(msg)
        reader = asyncio.StreamReader()
        reader.feed_data(raw)
        decoded = await read_lsp_message(reader)
        assert decoded == msg

    @pytest.mark.asyncio
    async def test_decode_eof_raises(self):
        reader = asyncio.StreamReader()
        reader.feed_eof()
        with pytest.raises(EOFError):
            await read_lsp_message(reader)

    @pytest.mark.asyncio
    async def test_roundtrip(self):
        """Encode then decode should give back the original message."""
        original = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "textDocument/hover",
            "params": {"textDocument": {"uri": "file:///test.lean"}, "position": {"line": 0, "character": 5}},
        }
        raw = encode_lsp_message(original)
        reader = asyncio.StreamReader()
        reader.feed_data(raw)
        decoded = await read_lsp_message(reader)
        assert decoded == original

    @pytest.mark.asyncio
    async def test_decode_multiple_messages(self):
        """Two messages concatenated in the stream should both decode."""
        msg1 = {"jsonrpc": "2.0", "id": 1, "result": "hello"}
        msg2 = {"jsonrpc": "2.0", "id": 2, "result": "world"}
        reader = asyncio.StreamReader()
        reader.feed_data(encode_lsp_message(msg1))
        reader.feed_data(encode_lsp_message(msg2))

        decoded1 = await read_lsp_message(reader)
        decoded2 = await read_lsp_message(reader)
        assert decoded1["result"] == "hello"
        assert decoded2["result"] == "world"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Test utility helpers in lean_lsp_tools."""

    def test_severity_labels(self):
        assert _severity_label(1) == "error"
        assert _severity_label(2) == "warning"
        assert _severity_label(3) == "info"
        assert _severity_label(4) == "hint"
        assert _severity_label(99) == "unknown"

    def test_format_diagnostic(self):
        raw = {
            "range": {
                "start": {"line": 5, "character": 3},
                "end": {"line": 5, "character": 10},
            },
            "severity": 1,
            "message": "unknown identifier 'foo'",
        }
        formatted = _format_diagnostic(raw)
        assert formatted["line"] == 5
        assert formatted["column"] == 3
        assert formatted["severity"] == "error"
        assert formatted["message"] == "unknown identifier 'foo'"

    def test_format_diagnostic_missing_fields(self):
        raw = {"message": "some hint"}
        formatted = _format_diagnostic(raw)
        assert formatted["line"] == 0
        assert formatted["column"] == 0
        # Default severity (when absent) is 4, which maps to "hint"
        assert formatted["severity"] == "hint"

    def test_ensure_imports_adds_header(self):
        code = "theorem foo : True := trivial"
        result = _ensure_imports(code)
        assert result.startswith("import Mathlib.Tactic")
        assert "theorem foo" in result

    def test_ensure_imports_preserves_existing(self):
        code = "import Mathlib\ntheorem foo : True := trivial"
        result = _ensure_imports(code)
        assert result == code


# ---------------------------------------------------------------------------
# LeanLSPSession dispatch tests (no real subprocess)
# ---------------------------------------------------------------------------

class TestSessionDispatch:
    """Test the internal message dispatch logic of LeanLSPSession."""

    def test_dispatch_response_resolves_future(self):
        session = LeanLSPSession()
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        session._pending[1] = future

        session._dispatch({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})

        assert future.done()
        result = future.result()
        assert result["result"] == {"ok": True}
        loop.close()

    def test_dispatch_diagnostics_notification(self):
        session = LeanLSPSession()
        uri = "file:///test.lean"
        session._diagnostics[uri] = []
        session._diag_events[uri] = asyncio.Event()

        diag_params = {
            "uri": uri,
            "diagnostics": [
                {
                    "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 5}},
                    "severity": 1,
                    "message": "type mismatch",
                },
            ],
        }
        session._dispatch({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": diag_params,
        })

        assert len(session._diagnostics[uri]) == 1
        assert session._diagnostics[uri][0]["message"] == "type mismatch"
        assert session._diag_events[uri].is_set()

    def test_dispatch_unknown_notification_no_error(self):
        session = LeanLSPSession()
        # Should not raise
        session._dispatch({
            "jsonrpc": "2.0",
            "method": "window/progress",
            "params": {"token": "x", "value": {}},
        })

    def test_dispatch_log_message(self):
        session = LeanLSPSession()
        # Should not raise
        session._dispatch({
            "jsonrpc": "2.0",
            "method": "window/logMessage",
            "params": {"type": 3, "message": "Loading Mathlib..."},
        })

    def test_dispatch_ignores_unmatched_id(self):
        session = LeanLSPSession()
        # No pending future for id=99 — should not raise
        session._dispatch({"jsonrpc": "2.0", "id": 99, "result": {}})


class TestSessionUri:
    """Test URI generation."""

    def test_make_uri_increments(self):
        session = LeanLSPSession()
        uri1 = session.make_uri()
        uri2 = session.make_uri()
        assert uri1 != uri2
        assert "scratch_1" in uri1
        assert "scratch_2" in uri2

    def test_make_uri_custom_tag(self):
        session = LeanLSPSession()
        uri = session.make_uri(tag="my_test")
        assert "my_test.lean" in uri


# ---------------------------------------------------------------------------
# Mock-based session lifecycle test
# ---------------------------------------------------------------------------

def _make_mock_proc():
    """Create a mock subprocess with stdin/stdout/stderr."""
    proc = AsyncMock()
    proc.returncode = None  # Still running

    # stdin
    proc.stdin = AsyncMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()

    # stdout — StreamReader that we can feed data into
    proc.stdout = asyncio.StreamReader()

    # stderr
    proc.stderr = asyncio.StreamReader()
    proc.stderr.feed_eof()

    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)

    return proc


class TestSessionLifecycle:
    """Test start / initialize / shutdown with a mocked subprocess."""

    @pytest.mark.asyncio
    async def test_start_sets_proc(self):
        session = LeanLSPSession()
        mock_proc = _make_mock_proc()

        with patch("bourbaki.tools.lean_lsp.shutil.which", return_value="/usr/bin/lake"), \
             patch("bourbaki.tools.lean_lsp.LEAN_PROJECT_DIR") as mock_dir, \
             patch("bourbaki.tools.lean_lsp.asyncio.create_subprocess_exec", return_value=mock_proc):
            mock_dir.is_dir.return_value = True
            await session.start()

            assert session.is_running
            assert session._proc is mock_proc

        await session.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_sends_handshake(self):
        session = LeanLSPSession()
        mock_proc = _make_mock_proc()

        # Feed initialize response and then EOF
        init_response = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        mock_proc.stdout.feed_data(encode_lsp_message(init_response))

        with patch("bourbaki.tools.lean_lsp.shutil.which", return_value="/usr/bin/lake"), \
             patch("bourbaki.tools.lean_lsp.LEAN_PROJECT_DIR") as mock_dir, \
             patch("bourbaki.tools.lean_lsp.asyncio.create_subprocess_exec", return_value=mock_proc):
            mock_dir.is_dir.return_value = True
            await session.start()

            result = await session.initialize()
            assert result == {"capabilities": {}}
            assert session._initialized

        # Feed EOF to the reader so it exits cleanly
        mock_proc.stdout.feed_eof()
        await session.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self):
        session = LeanLSPSession()
        mock_proc = _make_mock_proc()

        # Feed the init response
        init_response = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        mock_proc.stdout.feed_data(encode_lsp_message(init_response))

        with patch("bourbaki.tools.lean_lsp.shutil.which", return_value="/usr/bin/lake"), \
             patch("bourbaki.tools.lean_lsp.LEAN_PROJECT_DIR") as mock_dir, \
             patch("bourbaki.tools.lean_lsp.asyncio.create_subprocess_exec", return_value=mock_proc):
            mock_dir.is_dir.return_value = True
            await session.start()
            await session.initialize()

            # Feed shutdown response then EOF
            shutdown_response = {"jsonrpc": "2.0", "id": 2, "result": None}
            mock_proc.stdout.feed_data(encode_lsp_message(shutdown_response))
            mock_proc.stdout.feed_eof()

            await session.shutdown()

            assert not session._initialized
            assert session._proc is None


# ---------------------------------------------------------------------------
# Tool function tests with mocked session
# ---------------------------------------------------------------------------

def _mock_session():
    """Create a mock LeanLSPSession for tool function tests."""
    session = AsyncMock(spec=LeanLSPSession)
    session._file_counter = 0

    def _make_uri(tag=None):
        session._file_counter += 1
        name = tag or f"scratch_{session._file_counter}"
        return f"file:///tmp/bourbaki-lsp/{name}.lean"

    session.make_uri = _make_uri
    return session


class TestLspCheck:
    """Test the lsp_check tool function."""

    @pytest.mark.asyncio
    async def test_check_no_errors(self):
        session = _mock_session()
        session.get_diagnostics.return_value = []

        result = await lsp_check("theorem foo : True := trivial", session=session)

        assert result["success"] is True
        assert result["errors"] == []
        assert "duration" in result
        session.open_file.assert_called_once()
        session.close_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_with_errors(self):
        session = _mock_session()
        session.get_diagnostics.return_value = [
            {
                "range": {"start": {"line": 2, "character": 0}, "end": {"line": 2, "character": 5}},
                "severity": 1,
                "message": "unknown identifier 'bar'",
            },
        ]

        result = await lsp_check("noncomputable def bar := sorry", session=session)

        assert result["success"] is False
        assert len(result["errors"]) == 1
        assert result["errors"][0]["message"] == "unknown identifier 'bar'"

    @pytest.mark.asyncio
    async def test_check_mixed_diagnostics(self):
        session = _mock_session()
        session.get_diagnostics.return_value = [
            {
                "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 5}},
                "severity": 2,
                "message": "unused variable",
            },
            {
                "range": {"start": {"line": 3, "character": 0}, "end": {"line": 3, "character": 5}},
                "severity": 3,
                "message": "Nat.add_comm",
            },
        ]

        result = await lsp_check("example : True := trivial", session=session)

        assert result["success"] is True  # No errors, just warnings and info
        assert len(result["warnings"]) == 1
        assert len(result["infos"]) == 1


class TestLspCompletions:
    """Test the lsp_completions tool function."""

    @pytest.mark.asyncio
    async def test_completions_returns_labels(self):
        session = _mock_session()
        session.get_completions.return_value = [
            {"label": "ring", "kind": 1},
            {"label": "ring_nf", "kind": 1},
            {"label": "rfl", "kind": 1},
        ]

        result = await lsp_completions(
            "theorem foo : 1 + 1 = 2 := by sorry",
            line=0, col=35,
            session=session,
        )

        assert result == ["ring", "ring_nf", "rfl"]

    @pytest.mark.asyncio
    async def test_completions_empty(self):
        session = _mock_session()
        session.get_completions.return_value = []

        result = await lsp_completions("-- empty file", line=0, col=0, session=session)

        assert result == []


class TestLspGoal:
    """Test the lsp_goal tool function."""

    @pytest.mark.asyncio
    async def test_goal_returns_string(self):
        session = _mock_session()
        session.get_diagnostics.return_value = []
        session.get_goal.return_value = ["a b : Nat\n\u22a2 a + b = b + a"]

        result = await lsp_goal(
            "theorem foo (a b : Nat) : a + b = b + a := by sorry",
            line=0, col=52,
            session=session,
        )

        assert result is not None
        assert "\u22a2 a + b = b + a" in result

    @pytest.mark.asyncio
    async def test_goal_returns_none_when_empty(self):
        session = _mock_session()
        session.get_diagnostics.return_value = []
        session.get_goal.return_value = []

        result = await lsp_goal("-- no theorem", line=0, col=0, session=session)

        assert result is None


class TestLspHover:
    """Test the lsp_hover tool function."""

    @pytest.mark.asyncio
    async def test_hover_returns_info(self):
        session = _mock_session()
        session.get_diagnostics.return_value = []
        session.get_hover.return_value = "Nat.add_comm : \u2200 (a b : Nat), a + b = b + a"

        result = await lsp_hover(
            "theorem foo : True := trivial",
            line=0, col=10,
            session=session,
        )

        assert result is not None
        assert "Nat.add_comm" in result

    @pytest.mark.asyncio
    async def test_hover_returns_none(self):
        session = _mock_session()
        session.get_diagnostics.return_value = []
        session.get_hover.return_value = None

        result = await lsp_hover("-- nothing", line=0, col=0, session=session)

        assert result is None


class TestLspSuggestTactics:
    """Test the lsp_suggest_tactics tool function."""

    @pytest.mark.asyncio
    async def test_suggest_finds_tactics(self):
        session = _mock_session()
        session.get_diagnostics.return_value = []
        session.get_completions.return_value = [
            {"label": "simp"},
            {"label": "ring"},
            {"label": "omega"},
        ]

        result = await lsp_suggest_tactics(
            "theorem foo : 1 + 1 = 2 := by sorry",
            session=session,
        )

        assert "simp" in result
        assert "ring" in result

    @pytest.mark.asyncio
    async def test_suggest_with_bare_theorem(self):
        """A theorem without 'sorry' should still work."""
        session = _mock_session()
        session.get_diagnostics.return_value = []
        session.get_completions.return_value = [{"label": "decide"}]

        result = await lsp_suggest_tactics(
            "theorem foo : 1 + 1 = 2",
            session=session,
        )

        assert "decide" in result

    @pytest.mark.asyncio
    async def test_suggest_empty_when_no_sorry(self):
        """If sorry position can't be found (shouldn't normally happen), return []."""
        session = _mock_session()
        session.get_diagnostics.return_value = []
        session.get_completions.return_value = []

        # _ensure_imports + adding sorry should still produce a sorry
        result = await lsp_suggest_tactics(
            "theorem foo : True := by sorry",
            session=session,
        )

        # Even if completions are empty, the function should return []
        assert result == []


# ---------------------------------------------------------------------------
# Diagnostic event signalling
# ---------------------------------------------------------------------------

class TestDiagnosticEvents:
    """Test that diagnostics are collected and signalled properly."""

    @pytest.mark.asyncio
    async def test_diagnostics_arrive_and_signal(self):
        session = LeanLSPSession()
        uri = "file:///test.lean"
        session._diagnostics[uri] = []
        session._diag_events[uri] = asyncio.Event()

        # Simulate the server pushing diagnostics
        session._handle_diagnostics({
            "uri": uri,
            "diagnostics": [
                {
                    "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}},
                    "severity": 1,
                    "message": "test error",
                },
            ],
        })

        # The event should now be set
        assert session._diag_events[uri].is_set()
        assert len(session._diagnostics[uri]) == 1
        assert session._diagnostics[uri][0]["message"] == "test error"

    @pytest.mark.asyncio
    async def test_get_diagnostics_timeout_returns_empty(self):
        session = LeanLSPSession()
        uri = "file:///timeout.lean"
        session._diagnostics[uri] = []
        session._diag_events[uri] = asyncio.Event()

        # Don't signal the event — should timeout and return empty
        result = await session.get_diagnostics(uri, timeout=0.05)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_diagnostics_unknown_uri(self):
        session = LeanLSPSession()
        result = await session.get_diagnostics("file:///unknown.lean", timeout=0.01)
        assert result == []


# ---------------------------------------------------------------------------
# Error handling in session
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test error handling in the LSP session."""

    def test_is_running_false_initially(self):
        session = LeanLSPSession()
        assert not session.is_running

    @pytest.mark.asyncio
    async def test_request_when_not_running_raises(self):
        session = LeanLSPSession()
        with pytest.raises(RuntimeError, match="not running"):
            await session._request("test", {})

    @pytest.mark.asyncio
    async def test_notify_when_not_running_raises(self):
        session = LeanLSPSession()
        with pytest.raises(RuntimeError, match="not running"):
            await session._notify("test", {})

    @pytest.mark.asyncio
    async def test_start_without_lake_raises(self):
        session = LeanLSPSession()
        with patch("bourbaki.tools.lean_lsp.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="lake not found"):
                await session.start()

    @pytest.mark.asyncio
    async def test_start_without_project_dir_raises(self):
        session = LeanLSPSession()
        with patch("bourbaki.tools.lean_lsp.shutil.which", return_value="/usr/bin/lake"), \
             patch("bourbaki.tools.lean_lsp.LEAN_PROJECT_DIR") as mock_dir:
            mock_dir.is_dir.return_value = False
            with pytest.raises(RuntimeError, match="not found"):
                await session.start()
