"""Tests for REPL pipe corruption recovery.

Covers:
- _drain_stale_output() drains remaining response after timeout
- _handle_pipe_desync() drains or kills session
- _read_response() recovers from timeout (drains pipe instead of leaving it dirty)
- send_cmd()/send_tactic() handle external CancelledError (from caller's wait_for)
- lean_tactic() handles error dicts from send_cmd during initialization
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bourbaki.tools.lean_repl import LeanREPLSession, lean_tactic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_with_fake_proc(
    stdout_lines: list[bytes],
) -> LeanREPLSession:
    """Create a session with a fake subprocess whose stdout yields given lines."""
    session = LeanREPLSession()
    session._initialized = True

    proc = MagicMock()
    proc.returncode = None

    # Build a readline that yields lines from the list, then blocks forever
    remaining = list(stdout_lines)

    async def fake_readline() -> bytes:
        if remaining:
            return remaining.pop(0)
        # Simulate a stuck process — block until cancelled
        await asyncio.sleep(999)
        return b""

    proc.stdout = MagicMock()
    proc.stdout.readline = fake_readline

    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()

    proc.stderr = MagicMock()
    proc.stderr.readline = AsyncMock(return_value=b"")

    session.proc = proc
    session.env_id = 0
    return session


# ---------------------------------------------------------------------------
# _drain_stale_output() tests
# ---------------------------------------------------------------------------


class TestDrainStaleOutput:
    """Tests for LeanREPLSession._drain_stale_output()."""

    @pytest.mark.asyncio
    async def test_drain_succeeds_on_blank_line(self):
        """Drain reads until blank line separator and returns True."""
        session = _make_session_with_fake_proc([
            b'  "still": "producing",\n',
            b'  "output": true\n',
            b"}\n",
            b"\n",  # blank line = end of response
        ])

        result = await session._drain_stale_output(drain_timeout=2.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_drain_succeeds_on_eof(self):
        """Drain returns True when the process exits (EOF)."""
        session = _make_session_with_fake_proc([
            b'  "partial": "data"\n',
            b"",  # EOF
        ])

        result = await session._drain_stale_output(drain_timeout=2.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_drain_fails_on_stuck_process(self):
        """Drain returns False when the process doesn't produce output."""
        # No lines at all — fake_readline blocks forever
        session = _make_session_with_fake_proc([])

        result = await session._drain_stale_output(drain_timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_drain_returns_false_when_no_proc(self):
        """Drain returns False when the session has no process."""
        session = LeanREPLSession()
        session.proc = None

        result = await session._drain_stale_output()
        assert result is False


# ---------------------------------------------------------------------------
# _handle_pipe_desync() tests
# ---------------------------------------------------------------------------


class TestHandlePipeDesync:
    """Tests for LeanREPLSession._handle_pipe_desync()."""

    @pytest.mark.asyncio
    async def test_desync_drains_and_keeps_session(self):
        """When drain succeeds, session stays running."""
        session = _make_session_with_fake_proc([
            b'{"some": "leftover"}\n',
            b"\n",
        ])
        session.stop = AsyncMock()

        await session._handle_pipe_desync()

        session.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_desync_kills_session_on_stuck_drain(self):
        """When drain fails (timeout), session is stopped."""
        session = _make_session_with_fake_proc([])  # stuck
        session.stop = AsyncMock()

        # Override drain timeout to be very short
        with patch.object(
            session, "_drain_stale_output",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await session._handle_pipe_desync()

        session.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_desync_noop_when_not_running(self):
        """When session is not running, desync is a no-op."""
        session = LeanREPLSession()
        session.proc = None
        session.stop = AsyncMock()

        await session._handle_pipe_desync()

        session.stop.assert_not_called()


# ---------------------------------------------------------------------------
# _read_response() timeout recovery
# ---------------------------------------------------------------------------


class TestReadResponseTimeout:
    """Tests for _read_response() timeout handling with pipe recovery."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error_dict(self):
        """On timeout, _read_response returns error dict instead of raising."""
        session = _make_session_with_fake_proc([])  # will block

        # Mock the desync handler since the fake proc will timeout
        session._handle_pipe_desync = AsyncMock()

        result = await session._read_response(timeout=0.1)

        assert "error" in result
        assert "timed out" in result["error"].lower()
        session._handle_pipe_desync.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_drains_pipe(self):
        """On timeout, _read_response drains the stale output."""
        # First, the read starts and gets some data, then blocks (timeout)
        # After timeout, drain should read the remaining data
        session = _make_session_with_fake_proc([])  # stuck process
        session.stop = AsyncMock()

        # The drain will also timeout on a stuck process
        result = await session._read_response(timeout=0.1)
        assert "error" in result


# ---------------------------------------------------------------------------
# send_cmd() / send_tactic() cancellation handling
# ---------------------------------------------------------------------------


class TestExternalCancellation:
    """Tests for send_cmd/send_tactic handling external CancelledError."""

    @pytest.mark.asyncio
    async def test_send_tactic_drains_on_cancellation(self):
        """When send_tactic is cancelled externally, it drains before re-raising."""
        session = LeanREPLSession()
        session._initialized = True
        session.proc = MagicMock()
        session.proc.returncode = None
        session.proc.stdin = MagicMock()
        session.proc.stdin.write = MagicMock()
        session.proc.stdin.drain = AsyncMock()

        # _read_response will be cancelled from outside
        async def slow_read(timeout=120):
            await asyncio.sleep(999)

        session._read_response = slow_read
        session._handle_pipe_desync = AsyncMock()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.send_tactic("ring", 0),
                timeout=0.1,
            )

        # _handle_pipe_desync should have been called before CancelledError re-raised
        session._handle_pipe_desync.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_cmd_drains_on_cancellation(self):
        """When send_cmd is cancelled externally, it drains before re-raising."""
        session = LeanREPLSession()
        session._initialized = True
        session.proc = MagicMock()
        session.proc.returncode = None
        session.proc.stdin = MagicMock()
        session.proc.stdin.write = MagicMock()
        session.proc.stdin.drain = AsyncMock()

        async def slow_read(timeout=120):
            await asyncio.sleep(999)

        session._read_response = slow_read
        session._handle_pipe_desync = AsyncMock()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.send_cmd("example : True := trivial"),
                timeout=0.1,
            )

        session._handle_pipe_desync.assert_called_once()


# ---------------------------------------------------------------------------
# lean_tactic() init error handling
# ---------------------------------------------------------------------------


class TestLeanTacticInitError:
    """Tests for lean_tactic() handling error dicts during initialization."""

    @pytest.mark.asyncio
    async def test_init_timeout_returns_failure(self):
        """When send_cmd returns a timeout error, lean_tactic should report failure."""
        mock_session = AsyncMock(spec=LeanREPLSession)
        mock_session.is_running = True
        mock_session.env_id = 0
        mock_session._initialized = True

        # Simulate send_cmd returning a timeout error
        mock_session.send_cmd = AsyncMock(return_value={
            "error": "REPL timed out",
        })

        with patch(
            "bourbaki.tools.lean_repl._find_repl_binary",
            return_value=MagicMock(),
        ):
            result = await lean_tactic(
                goal="theorem foo : 1 + 1 = 2",
                tactic="sorry",
                proof_state=None,
                session=mock_session,
            )

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_init_pipe_error_returns_failure(self):
        """When send_cmd returns a JSON parse error, lean_tactic should report failure."""
        mock_session = AsyncMock(spec=LeanREPLSession)
        mock_session.is_running = True
        mock_session.env_id = 0
        mock_session._initialized = True

        mock_session.send_cmd = AsyncMock(return_value={
            "error": "Invalid JSON from REPL: garbage data",
        })

        with patch(
            "bourbaki.tools.lean_repl._find_repl_binary",
            return_value=MagicMock(),
        ):
            result = await lean_tactic(
                goal="theorem foo : 1 + 1 = 2",
                tactic="sorry",
                proof_state=None,
                session=mock_session,
            )

        assert result["success"] is False
        assert "Invalid JSON" in result["error"]
