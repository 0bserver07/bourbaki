"""Unit tests for the streaming Ollama /api/chat proposer adapter.

Mocks the upstream HTTP call with ``respx`` so we never touch the real
Ollama Cloud endpoint. The mock provides a custom :class:`AsyncByteStream`
that yields NDJSON chunks one at a time, mimicking the real
``stream: true`` response shape, and tracks how many chunks were consumed
plus whether the stream was closed early.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx
import pytest
import respx

from bourbaki.prover import streaming_proposer as stream_mod
from bourbaki.prover.parallel_proposer import _proposer_for
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
)
from bourbaki.prover.streaming_proposer import (
    OLLAMA_NATIVE_CHAT_URL,
    _try_parse_complete,
    run_proposer_stream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SpyStream(httpx.AsyncByteStream):
    """An async byte stream that yields pre-canned chunks, tracking consumption.

    Tests use ``consumed_count`` to assert that the early-cancel branch
    fires *before* the full chunk sequence is drained, and ``closed`` to
    assert that ``aclose`` ran (i.e. the upstream connection was released).
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.consumed_count = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for c in self._chunks:
            self.consumed_count += 1
            yield c

    async def aclose(self) -> None:
        self.closed = True


def _ndjson_chunk(content: str, *, done: bool = False) -> bytes:
    """Build one NDJSON chunk in Ollama /api/chat shape."""
    payload = {
        "model": "kimi-k2.6",
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done:
        payload["done_reason"] = "stop"
    return (json.dumps(payload) + "\n").encode("utf-8")


def _chunks_for(pieces: list[str], *, fence: bool = False) -> list[bytes]:
    """Render a list of content pieces as a complete NDJSON stream.

    The final chunk has ``done=True`` with empty content so the server's
    end-of-stream marker is faithfully reproduced.
    """
    if fence and pieces:
        pieces = ["```json\n", *pieces, "\n```"]
    chunks = [_ndjson_chunk(p) for p in pieces]
    chunks.append(_ndjson_chunk("", done=True))
    return chunks


def _make_state(
    iteration: int = 0,
    max_iterations: int = 8,
    target: str = "theorem mathd_demo : 1 + 1 = 2 := sorry",
) -> ProverState:
    return ProverState(
        problem_id="demo",
        target_theorem=target,
        full_file="-- complete file --",
        iteration=iteration,
        max_iterations=max_iterations,
    )


_VALID_PROPOSAL_PIECES: list[str] = [
    '{"reasoning":"Use decide",',
    '"imports":[],',
    '"opens":[],',
    '"updated_theorem":"theorem mathd_demo : 1 + 1 = 2 := by decide"}',
]


# ---------------------------------------------------------------------------
# 1. Happy path — full stream completes naturally with valid content
# ---------------------------------------------------------------------------


@respx.mock
async def test_happy_path_full_stream(monkeypatch):
    """Stream emits a valid JSON in 4 content chunks then done=true; we
    should accept the proposal whether via early-cancel or the final-parse
    branch."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    chunks = _chunks_for(_VALID_PROPOSAL_PIECES)
    stream = _SpyStream(chunks)
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=stream)
    )

    state = _make_state(iteration=2)
    out = await run_proposer_stream(state, model="ollama-stream:kimi-k2.6")

    assert route.called
    assert isinstance(out, ProposalMessage)
    assert out.reasoning == "Use decide"
    assert out.code == "theorem mathd_demo : 1 + 1 = 2 := by decide"
    assert out.iteration == 3  # state.iteration + 1
    assert out.imports == []
    assert out.opens == []


# ---------------------------------------------------------------------------
# 2. Early-cancel — proposal-complete signal closes the upstream connection
#    before done=true ever fires.
# ---------------------------------------------------------------------------


@respx.mock
async def test_early_cancel_closes_stream(monkeypatch):
    """The accumulated buffer becomes a parseable, complete ProverResult
    after chunk #4. We must close the stream then — *before* the done=true
    sentinel arrives — and never consume the trailing chunks."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")

    # 4 content chunks that complete the JSON, plus three trailing noise
    # chunks that simulate the model still spewing (e.g. trailing fluff
    # the schema-format guard didn't suppress). If early-cancel works,
    # consumed_count should be 4 and the noise/done chunks should never
    # be touched.
    primary = _chunks_for(_VALID_PROPOSAL_PIECES)
    # Insert three noise chunks BEFORE the done=true sentinel.
    noise = [_ndjson_chunk(" trailing noise") for _ in range(3)]
    chunks = primary[:-1] + noise + [primary[-1]]

    stream = _SpyStream(chunks)
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=stream)
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, ProposalMessage)
    assert out.code == "theorem mathd_demo : 1 + 1 = 2 := by decide"
    # Early-cancel happened: we consumed only the 4 primary content
    # chunks, not the trailing noise or the done sentinel.
    assert stream.consumed_count == 4, (
        f"expected 4 chunks consumed before early-cancel, got "
        f"{stream.consumed_count} (total={len(chunks)})"
    )
    assert stream.closed is True


# ---------------------------------------------------------------------------
# 3. Markdown-fenced content — fences stripped before parse attempts
# ---------------------------------------------------------------------------


@respx.mock
async def test_markdown_fenced_streaming_content(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    pieces = [
        "```json\n",
        '{"reasoning":"r",',
        '"imports":[],',
        '"opens":[],',
        '"updated_theorem":"theorem t : True := trivial"}',
        "\n```",
    ]
    chunks = [_ndjson_chunk(p) for p in pieces]
    chunks.append(_ndjson_chunk("", done=True))
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=_SpyStream(chunks))
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, ProposalMessage)
    assert out.code == "theorem t : True := trivial"


# ---------------------------------------------------------------------------
# 4. String-imports coercion — applied before pydantic validation
# ---------------------------------------------------------------------------


@respx.mock
async def test_string_imports_coerced_before_validation(monkeypatch):
    """Kimi-style ``imports`` as a string (instead of list) must be
    coerced to a single-element list before validation, same as the
    non-streaming path."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    body = json.dumps(
        {
            "reasoning": "r",
            "imports": "import Mathlib",
            "opens": "open Real",
            "updated_theorem": "theorem t : True := trivial",
        }
    )
    chunks = [_ndjson_chunk(body), _ndjson_chunk("", done=True)]
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=_SpyStream(chunks))
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, ProposalMessage)
    assert out.imports == ["import Mathlib"]
    assert out.opens == ["open Real"]


# ---------------------------------------------------------------------------
# 5. Stream completes with empty content → FeedbackMessage
# ---------------------------------------------------------------------------


@respx.mock
async def test_empty_stream_returns_feedback(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    chunks = [_ndjson_chunk("", done=True)]
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=_SpyStream(chunks))
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "empty" in out.content.lower()


# ---------------------------------------------------------------------------
# 6. Stream completes with invalid JSON → FeedbackMessage
# ---------------------------------------------------------------------------


@respx.mock
async def test_invalid_json_returns_feedback(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    chunks = [
        _ndjson_chunk("not json at all"),
        _ndjson_chunk("", done=True),
    ]
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=_SpyStream(chunks))
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "json" in out.content.lower() or "parse" in out.content.lower()


# ---------------------------------------------------------------------------
# 7. Stream completes with valid JSON but missing required fields
# ---------------------------------------------------------------------------


@respx.mock
async def test_missing_required_field_returns_feedback(monkeypatch):
    """JSON parses fine but lacks ``updated_theorem`` — validation must
    surface a parse-failed feedback rather than crashing."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    body = json.dumps({"reasoning": "only this field"})
    chunks = [_ndjson_chunk(body), _ndjson_chunk("", done=True)]
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=_SpyStream(chunks))
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "validation" in out.content.lower() or "updated_theorem" in out.content


# ---------------------------------------------------------------------------
# 8. HTTP 429 before stream starts → rate-limited feedback
# ---------------------------------------------------------------------------


@respx.mock
async def test_rate_limit_429(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(429, text="Too Many Requests")
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "rate" in out.content.lower()


# ---------------------------------------------------------------------------
# 9. HTTP 500 → feedback
# ---------------------------------------------------------------------------


@respx.mock
async def test_http_500_returns_feedback(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(500, text="internal server error")
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "500" in out.content


# ---------------------------------------------------------------------------
# 10. Stream gets killed mid-flight (network error) → FeedbackMessage
# ---------------------------------------------------------------------------


@respx.mock
async def test_network_error_returns_feedback(monkeypatch):
    """The httpx client raises a network error before the stream can be
    opened. Must be caught and surfaced as a FeedbackMessage."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        side_effect=httpx.ConnectError("simulated connection reset")
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "network" in out.content.lower() or "connection" in out.content.lower()


# ---------------------------------------------------------------------------
# 11. Iteration budget exhausted → no HTTP call
# ---------------------------------------------------------------------------


@respx.mock
async def test_iteration_budget_exhausted_no_http(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    chunks = _chunks_for(_VALID_PROPOSAL_PIECES)
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=_SpyStream(chunks))
    )

    state = _make_state(iteration=8, max_iterations=8)
    out = await run_proposer_stream(state, model="ollama-stream:kimi-k2.6")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "max_iterations"
    assert out.is_terminal is True
    assert not route.called


# ---------------------------------------------------------------------------
# 12. enable_mathlib_search=True → WARN logged, proceeds without tools
# ---------------------------------------------------------------------------


@respx.mock
async def test_enable_mathlib_search_warns_and_continues(monkeypatch, caplog):
    """The streaming adapter doesn't support tools. Passing
    ``enable_mathlib_search=True`` must log a WARNING and proceed without
    tools in the request body."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    chunks = _chunks_for(_VALID_PROPOSAL_PIECES)
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=_SpyStream(chunks))
    )

    with caplog.at_level(logging.WARNING, logger=stream_mod.logger.name):
        out = await run_proposer_stream(
            _make_state(),
            model="ollama-stream:kimi-k2.6",
            enable_mathlib_search=True,
        )

    assert isinstance(out, ProposalMessage)
    assert any(
        "does not support tool calls" in rec.getMessage()
        for rec in caplog.records
    )
    # Body must NOT include a "tools" field.
    body = json.loads(route.calls.last.request.content)
    assert "tools" not in body


# ---------------------------------------------------------------------------
# 13. Empty model name after prefix strip → FeedbackMessage (no HTTP call)
# ---------------------------------------------------------------------------


@respx.mock
async def test_empty_model_name_no_http(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    chunks = _chunks_for(_VALID_PROPOSAL_PIECES)
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=_SpyStream(chunks))
    )

    out = await run_proposer_stream(_make_state(), model="ollama-stream:")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "empty model name" in out.content
    assert not route.called


# ---------------------------------------------------------------------------
# 14. Missing OLLAMA_CLOUD_API_KEY → FeedbackMessage
# ---------------------------------------------------------------------------


async def test_missing_api_key_returns_feedback(monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_API_KEY", raising=False)

    out = await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "OLLAMA_CLOUD_API_KEY" in out.content


# ---------------------------------------------------------------------------
# 15. Request body shape — stream:true, think:false, format=<schema>,
#     and the prefix stripped from the model name.
# ---------------------------------------------------------------------------


@respx.mock
async def test_request_body_shape(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    chunks = _chunks_for(_VALID_PROPOSAL_PIECES)
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, stream=_SpyStream(chunks))
    )

    await run_proposer_stream(_make_state(), model="ollama-stream:kimi-k2.6")

    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "kimi-k2.6"  # prefix stripped
    assert body["stream"] is True
    assert body["think"] is False
    assert "format" in body
    fmt = body["format"]
    assert fmt["type"] == "object"
    assert "updated_theorem" in fmt["properties"]
    assert "updated_theorem" in fmt["required"]
    assert body["options"]["num_predict"] >= 1


# ---------------------------------------------------------------------------
# 16. Wrong-prefix model rejected eagerly with FeedbackMessage
# ---------------------------------------------------------------------------


async def test_wrong_prefix_returns_feedback(monkeypatch):
    """Passing an ``ollama-native:`` model string (or anything without the
    streaming prefix) should be flagged immediately, not silently routed
    through the streaming path."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    out = await run_proposer_stream(
        _make_state(), model="ollama-native:kimi-k2.6"
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "ollama-stream:" in out.content


# ---------------------------------------------------------------------------
# 17. Dispatcher routes ollama-stream: through this module
# ---------------------------------------------------------------------------


def test_parallel_proposer_routes_to_streaming():
    """The ``_proposer_for`` dispatcher must return ``run_proposer_stream``
    for an ``ollama-stream:`` prefixed model. Smoke check that the wiring
    in ``parallel_proposer.py`` is in place."""
    assert _proposer_for("ollama-stream:kimi-k2.6") is run_proposer_stream


# ---------------------------------------------------------------------------
# 18. Early-cancel heuristic: empty updated_theorem keeps streaming
# ---------------------------------------------------------------------------


def test_try_parse_complete_rejects_empty_updated_theorem():
    """An otherwise-valid JSON whose ``updated_theorem`` is empty must
    NOT trigger early-cancel — it's a mid-stream artifact (the field is
    declared but the proof body hasn't been emitted yet)."""
    payload = json.dumps(
        {"reasoning": "r", "imports": [], "opens": [], "updated_theorem": ""}
    )
    result, reason = _try_parse_complete(payload)
    assert result is None
    assert reason == "empty-updated_theorem"


def test_try_parse_complete_accepts_complete_proposal():
    """The happy path of the heuristic: fully-formed JSON with a non-empty
    updated_theorem returns the validated ProverResult."""
    payload = json.dumps(
        {
            "reasoning": "r",
            "imports": [],
            "opens": [],
            "updated_theorem": "theorem t : True := trivial",
        }
    )
    result, reason = _try_parse_complete(payload)
    assert result is not None
    assert reason is None
    assert result.updated_theorem == "theorem t : True := trivial"


def test_try_parse_complete_handles_incomplete_json():
    """A buffer that doesn't yet close its JSON object returns ``None``
    with a diagnostic reason — the streaming loop should keep accumulating
    chunks in this case rather than returning a FeedbackMessage."""
    result, reason = _try_parse_complete('{"reasoning":"r","imports"')
    assert result is None
    assert reason == "json-incomplete"
