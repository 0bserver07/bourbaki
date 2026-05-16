"""Unit tests for the native Ollama /api/chat proposer adapter.

Mocks the upstream HTTP call with ``respx`` so we never touch the real
Ollama Cloud endpoint. Exercises happy paths, error paths, and the
request-shape contract that the brief specifies.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from bourbaki.prover import native_proposer as native_mod
from bourbaki.prover.native_proposer import (
    OLLAMA_NATIVE_CHAT_URL,
    run_proposer_native,
)
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
)


pytestmark = pytest.mark.asyncio


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


def _make_ok_response(
    reasoning: str = "Trivial arithmetic",
    updated_theorem: str = "theorem mathd_demo : 1 + 1 = 2 := by decide",
    *,
    fence: bool = False,
    extra_fields: dict | None = None,
) -> dict:
    payload: dict = {
        "reasoning": reasoning,
        "imports": [],
        "opens": [],
        "updated_theorem": updated_theorem,
    }
    if extra_fields:
        payload.update(extra_fields)
    body = json.dumps(payload)
    content = f"```json\n{body}\n```" if fence else body
    return {
        "model": "kimi-k2.6",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "eval_count": 1234,
    }


# ---------------------------------------------------------------------------
# 1. Happy path: plain JSON content
# ---------------------------------------------------------------------------


@respx.mock
async def test_happy_path_plain_json(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_make_ok_response(
                reasoning="Use decide.",
                updated_theorem="theorem mathd_demo : 1 + 1 = 2 := by decide",
            ),
        )
    )

    state = _make_state(iteration=2, max_iterations=8)
    out = await run_proposer_native(state, model="ollama-native:kimi-k2.6")

    assert route.called
    assert isinstance(out, ProposalMessage)
    assert out.reasoning == "Use decide."
    assert out.code == "theorem mathd_demo : 1 + 1 = 2 := by decide"
    assert out.iteration == 3  # state.iteration + 1
    assert out.imports == []
    assert out.opens == []


# ---------------------------------------------------------------------------
# 2. Happy path: markdown-fenced JSON content
# ---------------------------------------------------------------------------


@respx.mock
async def test_happy_path_markdown_fenced_json(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_make_ok_response(
                reasoning="Fence-wrapped output.",
                updated_theorem="theorem t : True := trivial",
                fence=True,
            ),
        )
    )

    out = await run_proposer_native(
        _make_state(), model="ollama-native:deepseek-v4-pro"
    )

    assert isinstance(out, ProposalMessage)
    assert out.reasoning == "Fence-wrapped output."
    assert out.code == "theorem t : True := trivial"


# ---------------------------------------------------------------------------
# 3. HTTP 429: rate limit
# ---------------------------------------------------------------------------


@respx.mock
async def test_rate_limit_429(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(429, text="Too Many Requests")
    )

    out = await run_proposer_native(
        _make_state(), model="ollama-native:kimi-k2.6"
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "rate" in out.content.lower()


# ---------------------------------------------------------------------------
# 4. HTTP 500: server error
# ---------------------------------------------------------------------------


@respx.mock
async def test_http_500_returns_feedback(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(500, text="internal server error")
    )

    out = await run_proposer_native(
        _make_state(), model="ollama-native:kimi-k2.6"
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "500" in out.content


# ---------------------------------------------------------------------------
# 5. Network timeout
# ---------------------------------------------------------------------------


@respx.mock
async def test_network_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        side_effect=httpx.TimeoutException("simulated read timeout")
    )

    out = await run_proposer_native(
        _make_state(), model="ollama-native:kimi-k2.6", llm_timeout=5.0
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "timed out" in out.content
    assert "5" in out.content


# ---------------------------------------------------------------------------
# 6. Invalid JSON content
# ---------------------------------------------------------------------------


@respx.mock
async def test_invalid_json_in_content(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "this is definitely not json",
                },
                "done": True,
            },
        )
    )

    out = await run_proposer_native(
        _make_state(), model="ollama-native:kimi-k2.6"
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "parse" in out.content.lower() or "json" in out.content.lower()


# ---------------------------------------------------------------------------
# 7. JSON missing a required field
# ---------------------------------------------------------------------------


@respx.mock
async def test_missing_required_field(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    # Missing `updated_theorem` — ProverResult validation fails.
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"reasoning": "only reasoning"}),
                },
                "done": True,
            },
        )
    )

    out = await run_proposer_native(
        _make_state(), model="ollama-native:kimi-k2.6"
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "validation" in out.content.lower() or "updated_theorem" in out.content


# ---------------------------------------------------------------------------
# 8. Missing OLLAMA_CLOUD_API_KEY env var
# ---------------------------------------------------------------------------


async def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_API_KEY", raising=False)

    out = await run_proposer_native(
        _make_state(), model="ollama-native:kimi-k2.6"
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "OLLAMA_CLOUD_API_KEY" in out.content


# ---------------------------------------------------------------------------
# 9. Request body asserts: think=false and format=<schema>
# ---------------------------------------------------------------------------


@respx.mock
async def test_request_body_shape(monkeypatch):
    """Verify the request body sent to Ollama has the load-bearing fields:
    ``think: false``, a ``format`` JSON Schema, ``stream: false``, and the
    correct model name with the prefix stripped.
    """
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_make_ok_response())
    )

    await run_proposer_native(
        _make_state(), model="ollama-native:kimi-k2.6"
    )

    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "kimi-k2.6"  # prefix stripped
    assert body["think"] is False
    assert body["stream"] is False
    assert "format" in body
    fmt = body["format"]
    assert fmt["type"] == "object"
    # The proposer's ProverResult schema must require updated_theorem.
    assert "updated_theorem" in fmt["properties"]
    assert "reasoning" in fmt["properties"]
    assert "updated_theorem" in fmt["required"]
    assert body["options"]["num_predict"] >= 1


# ---------------------------------------------------------------------------
# 10. Request body includes the target theorem
# ---------------------------------------------------------------------------


@respx.mock
async def test_request_body_includes_target_theorem(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    target = "theorem UNIQUE_marker_xyz : 2 + 2 = 4 := sorry"
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_make_ok_response())
    )

    await run_proposer_native(
        _make_state(target=target), model="ollama-native:kimi-k2.6"
    )

    body = json.loads(route.calls.last.request.content)
    user_msgs = [m for m in body["messages"] if m["role"] == "user"]
    assert user_msgs, "expected at least one user message"
    assert "UNIQUE_marker_xyz" in user_msgs[0]["content"]


# ---------------------------------------------------------------------------
# 11. enable_mathlib_search=True logs a warning and still runs
# ---------------------------------------------------------------------------


@respx.mock
async def test_enable_mathlib_search_warns_and_proceeds(monkeypatch, caplog):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_make_ok_response())
    )

    with caplog.at_level(logging.WARNING, logger=native_mod.logger.name):
        out = await run_proposer_native(
            _make_state(),
            model="ollama-native:kimi-k2.6",
            enable_mathlib_search=True,
        )

    assert isinstance(out, ProposalMessage)
    assert any(
        "enable_mathlib_search" in rec.getMessage()
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# 12. Iteration budget exhausted — no HTTP call is made
# ---------------------------------------------------------------------------


@respx.mock
async def test_iteration_budget_exhausted_no_http(monkeypatch):
    """Budget guard mirrors run_proposer: no network call when budget is gone."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_make_ok_response())
    )

    state = _make_state(iteration=8, max_iterations=8)
    out = await run_proposer_native(state, model="ollama-native:kimi-k2.6")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "max_iterations"
    assert out.is_terminal is True
    assert not route.called


# ---------------------------------------------------------------------------
# 13. Empty model name after prefix → FeedbackMessage (no HTTP call)
# ---------------------------------------------------------------------------


@respx.mock
async def test_empty_model_name(monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_make_ok_response())
    )

    out = await run_proposer_native(_make_state(), model="ollama-native:")

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "empty model name" in out.content
    assert not route.called
