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
    _coerce_list_fields,
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
# 11. enable_mathlib_search=True with no tool_calls — behaves like happy path
#     and logs an info-level "tool enabled" message.
# ---------------------------------------------------------------------------


@respx.mock
async def test_enable_mathlib_search_no_tool_calls(monkeypatch, caplog):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_make_ok_response())
    )

    with caplog.at_level(logging.INFO, logger=native_mod.logger.name):
        out = await run_proposer_native(
            _make_state(),
            model="ollama-native:kimi-k2.6",
            enable_mathlib_search=True,
        )

    assert isinstance(out, ProposalMessage)
    assert any(
        "mathlib_search tool enabled" in rec.getMessage()
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


# ---------------------------------------------------------------------------
# 14. _coerce_list_fields: tolerant parse for sloppy reasoning models
# ---------------------------------------------------------------------------


def test_coerce_list_fields_wraps_single_string():
    obj = {"reasoning": "x", "imports": "import Mathlib", "opens": "open Real", "updated_theorem": "t"}
    _coerce_list_fields(obj)
    assert obj["imports"] == ["import Mathlib"]
    assert obj["opens"] == ["open Real"]


def test_coerce_list_fields_splits_multiline_string():
    obj = {"imports": "import Mathlib\nimport Aesop", "opens": "open Real\nopen Set"}
    _coerce_list_fields(obj)
    assert obj["imports"] == ["import Mathlib", "import Aesop"]
    assert obj["opens"] == ["open Real", "open Set"]


def test_coerce_list_fields_leaves_lists_alone():
    obj = {"imports": ["import Mathlib"], "opens": ["open Real"]}
    _coerce_list_fields(obj)
    assert obj["imports"] == ["import Mathlib"]
    assert obj["opens"] == ["open Real"]


def test_coerce_list_fields_handles_missing_fields():
    obj = {"reasoning": "x"}
    _coerce_list_fields(obj)
    assert obj == {"reasoning": "x"}


@respx.mock
async def test_string_imports_coerced_before_validation(monkeypatch):
    """End-to-end: Kimi-style string imports get coerced to lists and validate."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    payload = {
        "reasoning": "x",
        "imports": "import Mathlib",
        "opens": "open Real Set",
        "updated_theorem": "theorem demo : True := trivial",
    }
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": json.dumps(payload)}},
        )
    )

    out = await run_proposer_native(_make_state(), model="ollama-native:kimi-k2.6")

    assert isinstance(out, ProposalMessage)
    assert out.imports == ["import Mathlib"]
    assert out.opens == ["open Real Set"]


# ---------------------------------------------------------------------------
# 15-20. mathlib_search tool-call loop (multi-round) — Phase 4 on native path
# ---------------------------------------------------------------------------


def _tool_call_response(
    calls: list[dict],
    *,
    content: str = "",
) -> dict:
    """Build a /api/chat response that asks for one or more tool calls."""
    return {
        "model": "kimi-k2.6",
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": calls,
        },
        "done": True,
        "done_reason": "tool_calls",
    }


@respx.mock
async def test_enable_mathlib_search_no_tools_field_when_disabled(monkeypatch):
    """Default path (enable_mathlib_search=False) must not advertise tools."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_make_ok_response())
    )

    out = await run_proposer_native(_make_state(), model="ollama-native:kimi-k2.6")

    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert "tools" not in body
    assert isinstance(out, ProposalMessage)


@respx.mock
async def test_enable_mathlib_search_adds_tools_field(monkeypatch):
    """enable_mathlib_search=True must include the tools schema in the body."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        return_value=httpx.Response(200, json=_make_ok_response())
    )

    out = await run_proposer_native(
        _make_state(),
        model="ollama-native:kimi-k2.6",
        enable_mathlib_search=True,
    )

    assert isinstance(out, ProposalMessage)
    body = json.loads(route.calls.last.request.content)
    assert "tools" in body
    assert isinstance(body["tools"], list) and len(body["tools"]) == 1
    fn = body["tools"][0]["function"]
    assert fn["name"] == "mathlib_search"
    assert "query" in fn["parameters"]["properties"]
    assert "mode" in fn["parameters"]["properties"]


@respx.mock
async def test_single_tool_call_round(monkeypatch):
    """Model asks for mathlib_search once, then returns a final proposal."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")

    fake_result = {"success": True, "results": [{"name": "Nat.add_comm"}], "count": 1}

    async def fake_mathlib_search(*, query, mode="name", max_results=5):
        assert query == "Nat.add_comm"
        assert mode == "name"
        return fake_result

    monkeypatch.setattr(
        "bourbaki.tools.mathlib_search.mathlib_search", fake_mathlib_search
    )

    responses = [
        httpx.Response(
            200,
            json=_tool_call_response([
                {
                    "function": {
                        "name": "mathlib_search",
                        "arguments": {"query": "Nat.add_comm", "mode": "name"},
                    }
                }
            ]),
        ),
        httpx.Response(
            200,
            json=_make_ok_response(
                reasoning="Used Nat.add_comm",
                updated_theorem="theorem t : 1 + 1 = 2 := by simp",
            ),
        ),
    ]
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(side_effect=responses)

    out = await run_proposer_native(
        _make_state(),
        model="ollama-native:kimi-k2.6",
        enable_mathlib_search=True,
    )

    assert isinstance(out, ProposalMessage)
    assert out.reasoning == "Used Nat.add_comm"
    assert out.code == "theorem t : 1 + 1 = 2 := by simp"
    assert route.call_count == 2

    # Second request must carry the assistant message with tool_calls and
    # the tool result.
    second = json.loads(route.calls[1].request.content)
    msgs = second["messages"]
    assert any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in msgs
    )
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["name"] == "mathlib_search"
    parsed_tool = json.loads(tool_msgs[0]["content"])
    assert parsed_tool == fake_result


@respx.mock
async def test_multiple_tool_calls_one_round(monkeypatch):
    """Model emits 2 tool_calls in a single round — both must be invoked
    and appended as separate role:tool messages."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")

    call_log: list[dict] = []

    async def fake_mathlib_search(*, query, mode="name", max_results=5):
        call_log.append({"query": query, "mode": mode})
        return {"success": True, "results": [], "count": 0, "query": query}

    monkeypatch.setattr(
        "bourbaki.tools.mathlib_search.mathlib_search", fake_mathlib_search
    )

    responses = [
        httpx.Response(
            200,
            json=_tool_call_response([
                {
                    "function": {
                        "name": "mathlib_search",
                        "arguments": {"query": "List.length", "mode": "name"},
                    }
                },
                {
                    "function": {
                        "name": "mathlib_search",
                        "arguments": {"query": "Nat.succ", "mode": "name"},
                    }
                },
            ]),
        ),
        httpx.Response(
            200,
            json=_make_ok_response(
                reasoning="Searched two lemmas.",
                updated_theorem="theorem t : True := trivial",
            ),
        ),
    ]
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(side_effect=responses)

    out = await run_proposer_native(
        _make_state(),
        model="ollama-native:kimi-k2.6",
        enable_mathlib_search=True,
    )

    assert isinstance(out, ProposalMessage)
    assert route.call_count == 2
    assert len(call_log) == 2
    assert {c["query"] for c in call_log} == {"List.length", "Nat.succ"}

    second = json.loads(route.calls[1].request.content)
    tool_msgs = [m for m in second["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 2


@respx.mock
async def test_tool_call_loop_caps_at_max_rounds(monkeypatch):
    """Model keeps asking for tool_calls forever — cap kicks in and we
    return a parse-failed FeedbackMessage instead of looping forever."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")

    async def fake_mathlib_search(*, query, mode="name", max_results=5):
        return {"success": True, "results": [], "count": 0}

    monkeypatch.setattr(
        "bourbaki.tools.mathlib_search.mathlib_search", fake_mathlib_search
    )

    looping_response = httpx.Response(
        200,
        json=_tool_call_response([
            {
                "function": {
                    "name": "mathlib_search",
                    "arguments": {"query": "Nat.succ", "mode": "name"},
                }
            }
        ]),
    )
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(
        side_effect=[looping_response] * 10
    )

    out = await run_proposer_native(
        _make_state(),
        model="ollama-native:kimi-k2.6",
        enable_mathlib_search=True,
    )

    assert isinstance(out, FeedbackMessage)
    assert out.kind == "structured_output_parsing_failed"
    assert "max tool-call rounds" in out.content
    # Exactly _MAX_TOOL_CALL_ROUNDS HTTP calls, no more.
    assert route.call_count == native_mod._MAX_TOOL_CALL_ROUNDS


@respx.mock
async def test_tool_call_malformed_arguments(monkeypatch):
    """Malformed tool_call.arguments (non-dict / missing required) must
    feed an error string back to the model rather than crashing."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")

    invoked: list[dict] = []

    async def fake_mathlib_search(*, query, mode="name", max_results=5):
        invoked.append({"query": query})
        return {"success": True, "results": [], "count": 0}

    monkeypatch.setattr(
        "bourbaki.tools.mathlib_search.mathlib_search", fake_mathlib_search
    )

    responses = [
        # Round 1: arguments are a non-JSON string — should yield an error
        # payload, not raise.
        httpx.Response(
            200,
            json=_tool_call_response([
                {
                    "function": {
                        "name": "mathlib_search",
                        "arguments": "this is not json",
                    }
                }
            ]),
        ),
        # Round 2: arguments missing `query` — also yields an error payload.
        httpx.Response(
            200,
            json=_tool_call_response([
                {
                    "function": {
                        "name": "mathlib_search",
                        "arguments": {"mode": "name"},
                    }
                }
            ]),
        ),
        # Round 3: model finally produces a valid proposal.
        httpx.Response(
            200,
            json=_make_ok_response(
                reasoning="Retried after malformed args.",
                updated_theorem="theorem t : True := trivial",
            ),
        ),
    ]
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(side_effect=responses)

    out = await run_proposer_native(
        _make_state(),
        model="ollama-native:kimi-k2.6",
        enable_mathlib_search=True,
    )

    assert isinstance(out, ProposalMessage)
    assert route.call_count == 3
    assert invoked == []  # mathlib_search never actually invoked

    # Inspect the error payloads sent back to the model.
    second = json.loads(route.calls[1].request.content)
    tool_msgs = [m for m in second["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    err = json.loads(tool_msgs[0]["content"])
    assert err["success"] is False
    assert "json" in err["error"].lower() or "json" in err["error"]

    third = json.loads(route.calls[2].request.content)
    tool_msgs = [m for m in third["messages"] if m.get("role") == "tool"]
    # All historical tool messages must remain (1 from round 1 + 1 from round 2).
    assert len(tool_msgs) == 2
    err2 = json.loads(tool_msgs[1]["content"])
    assert err2["success"] is False
    assert "query" in err2["error"]


@respx.mock
async def test_tool_call_unknown_function_name(monkeypatch):
    """Model asks for a tool that doesn't exist — surfaces an error
    payload and lets the model retry."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")

    async def fake_mathlib_search(*, query, mode="name", max_results=5):
        return {"success": True, "results": [], "count": 0}

    monkeypatch.setattr(
        "bourbaki.tools.mathlib_search.mathlib_search", fake_mathlib_search
    )

    responses = [
        httpx.Response(
            200,
            json=_tool_call_response([
                {
                    "function": {
                        "name": "bogus_tool",
                        "arguments": {"query": "x"},
                    }
                }
            ]),
        ),
        httpx.Response(
            200,
            json=_make_ok_response(
                reasoning="Recovered.",
                updated_theorem="theorem t : True := trivial",
            ),
        ),
    ]
    route = respx.post(OLLAMA_NATIVE_CHAT_URL).mock(side_effect=responses)

    out = await run_proposer_native(
        _make_state(),
        model="ollama-native:kimi-k2.6",
        enable_mathlib_search=True,
    )

    assert isinstance(out, ProposalMessage)
    assert route.call_count == 2

    second = json.loads(route.calls[1].request.content)
    tool_msgs = [m for m in second["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    err = json.loads(tool_msgs[0]["content"])
    assert err["success"] is False
    assert "bogus_tool" in err["error"]


@respx.mock
async def test_tool_call_arguments_as_json_string(monkeypatch):
    """Some Ollama configs serialize tool_call.arguments as a JSON string;
    we must json.loads them transparently before invoking the tool."""
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "test-key")

    calls: list[dict] = []

    async def fake_mathlib_search(*, query, mode="name", max_results=5):
        calls.append({"query": query, "mode": mode, "max_results": max_results})
        return {"success": True, "results": [], "count": 0}

    monkeypatch.setattr(
        "bourbaki.tools.mathlib_search.mathlib_search", fake_mathlib_search
    )

    responses = [
        httpx.Response(
            200,
            json=_tool_call_response([
                {
                    "function": {
                        "name": "mathlib_search",
                        # Note: arguments is a JSON string, not a dict.
                        "arguments": json.dumps(
                            {"query": "Nat.add_zero", "mode": "name"}
                        ),
                    }
                }
            ]),
        ),
        httpx.Response(
            200,
            json=_make_ok_response(),
        ),
    ]
    respx.post(OLLAMA_NATIVE_CHAT_URL).mock(side_effect=responses)

    out = await run_proposer_native(
        _make_state(),
        model="ollama-native:kimi-k2.6",
        enable_mathlib_search=True,
    )

    assert isinstance(out, ProposalMessage)
    assert calls == [{"query": "Nat.add_zero", "mode": "name", "max_results": 5}]
