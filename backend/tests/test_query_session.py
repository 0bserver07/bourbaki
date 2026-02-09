"""Integration test — session-aware /query with OpenRouter.

This test hits a real LLM (DeepSeek R1 via OpenRouter free tier).
Requires OPENROUTER_API_KEY in .env.

Run with: pytest tests/test_query_session.py -v
"""

from __future__ import annotations

import os

import pytest

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
skip_no_key = pytest.mark.skipif(not OPENROUTER_KEY, reason="OPENROUTER_API_KEY not set")


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    import json
    events = []
    current_event = None
    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: ") and current_event:
            try:
                data = json.loads(line[6:])
                events.append({"event": current_event, "data": data})
            except json.JSONDecodeError:
                pass
            current_event = None
    return events


@skip_no_key
async def test_session_query_flow(client):
    """Full flow: create session → query → verify persistence."""
    model = "openrouter:nvidia/nemotron-nano-9b-v2:free"

    # 1. Create session
    resp = await client.post("/sessions", json={"model": model})
    assert resp.status_code == 200
    session_id = resp.json()["id"]

    # 2. Send a simple query with session_id
    resp = await client.post("/query", json={
        "query": "Is 17 prime? Answer in one word.",
        "model": model,
        "session_id": session_id,
    })
    assert resp.status_code == 200

    # 3. Parse SSE stream
    events = _parse_sse(resp.text)
    event_types = [e["event"] for e in events]

    # Must have a done event
    assert "done" in event_types, f"No 'done' event in: {event_types}"

    # Extract the answer
    done_events = [e for e in events if e["event"] == "done"]
    answer = done_events[0]["data"].get("answer", "")
    assert len(answer) > 0, "Empty answer from agent"

    # 4. Verify session persisted both messages
    resp = await client.get(f"/sessions/{session_id}/messages")
    assert resp.status_code == 200
    messages = resp.json()

    # Should have at least user + assistant
    assert len(messages) >= 2, f"Expected >=2 messages, got {len(messages)}: {messages}"

    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles

    # User message should be our query
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert "17" in user_msgs[0]["content"]


@skip_no_key
async def test_multi_turn_context(client):
    """Two queries on same session — second should have context from first."""
    model = "openrouter:nvidia/nemotron-nano-9b-v2:free"

    # Create session
    resp = await client.post("/sessions", json={"model": model})
    session_id = resp.json()["id"]

    # First query — establish context
    await client.post("/query", json={
        "query": "Remember this number: 42. Just say OK.",
        "model": model,
        "session_id": session_id,
    })

    # Second query — test context recall
    resp = await client.post("/query", json={
        "query": "What number did I ask you to remember?",
        "model": model,
        "session_id": session_id,
    })
    events = _parse_sse(resp.text)
    done_events = [e for e in events if e["event"] == "done"]
    assert len(done_events) > 0
    answer = done_events[0]["data"].get("answer", "")
    assert "42" in answer, f"Expected '42' in answer, got: {answer}"

    # Verify 4 messages total (2 user + 2 assistant)
    resp = await client.get(f"/sessions/{session_id}/messages")
    messages = resp.json()
    assert len(messages) >= 4, f"Expected >=4 messages, got {len(messages)}"
