"""Test session CRUD — the new session-aware thin-client flow."""


async def test_create_session(client):
    """POST /sessions with JSON body creates a session."""
    resp = await client.post("/sessions", json={"model": "openrouter:test-model"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert "title" in data


async def test_create_session_default_model(client):
    """POST /sessions with empty body uses default model."""
    resp = await client.post("/sessions", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data


async def test_list_sessions(client):
    """GET /sessions returns a list."""
    # Create one first
    await client.post("/sessions", json={"model": "test"})
    resp = await client.get("/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert isinstance(sessions, list)
    assert len(sessions) >= 1
    # Check summary fields
    s = sessions[0]
    assert "id" in s
    assert "title" in s
    assert "messageCount" in s


async def test_get_session(client):
    """GET /sessions/{id} returns full session."""
    create_resp = await client.post("/sessions", json={"model": "test"})
    session_id = create_resp.json()["id"]

    resp = await client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == session_id
    assert "messages" in data
    assert isinstance(data["messages"], list)


async def test_get_session_not_found(client):
    """GET /sessions/{id} returns 404 for missing session."""
    resp = await client.get("/sessions/nonexistent-id")
    assert resp.status_code == 404


async def test_delete_session(client):
    """DELETE /sessions/{id} removes a session."""
    create_resp = await client.post("/sessions", json={"model": "test"})
    session_id = create_resp.json()["id"]

    resp = await client.delete(f"/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # Should be gone now
    resp = await client.get(f"/sessions/{session_id}")
    assert resp.status_code == 404


async def test_get_messages_empty_session(client):
    """GET /sessions/{id}/messages returns empty list for new session."""
    create_resp = await client.post("/sessions", json={"model": "test"})
    session_id = create_resp.json()["id"]

    resp = await client.get(f"/sessions/{session_id}/messages")
    assert resp.status_code == 200
    assert resp.json() == []
