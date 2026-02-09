"""Session management endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bourbaki.sessions.manager import session_manager

router = APIRouter(prefix="/sessions")


class CreateSessionRequest(BaseModel):
    model: str = "unknown"


@router.get("")
async def list_sessions() -> list[dict[str, Any]]:
    """List all sessions (most recent first)."""
    return session_manager.list_sessions()


@router.post("")
async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
    """Create a new session."""
    session = session_manager.start_session(req.model)
    return {"id": session.id, "title": session.title}


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Load a session by ID."""
    session = session_manager.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session.to_dict()


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, bool]:
    """Delete a session."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"deleted": deleted}


@router.get("/{session_id}/messages")
async def get_messages(session_id: str) -> list[dict[str, str]]:
    """Get messages for a session (for context)."""
    session = session_manager.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session_manager.get_messages_for_context(session)
