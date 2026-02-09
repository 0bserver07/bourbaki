"""Autonomous proof search endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bourbaki.autonomous.search import AutonomousSearch, AutonomousSearchConfig

router = APIRouter(prefix="/autonomous")

# Singleton search instance
_search = AutonomousSearch()


class StartSearchRequest(BaseModel):
    problem: dict[str, Any]
    max_iterations: int = 100
    max_hours: float = 4.0
    strategies: list[str] | None = None
    checkpoint_interval: int = 10


class ResumeRequest(BaseModel):
    session_id: str


@router.post("/start")
async def start_search(req: StartSearchRequest) -> dict[str, Any]:
    """Start an autonomous proof search."""
    config = AutonomousSearchConfig(
        max_iterations=req.max_iterations,
        max_hours=req.max_hours,
        strategies=req.strategies,
        checkpoint_interval=req.checkpoint_interval,
    )
    await _search.start(req.problem, config)
    return _search.get_progress().to_dict()


@router.post("/pause")
async def pause_search() -> dict[str, str]:
    """Pause the current search."""
    _search.pause()
    return {"status": "paused"}


@router.post("/resume")
async def resume_search(req: ResumeRequest) -> dict[str, Any]:
    """Resume a search from checkpoint."""
    success = await _search.resume(req.session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Could not resume session {req.session_id}")
    return _search.get_progress().to_dict()


@router.get("/progress")
async def get_progress() -> dict[str, Any]:
    """Get current search progress."""
    return _search.get_progress().to_dict()


@router.get("/insights")
async def get_insights() -> list[str]:
    """Get accumulated insights."""
    return _search.insights
