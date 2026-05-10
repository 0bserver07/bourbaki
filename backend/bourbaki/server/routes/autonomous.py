"""Autonomous proof search endpoints.

Phase 3 deprecation: the legacy autonomous proof search pipeline
(`bourbaki.autonomous.search.AutonomousSearch`, ``decomposer``, ``search_tree``,
``sketch``, ``formalizer``, ``scoring``, ``strategies``) has been removed in
favour of the proposer-builder-reviewer loop in ``bourbaki.prover``.

The ``/autonomous/*`` route paths remain registered so the TUI does not 404,
but every handler returns HTTP 410 Gone with a deprecation message pointing
clients at ``/query`` with ``use_loop=True`` (or the
``bourbaki.benchmarks.minif2f.attempt_proof_loop`` driver) instead.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/autonomous")

_DEPRECATION_BODY: dict[str, Any] = {
    "error": (
        "autonomous search route deprecated; use /query with use_loop instead"
    ),
}
_DEPRECATION_STATUS = 410  # Gone


def _deprecated() -> JSONResponse:
    """Return the standard 410 Gone deprecation response."""
    return JSONResponse(status_code=_DEPRECATION_STATUS, content=_DEPRECATION_BODY)


# Request models retained so OpenAPI schema and any TUI typing stays stable.
class StartSearchRequest(BaseModel):
    problem: dict[str, Any]
    max_iterations: int = 100
    max_hours: float = 4.0
    strategies: list[str] | None = None
    checkpoint_interval: int = 10


class ResumeRequest(BaseModel):
    session_id: str


@router.post("/start")
async def start_search(req: StartSearchRequest) -> JSONResponse:
    """Deprecated: use /query with use_loop=True."""
    return _deprecated()


@router.post("/pause")
async def pause_search() -> JSONResponse:
    """Deprecated: use /query with use_loop=True."""
    return _deprecated()


@router.post("/resume")
async def resume_search(req: ResumeRequest) -> JSONResponse:
    """Deprecated: use /query with use_loop=True."""
    return _deprecated()


@router.get("/progress")
async def get_progress() -> JSONResponse:
    """Deprecated: use /query with use_loop=True."""
    return _deprecated()


@router.get("/insights")
async def get_insights() -> JSONResponse:
    """Deprecated: use /query with use_loop=True."""
    return _deprecated()
