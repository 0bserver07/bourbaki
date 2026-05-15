"""POST /prove — standalone Lean 4 proof verification endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from bourbaki.tools.lean_prover import lean_prover

router = APIRouter()


class ProveRequest(BaseModel):
    code: str
    mode: str = "check"
    # Aligned with lean_prover's 240s default — see HANDOFF.md §4 #2.
    # A cold-cache `lake env lean + import Mathlib` takes 60-180s on a
    # quiet box; the prior 30s default silently timed out before Mathlib
    # finished loading, same #19-class footgun as the pre-7b07c07 reviewer.
    timeout: int = 240


@router.post("/prove")
async def prove(req: ProveRequest) -> dict[str, Any]:
    """Verify Lean 4 code."""
    return await lean_prover(code=req.code, mode=req.mode, timeout=req.timeout)
