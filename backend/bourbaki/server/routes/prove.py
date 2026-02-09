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
    timeout: int = 30


@router.post("/prove")
async def prove(req: ProveRequest) -> dict[str, Any]:
    """Verify Lean 4 code."""
    return await lean_prover(code=req.code, mode=req.mode, timeout=req.timeout)
