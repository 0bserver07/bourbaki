"""POST /compute — standalone symbolic computation endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from bourbaki.tools.symbolic_compute import symbolic_compute

router = APIRouter()


class ComputeRequest(BaseModel):
    operation: str
    expression: str = ""
    variable: str | None = None
    from_val: str | int | None = None
    to_val: str | int | None = None
    point: str | int | None = None
    matrix: list[list[float]] | None = None
    matrix2: list[list[float]] | None = None
    order: int = 6


@router.post("/compute")
async def compute(req: ComputeRequest) -> dict[str, Any]:
    """Run a symbolic computation."""
    return symbolic_compute(
        operation=req.operation,
        expression=req.expression,
        variable=req.variable,
        from_val=req.from_val,
        to_val=req.to_val,
        point=req.point,
        matrix=req.matrix,
        matrix2=req.matrix2,
        order=req.order,
    )
