"""Health check endpoint."""

import shutil
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Return backend health status."""
    lean_available = shutil.which("lean") is not None
    return {
        "status": "ok",
        "python": True,
        "lean": lean_available,
    }
