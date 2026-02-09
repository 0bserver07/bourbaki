"""GET /skills — List available proof technique skills."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from bourbaki.skills.registry import discover_skills

router = APIRouter(tags=["skills"])


@router.get("/skills")
async def list_skills() -> list[dict[str, Any]]:
    """List all available proof technique skills."""
    skills = discover_skills()
    return [
        {
            "name": s.name,
            "description": s.description,
            "source": s.source,
        }
        for s in sorted(skills, key=lambda s: s.name)
    ]
