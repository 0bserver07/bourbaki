"""Problem database endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from bourbaki.problems.database import (
    ALL_PROBLEMS,
    get_famous_problems,
    get_problem,
    get_problems_by_difficulty,
    get_problems_by_domain,
    get_problems_by_technique,
    get_random_problem,
)

router = APIRouter(prefix="/problems")


@router.get("")
async def list_problems(
    domain: str | None = None,
    technique: str | None = None,
    min_difficulty: int | None = None,
    max_difficulty: int | None = None,
    famous: bool = False,
) -> list[dict[str, Any]]:
    """List problems with optional filters."""
    if famous:
        problems = get_famous_problems()
    elif domain:
        problems = get_problems_by_domain(domain)
    elif technique:
        problems = get_problems_by_technique(technique)
    elif min_difficulty is not None and max_difficulty is not None:
        problems = get_problems_by_difficulty(min_difficulty, max_difficulty)
    else:
        problems = ALL_PROBLEMS
    return [p.to_dict() for p in problems]


@router.get("/random")
async def random_problem(
    domain: str | None = None,
    difficulty: int | None = None,
    technique: str | None = None,
) -> dict[str, Any]:
    """Get a random problem."""
    problem = get_random_problem(domain=domain, difficulty=difficulty, technique=technique)
    if not problem:
        raise HTTPException(status_code=404, detail="No matching problems found")
    return problem.to_dict()


@router.get("/{problem_id}")
async def get_problem_by_id(problem_id: str) -> dict[str, Any]:
    """Get a specific problem by ID."""
    problem = get_problem(problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail=f"Problem '{problem_id}' not found")
    return problem.to_dict()
