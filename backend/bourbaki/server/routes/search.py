"""POST /search — standalone sequence lookup and paper search endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from bourbaki.tools.paper_search import paper_search
from bourbaki.tools.sequence_lookup import sequence_lookup

router = APIRouter()


class SequenceRequest(BaseModel):
    mode: str = "identify"
    terms: list[int] | None = None
    query: str | None = None
    id: str | None = None


class PaperRequest(BaseModel):
    mode: str = "search"
    query: str | None = None
    arxiv_id: str | None = None
    category: str | None = None
    max_results: int = 5


@router.post("/search/sequence")
async def search_sequence(req: SequenceRequest) -> dict[str, Any]:
    """Look up integer sequences via OEIS."""
    return await sequence_lookup(
        mode=req.mode,
        terms=req.terms,
        query=req.query,
        id=req.id,
    )


@router.post("/search/paper")
async def search_paper(req: PaperRequest) -> dict[str, Any]:
    """Search arXiv for mathematical papers."""
    return await paper_search(
        mode=req.mode,
        query=req.query,
        arxiv_id=req.arxiv_id,
        category=req.category,
        max_results=req.max_results,
    )
