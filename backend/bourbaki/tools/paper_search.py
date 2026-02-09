"""arXiv paper search via httpx."""

from __future__ import annotations

import re
import time
from typing import Any
from xml.etree import ElementTree

import httpx

ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "Bourbaki/0.1.0 (Mathematical Reasoning Agent)"
TIMEOUT = 15

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

MATH_CATEGORIES = [
    "math.NT", "math.CO", "math.AG", "math.CA", "math.LO",
    "math.PR", "math.GR", "math.AT", "math.RT", "math.FA",
    "math.DG", "math.AP", "math.OA", "math.QA",
]


def _extract_id(entry_id: str) -> str:
    """Extract arXiv ID from full URL, removing version suffix."""
    # http://arxiv.org/abs/2301.12345v1 → 2301.12345
    match = re.search(r"(\d{4}\.\d{4,5})", entry_id)
    return match.group(1) if match else entry_id


def _text(element: ElementTree.Element | None) -> str:
    """Safely get text content from an XML element."""
    if element is None:
        return ""
    return (element.text or "").strip()


def _parse_entry(entry: ElementTree.Element) -> dict[str, Any]:
    """Parse a single Atom entry into a paper dict."""
    arxiv_id = _extract_id(_text(entry.find(f"{ATOM_NS}id")))

    authors = []
    for author_el in entry.findall(f"{ATOM_NS}author"):
        name = _text(author_el.find(f"{ATOM_NS}name"))
        if name:
            authors.append(name)

    categories = []
    for cat_el in entry.findall(f"{ARXIV_NS}primary_category"):
        term = cat_el.get("term", "")
        if term:
            categories.append(term)
    for cat_el in entry.findall(f"{ATOM_NS}category"):
        term = cat_el.get("term", "")
        if term and term not in categories:
            categories.append(term)

    abstract = _text(entry.find(f"{ATOM_NS}summary"))
    # Normalize whitespace in abstract
    abstract = re.sub(r"\s+", " ", abstract)

    return {
        "id": arxiv_id,
        "title": re.sub(r"\s+", " ", _text(entry.find(f"{ATOM_NS}title"))),
        "authors": authors[:3] + (["et al."] if len(authors) > 3 else []),
        "abstract": abstract[:300] + ("..." if len(abstract) > 300 else ""),
        "published": _text(entry.find(f"{ATOM_NS}published"))[:10],
        "updated": _text(entry.find(f"{ATOM_NS}updated"))[:10],
        "categories": categories,
        "pdfUrl": f"https://arxiv.org/pdf/{arxiv_id}",
        "arxivUrl": f"https://arxiv.org/abs/{arxiv_id}",
    }


async def paper_search(
    mode: str = "search",
    query: str | None = None,
    arxiv_id: str | None = None,
    category: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search arXiv for mathematical papers.

    Args:
        mode: 'search' for keyword search, 'get' for specific paper.
        query: Search terms (required for search mode).
        arxiv_id: arXiv paper ID like '2301.12345' (required for get mode).
        category: Optional math category filter (e.g. 'math.NT').
        max_results: Maximum results to return (default 5).

    Returns:
        Dict with success, count, papers, error fields.
    """
    start = time.monotonic()

    if mode == "get":
        if not arxiv_id:
            return {"success": False, "error": "arxiv_id required for get mode"}
        search_query = f"id:{arxiv_id}"
        max_results = 1
    elif mode == "search":
        if not query:
            return {"success": False, "error": "query required for search mode"}
        search_query = f"all:{query}"
        if category:
            search_query += f" AND cat:{category}"
    else:
        return {"success": False, "error": f"Unknown mode: {mode}"}

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                ARXIV_API,
                params={
                    "search_query": search_query,
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()

            root = ElementTree.fromstring(resp.text)
            entries = root.findall(f"{ATOM_NS}entry")

            papers = [_parse_entry(e) for e in entries]
            # Filter out error entries (arXiv returns error as an entry)
            papers = [p for p in papers if p["id"] and p["title"]]

            elapsed = int((time.monotonic() - start) * 1000)
            return {
                "success": True,
                "count": len(papers),
                "papers": papers,
                "duration": elapsed,
            }
    except httpx.HTTPError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return {"success": False, "error": f"arXiv API error: {e}", "duration": elapsed}
    except ElementTree.ParseError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return {"success": False, "error": f"Failed to parse arXiv response: {e}", "duration": elapsed}
