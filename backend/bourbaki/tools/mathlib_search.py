"""Mathlib search via Loogle (type/name), LeanSearch (natural language), LeanExplore (semantic), and local FAISS index."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

LOOGLE_API = "https://loogle.lean-lang.org/json"
LEANSEARCH_API = "https://leansearch.net/api/search"
LEANEXPLORE_API = "https://www.leanexplore.com/api/v2/search"
USER_AGENT = "Bourbaki/0.1.0 (Mathematical Reasoning Agent)"
TIMEOUT = 15


def _get_leanexplore_api_key() -> str | None:
    """Get LeanExplore API key from config or environment."""
    from bourbaki.config import settings
    return settings.leanexplore_api_key or os.environ.get("LEANEXPLORE_API_KEY")


async def mathlib_search(
    query: str,
    mode: str = "name",
    max_results: int = 5,
) -> dict[str, Any]:
    """Search Mathlib for lemmas by name, type signature, natural language, or semantic search.

    Args:
        query: Search query. For mode="name" or "type", use Loogle syntax
               (e.g. "Nat.add_comm", "_ * (_ ^ _)", "(?a -> ?b) -> List ?a -> List ?b").
               For mode="natural", use plain English (e.g. "product of positive numbers is positive").
               For mode="semantic", use natural language (uses LeanExplore's hybrid ranking).
               For mode="local", use natural language (uses local FAISS embedding index).
        mode: "name" or "type" for Loogle API, "natural" for LeanSearch API,
              "semantic" for LeanExplore API (hybrid semantic + BM25 + PageRank),
              "local" for offline FAISS embedding search (fastest, no network).
        max_results: Maximum results to return (default 5, max 10).

    Returns:
        Dict with success, results, query, mode, duration fields.
    """
    start = time.monotonic()
    max_results = min(max_results, 10)

    if mode in ("name", "type"):
        return await _search_loogle(query, max_results, mode, start)
    elif mode == "natural":
        return await _search_leansearch(query, max_results, start)
    elif mode == "semantic":
        return await _search_semantic(query, max_results, start)
    elif mode == "local":
        return await _search_local(query, max_results, start)
    else:
        return {
            "success": False,
            "error": (
                f"Unknown mode: {mode!r}. "
                "Use 'name', 'type', 'natural', 'semantic', or 'local'."
            ),
            "query": query,
            "mode": mode,
        }


async def _search_loogle(
    query: str, max_results: int, mode: str, start: float,
) -> dict[str, Any]:
    """Search Mathlib via Loogle (name/type signature search)."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                LOOGLE_API,
                params={"q": query},
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            data = resp.json()

        # Loogle returns {count, hits: [{name, module, type, doc}]}
        # or {error: "..."} on bad queries
        if "error" in data:
            elapsed = int((time.monotonic() - start) * 1000)
            return {
                "success": False,
                "error": f"Loogle: {data['error']}",
                "query": query,
                "mode": mode,
                "duration": elapsed,
            }

        hits = data.get("hits", [])[:max_results]
        results = []
        for hit in hits:
            results.append({
                "name": hit.get("name", ""),
                "module": hit.get("module", ""),
                "type": hit.get("type", ""),
                "doc": hit.get("doc", ""),
            })

        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "query": query,
            "mode": mode,
            "duration": elapsed,
        }

    except httpx.HTTPStatusError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        body = e.response.text[:200] if e.response else ""
        return {
            "success": False,
            "error": f"Loogle API error {e.response.status_code}: {body}",
            "query": query,
            "mode": mode,
            "duration": elapsed,
        }
    except httpx.HTTPError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": f"Loogle request failed: {e}",
            "query": query,
            "mode": mode,
            "duration": elapsed,
        }


async def _search_leansearch(
    query: str, max_results: int, start: float,
) -> dict[str, Any]:
    """Search Mathlib via LeanSearch (natural language search)."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                LEANSEARCH_API,
                params={"query": query, "results": max_results},
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            data = resp.json()

        # LeanSearch returns a list of results
        hits = data if isinstance(data, list) else data.get("results", [])
        results = []
        for hit in hits[:max_results]:
            results.append({
                "name": hit.get("name", ""),
                "module": hit.get("module", ""),
                "type": hit.get("type", ""),
                "doc": hit.get("doc", hit.get("docstring", "")),
            })

        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "query": query,
            "mode": "natural",
            "duration": elapsed,
        }

    except httpx.HTTPStatusError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        body = e.response.text[:200] if e.response else ""
        return {
            "success": False,
            "error": f"LeanSearch API error {e.response.status_code}: {body}",
            "query": query,
            "mode": "natural",
            "duration": elapsed,
        }
    except httpx.HTTPError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": f"LeanSearch request failed: {e}",
            "query": query,
            "mode": "natural",
            "duration": elapsed,
        }


async def _search_leanexplore(
    query: str, max_results: int, start: float,
) -> dict[str, Any]:
    """Search Mathlib via LeanExplore (semantic + BM25 + PageRank hybrid)."""
    api_key = _get_leanexplore_api_key()
    if api_key is None:
        return None  # type: ignore[return-value]

    try:
        headers = {
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {api_key}",
        }
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                LEANEXPLORE_API,
                params={"q": query, "limit": max_results},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        raw_results = data.get("results", [])[:max_results]
        results = []
        for hit in raw_results:
            results.append({
                "name": hit.get("name", ""),
                "module": hit.get("module", ""),
                # source_text serves as the type signature
                "type": hit.get("source_text", ""),
                "doc": hit.get("docstring") or hit.get("informalization") or "",
            })

        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "query": query,
            "mode": "semantic",
            "duration": elapsed,
        }

    except (httpx.HTTPStatusError, httpx.HTTPError):
        return None  # type: ignore[return-value]


async def _search_local(
    query: str, max_results: int, start: float,
) -> dict[str, Any]:
    """Search using the local FAISS embedding index (offline, fastest)."""
    from bourbaki.tools.mathlib_embeddings import search_local
    return await search_local(query, max_results)


async def _search_semantic(
    query: str, max_results: int, start: float,
) -> dict[str, Any]:
    """Semantic search: try local FAISS first, then LeanExplore, then LeanSearch."""
    # 1. Try local FAISS index (fastest, no network)
    try:
        from bourbaki.tools.mathlib_embeddings import is_index_available, search_local

        if is_index_available():
            result = await search_local(query, max_results)
            if result.get("success"):
                # Override mode to indicate it came through the semantic pathway
                result["mode"] = "semantic"
                logger.debug("Semantic search served by local FAISS index")
                return result
    except Exception as e:
        logger.debug("Local FAISS index not available: %s", e)

    # 2. Try LeanExplore if API key is available
    result = await _search_leanexplore(query, max_results, start)
    if result is not None:
        return result

    # 3. Fallback to LeanSearch (natural language)
    return await _search_leansearch(query, max_results, start)
