"""Mathlib search via Loogle (type/name) and LeanSearch (natural language) APIs."""

from __future__ import annotations

import time
from typing import Any

import httpx

LOOGLE_API = "https://loogle.lean-lang.org/json"
LEANSEARCH_API = "https://leansearch.net/api/search"
USER_AGENT = "Bourbaki/0.1.0 (Mathematical Reasoning Agent)"
TIMEOUT = 15


async def mathlib_search(
    query: str,
    mode: str = "name",
    max_results: int = 5,
) -> dict[str, Any]:
    """Search Mathlib for lemmas by name, type signature, or natural language.

    Args:
        query: Search query. For mode="name" or "type", use Loogle syntax
               (e.g. "Nat.add_comm", "_ * (_ ^ _)", "(?a -> ?b) -> List ?a -> List ?b").
               For mode="natural", use plain English (e.g. "product of positive numbers is positive").
        mode: "name" or "type" for Loogle API, "natural" for LeanSearch API.
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
    else:
        return {
            "success": False,
            "error": f"Unknown mode: {mode!r}. Use 'name', 'type', or 'natural'.",
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
