"""Web search via Exa API — academic and general web search for mathematics."""

from __future__ import annotations

import time
from typing import Any

import httpx

from bourbaki.config import settings

EXA_API = "https://api.exa.ai"
TIMEOUT = 15


async def web_search(
    query: str,
    num_results: int = 5,
    category: str = "research paper",
) -> dict[str, Any]:
    """Search the web for mathematical content using Exa.

    Args:
        query: Search query (e.g. "proof of Pythagorean theorem", "Lean 4 formalization").
        num_results: Maximum results to return (default 5, max 10).
        category: Exa search category — "research paper", "tweet", "company",
                  "news", "github", "pdf", or None for general web.

    Returns:
        Dict with success, count, results, error fields.
    """
    start = time.monotonic()

    api_key = settings.exasearch_api_key
    if not api_key:
        return {
            "success": False,
            "error": "EXA_API_KEY / EXASEARCH_API_KEY not configured. "
                     "Set it in .env to enable web search.",
        }

    num_results = min(num_results, 10)

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{EXA_API}/search",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "numResults": num_results,
                    "type": "auto",
                    "category": category,
                    "contents": {
                        "text": {"maxCharacters": 500},
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "text": (item.get("text", "") or "")[:500],
                "publishedDate": item.get("publishedDate", ""),
                "author": item.get("author", ""),
                "score": item.get("score"),
            })

        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": True,
            "count": len(results),
            "results": results,
            "duration": elapsed,
        }

    except httpx.HTTPStatusError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        body = e.response.text[:200] if e.response else ""
        return {
            "success": False,
            "error": f"Exa API error {e.response.status_code}: {body}",
            "duration": elapsed,
        }
    except httpx.HTTPError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": f"Exa request failed: {e}",
            "duration": elapsed,
        }
