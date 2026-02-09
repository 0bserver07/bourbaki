"""OEIS sequence lookup via httpx."""

from __future__ import annotations

import time
from typing import Any

import httpx

OEIS_API = "https://oeis.org/search"
USER_AGENT = "Bourbaki/0.1.0 (Mathematical Reasoning Agent)"
TIMEOUT = 10

# Built-in fallback database of common sequences
BUILTIN_SEQUENCES: dict[str, dict[str, Any]] = {
    "A000045": {
        "name": "Fibonacci numbers",
        "terms": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],
        "formula": "F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1",
    },
    "A000040": {
        "name": "The prime numbers",
        "terms": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
        "formula": None,
    },
    "A000079": {
        "name": "Powers of 2",
        "terms": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "formula": "a(n) = 2^n",
    },
    "A000290": {
        "name": "The squares",
        "terms": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144],
        "formula": "a(n) = n^2",
    },
    "A000217": {
        "name": "Triangular numbers",
        "terms": [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105],
        "formula": "a(n) = n*(n+1)/2",
    },
    "A000142": {
        "name": "Factorial numbers",
        "terms": [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880],
        "formula": "a(n) = n!",
    },
    "A000027": {
        "name": "The positive integers",
        "terms": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "formula": "a(n) = n",
    },
    "A000578": {
        "name": "The cubes",
        "terms": [0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000],
        "formula": "a(n) = n^3",
    },
    "A000396": {
        "name": "Perfect numbers",
        "terms": [6, 28, 496, 8128, 33550336],
        "formula": None,
    },
    "A000041": {
        "name": "Number of partitions of n",
        "terms": [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135],
        "formula": None,
    },
    "A000108": {
        "name": "Catalan numbers",
        "terms": [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796],
        "formula": "a(n) = C(2n,n)/(n+1)",
    },
    "A000010": {
        "name": "Euler totient function",
        "terms": [1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8],
        "formula": None,
    },
    "A000005": {
        "name": "Number of divisors",
        "terms": [1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6, 2, 4, 4],
        "formula": None,
    },
    "A000203": {
        "name": "Sum of divisors",
        "terms": [1, 3, 4, 7, 6, 12, 8, 15, 13, 18, 12, 28, 14, 24, 24],
        "formula": None,
    },
}


def _is_subsequence(needle: list[int], haystack: list[int]) -> bool:
    """Check if needle appears as a contiguous subsequence of haystack."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def _search_builtin(terms: list[int]) -> list[dict[str, Any]]:
    """Search the built-in sequence database."""
    matches = []
    for seq_id, seq in BUILTIN_SEQUENCES.items():
        if _is_subsequence(terms, seq["terms"]):
            matches.append({
                "id": seq_id,
                "name": seq["name"],
                "terms": seq["terms"],
                "formula": seq.get("formula"),
            })
    return matches


def _parse_oeis_results(data: dict) -> list[dict[str, Any]]:
    """Parse OEIS JSON response into sequence list."""
    results = data.get("results")
    if not results:
        return []
    matches = []
    for entry in results[:5]:
        terms_str = entry.get("data", "")
        terms = [int(t) for t in terms_str.split(",") if t.strip()] if terms_str else []
        formulae = entry.get("formula", [])
        matches.append({
            "id": f"A{entry['number']:06d}",
            "name": entry.get("name", ""),
            "terms": terms[:20],
            "formula": formulae[0] if formulae else None,
            "comments": (entry.get("comment") or [])[:3],
        })
    return matches


async def sequence_lookup(
    mode: str = "identify",
    terms: list[int] | None = None,
    query: str | None = None,
    id: str | None = None,
) -> dict[str, Any]:
    """Look up integer sequences via OEIS.

    Args:
        mode: 'identify' (match terms), 'search' (text query), or 'get' (by ID).
        terms: Integer sequence terms (required for identify, min 3).
        query: Search text (required for search mode).
        id: OEIS ID like 'A000045' (required for get mode).

    Returns:
        Dict with success, matches, error fields.
    """
    start = time.monotonic()

    if mode == "identify":
        if not terms or len(terms) < 3:
            return {"success": False, "error": "At least 3 terms required for identification"}

        # Try built-in first
        builtin_matches = _search_builtin(terms)
        if builtin_matches:
            elapsed = int((time.monotonic() - start) * 1000)
            return {"success": True, "matches": builtin_matches, "source": "builtin", "duration": elapsed}

        # Query OEIS
        oeis_query = ",".join(str(t) for t in terms)
        return await _fetch_oeis(oeis_query, start)

    if mode == "search":
        if not query:
            return {"success": False, "error": "query parameter required for search mode"}
        return await _fetch_oeis(query, start)

    if mode == "get":
        if not id:
            return {"success": False, "error": "id parameter required for get mode"}
        # Check builtin first
        if id in BUILTIN_SEQUENCES:
            seq = BUILTIN_SEQUENCES[id]
            elapsed = int((time.monotonic() - start) * 1000)
            return {
                "success": True,
                "matches": [{"id": id, **seq}],
                "source": "builtin",
                "duration": elapsed,
            }
        return await _fetch_oeis(id, start)

    return {"success": False, "error": f"Unknown mode: {mode}"}


async def _fetch_oeis(query: str, start: float) -> dict[str, Any]:
    """Fetch from OEIS API."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                OEIS_API,
                params={"fmt": "json", "q": query},
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            data = resp.json()
            matches = _parse_oeis_results(data)
            elapsed = int((time.monotonic() - start) * 1000)
            if matches:
                return {"success": True, "matches": matches, "source": "oeis", "duration": elapsed}
            return {"success": False, "error": "No matches found", "duration": elapsed}
    except httpx.HTTPError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return {"success": False, "error": f"OEIS API error: {e}", "duration": elapsed}
