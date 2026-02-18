"""Tests for mathlib_search — including semantic mode via LeanExplore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bourbaki.tools.mathlib_search import mathlib_search


def _make_mock_response(data, status_code=200):
    """Create a mock httpx.Response with .json() returning data."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    resp.text = ""
    return resp


@pytest.mark.asyncio
async def test_mathlib_search_semantic_mode_success():
    """Semantic mode should call LeanExplore API and parse results."""
    mock_response_data = {
        "query": "product of positive numbers",
        "results": [
            {
                "id": 42,
                "name": "Nat.mul_pos",
                "module": "Mathlib.Data.Nat.Basic",
                "source_text": "theorem Nat.mul_pos : 0 < m → 0 < n → 0 < m * n",
                "source_link": "https://github.com/leanprover-community/mathlib4/blob/main/...",
                "docstring": "The product of two positive naturals is positive.",
                "dependencies": None,
                "informalization": "If m and n are positive, then m * n is positive.",
            },
            {
                "id": 99,
                "name": "Int.mul_pos",
                "module": "Mathlib.Data.Int.Basic",
                "source_text": "theorem Int.mul_pos : 0 < m → 0 < n → 0 < m * n",
                "source_link": "https://github.com/...",
                "docstring": None,
                "dependencies": None,
                "informalization": None,
            },
        ],
        "count": 2,
        "processing_time_ms": 55,
    }

    mock_resp = _make_mock_response(mock_response_data)

    with patch("bourbaki.tools.mathlib_search.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        with patch("bourbaki.tools.mathlib_search._get_leanexplore_api_key", return_value="test-key"):
            result = await mathlib_search("product of positive numbers", mode="semantic")

    assert result["success"] is True
    assert result["mode"] == "semantic"
    assert result["count"] == 2
    assert len(result["results"]) == 2
    assert result["results"][0]["name"] == "Nat.mul_pos"
    assert result["results"][0]["module"] == "Mathlib.Data.Nat.Basic"
    assert result["results"][0]["type"] == "theorem Nat.mul_pos : 0 < m → 0 < n → 0 < m * n"
    assert result["results"][0]["doc"] == "The product of two positive naturals is positive."
    # Second result has no docstring, should fall back to informalization (also None)
    assert result["results"][1]["doc"] == ""
    assert "duration" in result


@pytest.mark.asyncio
async def test_mathlib_search_semantic_fallback_to_leansearch():
    """When LeanExplore fails, semantic mode should fall back to LeanSearch."""
    # LeanExplore fails with HTTP error
    leanexplore_resp = MagicMock()
    leanexplore_resp.status_code = 500
    leanexplore_resp.text = "Internal Server Error"
    leanexplore_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=leanexplore_resp,
    )

    # LeanSearch succeeds
    leansearch_data = [
        {
            "name": "Nat.mul_pos",
            "module": "Mathlib.Data.Nat.Basic",
            "type": "0 < m → 0 < n → 0 < m * n",
            "doc": "Product of positives is positive.",
        }
    ]
    leansearch_resp = _make_mock_response(leansearch_data)

    async def mock_get(url, **kwargs):
        if "leanexplore" in url:
            return leanexplore_resp
        return leansearch_resp

    with patch("bourbaki.tools.mathlib_search.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        with patch("bourbaki.tools.mathlib_search._get_leanexplore_api_key", return_value="test-key"):
            result = await mathlib_search("product of positive numbers", mode="semantic")

    assert result["success"] is True
    assert result["count"] >= 1


@pytest.mark.asyncio
async def test_mathlib_search_semantic_no_api_key():
    """Without API key, semantic mode should fall back to LeanSearch directly."""
    leansearch_data = [
        {
            "name": "List.length_append",
            "module": "Mathlib.Data.List.Basic",
            "type": "(l₁ ++ l₂).length = l₁.length + l₂.length",
            "doc": "",
        }
    ]
    leansearch_resp = _make_mock_response(leansearch_data)

    with patch("bourbaki.tools.mathlib_search.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = leansearch_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        with patch("bourbaki.tools.mathlib_search._get_leanexplore_api_key", return_value=None):
            result = await mathlib_search("list append length", mode="semantic")

    assert result["success"] is True
    assert result["count"] >= 1


@pytest.mark.asyncio
async def test_mathlib_search_unknown_mode():
    """Unknown modes should return an error."""
    result = await mathlib_search("test", mode="invalid")
    assert result["success"] is False
    assert "Unknown mode" in result["error"]
