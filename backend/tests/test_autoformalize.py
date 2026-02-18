"""Tests for autoformalize tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from bourbaki.tools.autoformalize import autoformalize, _extract_lean_code


def test_extract_lean_code_from_fenced_block():
    text = "Here is the code:\n```lean\ntheorem foo : 1 + 1 = 2 := by sorry\n```\nDone."
    assert _extract_lean_code(text) == "theorem foo : 1 + 1 = 2 := by sorry"


def test_extract_lean_code_generic_fence():
    text = "```\ntheorem bar : True := trivial\n```"
    assert _extract_lean_code(text) == "theorem bar : True := trivial"


def test_extract_lean_code_no_fence():
    text = "theorem baz : 1 = 1 := rfl"
    assert _extract_lean_code(text) == "theorem baz : 1 = 1 := rfl"


@pytest.mark.asyncio
async def test_autoformalize_statement_success():
    """Statement mode should call LLM and verify with lean_prover."""
    lean_code = "theorem sum_comm (a b : Nat) : a + b = b + a := by sorry"

    with patch("bourbaki.tools.autoformalize._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = f"```lean\n{lean_code}\n```"

        with patch("bourbaki.tools.autoformalize.lean_prover", new_callable=AsyncMock) as mock_prover:
            mock_prover.return_value = {"success": True, "env": 1}

            result = await autoformalize(
                "The sum of natural numbers is commutative",
                mode="statement",
            )

    assert result["success"] is True
    assert result["verified"] is True
    assert result["mode"] == "statement"
    assert "lean_code" in result
    assert "duration" in result


@pytest.mark.asyncio
async def test_autoformalize_statement_retry_on_failure():
    """Statement mode should retry once on Lean verification failure."""
    bad_code = "theorem bad : wrong := by sorry"
    good_code = "theorem good : 1 + 1 = 2 := by sorry"

    call_count = 0

    async def mock_llm(prompt, model=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return bad_code
        return good_code

    prover_call_count = 0

    async def mock_prover(code, mode="check"):
        nonlocal prover_call_count
        prover_call_count += 1
        if prover_call_count == 1:
            return {"success": False, "errors": [{"message": "unknown identifier 'wrong'"}]}
        return {"success": True, "env": 1}

    with patch("bourbaki.tools.autoformalize._call_llm", side_effect=mock_llm):
        with patch("bourbaki.tools.autoformalize.lean_prover", side_effect=mock_prover):
            result = await autoformalize("some math statement", mode="statement")

    assert result["success"] is True
    assert result["verified"] is True
    assert call_count == 2  # Original + retry


@pytest.mark.asyncio
async def test_autoformalize_proof_step():
    """Proof step mode should return a tactic."""
    with patch("bourbaki.tools.autoformalize._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = "ring"

        result = await autoformalize(
            "simplify using the ring axioms",
            mode="proof_step",
            context="⊢ a + b = b + a",
        )

    assert result["success"] is True
    assert result["lean_code"] == "ring"
    assert result["mode"] == "proof_step"


@pytest.mark.asyncio
async def test_autoformalize_unknown_mode():
    """Unknown mode should return error."""
    result = await autoformalize("test", mode="invalid")
    assert result["success"] is False
    assert "Unknown mode" in result["error"]


@pytest.mark.asyncio
async def test_autoformalize_llm_failure():
    """LLM failure should return error."""
    with patch("bourbaki.tools.autoformalize._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None

        result = await autoformalize("test", mode="statement")

    assert result["success"] is False
    assert "LLM failed" in result["error"]
