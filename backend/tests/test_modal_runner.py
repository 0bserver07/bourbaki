"""Tests for modal_runner strategy response parsing and verification."""

from bourbaki.autonomous.modal_runner import _parse_strategy_response


def test_parse_success_true():
    text = """```lean4
theorem foo : True := by trivial
```

**INSIGHT:** Direct proof works
**SUCCESS:** true
**PARTIAL_PROGRESS:** none"""
    result = _parse_strategy_response(text, "direct-proof", 100)
    assert result.success is True
    assert result.proof_code is not None
    assert "trivial" in result.proof_code
    assert result.insight == "Direct proof works"


def test_parse_success_false():
    text = """```lean4
-- attempt failed
```

**INSIGHT:** Induction didn't work
**SUCCESS:** false
**PARTIAL_PROGRESS:** Got base case"""
    result = _parse_strategy_response(text, "induction", 200)
    assert result.success is False
    assert result.partial_progress == "Got base case"


def test_parse_no_code_block():
    text = "I couldn't find a proof.\n**SUCCESS:** false"
    result = _parse_strategy_response(text, "direct-proof", 50)
    assert result.success is False
    assert result.proof_code is None


def test_verified_flag_default_false():
    """Strategy results should track whether code was verified against Lean."""
    text = "```lean4\nsorry\n```\n**SUCCESS:** true"
    result = _parse_strategy_response(text, "test", 100)
    assert result.verified is False
