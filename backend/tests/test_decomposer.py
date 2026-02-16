"""Tests for recursive subgoal decomposer."""

from bourbaki.autonomous.decomposer import (
    DecompositionConfig,
    DecompositionResult,
)


def test_config_defaults():
    config = DecompositionConfig()
    assert config.max_recursion_depth == 2
    assert config.max_sketches == 3
    assert config.subgoal_search_budget == 50
    assert config.subgoal_search_timeout == 60.0
    assert config.formalization_retries == 2


def test_result_success():
    result = DecompositionResult(
        success=True,
        proof_code="theorem foo : True := by trivial",
        subgoals_total=2,
        subgoals_solved=2,
    )
    assert result.success
    assert result.all_solved


def test_result_partial():
    result = DecompositionResult(
        success=False,
        subgoals_total=3,
        subgoals_solved=1,
        failed_subgoals=["step_1", "step_2"],
    )
    assert not result.success
    assert not result.all_solved
    assert result.solve_rate == 1 / 3


def test_result_to_dict():
    result = DecompositionResult(success=True, subgoals_total=1, subgoals_solved=1)
    d = result.to_dict()
    assert d["success"] is True
    assert d["subgoals_total"] == 1
