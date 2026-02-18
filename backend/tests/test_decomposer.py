"""Tests for recursive subgoal decomposer."""

from bourbaki.autonomous.decomposer import (
    DecompositionConfig,
    DecompositionResult,
    SubgoalResult,
    _budget_for_depth,
    _timeout_for_depth,
    _extract_tactics_from_proof,
)


def test_config_defaults():
    config = DecompositionConfig()
    assert config.max_decomposition_depth == 3
    assert config.max_sketches == 3
    assert config.subgoal_search_budget == 50
    assert config.subgoal_search_timeout == 60.0
    assert config.formalization_retries == 2
    assert config.budget_decay_factor == 0.7
    assert config.timeout_decay_factor == 0.7
    assert config.verify_stitched is True
    assert config.parallel_subgoals is True
    assert config.max_parallel == 4


def test_config_legacy_alias():
    """max_recursion_depth should alias max_decomposition_depth."""
    config = DecompositionConfig()
    assert config.max_recursion_depth == 3
    config.max_recursion_depth = 5
    assert config.max_decomposition_depth == 5
    assert config.max_recursion_depth == 5


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


def test_result_verified():
    result = DecompositionResult(
        success=True,
        verified=True,
        proof_code="theorem foo : True := by trivial",
        subgoals_total=1,
        subgoals_solved=1,
    )
    d = result.to_dict()
    assert d["verified"] is True


def test_result_to_dict():
    result = DecompositionResult(
        success=True,
        subgoals_total=1,
        subgoals_solved=1,
        subgoal_results=[
            SubgoalResult(
                label="step_0",
                lean_type="1 + 1 = 2",
                success=True,
                method="search",
                time_spent=1.23,
            ),
        ],
    )
    d = result.to_dict()
    assert d["success"] is True
    assert d["subgoals_total"] == 1
    assert len(d["subgoal_details"]) == 1
    assert d["subgoal_details"][0]["label"] == "step_0"
    assert d["subgoal_details"][0]["method"] == "search"


def test_result_to_dict_no_subgoal_results():
    result = DecompositionResult(success=False, subgoals_total=0, subgoals_solved=0)
    d = result.to_dict()
    assert d["subgoal_details"] == []


def test_budget_for_depth():
    """Budget should decay with depth but never go below 10."""
    assert _budget_for_depth(50, 0, 0.7) == 50
    assert _budget_for_depth(50, 1, 0.7) == 35
    assert _budget_for_depth(50, 2, 0.7) == 24
    assert _budget_for_depth(50, 3, 0.7) == 17
    # Very deep should floor at 10
    assert _budget_for_depth(50, 10, 0.7) >= 10


def test_timeout_for_depth():
    """Timeout should decay with depth but never go below 10.0."""
    assert _timeout_for_depth(60.0, 0, 0.7) == 60.0
    assert abs(_timeout_for_depth(60.0, 1, 0.7) - 42.0) < 0.01
    assert abs(_timeout_for_depth(60.0, 2, 0.7) - 29.4) < 0.01
    # Very deep should floor at 10.0
    assert _timeout_for_depth(60.0, 10, 0.7) >= 10.0


def test_budget_for_depth_no_decay():
    """With decay factor 1.0, budget should stay constant."""
    assert _budget_for_depth(50, 0, 1.0) == 50
    assert _budget_for_depth(50, 3, 1.0) == 50


def test_extract_tactics_from_proof():
    proof = "theorem foo : True := by\n  simp\n  ring"
    tactics = _extract_tactics_from_proof(proof)
    assert tactics == ["simp", "ring"]


def test_extract_tactics_from_proof_single():
    proof = "theorem foo : True := by trivial"
    tactics = _extract_tactics_from_proof(proof)
    assert tactics == ["trivial"]


def test_extract_tactics_skips_comments():
    proof = "theorem foo : True := by\n  -- comment\n  simp\n  -- another\n  ring"
    tactics = _extract_tactics_from_proof(proof)
    assert tactics == ["simp", "ring"]


def test_extract_tactics_no_by():
    proof = "theorem foo : True := trivial"
    tactics = _extract_tactics_from_proof(proof)
    assert len(tactics) >= 1


def test_subgoal_result_creation():
    sr = SubgoalResult(
        label="step_0",
        lean_type="1 + 1 = 2",
        success=True,
        tactics=["norm_num"],
        method="search",
        time_spent=1.5,
    )
    assert sr.success
    assert sr.method == "search"
    assert sr.error is None


def test_subgoal_result_failure():
    sr = SubgoalResult(
        label="step_1",
        lean_type="n * n ≥ 0",
        success=False,
        method="decompose",
        depth_reached=2,
        error="Max decomposition depth reached",
    )
    assert not sr.success
    assert sr.depth_reached == 2
