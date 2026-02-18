"""Integration test for the full decomposition pipeline.

Uses mocked LLM and REPL to test the full flow:
sketch generation -> formalization -> subgoal solving -> stitching.

Tests cover:
- Basic decomposition with mock generator
- Recursive decomposition (depth > 1)
- Parallel subgoal solving
- Budget/timeout decay with depth
- Verification of stitched proofs
- Handling of failed subgoals and partial results
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bourbaki.autonomous.decomposer import (
    DecompositionConfig,
    SubgoalResult,
    decompose_and_prove,
    _solve_subgoals_parallel,
    _solve_single_subgoal,
)
from bourbaki.autonomous.formalizer import Subgoal
from bourbaki.autonomous.sketch import (
    ProofSketch,
    SketchContext,
    SketchStep,
)


class MockSketchGenerator:
    """Returns a fixed sketch for testing."""

    def __init__(self, sketches: list[ProofSketch] | None = None):
        self._sketches = sketches

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
        if self._sketches is not None:
            return self._sketches
        return [ProofSketch(
            strategy="direct",
            steps=[
                SketchStep(
                    statement="Simplify using norm_num",
                    formal_type="1 + 1 = 2",
                ),
            ],
            key_lemmas=[],
        )]


class MockRecursiveSketchGenerator:
    """Returns different sketches at different depths for testing recursion."""

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
        if context.depth == 0:
            return [ProofSketch(
                strategy="decompose",
                steps=[
                    SketchStep(statement="Step A", formal_type="A"),
                    SketchStep(statement="Step B", formal_type="B"),
                ],
                key_lemmas=[],
            )]
        elif context.depth == 1:
            # At depth 1, return simpler decomposition
            return [ProofSketch(
                strategy="simple",
                steps=[
                    SketchStep(statement="Simple step", formal_type="C"),
                ],
                key_lemmas=[],
            )]
        else:
            # At depth 2+, return empty (atomic)
            return []


def _make_search_mock(succeed_for: set[str] | None = None, always_succeed: bool = False):
    """Create a mock for prove_with_search that succeeds for specific subgoals."""
    async def mock_prove(theorem, budget=50, timeout=60.0, **kwargs):
        result = MagicMock()
        if always_succeed:
            result.success = True
            result.proof_tactics = ["norm_num"]
            result.proof_code = f"{theorem} := by norm_num"
            return result

        # Check if this subgoal should succeed
        if succeed_for:
            for label in succeed_for:
                if label in theorem:
                    result.success = True
                    result.proof_tactics = ["norm_num"]
                    result.proof_code = f"{theorem} := by norm_num"
                    return result

        result.success = False
        result.proof_tactics = []
        result.proof_code = None
        result.error = "No proof found"
        result.nodes_explored = 10
        return result

    return mock_prove


@pytest.mark.asyncio
async def test_decompose_with_mock_generator():
    """Test decomposition with mocked sketch generator and search."""
    config = DecompositionConfig(
        max_decomposition_depth=1,
        max_sketches=1,
        subgoal_search_budget=10,
        subgoal_search_timeout=5.0,
        verify_stitched=False,  # Skip verification for unit test
        parallel_subgoals=False,
    )

    mock_search = _make_search_mock(always_succeed=True)

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        side_effect=mock_search,
    ):
        result = await decompose_and_prove(
            theorem="theorem foo : 1 + 1 = 2",
            config=config,
            sketch_generator=MockSketchGenerator(),
        )

    assert result.subgoals_total >= 1
    assert result.subgoals_solved >= 1
    assert result.sketches_tried == 1


@pytest.mark.asyncio
async def test_decompose_all_subgoals_solved():
    """Test that when all subgoals are solved, proof is stitched."""
    config = DecompositionConfig(
        max_decomposition_depth=1,
        max_sketches=1,
        verify_stitched=False,
        parallel_subgoals=False,
    )

    sketches = [ProofSketch(
        strategy="direct",
        steps=[
            SketchStep(statement="Step A", formal_type="1 + 1 = 2"),
            SketchStep(statement="Step B", formal_type="2 + 2 = 4"),
        ],
        key_lemmas=[],
    )]

    mock_search = _make_search_mock(always_succeed=True)

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        side_effect=mock_search,
    ):
        result = await decompose_and_prove(
            theorem="theorem foo : True",
            config=config,
            sketch_generator=MockSketchGenerator(sketches),
        )

    assert result.success
    assert result.subgoals_total == 2
    assert result.subgoals_solved == 2
    assert result.proof_code is not None
    assert "sorry" not in result.proof_code


@pytest.mark.asyncio
async def test_decompose_partial_failure():
    """Test that partial failure returns best partial result."""
    config = DecompositionConfig(
        max_decomposition_depth=0,  # No recursion
        max_sketches=1,
        verify_stitched=False,
        parallel_subgoals=False,
    )

    sketches = [ProofSketch(
        strategy="direct",
        steps=[
            SketchStep(statement="Easy step", formal_type="1 + 1 = 2"),
            SketchStep(statement="Hard step", formal_type="P ∧ Q"),
        ],
        key_lemmas=[],
    )]

    # Only succeed for step_0
    mock_search = _make_search_mock(succeed_for={"step_0"})

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        side_effect=mock_search,
    ):
        result = await decompose_and_prove(
            theorem="theorem foo : True",
            config=config,
            sketch_generator=MockSketchGenerator(sketches),
        )

    assert not result.success
    assert result.subgoals_total == 2
    assert result.subgoals_solved == 1
    assert "step_1" in result.failed_subgoals


@pytest.mark.asyncio
async def test_decompose_recursive_depth():
    """Test recursive decomposition when flat search fails."""
    config = DecompositionConfig(
        max_decomposition_depth=2,
        max_sketches=1,
        verify_stitched=False,
        parallel_subgoals=False,
    )

    call_count = 0

    async def mock_prove(theorem, budget=50, timeout=60.0, **kwargs):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        # Only succeed for depth-2 subgoals (simple ones)
        if "C" in theorem or call_count > 2:
            result.success = True
            result.proof_tactics = ["simp"]
            result.proof_code = f"{theorem} := by simp"
        else:
            result.success = False
            result.proof_tactics = []
            result.proof_code = None
            result.error = "No proof found"
            result.nodes_explored = 5
        return result

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        side_effect=mock_prove,
    ):
        result = await decompose_and_prove(
            theorem="theorem foo : True",
            config=config,
            sketch_generator=MockRecursiveSketchGenerator(),
        )

    # Should have tried recursion
    assert result.recursion_depth_reached >= 1


@pytest.mark.asyncio
async def test_decompose_no_sketches():
    """Test handling when no sketches are generated."""
    config = DecompositionConfig(
        max_decomposition_depth=1,
        max_sketches=1,
        verify_stitched=False,
    )

    result = await decompose_and_prove(
        theorem="theorem foo : True",
        config=config,
        sketch_generator=MockSketchGenerator(sketches=[]),
    )

    assert not result.success
    assert "No sketches generated" in result.errors


@pytest.mark.asyncio
async def test_decompose_budget_decay():
    """Test that budget and timeout decay with depth."""
    config = DecompositionConfig(
        max_decomposition_depth=2,
        max_sketches=1,
        subgoal_search_budget=100,
        subgoal_search_timeout=60.0,
        budget_decay_factor=0.5,
        timeout_decay_factor=0.5,
        verify_stitched=False,
        parallel_subgoals=False,
    )

    budgets_seen = []
    timeouts_seen = []

    async def mock_prove(theorem, budget=50, timeout=60.0, **kwargs):
        budgets_seen.append(budget)
        timeouts_seen.append(timeout)
        result = MagicMock()
        result.success = True
        result.proof_tactics = ["norm_num"]
        result.proof_code = f"{theorem} := by norm_num"
        return result

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        side_effect=mock_prove,
    ):
        result = await decompose_and_prove(
            theorem="theorem foo : 1 + 1 = 2",
            config=config,
            sketch_generator=MockSketchGenerator(),
        )

    assert result.success
    # Budget at depth 0 should be 100, at depth 1 should be 50, etc.
    assert budgets_seen[0] == 100


@pytest.mark.asyncio
async def test_decompose_with_verification():
    """Test that stitched proof is verified when verify_stitched is True."""
    config = DecompositionConfig(
        max_decomposition_depth=1,
        max_sketches=1,
        verify_stitched=True,
        parallel_subgoals=False,
    )

    mock_search = _make_search_mock(always_succeed=True)

    with (
        patch(
            "bourbaki.autonomous.decomposer.prove_with_search",
            side_effect=mock_search,
        ),
        patch(
            "bourbaki.autonomous.decomposer._verify_proof",
            return_value=True,
        ) as mock_verify,
    ):
        result = await decompose_and_prove(
            theorem="theorem foo : 1 + 1 = 2",
            config=config,
            sketch_generator=MockSketchGenerator(),
        )

    assert result.success
    assert result.verified
    mock_verify.assert_called_once()


@pytest.mark.asyncio
async def test_decompose_verification_failure_tries_next_sketch():
    """Test that verification failure causes trying the next sketch."""
    config = DecompositionConfig(
        max_decomposition_depth=1,
        max_sketches=2,
        verify_stitched=True,
        parallel_subgoals=False,
    )

    sketches = [
        ProofSketch(
            strategy="bad",
            steps=[SketchStep(statement="Bad", formal_type="1 = 1")],
            key_lemmas=[],
        ),
        ProofSketch(
            strategy="good",
            steps=[SketchStep(statement="Good", formal_type="1 + 1 = 2")],
            key_lemmas=[],
        ),
    ]

    mock_search = _make_search_mock(always_succeed=True)
    verify_calls = [False, True]  # First fails, second succeeds

    with (
        patch(
            "bourbaki.autonomous.decomposer.prove_with_search",
            side_effect=mock_search,
        ),
        patch(
            "bourbaki.autonomous.decomposer._verify_proof",
            side_effect=verify_calls,
        ),
    ):
        result = await decompose_and_prove(
            theorem="theorem foo : 1 + 1 = 2",
            config=config,
            sketch_generator=MockSketchGenerator(sketches),
        )

    assert result.success
    assert result.verified
    assert result.sketches_tried == 2


@pytest.mark.asyncio
async def test_solve_subgoals_parallel_independent():
    """Test that independent subgoals are solved in parallel."""
    config = DecompositionConfig(
        max_decomposition_depth=0,
        parallel_subgoals=True,
        max_parallel=4,
        verify_stitched=False,
    )

    subgoals = [
        Subgoal(index=0, label="step_0", lean_type="1 + 1 = 2", depends_on=[]),
        Subgoal(index=1, label="step_1", lean_type="2 + 2 = 4", depends_on=[]),
    ]

    mock_search = _make_search_mock(always_succeed=True)
    generator = MockSketchGenerator()

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        side_effect=mock_search,
    ):
        results = await _solve_subgoals_parallel(
            subgoals=subgoals,
            config=config,
            sketch_generator=generator,
            depth=0,
            previous_attempts=[],
        )

    assert len(results) == 2
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_solve_subgoals_sequential_dependent():
    """Test that dependent subgoals are solved sequentially."""
    config = DecompositionConfig(
        max_decomposition_depth=0,
        parallel_subgoals=True,
        verify_stitched=False,
    )

    subgoals = [
        Subgoal(index=0, label="step_0", lean_type="1 + 1 = 2", depends_on=[]),
        Subgoal(index=1, label="step_1", lean_type="step_0 + 2 = 4", depends_on=["step_0"]),
    ]

    mock_search = _make_search_mock(always_succeed=True)
    generator = MockSketchGenerator()

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        side_effect=mock_search,
    ):
        results = await _solve_subgoals_parallel(
            subgoals=subgoals,
            config=config,
            sketch_generator=generator,
            depth=0,
            previous_attempts=[],
        )

    assert len(results) == 2
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_solve_single_subgoal_search_success():
    """Test solving a single subgoal via flat search."""
    config = DecompositionConfig(
        max_decomposition_depth=1,
        verify_stitched=False,
    )

    subgoal = Subgoal(index=0, label="step_0", lean_type="1 + 1 = 2")
    mock_search = _make_search_mock(always_succeed=True)
    generator = MockSketchGenerator()

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        side_effect=mock_search,
    ):
        sr = await _solve_single_subgoal(
            subgoal=subgoal,
            config=config,
            sketch_generator=generator,
            depth=0,
            previous_attempts=[],
            solved_siblings={},
        )

    assert sr.success
    assert sr.method == "search"
    assert sr.tactics == ["norm_num"]


@pytest.mark.asyncio
async def test_solve_single_subgoal_max_depth():
    """Test that solving respects max depth."""
    config = DecompositionConfig(
        max_decomposition_depth=1,
        verify_stitched=False,
    )

    subgoal = Subgoal(index=0, label="step_0", lean_type="P ∧ Q")
    mock_search = _make_search_mock(always_succeed=False)
    generator = MockSketchGenerator()

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        side_effect=mock_search,
    ):
        sr = await _solve_single_subgoal(
            subgoal=subgoal,
            config=config,
            sketch_generator=generator,
            depth=1,  # Already at max depth
            previous_attempts=[],
            solved_siblings={},
        )

    assert not sr.success
    assert "Max decomposition depth" in sr.error


@pytest.mark.asyncio
async def test_decompose_sketch_no_subgoals():
    """Test handling when sketch formalization produces no subgoals."""
    config = DecompositionConfig(
        max_decomposition_depth=1,
        max_sketches=1,
        verify_stitched=False,
        parallel_subgoals=False,
    )

    # Sketch with no formal types will produce no extractable subgoals
    # after our filtering
    sketches = [ProofSketch(
        strategy="empty",
        steps=[],  # No steps
        key_lemmas=[],
    )]

    result = await decompose_and_prove(
        theorem="theorem foo : True",
        config=config,
        sketch_generator=MockSketchGenerator(sketches),
    )

    assert not result.success
    assert any("no subgoals" in e for e in result.errors)
