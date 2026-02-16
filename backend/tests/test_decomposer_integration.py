"""Integration test for the full decomposition pipeline.

Uses mocked LLM and REPL to test the full flow:
sketch generation -> formalization -> subgoal solving -> stitching.
"""

import pytest
from unittest.mock import AsyncMock, patch

from bourbaki.autonomous.decomposer import (
    DecompositionConfig,
    decompose_and_prove,
)
from bourbaki.autonomous.sketch import (
    ProofSketch,
    SketchContext,
    SketchStep,
)


class MockSketchGenerator:
    """Returns a fixed sketch for testing."""

    async def generate(self, context: SketchContext) -> list[ProofSketch]:
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


@pytest.mark.asyncio
async def test_decompose_with_mock_generator():
    """Test decomposition with mocked sketch generator and search."""
    config = DecompositionConfig(
        max_recursion_depth=1,
        max_sketches=1,
        subgoal_search_budget=10,
        subgoal_search_timeout=5.0,
    )

    # Mock prove_with_search to return success
    mock_search_result = AsyncMock()
    mock_search_result.return_value.success = True
    mock_search_result.return_value.proof_tactics = ["norm_num"]
    mock_search_result.return_value.proof_code = "theorem step_0 : 1 + 1 = 2 := by norm_num"

    with patch(
        "bourbaki.autonomous.decomposer.prove_with_search",
        mock_search_result,
    ):
        result = await decompose_and_prove(
            theorem="theorem foo : 1 + 1 = 2",
            config=config,
            sketch_generator=MockSketchGenerator(),
        )

    assert result.subgoals_total >= 1
    assert result.subgoals_solved >= 1
    assert result.sketches_tried == 1
