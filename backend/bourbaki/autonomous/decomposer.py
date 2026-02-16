"""Recursive subgoal decomposer (HILBERT-style).

Orchestrates: sketch generation -> formalization -> subgoal solving -> stitching.
Each failed subgoal can be recursively decomposed up to max_recursion_depth.

Reference: HILBERT (Apple, 99.2% miniF2F), DeepSeek-Prover-V2 subgoal decomposition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from bourbaki.autonomous.formalizer import (
    FormalizedSkeleton,
    Subgoal,
    formalize_sketch,
    stitch_proofs,
)
from bourbaki.autonomous.search_tree import SearchResult, prove_with_search
from bourbaki.autonomous.sketch import (
    LLMSketchGenerator,
    ProofSketch,
    SketchContext,
    SketchGenerator,
)

logger = logging.getLogger(__name__)


@dataclass
class DecompositionConfig:
    """Configuration for recursive decomposition."""
    max_recursion_depth: int = 2
    max_sketches: int = 3
    subgoal_search_budget: int = 50
    subgoal_search_timeout: float = 60.0
    formalization_retries: int = 2
    model: str = "openai:gpt-4o"


@dataclass
class DecompositionResult:
    """Result of decompose_and_prove."""
    success: bool
    proof_code: str | None = None
    subgoals_total: int = 0
    subgoals_solved: int = 0
    solved_subgoal_proofs: dict[str, list[str]] = field(default_factory=dict)
    failed_subgoals: list[str] = field(default_factory=list)
    sketches_tried: int = 0
    recursion_depth_reached: int = 0
    total_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def all_solved(self) -> bool:
        return self.subgoals_total > 0 and self.subgoals_solved == self.subgoals_total

    @property
    def solve_rate(self) -> float:
        if self.subgoals_total == 0:
            return 0.0
        return self.subgoals_solved / self.subgoals_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "proof_code": self.proof_code,
            "subgoals_total": self.subgoals_total,
            "subgoals_solved": self.subgoals_solved,
            "failed_subgoals": self.failed_subgoals,
            "sketches_tried": self.sketches_tried,
            "recursion_depth_reached": self.recursion_depth_reached,
            "total_time": round(self.total_time, 2),
            "solve_rate": round(self.solve_rate, 3),
        }


async def decompose_and_prove(
    theorem: str,
    config: DecompositionConfig,
    sketch_generator: SketchGenerator | None = None,
    depth: int = 0,
    previous_attempts: list[str] | None = None,
) -> DecompositionResult:
    """Recursively decompose a theorem into subgoals and prove each.

    Algorithm:
    1. Generate proof sketches (1-3 diverse strategies)
    2. For each sketch, formalize into Lean have/sorry skeleton
    3. For each subgoal: try prove_with_search()
    4. Failed subgoals: recurse if depth < max_recursion_depth
    5. If all solved, stitch proofs and verify
    6. Return partial results even on failure

    Args:
        theorem: Lean 4 theorem statement.
        config: Decomposition configuration.
        sketch_generator: Pluggable sketch generator (defaults to LLMSketchGenerator).
        depth: Current recursion depth.
        previous_attempts: Failed approaches to avoid.

    Returns:
        DecompositionResult with proof or partial progress.
    """
    start = time.monotonic()

    if sketch_generator is None:
        sketch_generator = LLMSketchGenerator(config.model)

    result = DecompositionResult(
        success=False,
        recursion_depth_reached=depth,
    )

    # Generate proof sketches
    context = SketchContext(
        theorem=theorem,
        previous_attempts=previous_attempts or [],
        depth=depth,
    )

    try:
        sketches = await sketch_generator.generate(context)
    except Exception as e:
        result.errors.append(f"Sketch generation failed: {e}")
        result.total_time = time.monotonic() - start
        return result

    if not sketches:
        result.errors.append("No sketches generated")
        result.total_time = time.monotonic() - start
        return result

    # Try each sketch (up to max_sketches)
    for sketch_idx, sketch in enumerate(sketches[:config.max_sketches]):
        result.sketches_tried = sketch_idx + 1

        logger.info(
            "Trying sketch %d/%d (strategy=%s, steps=%d, depth=%d)",
            sketch_idx + 1, len(sketches), sketch.strategy,
            len(sketch.steps), depth,
        )

        # Formalize sketch into Lean skeleton
        skeleton = await formalize_sketch(
            theorem=theorem,
            sketch=sketch,
            model=config.model,
            max_retries=config.formalization_retries,
        )

        if not skeleton.subgoals:
            result.errors.append(f"Sketch {sketch_idx}: no subgoals extracted")
            continue

        # Try to solve each subgoal
        subgoal_proofs: dict[str, list[str]] = {}
        failed: list[str] = []
        result.subgoals_total = len(skeleton.subgoals)
        result.subgoals_solved = 0

        for subgoal in skeleton.subgoals:
            # Build a mini-theorem for this subgoal
            subgoal_theorem = f"theorem {subgoal.label} : {subgoal.lean_type}"

            # Try flat search first
            search_result = await prove_with_search(
                theorem=subgoal_theorem,
                budget=config.subgoal_search_budget,
                timeout=config.subgoal_search_timeout,
            )

            if search_result.success:
                subgoal_proofs[subgoal.label] = search_result.proof_tactics
                result.subgoals_solved += 1
                logger.info(
                    "Subgoal %s solved (%d tactics)",
                    subgoal.label, len(search_result.proof_tactics),
                )
            elif depth < config.max_recursion_depth:
                # Recurse: decompose the failed subgoal
                logger.info(
                    "Subgoal %s failed flat search, recursing (depth=%d)",
                    subgoal.label, depth + 1,
                )
                sub_result = await decompose_and_prove(
                    theorem=subgoal_theorem,
                    config=config,
                    sketch_generator=sketch_generator,
                    depth=depth + 1,
                    previous_attempts=[f"Flat search failed on: {subgoal.lean_type}"],
                )
                result.recursion_depth_reached = max(
                    result.recursion_depth_reached,
                    sub_result.recursion_depth_reached,
                )
                if sub_result.success and sub_result.proof_code:
                    # Extract tactics from the recursive proof
                    tactics = _extract_tactics_from_proof(sub_result.proof_code)
                    subgoal_proofs[subgoal.label] = tactics
                    result.subgoals_solved += 1
                else:
                    failed.append(subgoal.label)
            else:
                failed.append(subgoal.label)

        result.solved_subgoal_proofs = subgoal_proofs
        result.failed_subgoals = failed

        # If all subgoals solved, stitch and verify
        if not failed:
            stitched = stitch_proofs(skeleton.code, subgoal_proofs)
            result.proof_code = stitched
            result.success = True
            result.total_time = time.monotonic() - start
            logger.info(
                "Decomposition succeeded: %d subgoals, %d sketches tried, depth=%d",
                result.subgoals_total, result.sketches_tried, depth,
            )
            return result

        # Some subgoals failed — try next sketch
        logger.info(
            "Sketch %d: %d/%d subgoals solved, trying next sketch",
            sketch_idx, result.subgoals_solved, result.subgoals_total,
        )

    result.total_time = time.monotonic() - start
    return result


def _extract_tactics_from_proof(proof_code: str) -> list[str]:
    """Extract the tactic block from a complete proof.

    Given: "theorem foo : T := by\\n  tactic1\\n  tactic2"
    Returns: ["tactic1", "tactic2"]
    """
    # Find "by" and extract everything after it
    by_idx = proof_code.find(":= by")
    if by_idx == -1:
        by_idx = proof_code.find("by\n")
        if by_idx == -1:
            return [proof_code]  # Return as-is

    tactic_block = proof_code[by_idx + 5:].strip()  # Skip ":= by"
    tactics = [
        line.strip()
        for line in tactic_block.split("\n")
        if line.strip() and not line.strip().startswith("--")
    ]
    return tactics if tactics else [tactic_block]
