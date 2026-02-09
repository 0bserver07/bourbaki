"""Parallel strategy execution for autonomous proof search."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic_ai import Agent

from bourbaki.autonomous.strategies import DeadEnd, StrategyResult

PROOF_SYSTEM_PROMPT = (
    "You are a mathematical proof assistant. Apply the requested proof technique rigorously. "
    "If successful, provide a complete Lean 4 proof. If partial progress, describe what you discovered. "
    "If the approach fails, explain WHY clearly — this helps avoid dead ends. "
    "Be honest about failures."
)


def _build_strategy_prompt(
    problem: dict[str, Any],
    strategy: dict[str, Any],
    proof_state: dict[str, Any] | None = None,
    dead_ends: list[dict[str, Any]] | None = None,
) -> str:
    """Build the prompt for a single strategy attempt."""
    parts = [
        f"**Problem:** {problem.get('statement', problem.get('title', ''))}",
        f"**Domain:** {problem.get('domain', 'unknown')}",
        f"**Tags:** {', '.join(problem.get('tags', []))}",
        "",
        f"**Strategy to apply:** {strategy.get('name', strategy.get('id', ''))}",
        strategy.get("description", ""),
    ]

    if proof_state:
        parts.extend([
            "",
            "**Current proof state:**",
            f"Goals: {proof_state.get('goals', 'none')}",
            f"Progress: {proof_state.get('steps', 'none')}",
        ])

    if dead_ends:
        parts.extend(["", "**Previous dead ends (avoid these approaches):**"])
        for de in dead_ends[:5]:
            parts.append(f"  - {de.get('approach', '')}: {de.get('reason', '')}")

    parts.extend([
        "",
        f"Attempt to prove using the {strategy.get('technique', '')} technique.",
        "If partial progress, explain what you discovered.",
        "If approach fails, explain why clearly.",
        "",
        "Format your response as:",
        "```lean4",
        "-- Your proof here",
        "```",
        "",
        "**INSIGHT:** [what you learned]",
        "**SUCCESS:** [true/false]",
        "**PARTIAL_PROGRESS:** [description if any]",
    ])

    return "\n".join(parts)


def _parse_strategy_response(
    text: str,
    strategy_id: str,
    elapsed_ms: int,
) -> StrategyResult:
    """Parse LLM response into a StrategyResult."""
    success = "**SUCCESS:** true" in text.lower() or "**success:** true" in text
    insight = None
    partial_progress = None
    proof_code = None

    # Extract insight
    for line in text.split("\n"):
        lower = line.strip().lower()
        if lower.startswith("**insight:**"):
            insight = line.split(":", 1)[1].strip() if ":" in line else None
        elif lower.startswith("**partial_progress:**"):
            partial_progress = line.split(":", 1)[1].strip() if ":" in line else None

    # Extract Lean code block
    if "```lean" in text:
        start = text.index("```lean")
        end_marker = text.find("```", start + 7)
        if end_marker > start:
            code_block = text[start + 7:end_marker].strip()
            # Remove the optional "4" after "lean"
            if code_block.startswith("4\n"):
                code_block = code_block[2:]
            proof_code = code_block

    return StrategyResult(
        strategy_id=strategy_id,
        success=success,
        partial_progress=partial_progress,
        error=None if success else "Strategy did not produce a valid proof",
        insight=insight,
        proof_code=proof_code,
        time_spent=elapsed_ms,
    )


async def execute_strategy_local(
    problem: dict[str, Any],
    strategy: dict[str, Any],
    model: str,
    proof_state: dict[str, Any] | None = None,
    dead_ends: list[dict[str, Any]] | None = None,
) -> StrategyResult:
    """Execute a single strategy locally using Pydantic AI."""
    prompt = _build_strategy_prompt(problem, strategy, proof_state, dead_ends)
    start = time.monotonic()

    try:
        agent: Agent[None, str] = Agent(model, system_prompt=PROOF_SYSTEM_PROMPT)
        result = await agent.run(prompt)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return _parse_strategy_response(result.output, strategy["id"], elapsed_ms)
    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return StrategyResult(
            strategy_id=strategy["id"],
            success=False,
            error=str(e),
            time_spent=elapsed_ms,
        )


async def run_parallel_strategies(
    problem: dict[str, Any],
    strategies: list[dict[str, Any]],
    model: str,
    max_parallel: int = 5,
    proof_state: dict[str, Any] | None = None,
    dead_ends: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run multiple strategies in parallel via asyncio.gather.

    Returns:
        Dict with results, newDeadEnds, insights, bestResult, proofFound.
    """
    # Local parallel execution via asyncio
    batch = strategies[:max_parallel]
    tasks = [
        execute_strategy_local(problem, s, model, proof_state, dead_ends)
        for s in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    strategy_results: list[StrategyResult] = []
    new_dead_ends: list[DeadEnd] = []
    all_insights: list[str] = []
    best_result: StrategyResult | None = None
    proof_found = False

    for r in results:
        if isinstance(r, Exception):
            continue
        strategy_results.append(r)

        if r.insight:
            all_insights.append(r.insight)

        if r.success:
            proof_found = True
            best_result = r
        elif not best_result or (r.partial_progress and not best_result.success):
            best_result = r

        if not r.success and r.error:
            new_dead_ends.append(DeadEnd(
                strategy_id=r.strategy_id,
                problem_id=problem.get("id", ""),
                approach=r.strategy_id,
                reason=r.error,
            ))

    return {
        "results": strategy_results,
        "newDeadEnds": new_dead_ends,
        "insights": all_insights,
        "bestResult": best_result,
        "proofFound": proof_found,
    }



