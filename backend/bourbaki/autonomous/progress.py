"""Progress tracking for autonomous proof search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgressReport:
    """Current state of an autonomous proof search."""
    session_id: str
    problem_id: str
    status: str  # 'running', 'paused', 'completed', 'failed'
    iteration: int = 0
    max_iterations: int = 100
    elapsed_seconds: float = 0.0
    max_hours: float = 4.0
    current_strategy: str | None = None
    strategies_tried: int = 0
    strategies_remaining: int = 0
    dead_ends: int = 0
    insights: list[str] = field(default_factory=list)
    partial_progress: list[str] = field(default_factory=list)
    proof_found: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "problemId": self.problem_id,
            "status": self.status,
            "iteration": self.iteration,
            "maxIterations": self.max_iterations,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
            "maxHours": self.max_hours,
            "currentStrategy": self.current_strategy,
            "strategiesTried": self.strategies_tried,
            "strategiesRemaining": self.strategies_remaining,
            "deadEnds": self.dead_ends,
            "insights": self.insights,
            "partialProgress": self.partial_progress,
            "proofFound": self.proof_found,
        }

    def format_string(self) -> str:
        """Human-readable progress summary."""
        pct = (self.iteration / self.max_iterations * 100) if self.max_iterations else 0
        elapsed_min = self.elapsed_seconds / 60
        parts = [
            f"[{self.status.upper()}] {self.problem_id}",
            f"Iteration {self.iteration}/{self.max_iterations} ({pct:.0f}%)",
            f"Time: {elapsed_min:.1f}min / {self.max_hours * 60:.0f}min",
            f"Strategy: {self.current_strategy or 'none'}",
            f"Tried: {self.strategies_tried} | Dead ends: {self.dead_ends}",
        ]
        if self.insights:
            parts.append(f"Insights: {len(self.insights)}")
        if self.proof_found:
            parts.append("PROOF FOUND!")
        return " | ".join(parts)
