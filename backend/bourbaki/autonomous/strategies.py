"""Strategy definitions, queue, and dead-end tracking.

Ported from src/autonomous/strategies.ts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Strategy:
    id: str
    name: str
    technique: str
    priority: int
    applicable_domains: list[str]
    relevant_tags: list[str]
    description: str
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class StrategyResult:
    strategy_id: str
    success: bool
    partial_progress: str | None = None
    error: str | None = None
    insight: str | None = None
    proof_code: str | None = None
    time_spent: int = 0  # milliseconds


@dataclass
class DeadEnd:
    strategy_id: str
    problem_id: str
    approach: str
    reason: str
    timestamp: str = ""
    attempts: int = 1
    failure_features: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


DEFAULT_STRATEGIES: list[Strategy] = [
    Strategy("direct-computation", "Direct Computation", "direct", 100,
             ["number-theory", "algebra"], ["computation", "evaluate"], "Compute the answer directly."),
    Strategy("direct-proof", "Direct Proof", "direct", 90,
             ["number-theory", "algebra", "analysis"], ["proof"], "Prove by direct deduction from hypotheses."),
    Strategy("simple-induction", "Simple Induction", "induction", 85,
             ["number-theory", "combinatorics"], ["induction", "natural-numbers"], "Prove by mathematical induction on n."),
    Strategy("strong-induction", "Strong Induction", "strong-induction", 80,
             ["number-theory", "combinatorics"], ["induction", "well-ordering"], "Use strong/complete induction."),
    Strategy("structural-induction", "Structural Induction", "induction", 75,
             ["combinatorics", "algebra"], ["induction", "recursive"], "Induction on recursive structure."),
    Strategy("contradiction", "Proof by Contradiction", "contradiction", 70,
             ["number-theory", "algebra", "analysis"], ["impossibility", "negation"], "Assume negation, derive contradiction."),
    Strategy("infinite-descent", "Infinite Descent", "contradiction", 65,
             ["number-theory"], ["descent", "minimal"], "Fermat's infinite descent method."),
    Strategy("double-counting", "Double Counting", "counting", 80,
             ["combinatorics"], ["counting", "bijection"], "Count the same quantity two ways."),
    Strategy("pigeonhole", "Pigeonhole Principle", "pigeonhole", 85,
             ["combinatorics", "number-theory"], ["pigeonhole", "existence"], "If n+1 items in n boxes, some box has two."),
    Strategy("generalized-pigeonhole", "Generalized Pigeonhole", "pigeonhole", 75,
             ["combinatorics", "number-theory"], ["pigeonhole", "counting"], "Generalized pigeonhole principle."),
    Strategy("case-analysis", "Case Analysis", "cases", 60,
             ["number-theory", "combinatorics", "algebra"], ["cases", "exhaustive"], "Split into exhaustive cases."),
    Strategy("probabilistic-method", "Probabilistic Method", "probabilistic", 40,
             ["combinatorics"], ["random", "expectation"], "Prove existence via probability."),
    Strategy("algebraic-manipulation", "Algebraic Manipulation", "algebra", 55,
             ["algebra", "number-theory"], ["factor", "simplify"], "Simplify or transform algebraically."),
    Strategy("extremal-principle", "Extremal Principle", "extremal", 50,
             ["combinatorics", "geometry"], ["minimal", "maximal"], "Consider minimal or maximal element."),
    Strategy("similar-problems", "Search Similar Problems", "meta", 45,
             [], ["reference", "known-result"], "Look for similar known results."),
    Strategy("generalize", "Generalize the Problem", "meta", 35,
             [], ["generalize"], "Solve a more general version."),
    Strategy("specialize", "Specialize the Problem", "meta", 30,
             [], ["specialize", "small-cases"], "Test small/special cases first."),
    Strategy("counterexample-search", "Search for Counterexamples", "meta", 25,
             [], ["counterexample"], "Search for counterexamples before proving."),
]

_STRATEGY_MAP: dict[str, Strategy] = {s.id: s for s in DEFAULT_STRATEGIES}


def get_strategy(strategy_id: str) -> Strategy | None:
    return _STRATEGY_MAP.get(strategy_id)


class StrategyQueue:
    """Prioritized queue of strategies for a problem."""

    def __init__(self, strategies: list[Strategy] | None = None) -> None:
        self.strategies = list(strategies or DEFAULT_STRATEGIES)
        self.attempted: dict[str, list[StrategyResult]] = {}
        self._current_index = 0

    def get_strategies_for_problem(self, problem: dict[str, Any]) -> list[Strategy]:
        """Filter and prioritize strategies for a specific problem."""
        domain = problem.get("domain", "")
        tags = problem.get("tags", [])
        techniques = problem.get("techniques", [])

        applicable = []
        for s in self.strategies:
            # Meta strategies apply to all domains
            if s.applicable_domains and domain not in s.applicable_domains:
                continue
            # Calculate dynamic priority
            priority = s.priority
            if s.technique in techniques:
                priority += 50
            priority += sum(10 for t in s.relevant_tags if t in tags)
            # Penalize failed attempts
            attempts = self.attempted.get(s.id, [])
            failed = sum(1 for a in attempts if not a.success)
            priority -= failed * 20
            # Boost partial progress
            if any(a.partial_progress for a in attempts):
                priority += 15

            applicable.append((priority, s))

        applicable.sort(key=lambda x: -x[0])
        return [s for _, s in applicable]

    def get_next(self, problem: dict[str, Any]) -> Strategy | None:
        """Get the next unattempted or unexhausted strategy."""
        ranked = self.get_strategies_for_problem(problem)
        for s in ranked:
            attempts = self.attempted.get(s.id, [])
            failed = sum(1 for a in attempts if not a.success)
            if failed < 2:
                return s
        return None

    def record_attempt(self, result: StrategyResult) -> None:
        self.attempted.setdefault(result.strategy_id, []).append(result)

    def get_attempt_history(self) -> dict[str, list[StrategyResult]]:
        return dict(self.attempted)

    def get_promising_strategies(self) -> list[Strategy]:
        """Strategies with partial progress."""
        promising = []
        for sid, results in self.attempted.items():
            if any(r.partial_progress for r in results):
                s = get_strategy(sid)
                if s:
                    promising.append(s)
        return promising

    def reset(self) -> None:
        self.attempted.clear()
        self._current_index = 0


class DeadEndDatabase:
    """Tracks dead ends to avoid repeating failed approaches."""

    def __init__(self, max_attempts: int = 3) -> None:
        self.dead_ends: dict[str, list[DeadEnd]] = {}
        self.max_attempts = max_attempts

    def record_dead_end(self, dead_end: DeadEnd) -> None:
        key = f"{dead_end.problem_id}:{dead_end.strategy_id}"
        existing = self.dead_ends.setdefault(key, [])
        # Check for same approach
        for de in existing:
            if de.approach == dead_end.approach:
                de.attempts += 1
                de.reason = dead_end.reason
                return
        existing.append(dead_end)

    def is_dead_end(
        self, problem_id: str, strategy_id: str, approach: str | None = None,
    ) -> bool:
        key = f"{problem_id}:{strategy_id}"
        dead_ends = self.dead_ends.get(key, [])
        if approach:
            return any(
                de.approach == approach and de.attempts >= self.max_attempts
                for de in dead_ends
            )
        return any(de.attempts >= self.max_attempts for de in dead_ends)

    def get_dead_ends_for_problem(self, problem_id: str) -> list[DeadEnd]:
        result = []
        for key, des in self.dead_ends.items():
            if key.startswith(f"{problem_id}:"):
                result.extend(des)
        return result

    def get_insights(self, problem_id: str) -> list[str]:
        dead_ends = self.get_dead_ends_for_problem(problem_id)
        return [de.reason for de in dead_ends if de.reason]

    def export_data(self) -> dict[str, list[dict]]:
        return {
            key: [
                {
                    "strategy_id": de.strategy_id,
                    "problem_id": de.problem_id,
                    "approach": de.approach,
                    "reason": de.reason,
                    "timestamp": de.timestamp,
                    "attempts": de.attempts,
                    "failure_features": de.failure_features,
                }
                for de in des
            ]
            for key, des in self.dead_ends.items()
        }

    def import_data(self, data: dict[str, list[dict]]) -> None:
        for key, des_data in data.items():
            self.dead_ends[key] = [
                DeadEnd(**d) for d in des_data
            ]

    def clear_for_problem(self, problem_id: str) -> None:
        keys_to_remove = [k for k in self.dead_ends if k.startswith(f"{problem_id}:")]
        for k in keys_to_remove:
            del self.dead_ends[k]


def select_initial_strategies(problem: dict[str, Any]) -> list[str]:
    """Select initial strategy IDs based on problem characteristics."""
    techniques = problem.get("techniques", [])
    domain = problem.get("domain", "")

    technique_map: dict[str, str] = {
        "induction": "simple-induction",
        "strong-induction": "strong-induction",
        "contradiction": "contradiction",
        "counting": "double-counting",
        "double-counting": "double-counting",
        "pigeonhole": "pigeonhole",
        "cases": "case-analysis",
    }

    selected = [technique_map[t] for t in techniques if t in technique_map]

    if not selected:
        domain_defaults: dict[str, list[str]] = {
            "number-theory": ["direct-computation", "simple-induction"],
            "combinatorics": ["double-counting", "pigeonhole"],
            "algebra": ["algebraic-manipulation", "direct-proof"],
            "analysis": ["direct-proof", "contradiction"],
        }
        selected = domain_defaults.get(domain, ["direct-proof"])

    selected.extend(["similar-problems", "specialize"])
    return selected
