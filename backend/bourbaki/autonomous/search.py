"""Autonomous proof search engine.

Ported from src/autonomous/search.ts — manages long-running proof attempts
with strategy rotation, dead-end tracking, and checkpointing.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bourbaki.autonomous.modal_runner import execute_strategy_local
from bourbaki.autonomous.progress import ProgressReport
from bourbaki.autonomous.search_tree import prove_with_search
from bourbaki.autonomous.strategies import (
    DEFAULT_STRATEGIES,
    DeadEnd,
    DeadEndDatabase,
    StrategyQueue,
    StrategyResult,
    select_initial_strategies,
)
from bourbaki.config import settings

SearchEvent = dict[str, Any]
SearchEventHandler = Callable[[SearchEvent], None]


class AutonomousSearchConfig:
    """Configuration for an autonomous search."""

    def __init__(
        self,
        max_iterations: int = 100,
        max_hours: float = 4.0,
        strategies: list[str] | None = None,
        checkpoint_interval: int = 10,
        auto_resume: bool = True,
        max_dead_ends_per_strategy: int = 3,
        use_search_tree: bool = False,
        search_tree_budget: int = 100,
        search_tree_max_depth: int = 30,
        # Phase 0: Recursive decomposition
        use_decomposition: bool = True,
        decomposition_max_depth: int = 2,
        decomposition_max_sketches: int = 3,
        decomposition_subgoal_budget: int = 50,
    ):
        self.max_iterations = max_iterations
        self.max_hours = max_hours
        self.strategies = strategies
        self.checkpoint_interval = checkpoint_interval
        self.auto_resume = auto_resume
        self.max_dead_ends_per_strategy = max_dead_ends_per_strategy
        self.use_search_tree = use_search_tree
        self.search_tree_budget = search_tree_budget
        self.search_tree_max_depth = search_tree_max_depth
        self.use_decomposition = use_decomposition
        self.decomposition_max_depth = decomposition_max_depth
        self.decomposition_max_sketches = decomposition_max_sketches
        self.decomposition_subgoal_budget = decomposition_subgoal_budget


class AutonomousSearch:
    """Long-running proof search with strategy management."""

    def __init__(self) -> None:
        self._problem: dict[str, Any] | None = None
        self._session_id: str | None = None
        self._status: str = "paused"  # paused, running, completed, failed
        self._iteration: int = 0
        self._start_time: float = 0
        self._pause_time: float = 0
        self._total_paused: float = 0
        self._insights: list[str] = []
        self._proof_state: dict[str, Any] | None = None
        self._event_handler: SearchEventHandler | None = None
        self._abort: asyncio.Event = asyncio.Event()
        self._strategy_queue: StrategyQueue | None = None
        self._dead_ends: DeadEndDatabase | None = None
        self._config: AutonomousSearchConfig = AutonomousSearchConfig()
        self._progress_dir = settings.bourbaki_path / "progress"

    def on_event(self, handler: SearchEventHandler) -> None:
        self._event_handler = handler

    def _emit(self, event: SearchEvent) -> None:
        if self._event_handler:
            self._event_handler(event)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def status(self) -> str:
        return self._status

    @property
    def insights(self) -> list[str]:
        return list(self._insights)

    @property
    def proof_state(self) -> dict[str, Any] | None:
        return self._proof_state

    def get_elapsed_time(self) -> float:
        """Elapsed time in seconds, excluding paused periods."""
        if self._status == "running":
            return time.monotonic() - self._start_time - self._total_paused
        if self._pause_time:
            return self._pause_time - self._start_time - self._total_paused
        return 0

    def get_progress(self) -> ProgressReport:
        """Get current progress report."""
        return ProgressReport(
            session_id=self._session_id or "",
            problem_id=self._problem.get("id", "") if self._problem else "",
            status=self._status,
            iteration=self._iteration,
            max_iterations=self._config.max_iterations,
            elapsed_seconds=self.get_elapsed_time(),
            max_hours=self._config.max_hours,
            current_strategy=(
                self._strategy_queue.get_next(self._problem).id
                if self._strategy_queue and self._problem
                else None
            ) if self._status == "running" else None,
            strategies_tried=len(self._strategy_queue.attempted) if self._strategy_queue else 0,
            dead_ends=sum(
                len(des) for des in self._dead_ends.dead_ends.values()
            ) if self._dead_ends else 0,
            insights=list(self._insights),
            proof_found=bool(self._proof_state and self._proof_state.get("complete")),
        )

    async def start(
        self,
        problem: dict[str, Any],
        config: AutonomousSearchConfig | None = None,
    ) -> None:
        """Start an autonomous proof search."""
        self._config = config or AutonomousSearchConfig()
        self._problem = problem
        self._session_id = uuid.uuid4().hex[:8]
        self._status = "running"
        self._iteration = 0
        self._start_time = time.monotonic()
        self._total_paused = 0
        self._insights = []
        self._proof_state = None
        self._abort = asyncio.Event()

        # Initialize strategy queue with problem-specific strategies
        initial_ids = self._config.strategies or select_initial_strategies(problem)
        initial_strategies = [
            s for s in DEFAULT_STRATEGIES if s.id in initial_ids
        ]
        # Include remaining strategies as fallback (lower priority)
        remaining = [s for s in DEFAULT_STRATEGIES if s.id not in initial_ids]
        self._strategy_queue = StrategyQueue(initial_strategies + remaining)
        self._dead_ends = DeadEndDatabase(max_attempts=self._config.max_dead_ends_per_strategy)

        self._emit({"type": "started", "sessionId": self._session_id, "problem": problem})

        await self._run_loop()

    async def resume(self, session_id: str | None = None) -> bool:
        """Resume from a checkpoint."""
        sid = session_id or self._session_id
        if not sid:
            return False

        state = self._load_checkpoint(sid)
        if not state:
            return False

        self._session_id = sid
        self._problem = state.get("problem")
        self._iteration = state.get("iteration", 0)
        self._insights = state.get("insights", [])
        self._proof_state = state.get("proofState")

        # Restore dead ends
        self._dead_ends = DeadEndDatabase()
        if "deadEnds" in state:
            self._dead_ends.import_data(state["deadEnds"])

        # Restore strategy queue
        self._strategy_queue = StrategyQueue()
        # Restore attempt history
        for _, results in state.get("attemptHistory", {}).items():
            for r in results:
                self._strategy_queue.record_attempt(StrategyResult(**r))

        self._status = "running"
        self._start_time = time.monotonic()
        paused_elapsed = state.get("elapsedBeforePause", 0)
        self._total_paused = -paused_elapsed  # So elapsed calculation works

        self._emit({"type": "resumed", "sessionId": self._session_id})
        await self._run_loop()
        return True

    def pause(self) -> None:
        """Pause the search."""
        if self._status != "running":
            return
        self._status = "paused"
        self._pause_time = time.monotonic()
        self._save_checkpoint()
        self._emit({"type": "paused", "reason": "manual"})

    def record_attempt(self, result: StrategyResult) -> None:
        """Record a strategy attempt result."""
        if self._strategy_queue:
            self._strategy_queue.record_attempt(result)

        if result.insight:
            self._insights.append(result.insight)
            self._emit({"type": "insight", "insight": result.insight})

        if not result.success and result.error and self._problem:
            dead_end = DeadEnd(
                strategy_id=result.strategy_id,
                problem_id=self._problem.get("id", ""),
                approach=result.strategy_id,
                reason=result.error,
            )
            if self._dead_ends:
                self._dead_ends.record_dead_end(dead_end)
            self._emit({"type": "dead_end", "strategy": result.strategy_id, "reason": result.error})

    def update_proof_state(self, state: dict[str, Any]) -> None:
        """Update the current proof state."""
        self._proof_state = state

    async def _run_loop(self) -> None:
        """Main search loop — gets strategies and executes them via LLM.

        If use_search_tree is enabled, first attempts best-first tactic search
        via the Lean REPL before falling back to strategy rotation.
        """
        # Phase 0: Recursive subgoal decomposition
        if (
            self._config.use_decomposition
            and self._problem
            and self._problem.get("lean_statement")
        ):
            from bourbaki.autonomous.decomposer import (
                DecompositionConfig,
                decompose_and_prove,
            )

            self._emit({
                "type": "strategy_attempt",
                "strategy": "recursive-decomposition",
                "approach": "HILBERT-style recursive subgoal decomposition",
            })

            decomp_config = DecompositionConfig(
                max_recursion_depth=self._config.decomposition_max_depth,
                max_sketches=self._config.decomposition_max_sketches,
                subgoal_search_budget=self._config.decomposition_subgoal_budget,
                model=settings.default_model,
            )

            decomp_result = await decompose_and_prove(
                theorem=self._problem["lean_statement"],
                config=decomp_config,
            )

            self._iteration += 1

            if decomp_result.success and decomp_result.proof_code:
                self._proof_state = {
                    "complete": True,
                    "proof_code": decomp_result.proof_code,
                }
                self._insights.append(
                    f"Decomposition found proof: {decomp_result.subgoals_solved} subgoals, "
                    f"{decomp_result.sketches_tried} sketches tried, "
                    f"depth={decomp_result.recursion_depth_reached}"
                )
                self._emit({
                    "type": "completed",
                    "success": True,
                    "result": decomp_result.proof_code,
                })
                self._status = "completed"
                return
            else:
                insight = (
                    f"Decomposition partial: {decomp_result.subgoals_solved}/{decomp_result.subgoals_total} "
                    f"subgoals solved ({decomp_result.sketches_tried} sketches tried)"
                )
                self._insights.append(insight)
                self._emit({
                    "type": "strategy_result",
                    "strategy": "recursive-decomposition",
                    "success": False,
                    "insight": insight,
                    "partial_progress": decomp_result.to_dict(),
                })

        # Phase 1: Try best-first proof search tree (if enabled and problem has Lean statement)
        if (
            self._config.use_search_tree
            and self._problem
            and self._problem.get("lean_statement")
        ):
            self._emit({
                "type": "strategy_attempt",
                "strategy": "best-first-search",
                "approach": "Tactic-by-tactic best-first search using Lean REPL",
            })

            search_result = await prove_with_search(
                theorem=self._problem["lean_statement"],
                budget=self._config.search_tree_budget,
                timeout=self._config.max_hours * 3600,
                max_depth=self._config.search_tree_max_depth,
                use_mathlib=True,
            )

            self._iteration += 1

            if search_result.success:
                self._proof_state = {
                    "complete": True,
                    "proof_code": search_result.proof_code,
                    "tactics": search_result.proof_tactics,
                }
                self._insights.append(
                    f"Best-first search found proof: {len(search_result.proof_tactics)} tactics, "
                    f"{search_result.nodes_explored} nodes explored"
                )
                self._emit({
                    "type": "completed",
                    "success": True,
                    "result": search_result.proof_code,
                })
                self._status = "completed"
                return
            else:
                self._insights.append(
                    f"Best-first search failed: {search_result.error} "
                    f"({search_result.nodes_explored} nodes explored)"
                )
                self._emit({
                    "type": "strategy_result",
                    "strategy": "best-first-search",
                    "success": False,
                    "insight": f"Explored {search_result.nodes_explored} proof states without finding a proof",
                })

        # Phase 2: Strategy rotation (original approach)
        while self._status == "running":
            # Check limits
            if self._iteration >= self._config.max_iterations:
                self._emit({"type": "completed", "success": False, "result": "Iteration limit reached"})
                self._status = "completed"
                break

            elapsed_hours = self.get_elapsed_time() / 3600
            if elapsed_hours >= self._config.max_hours:
                self._emit({"type": "completed", "success": False, "result": "Time limit reached"})
                self._status = "completed"
                break

            if self._abort.is_set():
                self.pause()
                break

            # Get next strategy
            if not self._strategy_queue or not self._problem:
                break

            strategy = self._strategy_queue.get_next(self._problem)
            if not strategy:
                self._emit({"type": "completed", "success": False, "result": "No more strategies"})
                self._status = "completed"
                break

            # Check if this strategy is a known dead end
            problem_id = self._problem.get("id", "")
            if self._dead_ends and self._dead_ends.is_dead_end(problem_id, strategy.id):
                continue  # Skip exhausted strategies

            self._iteration += 1
            self._emit({
                "type": "iteration",
                "iteration": self._iteration,
                "strategy": strategy.id,
            })

            self._emit({
                "type": "strategy_attempt",
                "strategy": strategy.id,
                "approach": strategy.description,
            })

            # Execute the strategy via LLM
            dead_ends_for_prompt = (
                [{"approach": de.approach, "reason": de.reason}
                 for de in self._dead_ends.get_dead_ends_for_problem(problem_id)]
                if self._dead_ends else []
            )

            strategy_dict = {
                "id": strategy.id,
                "name": strategy.name,
                "technique": strategy.technique,
                "description": strategy.description,
            }

            result = await execute_strategy_local(
                problem=self._problem,
                strategy=strategy_dict,
                model=settings.default_model,
                proof_state=self._proof_state,
                dead_ends=dead_ends_for_prompt or None,
            )

            # Record the attempt
            self.record_attempt(result)

            self._emit({
                "type": "strategy_result",
                "strategy": strategy.id,
                "success": result.success,
                "insight": result.insight,
                "partial_progress": result.partial_progress,
            })

            # Check if proof was found
            if result.success and result.proof_code:
                self._proof_state = {"complete": True, "proof_code": result.proof_code}
                self._emit({
                    "type": "completed",
                    "success": True,
                    "result": result.proof_code,
                })
                self._status = "completed"
                break

            # Update proof state with partial progress
            if result.partial_progress and self._proof_state is None:
                self._proof_state = {"steps": result.partial_progress}

            # Checkpoint
            if self._iteration % self._config.checkpoint_interval == 0:
                self._save_checkpoint()
                self._emit({"type": "checkpoint", "path": str(self._checkpoint_path())})

            # Small delay to allow abort and avoid tight loops
            await asyncio.sleep(0.1)

        # Final checkpoint
        self._save_checkpoint()

    def _checkpoint_path(self) -> Path:
        self._progress_dir.mkdir(parents=True, exist_ok=True)
        return self._progress_dir / f"{self._session_id}.json"

    def _save_checkpoint(self) -> None:
        if not self._session_id:
            return
        state = {
            "sessionId": self._session_id,
            "problem": self._problem,
            "iteration": self._iteration,
            "status": self._status,
            "insights": self._insights,
            "proofState": self._proof_state,
            "elapsedBeforePause": self.get_elapsed_time(),
            "deadEnds": self._dead_ends.export_data() if self._dead_ends else {},
            "attemptHistory": {
                sid: [
                    {
                        "strategy_id": r.strategy_id,
                        "success": r.success,
                        "partial_progress": r.partial_progress,
                        "error": r.error,
                        "insight": r.insight,
                        "time_spent": r.time_spent,
                    }
                    for r in results
                ]
                for sid, results in (self._strategy_queue.get_attempt_history().items() if self._strategy_queue else {})
            },
        }
        path = self._checkpoint_path()
        path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def _load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        path = self._progress_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
