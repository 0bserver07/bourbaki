"""ProverLoop — proposer/builder/reviewer driver.

Phase 1 scaffold: signatures only. Phase 2 fills in the node bodies and
routing functions. See `.bourbaki/plans/proposer-builder-loop.md` for the
design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
)

if TYPE_CHECKING:
    from bourbaki.benchmarks.loader import MiniF2FProblem
    from bourbaki.tools.lean_repl import LeanREPLSession


class ProverConfig(BaseModel):
    """Tunable knobs for one ProverLoop run."""

    model: str = "glm:glm-5.1"
    max_iterations: int = 50
    reviewer_model: str | None = None
    memory_cls: str = "ExperienceMemory"
    memory_k: int = 2
    enable_tools: bool = True
    verify_on_approve: bool = True
    build_timeout: float = 60.0


class ProverLoop:
    """Drives the proposer → builder → reviewer cycle for a single theorem.

    All node bodies and routing helpers are stubbed in Phase 1 — they raise
    ``NotImplementedError`` until Phase 2 wires the actual loop logic.
    """

    def __init__(self, config: ProverConfig, session: LeanREPLSession) -> None:
        self.config = config
        self.session = session

    async def run(self, problem: MiniF2FProblem) -> ProverState:
        """Main entry point. Drives the loop until terminal state and returns
        the final :class:`ProverState`.
        """
        raise NotImplementedError("Phase 2 will implement ProverLoop.run")

    async def _proposer(self, state: ProverState) -> ProposalMessage | FeedbackMessage:
        raise NotImplementedError("Phase 2 will implement ProverLoop._proposer")

    async def _builder(self, state: ProverState) -> FeedbackMessage:
        raise NotImplementedError("Phase 2 will implement ProverLoop._builder")

    async def _reviewer(self, state: ProverState) -> FeedbackMessage:
        raise NotImplementedError("Phase 2 will implement ProverLoop._reviewer")

    async def _memory(self, state: ProverState) -> str:
        raise NotImplementedError("Phase 2 will implement ProverLoop._memory")

    def _route_proposer(
        self, state: ProverState
    ) -> Literal["continue", "retry", "end"]:
        raise NotImplementedError("Phase 2 will implement _route_proposer")

    def _route_builder(self, state: ProverState) -> Literal["continue", "retry"]:
        raise NotImplementedError("Phase 2 will implement _route_builder")

    def _route_reviewer(self, state: ProverState) -> Literal["continue", "retry"]:
        raise NotImplementedError("Phase 2 will implement _route_reviewer")
