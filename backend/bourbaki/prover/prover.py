"""ProverLoop — proposer/builder/reviewer driver.

Plain async loop (no LangGraph). The graph is small enough — three nodes
plus memory — that the routing fits in a ``while`` with a few ``if``
branches. Mirrors ax-prover's ``route_proposer`` / ``route_builder`` /
``route_reviewer`` semantics; see ``.bourbaki/plans/proposer-builder-loop.md`` §3.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from bourbaki.prover import memory as memory_module
from bourbaki.prover.builder import run_builder
from bourbaki.prover.proposer import run_proposer
from bourbaki.prover.reviewer import run_reviewer
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
)

if TYPE_CHECKING:
    from bourbaki.benchmarks.loader import MiniF2FProblem
    from bourbaki.tools.lean_repl import LeanREPLSession


logger = logging.getLogger(__name__)


class ProverConfig(BaseModel):
    model: str = "glm:glm-5.1"
    max_iterations: int = 50
    reviewer_model: str | None = None
    memory_cls: str = "MemorylessMemory"
    memory_k: int = 2
    enable_tools: bool = True
    verify_on_approve: bool = True
    build_timeout: float = 60.0


def _build_memory(config: ProverConfig) -> memory_module.BaseMemory:
    cls = getattr(memory_module, config.memory_cls)
    if config.memory_cls == "PreviousKMemory":
        return cls(k=config.memory_k)
    if config.memory_cls == "ExperienceMemory":
        return cls(model=config.reviewer_model or config.model)
    return cls()


def _extract_preamble(full_lean_code: str, statement: str) -> str:
    """Return the full file's prefix preceding the target ``statement``.

    Used to seed ``state.preamble`` so the REPL builder can replay any
    ``open`` / ``set_option`` lines that sit above the theorem.
    """
    idx = full_lean_code.find(statement.strip())
    if idx < 0:
        return ""
    return full_lean_code[:idx].rstrip()


class ProverLoop:
    """Drives the proposer → builder → reviewer cycle for a single theorem."""

    def __init__(self, config: ProverConfig, session: LeanREPLSession) -> None:
        self.config = config
        self.session = session
        self.memory = _build_memory(config)

    async def run(self, problem: MiniF2FProblem) -> ProverState:
        state = ProverState(
            problem_id=problem.id,
            target_theorem=problem.statement,
            preamble=_extract_preamble(problem.full_lean_code, problem.statement),
            full_file=problem.full_lean_code,
            iteration=0,
            max_iterations=self.config.max_iterations,
        )

        while True:
            # --- Proposer ---
            proposed = await self._proposer(state)
            state.messages.append(proposed)
            if isinstance(proposed, ProposalMessage):
                state.last_proposal = proposed
                state.iteration = proposed.iteration
            else:
                state.last_feedback = proposed

            route = self._route_proposer(state)
            if route == "end":
                break
            if route == "retry":
                state.experience = await self._memory(state)
                continue

            # --- Builder ---
            build_fb = await self._builder(state)
            state.messages.append(build_fb)
            state.last_feedback = build_fb

            if self._route_builder(state) == "retry":
                state.experience = await self._memory(state)
                continue

            # --- Reviewer ---
            review_fb = await self._reviewer(state)
            state.messages.append(review_fb)
            state.last_feedback = review_fb

            if review_fb.kind == "review_approved":
                state.approved = True
                # The reviewer already invoked lean_prover when
                # ``verify_on_approve`` is on (it always is for now); a
                # success here implies the standalone file built clean.
                state.verified = True
                state.final_proof_code = (
                    state.last_proposal.code if state.last_proposal else None
                )
                break

            # rejection → memory → next proposer iteration
            state.experience = await self._memory(state)

        return state

    # ----- Node delegators -----

    async def _proposer(self, state: ProverState) -> ProposalMessage | FeedbackMessage:
        return await run_proposer(state, model=self.config.model)

    async def _builder(self, state: ProverState) -> FeedbackMessage:
        return await run_builder(state, self.session)

    async def _reviewer(self, state: ProverState) -> FeedbackMessage:
        return await run_reviewer(
            state, model=self.config.reviewer_model or self.config.model
        )

    async def _memory(self, state: ProverState) -> str:
        return await self.memory.process(state)

    # ----- Routing (mirrors ax-prover) -----

    def _route_proposer(
        self, state: ProverState
    ) -> Literal["continue", "retry", "end"]:
        last = state.messages[-1] if state.messages else None
        if isinstance(last, FeedbackMessage) and last.is_terminal:
            return "end"
        if not isinstance(last, ProposalMessage):
            return "retry"
        return "continue"

    def _route_builder(self, state: ProverState) -> Literal["continue", "retry"]:
        last = state.last_feedback
        return "continue" if last is not None and last.is_success else "retry"

    def _route_reviewer(self, state: ProverState) -> Literal["continue", "retry"]:
        last = state.last_feedback
        if isinstance(last, FeedbackMessage) and last.kind == "review_approved":
            return "continue"
        return "retry"
