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

from bourbaki.prover import feedback as feedback_module
from bourbaki.prover import memory as memory_module
from bourbaki.prover.builder import run_builder
from bourbaki.prover.feedback import max_iterations as max_iterations_feedback
from bourbaki.prover.parallel_proposer import parallel_propose
from bourbaki.prover.proposer import run_proposer
from bourbaki.prover.reviewer import _summarize_lean_errors, run_reviewer
from bourbaki.prover.reviewer_fanout import fanout_review
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverState,
)
from bourbaki.tools.lean_prover import lean_prover
from bourbaki.tools.proof_code_builder import assemble_standalone_proof

if TYPE_CHECKING:
    from bourbaki.benchmarks.loader import MiniF2FProblem
    from bourbaki.prover.parallel_proposer import ProposerConfig
    from bourbaki.prover.reviewer_fanout import ReviewerConfig
    from bourbaki.tools.lean_repl import LeanREPLSession


logger = logging.getLogger(__name__)


class ProverConfig(BaseModel):
    model: str = "glm:glm-5.1"
    max_iterations: int = 50
    reviewer_model: str | None = None
    memory_cls: str = "MemorylessMemory"
    memory_k: int = 2
    enable_tools: bool = True
    # Phase 4: opt-in registration of the mathlib_search tool on the
    # proposer agent. Off by default — the loop already hits 90% on the
    # 10-problem subset without it, so enabling tool calls is an
    # experimental dial, not a baseline.
    enable_mathlib_search: bool = False
    verify_on_approve: bool = True
    build_timeout: float = 60.0
    # Per-LLM-call timeouts on the proposer / reviewer nodes. Defaults
    # match the module-level constants in proposer.py / reviewer.py so
    # legacy callers see identical behaviour; frontier models (e.g.
    # Kimi K2.6 1T, gpt-oss:120b) routinely exceed 90s on a single
    # proposer call, so the timeout has to be liftable from the config
    # layer rather than baked into the node modules.
    proposer_llm_timeout: float = 90.0
    reviewer_llm_timeout: float = 60.0
    # Multi-provider parallel-fanout proposer. When ``None`` (default),
    # ``ProverLoop._proposer`` calls :func:`run_proposer` once with
    # ``model`` / ``proposer_llm_timeout`` (single-model legacy path).
    # When a non-empty list is supplied, the loop routes through
    # :func:`parallel_propose` and races those configs; the single-model
    # ``model`` / ``proposer_llm_timeout`` fields are ignored for the
    # proposer step. Reviewer is unaffected — always single-model.
    proposer_configs: list["ProposerConfig"] | None = None
    # Race mode forwarded to :func:`parallel_propose`. Only consulted
    # when ``proposer_configs`` is non-empty. ``"first_completed"``
    # (default) preserves the original speed-wins-the-race primitive.
    # ``"first_verified"`` gates each leg on ``lean_prover`` before
    # declaring a winner — trades latency for compile-correctness.
    proposer_race_mode: Literal["first_completed", "first_verified"] = (
        "first_completed"
    )
    # Multi-model reviewer fanout. When ``None`` (default),
    # ``ProverLoop._reviewer`` calls :func:`run_reviewer` once with
    # ``reviewer_model`` / ``reviewer_llm_timeout`` (single-model legacy
    # path). When a non-empty list is supplied, the loop routes through
    # :func:`fanout_review` and the single-model fields are ignored for
    # the reviewer step. The ``lean_prover`` final gate is run exactly
    # once after the ensemble's decision — never per leg — so adding
    # legs costs N LLM calls but only 1 standalone compile.
    reviewer_configs: list["ReviewerConfig"] | None = None
    # Combination strategy for the reviewer fanout. Only consulted when
    # ``reviewer_configs`` is non-empty. ``"first_approved"`` (default)
    # races the legs and accepts the first APPROVE; pending legs are
    # cancelled. ``"all_must_approve"`` waits for every leg and accepts
    # only if all approve — strictly more conservative.
    reviewer_review_mode: Literal["first_approved", "all_must_approve"] = (
        "first_approved"
    )

    model_config = {"arbitrary_types_allowed": True}


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


def _last_feedback_kind(state: ProverState) -> str:
    """Return the ``kind`` of the most recent feedback message, for logging.

    Routing decisions are driven by the latest feedback (builder/reviewer)
    or by a non-proposal proposer result; surfacing the kind makes a route
    log line self-explanatory. Returns ``"-"`` when no feedback is set.
    """
    fb = state.last_feedback
    return fb.kind if isinstance(fb, FeedbackMessage) else "-"


class ProverLoop:
    """Drives the proposer → builder → reviewer cycle for a single theorem."""

    def __init__(self, config: ProverConfig, session: LeanREPLSession) -> None:
        self.config = config
        self.session = session
        self.memory = _build_memory(config)

    async def run(self, problem: MiniF2FProblem) -> ProverState:
        # Reconstruct the placeholder body the loader stripped so the
        # proposer's <target> block has an explicit ``:= sorry`` to replace.
        target_with_placeholder = f"{problem.statement} := by\n  sorry"

        state = ProverState(
            problem_id=problem.id,
            target_theorem=target_with_placeholder,
            preamble=_extract_preamble(problem.full_lean_code, problem.statement),
            full_file=problem.full_lean_code,
            iteration=0,
            max_iterations=self.config.max_iterations,
        )

        logger.info(
            "prover loop: entering for problem '%s' (max_iterations=%d, "
            "proposer=%s, reviewer=%s)",
            problem.id,
            state.max_iterations,
            "fanout" if self.config.proposer_configs else "single",
            "fanout" if self.config.reviewer_configs else "single",
        )

        # Loop-owned attempt counter — bumped on every proposer call so a
        # streak of parsing failures (which never bump ``state.iteration``,
        # since that tracks successful proposals) can't loop forever.
        attempts = 0

        while True:
            attempts += 1
            if attempts > state.max_iterations:
                terminal = max_iterations_feedback(state.max_iterations)
                state.messages.append(terminal)
                state.last_feedback = terminal
                logger.info(
                    "prover loop: '%s' attempts=%d exceeded max_iterations=%d "
                    "— ending",
                    state.problem_id,
                    attempts,
                    state.max_iterations,
                )
                break

            # --- Proposer ---
            proposed = await self._proposer(state)
            state.messages.append(proposed)
            if isinstance(proposed, ProposalMessage):
                state.last_proposal = proposed
                state.iteration = proposed.iteration
            else:
                state.last_feedback = proposed

            route = self._route_proposer(state)
            logger.info(
                "prover route: node=proposer outcome=%s problem='%s' "
                "iteration=%d attempts=%d feedback_kind=%s",
                route,
                state.problem_id,
                state.iteration,
                attempts,
                _last_feedback_kind(state),
            )
            if route == "end":
                break
            if route == "retry":
                state.experience = await self._memory(state)
                continue

            # --- Builder ---
            build_fb = await self._builder(state)
            state.messages.append(build_fb)
            state.last_feedback = build_fb

            build_route = self._route_builder(state)
            logger.info(
                "prover route: node=builder outcome=%s problem='%s' "
                "iteration=%d attempts=%d feedback_kind=%s",
                build_route,
                state.problem_id,
                state.iteration,
                attempts,
                _last_feedback_kind(state),
            )
            if build_route == "end":
                break
            if build_route == "retry":
                state.experience = await self._memory(state)
                continue

            # --- Reviewer ---
            review_fb = await self._reviewer(state)
            state.messages.append(review_fb)
            state.last_feedback = review_fb

            review_route = self._route_reviewer(state)
            logger.info(
                "prover route: node=reviewer outcome=%s problem='%s' "
                "iteration=%d attempts=%d feedback_kind=%s",
                review_route,
                state.problem_id,
                state.iteration,
                attempts,
                _last_feedback_kind(state),
            )

            if review_fb.kind == "review_approved":
                state.approved = True
                # The reviewer already invoked lean_prover with the same
                # standalone source we persist here, so a success implies
                # the file builds clean. Persisting the assembled source
                # (preamble + proposal) is critical: the outer benchmark
                # verifier checks ``proof_code`` directly, and missing
                # ``set_option maxHeartbeats 0`` from the preamble can
                # turn an honest success into a phantom false positive.
                state.verified = True
                state.final_proof_code = (
                    assemble_standalone_proof(
                        state.preamble, state.last_proposal.code
                    )
                    if state.last_proposal
                    else None
                )
                break

            # rejection → memory → next proposer iteration
            state.experience = await self._memory(state)

        logger.info(
            "prover loop: exiting '%s' — verified=%s approved=%s "
            "iteration=%d attempts=%d",
            state.problem_id,
            state.verified,
            state.approved,
            state.iteration,
            attempts,
        )
        return state

    # ----- Node delegators -----

    async def _proposer(self, state: ProverState) -> ProposalMessage | FeedbackMessage:
        # Multi-provider fanout: when ``proposer_configs`` is set, race
        # the supplied legs and take the first valid ProposalMessage.
        # The single-model ``model`` / ``proposer_llm_timeout`` /
        # ``enable_mathlib_search`` fields are ignored in this path —
        # each ``ProposerConfig`` carries its own.  No outer cap is
        # imposed here: per-leg ``llm_timeout`` governs.
        if self.config.proposer_configs:
            return await parallel_propose(
                state,
                self.config.proposer_configs,
                race_mode=self.config.proposer_race_mode,
            )
        return await run_proposer(
            state,
            model=self.config.model,
            llm_timeout=self.config.proposer_llm_timeout,
            enable_mathlib_search=self.config.enable_mathlib_search,
        )

    async def _builder(self, state: ProverState) -> FeedbackMessage:
        return await run_builder(
            state, self.session, timeout=self.config.build_timeout
        )

    async def _reviewer(self, state: ProverState) -> FeedbackMessage:
        # Multi-model reviewer fanout: when ``reviewer_configs`` is set,
        # race the supplied legs and combine their decisions per
        # ``reviewer_review_mode``. The single-model ``reviewer_model`` /
        # ``reviewer_llm_timeout`` fields are ignored in this path —
        # each ``ReviewerConfig`` carries its own. The fanout itself
        # only invokes :func:`run_reviewer` with ``decide_only=True``,
        # so the per-leg behaviour stops at the LLM judgement; we run
        # :func:`lean_prover` here exactly once after the ensemble has
        # agreed to spare the linear-with-leg-count compile cost.
        if self.config.reviewer_configs:
            ensemble = await fanout_review(
                state,
                self.config.reviewer_configs,
                mode=self.config.reviewer_review_mode,
            )
            logger.info(
                "reviewer fanout: ensemble for '%s' returned kind=%s",
                state.problem_id,
                ensemble.kind,
            )
            if ensemble.kind != "review_approved":
                return ensemble
            return await self._final_lean_gate(state, ensemble)
        return await run_reviewer(
            state,
            model=self.config.reviewer_model or self.config.model,
            llm_timeout=self.config.reviewer_llm_timeout,
        )

    async def _final_lean_gate(
        self, state: ProverState, ensemble_approval: FeedbackMessage,
    ) -> FeedbackMessage:
        """Run the ground-truth ``lean_prover`` compile once after the
        multi-leg ensemble has approved.

        Mirrors the second half of :func:`run_reviewer`: same assembled
        source, same ``proofComplete`` check, same error rendering. The
        only reason this lives on :class:`ProverLoop` rather than in
        :mod:`reviewer_fanout` is that the fanout module is intentionally
        Lean-agnostic — it composes :func:`run_reviewer` results and
        nothing else.
        """
        if state.last_proposal is None:
            return feedback_module.review_rejected(
                "reviewer fanout approved but state.last_proposal is None; "
                "cannot run final lean gate."
            )
        full_source = assemble_standalone_proof(
            state.preamble, state.last_proposal.code
        )
        try:
            verify_result = await lean_prover(
                code=full_source, mode="check", timeout=240,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "Reviewer-fanout final lean_prover call raised"
            )
            return feedback_module.review_rejected(
                f"final lean_prover verification failed: {e}"
            )

        if verify_result.get("proofComplete"):
            logger.info(
                "reviewer fanout: final lean gate PASSED for '%s'",
                state.problem_id,
            )
            return ensemble_approval

        error_summary = _summarize_lean_errors(verify_result)
        logger.info(
            "reviewer fanout: final lean gate FAILED for '%s' — %s",
            state.problem_id,
            error_summary,
        )
        return feedback_module.review_rejected(
            f"final lean_prover verification failed: {error_summary}"
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

    def _route_builder(
        self, state: ProverState
    ) -> Literal["continue", "retry", "end"]:
        last = state.last_feedback
        if last is None:
            return "retry"
        if last.is_terminal:
            return "end"
        return "continue" if last.is_success else "retry"

    def _route_reviewer(
        self, state: ProverState
    ) -> Literal["continue", "retry", "end"]:
        last = state.last_feedback
        if last is None:
            return "retry"
        if last.is_terminal:
            return "end"
        if isinstance(last, FeedbackMessage) and last.kind == "review_approved":
            return "continue"
        return "retry"


# Resolve the forward references on ``ProverConfig.proposer_configs`` /
# ``ProverConfig.reviewer_configs`` — ``ProposerConfig`` and
# ``ReviewerConfig`` are frozen dataclasses living in
# ``parallel_proposer`` / ``reviewer_fanout`` and are only imported
# under ``TYPE_CHECKING`` above to avoid mixing the Pydantic field
# type-resolution with runtime imports.  Without this rebuild Pydantic
# raises ``PydanticUserError`` on first instantiation.
from bourbaki.prover.parallel_proposer import ProposerConfig as _ProposerConfig  # noqa: E402
from bourbaki.prover.reviewer_fanout import ReviewerConfig as _ReviewerConfig  # noqa: E402

ProverConfig.model_rebuild(
    _types_namespace={
        "ProposerConfig": _ProposerConfig,
        "ReviewerConfig": _ReviewerConfig,
    }
)
