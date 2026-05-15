"""Memory strategies for the proposer-builder-reviewer loop.

Three policies, mirroring ax-prover:

- :class:`MemorylessMemory` — forget everything, return empty experience.
- :class:`PreviousKMemory` — render the last K (proposal, feedback) pairs verbatim.
- :class:`ExperienceMemory` — call GLM-5.1 once per retry to compress prior lessons
  into an `<experience>...</experience>` block.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from pydantic_ai import Agent

from bourbaki.providers import resolve_model_object
from bourbaki.prover import prompts
from bourbaki.prover.state import FeedbackMessage, ProposalMessage, ProverState

logger = logging.getLogger(__name__)


# Kept as a module-level alias so existing tests and external callers that
# import ``memory._resolve_model_object`` (and monkeypatch it) keep working.
# See :mod:`bourbaki.providers` for the routing logic.
_resolve_model_object = resolve_model_object


class BaseMemory(ABC):
    """Strategy that turns recent loop history into a fresh experience string."""

    @abstractmethod
    async def process(self, state: ProverState) -> str:
        """Return the new value for `state.experience`."""


class MemorylessMemory(BaseMemory):
    """Returns an empty experience string. Default for the first cut."""

    async def process(self, state: ProverState) -> str:
        return ""


class PreviousKMemory(BaseMemory):
    """Renders the last K (ProposalMessage, FeedbackMessage) pairs verbatim."""

    def __init__(self, k: int = 2) -> None:
        self.k = k

    async def process(self, state: ProverState) -> str:
        msgs = state.messages
        pairs: list[tuple[ProposalMessage, FeedbackMessage]] = []

        # Walk messages backwards, pairing each FeedbackMessage with its
        # most-recent preceding ProposalMessage.
        i = len(msgs) - 1
        while i >= 0 and len(pairs) < self.k:
            msg = msgs[i]
            if isinstance(msg, FeedbackMessage):
                for j in range(i - 1, -1, -1):
                    candidate = msgs[j]
                    if isinstance(candidate, ProposalMessage):
                        pairs.append((candidate, msg))
                        break
            i -= 1

        if not pairs:
            return ""

        # Render in chronological (oldest-first) order.
        pairs.reverse()
        rendered = "\n\n".join(
            prompts.ATTEMPT_TEMPLATE.format(
                reasoning=proposal.reasoning,
                code=proposal.code,
                feedback=fb.content,
            )
            for proposal, fb in pairs
        )
        return f"<previous-attempts>\n{rendered}\n</previous-attempts>"


_EXPERIENCE_LLM_TIMEOUT = 60.0  # seconds — per-call cap on the LLM summary


class ExperienceMemory(BaseMemory):
    """LLM-summarized experience block, regenerated each iteration."""

    def __init__(
        self,
        model: str = "glm:glm-5.1",
        llm_timeout: float = _EXPERIENCE_LLM_TIMEOUT,
    ) -> None:
        self.model = model
        self.llm_timeout = llm_timeout

    async def process(self, state: ProverState) -> str:
        # Defensive: if we don't have a full last attempt to summarize,
        # leave the existing experience untouched.
        if state.last_proposal is None or state.last_feedback is None:
            return state.experience

        last_attempt = prompts.ATTEMPT_TEMPLATE.format(
            reasoning=state.last_proposal.reasoning,
            code=state.last_proposal.code,
            feedback=state.last_feedback.content,
        )
        user_prompt = prompts.EXPERIENCE_USER_PROMPT.format(
            attempt_template=last_attempt,
            previous_context=state.experience or "",
        )

        try:
            resolved_model = _resolve_model_object(self.model)
            agent: Agent[None, str] = Agent(
                resolved_model,
                output_type=str,
                system_prompt=prompts.EXPERIENCE_SYSTEM_PROMPT,
            )
            result = await asyncio.wait_for(
                agent.run(user_prompt), timeout=self.llm_timeout
            )
            summary = result.output
        except asyncio.TimeoutError:
            # Surface the duration so a load-induced stall is observable in
            # the logs rather than silently falling back to stale experience.
            logger.warning(
                "ExperienceMemory LLM call exceeded %.0fs timeout — "
                "keeping prior experience",
                self.llm_timeout,
            )
            return state.experience
        except Exception:  # noqa: BLE001 — last-resort safety net
            logger.exception(
                "ExperienceMemory LLM call failed; keeping prior experience"
            )
            # Fall back to whatever experience we already had so the loop
            # doesn't lose context due to a transient API hiccup or hang.
            return state.experience

        return f"<experience>\n{summary}\n</experience>"
