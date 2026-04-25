"""Memory strategies for the proposer-builder-reviewer loop.

Three policies, mirroring ax-prover:

- :class:`MemorylessMemory` — forget everything, return empty experience.
- :class:`PreviousKMemory` — render the last K (proposal, feedback) pairs verbatim.
- :class:`ExperienceMemory` — call GLM-5.1 once per retry to compress prior lessons
  into an `<experience>...</experience>` block.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from bourbaki.prover import prompts
from bourbaki.prover.state import FeedbackMessage, ProposalMessage, ProverState

logger = logging.getLogger(__name__)


def _resolve_model_object(model: str) -> str | OpenAIModel | AnthropicModel:
    """Resolve a model string to a Pydantic AI model object.

    Duplicated from :mod:`bourbaki.prover.reviewer` deliberately — the design
    doc keeps memory and reviewer free of cross-module dependencies during
    Phase 2 so the two can ship and be tested independently.
    """
    if model.startswith("ollama-cloud:"):
        model_name = model.removeprefix("ollama-cloud:")
        api_key = os.environ.get("OLLAMA_CLOUD_API_KEY", "ollama")
        provider = OpenAIProvider(
            base_url="https://ollama.com/v1",
            api_key=api_key,
        )
        return OpenAIModel(model_name, provider=provider)

    if model.startswith("glm:"):
        model_name = model.removeprefix("glm:")
        api_key = os.environ.get("GLM_API_KEY", "")
        provider = AnthropicProvider(
            base_url="https://api.z.ai/api/anthropic",
            api_key=api_key,
        )
        return AnthropicModel(model_name, provider=provider)

    return model


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


class ExperienceMemory(BaseMemory):
    """LLM-summarized experience block, regenerated each iteration."""

    def __init__(self, model: str = "glm:glm-5.1") -> None:
        self.model = model

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
            result = await agent.run(user_prompt)
            summary = result.output
        except Exception as e:  # noqa: BLE001
            logger.exception("ExperienceMemory LLM call failed; keeping prior experience")
            # Fall back to whatever experience we already had so the loop
            # doesn't lose context due to a transient API hiccup.
            return state.experience

        return f"<experience>\n{summary}\n</experience>"
