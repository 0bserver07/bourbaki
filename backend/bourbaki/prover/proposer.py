"""Proposer node for the proposer-builder-reviewer loop.

This module implements the LLM-driven proposer step. It builds the
user/system prompt from the current :class:`ProverState`, calls the model
through Pydantic AI with a strict ``ProverResult`` output schema, and
returns either a :class:`ProposalMessage` (success) or a
:class:`FeedbackMessage` (terminal / parse error).

Mirrors ``ax-prover``'s ``_proposer_node`` (LangGraph) but uses Pydantic AI
directly — no LangChain plumbing.
"""

from __future__ import annotations

import asyncio
import logging
import os

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from bourbaki.prover import feedback, prompts
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverResult,
    ProverState,
)

logger = logging.getLogger(__name__)


def _resolve_model_object(model_str: str) -> str | OpenAIChatModel | AnthropicModel:
    """Resolve a model string into a Pydantic AI model object.

    Supports the same ``glm:`` and ``ollama-cloud:`` prefixes as
    :func:`bourbaki.agent.core._resolve_model_object`. Anything else falls
    through and is handed to Pydantic AI as a plain provider string
    (e.g. ``openai:gpt-4o``).
    """

    if model_str.startswith("ollama-cloud:"):
        model_name = model_str.removeprefix("ollama-cloud:")
        api_key = os.environ.get("OLLAMA_CLOUD_API_KEY", "ollama")
        provider = OpenAIProvider(
            base_url="https://ollama.com/v1",
            api_key=api_key,
        )
        return OpenAIChatModel(model_name, provider=provider)

    if model_str.startswith("glm:"):
        model_name = model_str.removeprefix("glm:")
        api_key = os.environ.get("GLM_API_KEY", "")
        provider = AnthropicProvider(
            base_url="https://api.z.ai/api/anthropic",
            api_key=api_key,
        )
        return AnthropicModel(model_name, provider=provider)

    return model_str


def _build_user_message(state: ProverState) -> str:
    """Render the proposer user prompt from the current state.

    Order: base prompt → previous-attempt block (if any) → experience block
    (if any). Joined with double newlines, matching ax-prover.
    """

    parts: list[str] = [
        prompts.PROPOSER_USER_PROMPT.format(
            target_theorem=state.target_theorem,
            complete_file=state.full_file,
        )
    ]

    if state.last_proposal is not None and state.last_feedback is not None:
        attempt = prompts.ATTEMPT_TEMPLATE.format(
            reasoning=state.last_proposal.reasoning,
            code=state.last_proposal.code,
            feedback=state.last_feedback.content,
        )
        parts.append(prompts.PREVIOUS_ATTEMPT_USER_PROMPT.format(attempt=attempt))

    if state.experience:
        parts.append(state.experience)

    return "\n\n".join(parts)


_PROPOSER_LLM_TIMEOUT = 90.0  # seconds — per-call cap to prevent hung iterations


async def run_proposer(
    state: ProverState,
    model: str = "glm:glm-5.1",
    proposer_tools: list | None = None,
    llm_timeout: float = _PROPOSER_LLM_TIMEOUT,
) -> ProposalMessage | FeedbackMessage:
    """Run one proposer step.

    Returns:
        - :class:`ProposalMessage` on a successful structured-output call.
        - terminal :class:`FeedbackMessage` (``max_iterations``) when the
          iteration budget is exhausted — no LLM call is made.
        - :class:`FeedbackMessage` of kind ``structured_output_parsing_failed``
          on validation / model errors.

    Args:
        state: Current loop state. ``state.iteration`` is the count of
            *completed* iterations; the proposal returned will carry
            ``iteration = state.iteration + 1``.
        model: Pydantic AI model string. Custom prefixes ``glm:`` and
            ``ollama-cloud:`` are resolved via :func:`_resolve_model_object`.
        proposer_tools: Reserved for future tool wiring (e.g. ``mathlib_search``).
            Currently ignored.
    """

    # 1. Iteration-limit guard — never call the LLM if we're out of budget.
    if state.iteration >= state.max_iterations:
        logger.warning(
            "Proposer iteration budget exhausted (%s/%s); returning terminal feedback.",
            state.iteration,
            state.max_iterations,
        )
        return feedback.max_iterations(state.max_iterations)

    # 2. Build user message.
    user_msg = _build_user_message(state)

    # 3. Resolve the model.
    resolved_model = _resolve_model_object(model)

    # 4. Pick the system prompt (single-shot vs iterative).
    sys_prompt = (
        prompts.PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT
        if state.max_iterations == 1
        else prompts.PROPOSER_SYSTEM_PROMPT
    )

    # TODO: register proposer_tools when Phase 4 lands (mathlib_search, etc.)
    _ = proposer_tools  # silence "unused" linters until Phase 4

    # 5. Create the Pydantic AI agent with strict structured output.
    agent: Agent[None, ProverResult] = Agent(
        resolved_model,
        system_prompt=sys_prompt,
        output_type=ProverResult,
        retries=2,
    )

    # 6. Call the LLM and validate. Per-call timeout prevents one slow
    # proposal from consuming the whole per-problem budget (we saw
    # mathd_algebra_31 hang for 300s with attempts=0).
    try:
        result = await asyncio.wait_for(agent.run(user_msg), timeout=llm_timeout)
        output: ProverResult = result.output
    except asyncio.TimeoutError:
        logger.error("Proposer LLM call exceeded %.0fs timeout", llm_timeout)
        return feedback.structured_output_parsing_failed(
            f"LLM call timed out after {llm_timeout:.0f}s"
        )
    except ValidationError as exc:
        logger.error("Proposer structured output validation failed: %s", exc)
        return feedback.structured_output_parsing_failed(str(exc))
    except UnexpectedModelBehavior as exc:
        logger.error("Proposer model returned unexpected behavior: %s", exc)
        return feedback.structured_output_parsing_failed(str(exc))
    except Exception as exc:  # noqa: BLE001 — must not let exceptions bubble up
        logger.exception("Proposer call failed: %s", exc)
        return feedback.structured_output_parsing_failed(str(exc))

    # 7. Build the proposal.
    return ProposalMessage(
        reasoning=output.reasoning,
        imports=output.imports,
        opens=output.opens,
        code=output.updated_theorem,
        iteration=state.iteration + 1,
    )
