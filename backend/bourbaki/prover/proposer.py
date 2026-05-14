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
import json
import logging
import os

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import Tool

from bourbaki.prover import feedback, prompts
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverResult,
    ProverState,
)

logger = logging.getLogger(__name__)


# z.ai's OpenAI-compatible chat-completions endpoint. Documented base URL,
# used with ``OpenAIProvider`` so the Anthropic adapter's
# ``args_as_dict`` retry-mapping path (the one that explodes on z.ai's
# tool-use response shape) is never hit. See issue #13.
_ZAI_OPENAI_BASE_URL = "https://api.z.ai/api/paas/v4/"


def _resolve_model_object(model_str: str) -> str | OpenAIChatModel:
    """Resolve a model string into a Pydantic AI model object.

    Supports the same ``glm:`` and ``ollama-cloud:`` prefixes as
    :func:`bourbaki.agent.core._resolve_model_object`. Anything else falls
    through and is handed to Pydantic AI as a plain provider string
    (e.g. ``openai:gpt-4o``).

    ``glm:`` was previously routed through ``AnthropicProvider`` against
    z.ai's Anthropic-compatible endpoint (``https://api.z.ai/api/anthropic``).
    That path crashes during pydantic_ai's retry-message re-mapping with
    ``TypeError: Expected bytes, bytearray or str`` inside
    ``messages.py:args_as_dict`` whenever z.ai's structured-output
    response feeds a non-string/non-dict back into a ``ToolCallPart``.
    PutnamBench problems reliably trigger it (miniF2F mostly does not).
    Switching to z.ai's OpenAI-compatible endpoint side-steps the bug
    entirely — the OpenAI message-mapping code does not use
    ``args_as_dict`` in the same way. See issue #13.
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
        provider = OpenAIProvider(
            base_url=_ZAI_OPENAI_BASE_URL,
            api_key=api_key,
        )
        return OpenAIChatModel(model_name, provider=provider)

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


async def _search_mathlib(
    query: str,
    mode: str = "name",
    max_results: int = 5,
) -> str:
    """Search Mathlib for lemmas.

    Use this tool to find existing lemmas by name (e.g. ``Nat.add_comm``),
    by type signature in Loogle syntax (e.g. ``_ * (_ ^ _)``), or in plain
    English. Modes:

    - ``name`` (default): exact / partial name lookup via Loogle.
    - ``type``: type-signature search via Loogle.
    - ``natural``: natural-language query via LeanSearch.
    - ``semantic``: semantic search via LeanExplore (hybrid ranking).
    - ``local``: offline FAISS embedding index (fastest).

    Returns a JSON string with ``success``, ``results`` (a list of
    ``{name, module, type, doc}``), ``count``, ``query``, ``mode``,
    ``duration``. On failure returns ``{success: False, error: ...}``.
    """
    from bourbaki.tools.mathlib_search import mathlib_search

    result = await mathlib_search(query=query, mode=mode, max_results=max_results)
    return json.dumps(result, default=str)


def _build_proposer_tools(
    enable_mathlib_search: bool,
    proposer_tools: list | None,
) -> list[Tool]:
    """Resolve the tool list passed to the Pydantic AI agent.

    When ``enable_mathlib_search`` is True, the ``mathlib_search`` tool is
    registered unconditionally (and ``proposer_tools`` is ignored). When
    False, fall back to ``proposer_tools``; ``None`` or empty yields no
    tools.
    """
    if enable_mathlib_search:
        return [Tool(_search_mathlib, name="mathlib_search")]
    if proposer_tools:
        return list(proposer_tools)
    return []


async def run_proposer(
    state: ProverState,
    model: str = "glm:glm-5.1",
    proposer_tools: list | None = None,
    llm_timeout: float = _PROPOSER_LLM_TIMEOUT,
    enable_mathlib_search: bool = False,
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
        proposer_tools: Optional pre-built ``Tool`` list passed straight
            to ``Agent(..., tools=...)``. Used only when
            ``enable_mathlib_search`` is False.
        llm_timeout: Per-LLM-call timeout in seconds.
        enable_mathlib_search: When True, register the built-in
            ``mathlib_search`` tool (Loogle + LeanSearch + LeanExplore +
            local FAISS) so the proposer can look up Mathlib lemmas
            mid-iteration. Overrides ``proposer_tools``.
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

    # Phase 4: optionally register mathlib_search (and any other caller-
    # supplied tools) so the proposer can ground its proposals in real
    # Mathlib lemmas before emitting structured output.
    tools = _build_proposer_tools(enable_mathlib_search, proposer_tools)

    # 5. Create the Pydantic AI agent with strict structured output.
    agent: Agent[None, ProverResult] = Agent(
        resolved_model,
        system_prompt=sys_prompt,
        output_type=ProverResult,
        tools=tools,
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
