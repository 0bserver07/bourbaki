"""Reviewer node for the proposer-builder-reviewer loop.

Calls GLM-5.1 (or another configurable model) with a structured-output
schema (:class:`ReviewDecision`) to check the proposed proof against two
hard rules: (1) the theorem signature is unchanged, (2) the proof body
contains no `sorry`/`admit`. Approval is derived from `check_1 AND check_2`
— `check_3` and `approved` are honeypots and ignored, mirroring ax-prover.

When the reviewer approves, we run :func:`lean_prover` once as a final
ground-truth gate. This catches cases where the warm REPL session reported
no errors but the standalone build would fail (false positives we have
already paid for elsewhere in the codebase).
"""

from __future__ import annotations

import logging
import os

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from bourbaki.prover import feedback, prompts
from bourbaki.prover.state import FeedbackMessage, ProverState, ReviewDecision
from bourbaki.tools.lean_prover import lean_prover

logger = logging.getLogger(__name__)


def _resolve_model_object(model: str) -> str | OpenAIModel | AnthropicModel:
    """Resolve a model string to a Pydantic AI model object.

    Mirrors :func:`bourbaki.agent.core._resolve_model_object` for the two
    custom providers we route through (`glm:` via z.ai's Anthropic-compatible
    endpoint, `ollama-cloud:` via OpenAI-compatible endpoint). Other strings
    pass through untouched so Pydantic AI handles them natively.
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


def _summarize_lean_errors(result: dict) -> str:
    """Render the most useful error info from a `lean_prover` result."""
    errors = result.get("errors") or []
    if errors:
        head = errors[:3]
        rendered = "; ".join(
            f"line {e.get('line', '?')}: {e.get('message', '')}" for e in head
        )
        if len(errors) > 3:
            rendered += f" (+{len(errors) - 3} more)"
        return rendered
    if result.get("error"):
        return str(result["error"])
    raw = result.get("rawOutput") or ""
    if raw:
        return raw[:400]
    return "unknown error"


async def run_reviewer(
    state: ProverState, model: str = "glm:glm-5.1"
) -> FeedbackMessage:
    """Run the reviewer node and return a typed feedback message.

    Returns one of: :func:`feedback.review_approved`, :func:`feedback.review_rejected`.
    Errors from the LLM call (validation failures, unexpected behaviour, generic
    exceptions) collapse into a `review_rejected` so the loop can retry rather
    than crash.
    """
    if state.last_proposal is None:
        return feedback.review_rejected(
            "reviewer invoked with no proposal in state; nothing to review."
        )

    user_prompt = prompts.REVIEWER_USER_PROMPT.format(
        original_theorem=state.target_theorem,
        proposed_proof=state.last_proposal.code,
    )

    try:
        resolved_model = _resolve_model_object(model)
        agent: Agent[None, ReviewDecision] = Agent(
            resolved_model,
            output_type=ReviewDecision,
            system_prompt=prompts.REVIEWER_SYSTEM_PROMPT,
        )
        result = await agent.run(user_prompt)
        decision: ReviewDecision = result.output
    except ValidationError as e:
        logger.warning("Reviewer structured output failed validation: %s", e)
        return feedback.review_rejected(f"reviewer LLM error: {e}")
    except UnexpectedModelBehavior as e:
        logger.warning("Reviewer unexpected model behaviour: %s", e)
        return feedback.review_rejected(f"reviewer LLM error: {e}")
    except Exception as e:  # noqa: BLE001 - last-resort safety net
        logger.exception("Reviewer LLM call failed")
        return feedback.review_rejected(f"reviewer LLM error: {e}")

    # Approval is derived from check_1 AND check_2 only. check_3 and the
    # `approved` field are honeypots — see plan §5.2.
    approved = bool(decision.check_1 and decision.check_2)

    if not approved:
        return feedback.review_rejected(decision.reasoning)

    # Final ground-truth gate: standalone lean_prover compile.
    parts: list[str] = []
    if "import " not in state.last_proposal.code:
        parts.append("import Mathlib")
    if state.preamble.strip():
        parts.append(state.preamble.strip())
    parts.append(state.last_proposal.code)
    full_source = "\n\n".join(parts)

    try:
        verify_result = await lean_prover(code=full_source, mode="check")
    except Exception as e:  # noqa: BLE001
        logger.exception("Reviewer final lean_prover call raised")
        return feedback.review_rejected(
            f"final lean_prover verification failed: {e}"
        )

    if verify_result.get("success"):
        return feedback.review_approved(decision.reasoning)

    error_summary = _summarize_lean_errors(verify_result)
    return feedback.review_rejected(
        f"final lean_prover verification failed: {error_summary}"
    )
