"""Native Ollama adapter for thinking-capable reasoning models.

Bypasses pydantic_ai's OpenAI-compat path because Ollama Cloud's
/v1/chat/completions endpoint silently ignores the ``think`` parameter,
causing reasoning models (Kimi K2.6, DeepSeek V4-pro, etc.) to emit
infinite reasoning chains and never produce structured content. The
native /api/chat endpoint respects ``think: false``.

Routes activated by the ``ollama-native:<model>`` prefix.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

import httpx
from pydantic import ValidationError

from bourbaki.prover import feedback, prompts
from bourbaki.prover.proposer import _build_user_message
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverResult,
    ProverState,
)

logger = logging.getLogger(__name__)


OLLAMA_NATIVE_CHAT_URL = "https://ollama.com/api/chat"
_PREFIX = "ollama-native:"
_DEFAULT_NUM_PREDICT = 8000
_DEFAULT_LLM_TIMEOUT = 90.0


_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)


def _coerce_list_fields(parsed_obj: dict) -> None:
    """Coerce single-string ``imports`` / ``opens`` to single-element lists.

    Reasoning models (Kimi K2.6 in particular) sometimes emit these
    fields as a string like ``"import Mathlib"`` instead of a list,
    despite the JSON Schema spec. Mutate the dict in place so pydantic
    validation passes. Multi-line strings are split on newlines.
    """
    for field in ("imports", "opens"):
        val = parsed_obj.get(field)
        if isinstance(val, str):
            parsed_obj[field] = [
                line.strip() for line in val.split("\n") if line.strip()
            ]


def _strip_markdown_fences(content: str) -> str:
    """Strip ```json ... ``` fences (any case) from a model response.

    Ollama Cloud's reasoning models often wrap their structured-output
    content in markdown code fences despite the ``format`` JSON Schema
    constraint. Strip them before ``json.loads``.
    """
    match = _FENCE_PATTERN.match(content)
    if match:
        return match.group(1).strip()
    return content.strip()


def _build_request_body(
    model_name: str,
    system_prompt: str,
    user_msg: str,
) -> dict:
    """Compose the native ``/api/chat`` body.

    Uses the pydantic-generated ``ProverResult.model_json_schema()`` as
    the ``format`` constraint. ``think: false`` is the load-bearing flag
    that prevents reasoning models from looping forever.
    """
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "think": False,
        "format": ProverResult.model_json_schema(),
        "stream": False,
        "options": {"num_predict": _DEFAULT_NUM_PREDICT},
    }


async def run_proposer_native(
    state: ProverState,
    model: str,
    llm_timeout: float = _DEFAULT_LLM_TIMEOUT,
    enable_mathlib_search: bool = False,
) -> ProposalMessage | FeedbackMessage:
    """Run one proposer step via Ollama's native /api/chat endpoint.

    Mirrors :func:`bourbaki.prover.proposer.run_proposer`'s contract:
    same input types, same return union, never raises.

    Args:
        state: Current loop state. ``state.iteration`` is the count of
            completed iterations; the proposal returned will carry
            ``iteration = state.iteration + 1``.
        model: Must begin with ``ollama-native:`` (e.g.
            ``ollama-native:kimi-k2.6``).
        llm_timeout: Per-LLM-call timeout in seconds; passed as the
            httpx request timeout.
        enable_mathlib_search: Accepted for API parity with
            ``run_proposer``. Native path does NOT yet support proposer
            tools — logs a WARNING and proceeds as if False.
    """

    if state.iteration >= state.max_iterations:
        logger.warning(
            "Proposer iteration budget exhausted (%s/%s); returning terminal feedback.",
            state.iteration,
            state.max_iterations,
        )
        return feedback.max_iterations(state.max_iterations)

    if not model.startswith(_PREFIX):
        return feedback.structured_output_parsing_failed(
            f"run_proposer_native expects model prefixed with '{_PREFIX}', got {model!r}"
        )
    model_name = model.removeprefix(_PREFIX).strip()
    if not model_name:
        return feedback.structured_output_parsing_failed(
            f"run_proposer_native: empty model name after stripping '{_PREFIX}' "
            f"prefix from {model!r}"
        )

    api_key = os.environ.get("OLLAMA_CLOUD_API_KEY")
    if not api_key:
        return feedback.structured_output_parsing_failed(
            "OLLAMA_CLOUD_API_KEY env var is not set; native Ollama path "
            "requires an Ollama Cloud API key."
        )

    if enable_mathlib_search:
        logger.warning(
            "run_proposer_native: enable_mathlib_search=True is not supported on "
            "the native path yet; proceeding without proposer tools."
        )

    user_msg = _build_user_message(state)
    sys_prompt = (
        prompts.PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT
        if state.max_iterations == 1
        else prompts.PROPOSER_SYSTEM_PROMPT
    )
    body = _build_request_body(model_name, sys_prompt, user_msg)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(llm_timeout)) as client:
            response = await client.post(
                OLLAMA_NATIVE_CHAT_URL,
                json=body,
                headers=headers,
            )
    except (httpx.TimeoutException, asyncio.TimeoutError):
        logger.warning(
            "run_proposer_native: %s call timed out after %.0fs", model, llm_timeout
        )
        return feedback.structured_output_parsing_failed(
            f"ollama-native LLM call timed out after {llm_timeout:.0f}s"
        )
    except Exception as exc:  # noqa: BLE001 — never let exceptions bubble up
        logger.exception("run_proposer_native: %s call failed", model)
        return feedback.structured_output_parsing_failed(
            f"native call failed: {exc}"
        )

    if response.status_code == 429:
        logger.warning("run_proposer_native: %s rate limited (429)", model)
        return feedback.structured_output_parsing_failed(
            "ollama-native rate limited (429); retry on next iteration."
        )
    if response.status_code >= 400:
        snippet = response.text[:200].replace("\n", " ")
        logger.error(
            "run_proposer_native: %s returned HTTP %s — %s",
            model,
            response.status_code,
            snippet,
        )
        return feedback.structured_output_parsing_failed(
            f"ollama-native HTTP {response.status_code} error: {snippet}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        logger.error("run_proposer_native: non-JSON response: %s", exc)
        return feedback.structured_output_parsing_failed(
            f"ollama-native returned non-JSON response: {exc}"
        )

    try:
        raw_content = payload["message"]["content"]
    except (KeyError, TypeError) as exc:
        logger.error(
            "run_proposer_native: missing message.content in response (%s)", exc
        )
        return feedback.structured_output_parsing_failed(
            f"ollama-native response missing message.content: {exc}"
        )

    cleaned = _strip_markdown_fences(raw_content)
    try:
        parsed_obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "run_proposer_native: failed to parse content as JSON: %s", exc
        )
        return feedback.structured_output_parsing_failed(
            f"failed to parse ollama-native content as JSON: {exc}"
        )

    _coerce_list_fields(parsed_obj)

    try:
        output = ProverResult.model_validate(parsed_obj)
    except ValidationError as exc:
        logger.error("run_proposer_native: validation failed: %s", exc)
        return feedback.structured_output_parsing_failed(
            f"ollama-native output failed schema validation: {exc}"
        )

    return ProposalMessage(
        reasoning=output.reasoning,
        imports=output.imports,
        opens=output.opens,
        code=output.updated_theorem,
        iteration=state.iteration + 1,
    )
