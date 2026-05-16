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
import time
from typing import Any

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
_MAX_TOOL_CALL_ROUNDS = 5


_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "mathlib_search",
            "description": (
                "Search Mathlib for lemmas. Modes: 'name' (Loogle exact/partial), "
                "'type' (Loogle type signature), 'natural' (LeanSearch), 'semantic' "
                "(LeanExplore), 'local' (offline FAISS). Returns JSON with "
                "{success, results, count}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["name", "type", "natural", "semantic", "local"],
                    },
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    }
]


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
    messages: list[dict[str, Any]],
    *,
    include_tools: bool,
) -> dict[str, Any]:
    """Compose the native ``/api/chat`` body.

    Uses the pydantic-generated ``ProverResult.model_json_schema()`` as
    the ``format`` constraint. ``think: false`` is the load-bearing flag
    that prevents reasoning models from looping forever.
    """
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "think": False,
        "format": ProverResult.model_json_schema(),
        "stream": False,
        "options": {"num_predict": _DEFAULT_NUM_PREDICT},
    }
    if include_tools:
        body["tools"] = TOOLS_SCHEMA
    return body


async def _invoke_tool_call(tool_call: dict[str, Any]) -> str:
    """Execute one tool call and return a JSON-string result.

    Unknown tool names and malformed arguments return a JSON error
    payload instead of raising — the LLM should observe the failure and
    retry on the next round.
    """
    from bourbaki.tools.mathlib_search import mathlib_search

    function = tool_call.get("function") or {}
    name = function.get("name", "")
    raw_args = function.get("arguments", {})

    # Ollama may serialize arguments as a JSON string in some configs.
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            return json.dumps(
                {
                    "success": False,
                    "error": f"tool arguments not valid JSON: {exc}",
                }
            )

    if not isinstance(raw_args, dict):
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"tool arguments must be an object, got "
                    f"{type(raw_args).__name__}"
                ),
            }
        )

    if name != "mathlib_search":
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"unknown tool {name!r}; only 'mathlib_search' is available "
                    "on the native path."
                ),
            }
        )

    query = raw_args.get("query")
    if not isinstance(query, str) or not query:
        return json.dumps(
            {
                "success": False,
                "error": "mathlib_search: missing required 'query' string argument.",
            }
        )

    mode = raw_args.get("mode", "name")
    if not isinstance(mode, str):
        mode = "name"

    max_results = raw_args.get("max_results", 5)
    if not isinstance(max_results, int) or isinstance(max_results, bool):
        max_results = 5

    try:
        result = await mathlib_search(
            query=query, mode=mode, max_results=max_results
        )
    except Exception as exc:  # noqa: BLE001 — never bubble up
        logger.exception("mathlib_search tool invocation failed")
        return json.dumps(
            {"success": False, "error": f"mathlib_search raised: {exc}"}
        )

    return json.dumps(result, default=str)


def _parse_proposal(
    raw_content: str, state: ProverState
) -> ProposalMessage | FeedbackMessage:
    """Strip fences, parse JSON, coerce sloppy fields, and validate.

    Returns either a fully-formed :class:`ProposalMessage` or a
    parse-failed :class:`FeedbackMessage` — never raises.
    """
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
        llm_timeout: Total time budget across all tool-call rounds, in
            seconds. Each round shares the global budget — when it is
            exhausted we return a parse-failed feedback.
        enable_mathlib_search: When True, advertise the ``mathlib_search``
            tool in the request and run the tool-call loop (max
            :data:`_MAX_TOOL_CALL_ROUNDS` rounds) until the model emits a
            final structured-output response without ``tool_calls``.
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
        logger.info(
            "native path: mathlib_search tool enabled (max %d rounds)",
            _MAX_TOOL_CALL_ROUNDS,
        )

    user_msg = _build_user_message(state)
    sys_prompt = (
        prompts.PROPOSER_SYSTEM_PROMPT_SINGLE_SHOT
        if state.max_iterations == 1
        else prompts.PROPOSER_SYSTEM_PROMPT
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    deadline = time.monotonic() + llm_timeout

    # Tool-call loop. When enable_mathlib_search is False this runs
    # exactly once because the model can't request a tool. The cap
    # protects against pathological loops where the model keeps asking
    # for searches forever.
    async with httpx.AsyncClient(timeout=httpx.Timeout(llm_timeout)) as client:
        for round_idx in range(_MAX_TOOL_CALL_ROUNDS + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "run_proposer_native: %s exceeded total %.0fs budget across "
                    "%d round(s)",
                    model,
                    llm_timeout,
                    round_idx,
                )
                return feedback.structured_output_parsing_failed(
                    f"ollama-native exceeded total {llm_timeout:.0f}s budget "
                    f"across {round_idx} tool-call round(s)"
                )

            if round_idx == _MAX_TOOL_CALL_ROUNDS:
                logger.warning(
                    "run_proposer_native: %s hit max tool-call rounds (%d); "
                    "returning parse-failed feedback",
                    model,
                    _MAX_TOOL_CALL_ROUNDS,
                )
                return feedback.structured_output_parsing_failed(
                    f"ollama-native exceeded max tool-call rounds "
                    f"({_MAX_TOOL_CALL_ROUNDS})"
                )

            body = _build_request_body(
                model_name, messages, include_tools=enable_mathlib_search
            )

            try:
                response = await client.post(
                    OLLAMA_NATIVE_CHAT_URL,
                    json=body,
                    headers=headers,
                    timeout=httpx.Timeout(remaining),
                )
            except (httpx.TimeoutException, asyncio.TimeoutError):
                logger.warning(
                    "run_proposer_native: %s call timed out after %.0fs",
                    model,
                    llm_timeout,
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
                message = payload["message"]
            except (KeyError, TypeError) as exc:
                logger.error(
                    "run_proposer_native: missing message in response (%s)", exc
                )
                return feedback.structured_output_parsing_failed(
                    f"ollama-native response missing message: {exc}"
                )

            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            if tool_calls:
                # Append the assistant message verbatim (including the
                # tool_calls field, even if content is empty) and one
                # role:tool message per call, then loop.
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content", "") or "",
                        "tool_calls": tool_calls,
                    }
                )
                for tc in tool_calls:
                    tool_content = await _invoke_tool_call(tc)
                    fn = tc.get("function") or {}
                    tool_name = fn.get("name", "")
                    messages.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_content,
                        }
                    )
                continue

            try:
                raw_content = message["content"]
            except (KeyError, TypeError) as exc:
                logger.error(
                    "run_proposer_native: missing message.content in response (%s)",
                    exc,
                )
                return feedback.structured_output_parsing_failed(
                    f"ollama-native response missing message.content: {exc}"
                )

            return _parse_proposal(raw_content, state)

    # Unreachable: the loop either returns inside the body or hits the
    # max-rounds branch. Belt-and-braces fallback for type-checkers.
    return feedback.structured_output_parsing_failed(
        "ollama-native tool-call loop terminated unexpectedly"
    )
