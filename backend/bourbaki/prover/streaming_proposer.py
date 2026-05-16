"""Streaming proposer adapter — early-cancel for slow reasoning models.

Streams the native /api/chat response token-by-token, accumulates the
``message.content`` field, and cancels the upstream connection as soon as
the structured JSON output is parseable AND the ``code`` field
(``updated_theorem``) is non-empty.

Activated by the ``ollama-stream:<model>`` prefix (parallels the existing
``ollama-native:`` non-streaming adapter in :mod:`native_proposer`).

Why bother
----------
Even with ``think: false``, Ollama Cloud's 1T reasoning models (Kimi K2.6,
DeepSeek V4-pro) take ~20-30s to emit a full proof prompt response because
the model still serializes a long ``reasoning`` field before the
load-bearing ``updated_theorem`` field. By streaming and parsing
incrementally, we can close the connection the moment ``updated_theorem``
appears with a non-empty value — typically 3-5x faster than waiting for
the full response.

Limitations (v1)
----------------
- **No tool-call support.** Tool calls require a multi-round conversation
  and don't play well with streaming + early-cancel. Pass
  ``enable_mathlib_search=False`` (the default), or use the non-streaming
  ``ollama-native:`` adapter if you need ``mathlib_search``. Passing
  ``enable_mathlib_search=True`` logs a WARNING and continues without
  tools.
- **Heuristic early-cancel.** We don't have a streaming JSON parser; we
  attempt a full ``json.loads`` after every chunk. For typical proposer
  outputs (1-4kB of content), this is cheap.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator

import httpx
from pydantic import ValidationError

from bourbaki.prover import feedback, prompts
from bourbaki.prover.native_proposer import (
    OLLAMA_NATIVE_CHAT_URL,
    _coerce_list_fields,
    _strip_markdown_fences,
)
from bourbaki.prover.proposer import _build_user_message
from bourbaki.prover.state import (
    FeedbackMessage,
    ProposalMessage,
    ProverResult,
    ProverState,
)

logger = logging.getLogger(__name__)


_PREFIX = "ollama-stream:"
_DEFAULT_NUM_PREDICT = 8000
_DEFAULT_LLM_TIMEOUT = 60.0  # lower than native: streaming is meant to be faster


def _build_stream_request_body(
    model_name: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compose the streaming ``/api/chat`` body.

    ``stream: true`` is what flips this from the non-streaming path.
    ``think: false`` and ``format=<schema>`` are inherited from the native
    body shape — they're load-bearing for reasoning models on Ollama Cloud.
    """
    return {
        "model": model_name,
        "messages": messages,
        "think": False,
        "format": ProverResult.model_json_schema(),
        "stream": True,
        "options": {"num_predict": _DEFAULT_NUM_PREDICT},
    }


def _try_parse_complete(
    accumulated: str,
) -> tuple[ProverResult | None, str | None]:
    """Attempt to parse the accumulated buffer as a complete ProverResult.

    Returns ``(result, None)`` on success — a fully validated structured
    output we can early-cancel on. Returns ``(None, reason)`` if the buffer
    isn't (yet) a valid+complete ProverResult; ``reason`` is a short
    diagnostic string for debug logging only.

    "Complete" here means:
      1. The accumulated content (with fences stripped) parses as JSON.
      2. The parsed object validates against :class:`ProverResult`.
      3. ``updated_theorem`` is non-empty (after ``.strip()``).

    All three checks must pass before we trigger early-cancel — a parseable
    JSON object missing the ``code`` field (or with an empty one) means the
    stream is mid-emit and we should keep accumulating.
    """
    cleaned = _strip_markdown_fences(accumulated)
    if not cleaned:
        return None, "empty"
    try:
        parsed_obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Mid-stream — JSON isn't closed yet. Common case.
        return None, "json-incomplete"
    if not isinstance(parsed_obj, dict):
        return None, "non-object-json"

    _coerce_list_fields(parsed_obj)

    try:
        result = ProverResult.model_validate(parsed_obj)
    except ValidationError:
        # JSON parses but is missing required fields. Mid-stream too.
        return None, "missing-fields"

    if not result.updated_theorem.strip():
        # JSON parses, all fields present, but the load-bearing field is
        # still empty. Keep streaming — the model is presumably about to
        # fill it in.
        return None, "empty-updated_theorem"

    return result, None


def _result_to_proposal(
    result: ProverResult, state: ProverState
) -> ProposalMessage:
    """Build a :class:`ProposalMessage` from a validated ProverResult."""
    return ProposalMessage(
        reasoning=result.reasoning,
        imports=result.imports,
        opens=result.opens,
        code=result.updated_theorem,
        iteration=state.iteration + 1,
    )


def _parse_final(
    accumulated: str, state: ProverState
) -> ProposalMessage | FeedbackMessage:
    """Parse the full content once the stream has ended naturally.

    Mirrors :func:`bourbaki.prover.native_proposer._parse_proposal` but
    inlined here so the streaming module doesn't pull in any of the
    non-streaming module's private state. Returns either a fully-formed
    :class:`ProposalMessage` or a parse-failed :class:`FeedbackMessage`.
    """
    cleaned = _strip_markdown_fences(accumulated)
    if not cleaned:
        logger.error("run_proposer_stream: stream ended with empty content")
        return feedback.structured_output_parsing_failed(
            "ollama-stream returned an empty content stream"
        )

    try:
        parsed_obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "run_proposer_stream: failed to parse final content as JSON: %s",
            exc,
        )
        return feedback.structured_output_parsing_failed(
            f"failed to parse ollama-stream content as JSON: {exc}"
        )

    if not isinstance(parsed_obj, dict):
        return feedback.structured_output_parsing_failed(
            f"ollama-stream content is JSON but not an object: "
            f"got {type(parsed_obj).__name__}"
        )

    _coerce_list_fields(parsed_obj)

    try:
        output = ProverResult.model_validate(parsed_obj)
    except ValidationError as exc:
        logger.error("run_proposer_stream: validation failed: %s", exc)
        return feedback.structured_output_parsing_failed(
            f"ollama-stream output failed schema validation: {exc}"
        )

    return _result_to_proposal(output, state)


async def _iter_chunks(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, bool]]:
    """Iterate NDJSON lines from ``response`` and yield (content, done) pairs.

    Each NDJSON line is one ``/api/chat`` chunk shaped like::

        {"model":"...", "message":{"role":"assistant","content":"..."}, "done":false}

    We extract ``message.content`` (an additive token-or-token-group string)
    and the ``done`` flag, ignoring everything else. Malformed lines are
    skipped silently — they're rare and the stream has more chunks coming.

    Implementation note: we use ``aiter_lines`` rather than ``aiter_bytes``
    + manual splitting because the Ollama server emits one JSON object per
    line and ``aiter_lines`` already handles partial-buffer line buffering.
    """
    async for line in response.aiter_lines():
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            # Stray non-JSON line — skip and keep iterating. The server
            # shouldn't emit these but it's cheap to be defensive.
            continue
        if not isinstance(chunk, dict):
            continue
        msg = chunk.get("message")
        content_piece = ""
        if isinstance(msg, dict):
            raw = msg.get("content")
            if isinstance(raw, str):
                content_piece = raw
        done = bool(chunk.get("done"))
        yield content_piece, done
        if done:
            return


async def run_proposer_stream(
    state: ProverState,
    model: str,
    llm_timeout: float = _DEFAULT_LLM_TIMEOUT,
    enable_mathlib_search: bool = False,
) -> ProposalMessage | FeedbackMessage:
    """Stream from Ollama native ``/api/chat`` with early cancel.

    Same input/output contract as :func:`run_proposer` /
    :func:`run_proposer_native`. Strips the ``ollama-stream:`` prefix to
    get the underlying model name. Never raises — every error path returns
    a :class:`FeedbackMessage`.

    Strategy
    --------
    1. Open an async stream against ``/api/chat`` with ``stream: true``.
    2. Accumulate ``message.content`` chunks into a buffer.
    3. After each chunk, attempt to parse the buffer as a complete
       :class:`ProverResult` (fences stripped, list fields coerced,
       pydantic-validated). On success **AND** when ``updated_theorem`` is
       non-empty: close the stream early and return the proposal.
    4. If the stream finishes naturally with ``done=true``, parse the
       complete content the same way the non-streaming path does.

    Args:
        state: Current loop state. Same semantics as
            :func:`run_proposer_native`.
        model: Must begin with ``ollama-stream:`` (e.g.
            ``ollama-stream:kimi-k2.6``).
        llm_timeout: Total wall-clock budget for the streaming call, in
            seconds. Applied via ``httpx.Timeout`` across connect, read,
            and pool.
        enable_mathlib_search: **Ignored on the streaming path.** If True,
            logs a WARNING and proceeds without tools. Use
            :func:`run_proposer_native` if you need ``mathlib_search``.
    """
    if state.iteration >= state.max_iterations:
        logger.warning(
            "Streaming proposer iteration budget exhausted (%s/%s); "
            "returning terminal feedback.",
            state.iteration,
            state.max_iterations,
        )
        return feedback.max_iterations(state.max_iterations)

    if not model.startswith(_PREFIX):
        return feedback.structured_output_parsing_failed(
            f"run_proposer_stream expects model prefixed with '{_PREFIX}', "
            f"got {model!r}"
        )
    model_name = model.removeprefix(_PREFIX).strip()
    if not model_name:
        return feedback.structured_output_parsing_failed(
            f"run_proposer_stream: empty model name after stripping "
            f"'{_PREFIX}' prefix from {model!r}"
        )

    api_key = os.environ.get("OLLAMA_CLOUD_API_KEY")
    if not api_key:
        return feedback.structured_output_parsing_failed(
            "OLLAMA_CLOUD_API_KEY env var is not set; streaming Ollama path "
            "requires an Ollama Cloud API key."
        )

    if enable_mathlib_search:
        logger.warning(
            "ollama-stream path does not support tool calls; "
            "enable_mathlib_search=True is ignored. Use ollama-native: "
            "if you need mathlib_search."
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
    body = _build_stream_request_body(model_name, messages)

    accumulated = ""
    early_cancelled = False

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(llm_timeout)) as client:
            async with client.stream(
                "POST",
                OLLAMA_NATIVE_CHAT_URL,
                json=body,
                headers=headers,
            ) as response:
                # Eagerly surface upstream errors so we don't burn the
                # whole timeout budget waiting for chunks that will never
                # arrive (e.g. 429 / 5xx returns no body in stream mode).
                if response.status_code == 429:
                    logger.warning(
                        "run_proposer_stream: %s rate limited (429)", model
                    )
                    return feedback.structured_output_parsing_failed(
                        "ollama-stream rate limited (429); retry on next iteration."
                    )
                if response.status_code >= 400:
                    # We have to ``aread()`` because the body hasn't been
                    # loaded yet under ``stream=True``.
                    await response.aread()
                    snippet = response.text[:200].replace("\n", " ")
                    logger.error(
                        "run_proposer_stream: %s returned HTTP %s — %s",
                        model,
                        response.status_code,
                        snippet,
                    )
                    return feedback.structured_output_parsing_failed(
                        f"ollama-stream HTTP {response.status_code} error: "
                        f"{snippet}"
                    )

                # === EARLY-CANCEL LOOP ===
                # For each chunk we (a) append to accumulated content,
                # (b) attempt full ProverResult parse, (c) on success
                # break out of the iterator — exiting the `async with`
                # closes the upstream connection.
                async for content_piece, done in _iter_chunks(response):
                    if content_piece:
                        accumulated += content_piece

                    if done:
                        # Server signalled end of stream. Fall through to
                        # final-parse below.
                        break

                    if not content_piece:
                        # No new content this chunk; skip the parse attempt.
                        continue

                    result, _reason = _try_parse_complete(accumulated)
                    if result is not None:
                        # Early-cancel: structured output is fully formed
                        # and updated_theorem is non-empty.
                        logger.info(
                            "run_proposer_stream: %s early-cancelled after "
                            "%d bytes of content",
                            model,
                            len(accumulated),
                        )
                        early_cancelled = True
                        return _result_to_proposal(result, state)

    except (httpx.TimeoutException, asyncio.TimeoutError):
        logger.warning(
            "run_proposer_stream: %s call timed out after %.0fs",
            model,
            llm_timeout,
        )
        return feedback.structured_output_parsing_failed(
            f"ollama-stream LLM call timed out after {llm_timeout:.0f}s"
        )
    except httpx.RequestError as exc:
        # Network died mid-stream (connection reset, DNS, etc).
        logger.warning(
            "run_proposer_stream: %s network error: %s", model, exc
        )
        return feedback.structured_output_parsing_failed(
            f"ollama-stream network error: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 — never let exceptions bubble up
        logger.exception("run_proposer_stream: %s call failed", model)
        return feedback.structured_output_parsing_failed(
            f"streaming call failed: {exc}"
        )

    # Reached when we exited the iterator without an early-cancel return —
    # either ``done=True`` arrived or the iterator ran dry. Parse what we
    # got with the same logic the non-streaming path uses.
    if early_cancelled:
        # Belt-and-braces: the early-cancel branch returns directly, so we
        # should never reach here with it set. Keeping this as a type-checker
        # hint and defensive guard.
        return feedback.structured_output_parsing_failed(
            "ollama-stream: unreachable post-cancel state"
        )

    return _parse_final(accumulated, state)
