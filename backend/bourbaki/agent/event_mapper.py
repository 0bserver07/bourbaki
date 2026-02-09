"""Map Pydantic AI iteration nodes to Bourbaki AgentEvents."""

from __future__ import annotations

import json
from typing import Any

from bourbaki.events import (
    AnswerStartEvent,
    DoneEvent,
    ThinkingEvent,
    ToolCallRecord,
    ToolEndEvent,
    ToolErrorEvent,
    ToolLimitEvent,
    ToolStartEvent,
)


def make_tool_start(tool_name: str, args: dict[str, Any]) -> ToolStartEvent:
    return ToolStartEvent(tool=tool_name, args=args)


def make_tool_end(
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    duration_ms: int,
) -> ToolEndEvent:
    result_str = result if isinstance(result, str) else json.dumps(result, default=str)
    return ToolEndEvent(tool=tool_name, args=args, result=result_str, duration=duration_ms)


def make_tool_error(tool_name: str, error: str) -> ToolErrorEvent:
    return ToolErrorEvent(tool=tool_name, error=error)


def make_tool_limit(
    tool_name: str,
    warning: str | None,
    block_reason: str | None,
    blocked: bool,
) -> ToolLimitEvent:
    return ToolLimitEvent(tool=tool_name, warning=warning, blockReason=block_reason, blocked=blocked)


def make_thinking(message: str) -> ThinkingEvent:
    return ThinkingEvent(message=message)


def make_answer_start() -> AnswerStartEvent:
    return AnswerStartEvent()


def make_done(
    answer: str,
    tool_calls: list[dict[str, Any]],
    iterations: int,
) -> DoneEvent:
    records = [
        ToolCallRecord(tool=tc["tool"], args=tc["args"], result=tc["result"])
        for tc in tool_calls
    ]
    return DoneEvent(answer=answer, toolCalls=records, iterations=iterations)
