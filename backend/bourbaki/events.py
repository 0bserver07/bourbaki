"""Agent event models matching src/agent/types.ts.

These Pydantic models define the SSE wire format between the Python backend
and the React + Ink TUI. The TypeScript types in src/agent/types.ts are the
source of truth — these must stay in sync.
"""

from __future__ import annotations

from typing import Any, Literal, Union
from pydantic import BaseModel


class ThinkingEvent(BaseModel):
    """Agent is processing/thinking."""
    type: Literal["thinking"] = "thinking"
    message: str


class ToolStartEvent(BaseModel):
    """Tool execution started."""
    type: Literal["tool_start"] = "tool_start"
    tool: str
    args: dict[str, Any]


class ToolEndEvent(BaseModel):
    """Tool execution completed successfully."""
    type: Literal["tool_end"] = "tool_end"
    tool: str
    args: dict[str, Any]
    result: str
    duration: int  # milliseconds


class ToolErrorEvent(BaseModel):
    """Tool execution failed."""
    type: Literal["tool_error"] = "tool_error"
    tool: str
    error: str


class ToolLimitEvent(BaseModel):
    """Tool call was blocked or warned due to retry limits."""
    type: Literal["tool_limit"] = "tool_limit"
    tool: str
    warning: str | None = None
    blockReason: str | None = None
    blocked: bool


class AnswerStartEvent(BaseModel):
    """Final answer generation started."""
    type: Literal["answer_start"] = "answer_start"


class ToolCallRecord(BaseModel):
    """Record of a tool call for the done event."""
    tool: str
    args: dict[str, Any]
    result: str


class DoneEvent(BaseModel):
    """Agent completed with final result."""
    type: Literal["done"] = "done"
    answer: str
    toolCalls: list[ToolCallRecord]
    iterations: int


class CheckpointEvent(BaseModel):
    """Checkpoint event — agent paused for human review."""
    type: Literal["checkpoint"] = "checkpoint"
    checkpointId: str
    iteration: int
    reason: Literal["interval", "stuck", "technique_switch", "manual", "error"]
    filepath: str
    message: str | None = None


class ResumeEvent(BaseModel):
    """Resume event — agent resumed from checkpoint."""
    type: Literal["resume"] = "resume"
    checkpointId: str
    iteration: int


AgentEvent = Union[
    ThinkingEvent,
    ToolStartEvent,
    ToolEndEvent,
    ToolErrorEvent,
    ToolLimitEvent,
    AnswerStartEvent,
    DoneEvent,
    CheckpointEvent,
    ResumeEvent,
]


def event_to_sse(event: AgentEvent) -> str:
    """Format an AgentEvent as an SSE message string."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
