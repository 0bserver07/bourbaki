"""RunContext dependencies for Pydantic AI agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from bourbaki.agent.scratchpad import Scratchpad


@dataclass
class AgentDependencies:
    """Dependencies injected into the Pydantic AI agent via RunContext.

    These are available to all tools via `ctx.deps`.
    """

    query: str
    model: str
    scratchpad: Scratchpad = field(default_factory=Scratchpad)
    session_id: str | None = None
    chat_history: list[dict] = field(default_factory=list)
