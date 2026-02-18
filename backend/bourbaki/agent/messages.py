"""Agent message protocol for multi-agent coordination.

Provides a simple message bus for inter-agent communication during
coordinated proof construction.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    """A message between agents."""
    from_agent: str
    to_agent: str  # Use "*" for broadcast
    msg_type: str  # "strategy" | "lemma_list" | "proof_state" | "error" | "verified" | "subgoal"
    content: dict[str, Any]
    timestamp: float = field(default_factory=time.monotonic)


class MessageBus:
    """Simple message routing for multi-agent coordination."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[AgentMessage]] = {}
        self._history: dict[str, list[AgentMessage]] = {}

    def _ensure_queue(self, agent_name: str) -> asyncio.Queue[AgentMessage]:
        if agent_name not in self._queues:
            self._queues[agent_name] = asyncio.Queue()
        return self._queues[agent_name]

    async def send(self, msg: AgentMessage) -> None:
        """Send a message to the target agent (or broadcast to all)."""
        # Record in history
        self._history.setdefault(msg.to_agent, []).append(msg)

        if msg.to_agent == "*":
            # Broadcast: send to all known queues + store for future agents
            for name, queue in self._queues.items():
                await queue.put(msg)
            # Also store in broadcast history for agents that join later
            self._history.setdefault("*", []).append(msg)
        else:
            queue = self._ensure_queue(msg.to_agent)
            await queue.put(msg)

    async def receive(
        self, agent_name: str, timeout: float = 30.0,
    ) -> AgentMessage | None:
        """Receive the next message for an agent.

        Returns None if no message arrives within timeout.
        """
        queue = self._ensure_queue(agent_name)

        # Check for undelivered broadcast messages
        broadcast_history = self._history.get("*", [])
        agent_history = self._history.get(agent_name, [])
        for bmsg in broadcast_history:
            if bmsg not in agent_history:
                self._history.setdefault(agent_name, []).append(bmsg)
                return bmsg

        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def get_history(self, agent_name: str) -> list[AgentMessage]:
        """Get all messages that were sent to an agent."""
        return list(self._history.get(agent_name, []))
