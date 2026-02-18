"""Tests for agent message protocol."""

from __future__ import annotations

import asyncio

import pytest

from bourbaki.agent.messages import AgentMessage, MessageBus


def test_agent_message_creation():
    msg = AgentMessage(
        from_agent="strategist",
        to_agent="prover",
        msg_type="strategy",
        content={"sketch": ["intro n", "induction n"]},
    )
    assert msg.from_agent == "strategist"
    assert msg.to_agent == "prover"
    assert msg.msg_type == "strategy"
    assert msg.timestamp > 0


@pytest.mark.asyncio
async def test_message_bus_send_receive():
    bus = MessageBus()
    msg = AgentMessage(
        from_agent="strategist",
        to_agent="prover",
        msg_type="strategy",
        content={"sketch": ["ring"]},
    )
    await bus.send(msg)
    received = await bus.receive("prover", timeout=1.0)
    assert received is not None
    assert received.from_agent == "strategist"
    assert received.content["sketch"] == ["ring"]


@pytest.mark.asyncio
async def test_message_bus_receive_timeout():
    bus = MessageBus()
    received = await bus.receive("prover", timeout=0.1)
    assert received is None


@pytest.mark.asyncio
async def test_message_bus_routing():
    """Messages should only be received by the target agent."""
    bus = MessageBus()
    msg = AgentMessage(
        from_agent="strategist",
        to_agent="prover",
        msg_type="strategy",
        content={"step": 1},
    )
    await bus.send(msg)

    # Searcher should not receive this
    received = await bus.receive("searcher", timeout=0.1)
    assert received is None

    # Prover should receive it
    received = await bus.receive("prover", timeout=1.0)
    assert received is not None


@pytest.mark.asyncio
async def test_message_bus_history():
    bus = MessageBus()
    msg1 = AgentMessage(from_agent="a", to_agent="b", msg_type="x", content={})
    msg2 = AgentMessage(from_agent="c", to_agent="b", msg_type="y", content={})
    await bus.send(msg1)
    await bus.send(msg2)

    # Drain messages
    await bus.receive("b", timeout=0.1)
    await bus.receive("b", timeout=0.1)

    history = bus.get_history("b")
    assert len(history) == 2


@pytest.mark.asyncio
async def test_message_bus_broadcast():
    """Broadcast messages should be receivable by any agent."""
    bus = MessageBus()
    msg = AgentMessage(
        from_agent="coordinator",
        to_agent="*",
        msg_type="shutdown",
        content={},
    )
    await bus.send(msg)

    received = await bus.receive("prover", timeout=1.0)
    assert received is not None
    assert received.msg_type == "shutdown"
