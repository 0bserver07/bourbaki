"""Unit tests for Phase 4: mathlib_search wired as a proposer tool.

Covers:
- by default, no tools are registered on the proposer agent
- ``enable_mathlib_search=True`` registers a tool named ``mathlib_search``
- the helper ``_search_mathlib`` delegates to ``mathlib_search`` and
  JSON-encodes the result (independent of pydantic-ai plumbing)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from bourbaki.prover import proposer as proposer_mod
from bourbaki.prover.proposer import _search_mathlib, run_proposer
from bourbaki.prover.state import ProposalMessage, ProverResult, ProverState


def _agent_tool_names(agent: Agent) -> list[str]:
    """Pull the tool names off a Pydantic AI 1.56 agent.

    The function toolset stores tools in ``_function_toolset.tools`` keyed
    by tool name. This is internal but stable enough for tests pinned to
    pydantic-ai 1.56.
    """
    toolset = agent._function_toolset  # type: ignore[attr-defined]
    return list(toolset.tools.keys())


# ---------------------------------------------------------------------------
# Agent tool registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mathlib_search_not_registered_by_default(monkeypatch):
    """Without ``enable_mathlib_search``, no tools should be wired onto the
    proposer agent.
    """

    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    captured: dict[str, list[str]] = {}
    original_init = Agent.__init__

    def capture_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["tool_names"] = _agent_tool_names(self)

    monkeypatch.setattr(Agent, "__init__", capture_init)

    async def fake_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            output=ProverResult(
                reasoning="r",
                imports=[],
                opens=[],
                updated_theorem="theorem t : True := trivial",
            )
        )

    monkeypatch.setattr(Agent, "run", fake_run)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=0,
        max_iterations=8,
    )

    out = await run_proposer(state, model="test")
    assert isinstance(out, ProposalMessage)
    assert captured["tool_names"] == []


@pytest.mark.asyncio
async def test_mathlib_search_registered_when_enabled(monkeypatch):
    """``enable_mathlib_search=True`` should register a tool literally
    named ``mathlib_search`` on the agent.
    """

    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    captured: dict[str, list[str]] = {}
    original_init = Agent.__init__

    def capture_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["tool_names"] = _agent_tool_names(self)

    monkeypatch.setattr(Agent, "__init__", capture_init)

    async def fake_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            output=ProverResult(
                reasoning="r",
                imports=[],
                opens=[],
                updated_theorem="theorem t : True := trivial",
            )
        )

    monkeypatch.setattr(Agent, "run", fake_run)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=0,
        max_iterations=8,
    )

    out = await run_proposer(state, model="test", enable_mathlib_search=True)
    assert isinstance(out, ProposalMessage)
    assert "mathlib_search" in captured["tool_names"]


@pytest.mark.asyncio
async def test_enable_flag_overrides_proposer_tools(monkeypatch):
    """When the flag is on, ``proposer_tools`` is ignored — the registered
    tool list contains exactly the built-in ``mathlib_search``.
    """
    from pydantic_ai.tools import Tool

    async def _decoy(x: str) -> str:
        """A decoy tool that should NOT win when the flag is on."""
        return x

    monkeypatch.setattr(
        proposer_mod, "_resolve_model_object", lambda _m: TestModel()
    )

    captured: dict[str, list[str]] = {}
    original_init = Agent.__init__

    def capture_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured["tool_names"] = _agent_tool_names(self)

    monkeypatch.setattr(Agent, "__init__", capture_init)

    async def fake_run(self, user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            output=ProverResult(
                reasoning="r",
                imports=[],
                opens=[],
                updated_theorem="theorem t : True := trivial",
            )
        )

    monkeypatch.setattr(Agent, "run", fake_run)

    state = ProverState(
        problem_id="p1",
        target_theorem="theorem t : True := sorry",
        iteration=0,
        max_iterations=8,
    )

    decoy_tool = Tool(_decoy, name="decoy")

    await run_proposer(
        state,
        model="test",
        proposer_tools=[decoy_tool],
        enable_mathlib_search=True,
    )

    assert captured["tool_names"] == ["mathlib_search"]


# ---------------------------------------------------------------------------
# Helper delegates correctly to mathlib_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_helper_calls_underlying_tool(monkeypatch):
    """``_search_mathlib`` should hand its kwargs straight through to
    ``bourbaki.tools.mathlib_search.mathlib_search`` and JSON-encode the
    result.
    """

    fake_result = {
        "success": True,
        "results": [{"name": "Nat.succ_add", "module": "Mathlib", "type": "...", "doc": ""}],
        "count": 1,
        "query": "Nat.succ_add",
        "mode": "name",
        "duration": 5,
    }

    captured: dict[str, object] = {}

    async def fake_search(*, query: str, mode: str, max_results: int):
        captured["query"] = query
        captured["mode"] = mode
        captured["max_results"] = max_results
        return fake_result

    # Patch the symbol in the module that ``_search_mathlib`` imports it
    # from at call time.
    import bourbaki.tools.mathlib_search as ms_mod

    monkeypatch.setattr(ms_mod, "mathlib_search", fake_search)

    out = await _search_mathlib("Nat.succ_add", mode="name")

    assert captured == {
        "query": "Nat.succ_add",
        "mode": "name",
        "max_results": 5,
    }
    assert json.loads(out) == fake_result
