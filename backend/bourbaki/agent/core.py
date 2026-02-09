"""Agent runner using Pydantic AI — the heart of the backend.

Uses `agent.iter()` for node-by-node control, yielding AgentEvents
for SSE streaming to the TUI.

Node flow: UserPromptNode → ModelRequestNode → CallToolsNode → (loop or End)

- CallToolsNode.model_response holds the LLM's response (text + tool calls)
- After CallToolsNode runs, tool results appear in the next ModelRequestNode
- End(data=FinalResult(output=...)) holds the final answer
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_graph.nodes import End

from bourbaki.agent.context import AgentDependencies
from bourbaki.agent.event_mapper import (
    make_answer_start,
    make_done,
    make_thinking,
    make_tool_end,
    make_tool_error,
    make_tool_start,
)
from bourbaki.agent.prompts import build_system_prompt
from bourbaki.agent.scratchpad import Scratchpad
from bourbaki.config import settings
from bourbaki.events import AgentEvent
from bourbaki.tools.lean_prover import lean_prover
from bourbaki.tools.paper_search import paper_search
from bourbaki.tools.sequence_lookup import sequence_lookup
from bourbaki.tools.skill_tool import skill_invoke
from bourbaki.tools.symbolic_compute import symbolic_compute
from bourbaki.tools.web_search import web_search

logger = logging.getLogger(__name__)


def _check_scratchpad(ctx: RunContext[AgentDependencies], tool_name: str, query: str | None = None) -> str | None:
    """Check scratchpad limits. Returns error JSON if blocked, None if allowed."""
    check = ctx.deps.scratchpad.can_call_tool(tool_name, query)
    if not check["allowed"]:
        return json.dumps({"success": False, "error": check["blockReason"], "blocked": True})
    if check.get("warning"):
        logger.info("Scratchpad warning for %s: %s", tool_name, check["warning"])
    return None


def _record_call(ctx: RunContext[AgentDependencies], tool_name: str, args: dict, result_str: str, query: str | None = None) -> None:
    """Record a tool call in the scratchpad."""
    summary = f"{tool_name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
    ctx.deps.scratchpad.record_tool_call(
        tool_name=tool_name, args=args, result=result_str,
        summary=summary, query=query,
    )


def _create_agent(model: str) -> Agent[AgentDependencies, str]:
    """Create a Pydantic AI agent with all tools registered."""
    system_prompt = build_system_prompt()

    agent: Agent[AgentDependencies, str] = Agent(
        model,
        system_prompt=system_prompt,
        retries=2,
    )

    # Register tools — using @agent.tool to get RunContext for scratchpad access
    @agent.tool(strict=False)
    async def tool_symbolic_compute(
        ctx: RunContext[AgentDependencies],
        operation: str,
        expression: str = "",
        variable: str | None = None,
        from_val: str | int | None = None,
        to_val: str | int | None = None,
        point: str | int | None = None,
        matrix: list[list[float]] | None = None,
        matrix2: list[list[float]] | None = None,
        order: int = 6,
    ) -> str:
        """Execute symbolic math with SymPy: factor, solve, integrate, matrix ops, number theory, and more."""
        blocked = _check_scratchpad(ctx, "symbolic_compute", expression)
        if blocked:
            return blocked
        result = symbolic_compute(
            operation=operation, expression=expression, variable=variable,
            from_val=from_val, to_val=to_val, point=point,
            matrix=matrix, matrix2=matrix2, order=order,
        )
        result_str = json.dumps(result, default=str)
        _record_call(ctx, "symbolic_compute", {"operation": operation, "expression": expression}, result_str, expression)
        return result_str

    @agent.tool(strict=False)
    async def tool_lean_prover(
        ctx: RunContext[AgentDependencies],
        code: str,
        mode: str = "check",
    ) -> str:
        """Verify Lean 4 proofs. Submit Lean code and get verification results."""
        blocked = _check_scratchpad(ctx, "lean_prover", code[:100])
        if blocked:
            return blocked
        result = await lean_prover(code=code, mode=mode)
        result_str = json.dumps(result, default=str)
        _record_call(ctx, "lean_prover", {"mode": mode}, result_str, code[:100])
        return result_str

    @agent.tool(strict=False)
    async def tool_sequence_lookup(
        ctx: RunContext[AgentDependencies],
        mode: str = "identify",
        terms: list[int] | None = None,
        query: str | None = None,
        id: str | None = None,
    ) -> str:
        """Look up integer sequences in OEIS — identify from terms, search, or get by ID."""
        blocked = _check_scratchpad(ctx, "sequence_lookup", query or str(terms))
        if blocked:
            return blocked
        result = await sequence_lookup(mode=mode, terms=terms, query=query, id=id)
        result_str = json.dumps(result, default=str)
        _record_call(ctx, "sequence_lookup", {"mode": mode, "query": query}, result_str, query)
        return result_str

    @agent.tool(strict=False)
    async def tool_paper_search(
        ctx: RunContext[AgentDependencies],
        mode: str = "search",
        query: str | None = None,
        arxiv_id: str | None = None,
        category: str | None = None,
        max_results: int = 5,
    ) -> str:
        """Search arXiv for mathematical papers by keyword or retrieve by ID."""
        blocked = _check_scratchpad(ctx, "paper_search", query)
        if blocked:
            return blocked
        result = await paper_search(
            mode=mode, query=query, arxiv_id=arxiv_id,
            category=category, max_results=max_results,
        )
        result_str = json.dumps(result, default=str)
        _record_call(ctx, "paper_search", {"mode": mode, "query": query}, result_str, query)
        return result_str

    @agent.tool(strict=False)
    async def tool_web_search(
        ctx: RunContext[AgentDependencies],
        query: str,
        num_results: int = 5,
        category: str = "research paper",
    ) -> str:
        """Search the web for mathematical content — papers, proofs, references."""
        blocked = _check_scratchpad(ctx, "web_search", query)
        if blocked:
            return blocked
        result = await web_search(query=query, num_results=num_results, category=category)
        result_str = json.dumps(result, default=str)
        _record_call(ctx, "web_search", {"query": query, "category": category}, result_str, query)
        return result_str

    @agent.tool(strict=False)
    async def tool_skill_invoke(
        ctx: RunContext[AgentDependencies],
        skill_name: str,
    ) -> str:
        """Load a proof technique skill (e.g. 'proof-by-induction', 'proof-by-contradiction')."""
        # Skills don't count toward normal tool limits — they're instruction loaders
        ctx.deps.scratchpad.mark_skill_executed(skill_name)
        result = skill_invoke(skill_name=skill_name)
        return json.dumps(result, default=str)

    return agent


async def run_agent(
    query: str,
    model: str | None = None,
    chat_history: list[dict] | None = None,
    max_iterations: int | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Run the agent and yield AgentEvents for SSE streaming.

    Iterates through Pydantic AI nodes, emitting tool_start/tool_end events
    as tool calls happen, then a final done event with the answer.

    Args:
        query: User's math question or proof request.
        model: Pydantic AI model string (e.g. 'openai:gpt-4o').
        chat_history: Previous conversation messages.
        max_iterations: Override max iterations.

    Yields:
        AgentEvent objects for SSE transmission.
    """
    model = model or settings.default_model

    scratchpad = Scratchpad(tool_call_limit=settings.tool_call_limit)
    deps = AgentDependencies(query=query, model=model, scratchpad=scratchpad)

    agent = _create_agent(model)

    # Convert chat history to Pydantic AI message format
    message_history: list[ModelMessage] | None = None
    if chat_history:
        message_history = _convert_chat_history(chat_history)

    iteration = 0
    all_tool_calls: list[dict[str, Any]] = []
    # Track which tool calls we've already emitted start events for
    pending_tool_calls: dict[str, dict[str, Any]] = {}  # tool_call_id → {name, args}

    async with agent.iter(query, deps=deps, message_history=message_history) as run:
        async for node in run:
            if isinstance(node, CallToolsNode):
                iteration += 1
                # The model_response contains the LLM's response
                response = node.model_response
                for part in response.parts:
                    if isinstance(part, TextPart) and part.content:
                        # LLM is thinking/reasoning before tool calls
                        yield make_thinking(part.content)

                    if isinstance(part, ToolCallPart):
                        tool_name = _clean_tool_name(part.tool_name)
                        try:
                            args = (
                                json.loads(part.args)
                                if isinstance(part.args, str)
                                else (part.args if isinstance(part.args, dict) else {})
                            )
                        except (json.JSONDecodeError, TypeError):
                            args = {}

                        yield make_tool_start(tool_name, args)
                        # Track for pairing with results
                        pending_tool_calls[part.tool_call_id or tool_name] = {
                            "name": tool_name,
                            "args": args,
                        }

            elif isinstance(node, ModelRequestNode):
                # This node contains tool results from the previous iteration
                request = node.request
                for part in request.parts:
                    if isinstance(part, ToolReturnPart):
                        tool_name = _clean_tool_name(part.tool_name)
                        result_str = (
                            part.content
                            if isinstance(part.content, str)
                            else json.dumps(part.content, default=str)
                        )

                        # Look up the original args from pending
                        call_info = pending_tool_calls.pop(
                            part.tool_call_id or tool_name,
                            {"name": tool_name, "args": {}},
                        )

                        # Emit tool_end or tool_error
                        try:
                            result_data = json.loads(result_str)
                            if isinstance(result_data, dict) and result_data.get("success") is False:
                                yield make_tool_error(tool_name, result_data.get("error", "Unknown error"))
                            else:
                                duration = result_data.get("duration", 0) if isinstance(result_data, dict) else 0
                                yield make_tool_end(tool_name, call_info["args"], result_str, duration)
                        except (json.JSONDecodeError, TypeError):
                            yield make_tool_end(tool_name, call_info["args"], result_str, 0)

                        all_tool_calls.append({
                            "tool": tool_name,
                            "args": call_info["args"],
                            "result": result_str,
                        })

            elif isinstance(node, End):
                # Final result
                pass

    # Extract final answer from the result
    final_answer = ""
    if run.result:
        final_answer = run.result.output

    # Emit final events
    yield make_answer_start()

    yield make_done(
        answer=final_answer,
        tool_calls=all_tool_calls,
        iterations=iteration,
    )


def _clean_tool_name(name: str) -> str:
    """Remove 'tool_' prefix from Pydantic AI tool names."""
    if name.startswith("tool_"):
        return name[5:]
    return name


def _convert_chat_history(history: list[dict]) -> list[ModelMessage]:
    """Convert simple chat history dicts to Pydantic AI ModelMessage list."""
    messages: list[ModelMessage] = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            messages.append(ModelResponse(parts=[TextPart(content=content)]))
        elif role == "system":
            # System messages (e.g. compaction summaries) are injected as
            # user-prompt parts so the LLM sees the context.
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"[Context summary]: {content}")]))
    return messages
