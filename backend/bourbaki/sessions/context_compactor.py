"""Context compaction — summarizes older messages when context fills up.

Ported from src/agent/context-compactor.ts.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent

from bourbaki.utils.tokens import estimate_tokens

COMPACTION_THRESHOLD = 100_000  # tokens
KEEP_RECENT_MESSAGES = 6

SUMMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant that creates concise summaries of mathematical conversations. "
    "Focus on: problems being worked on, key results and formulas, current proof state, "
    "and what the user is trying to accomplish. Keep to 2-4 paragraphs."
)


def needs_compaction(token_count: int) -> bool:
    """Check if context exceeds compaction threshold."""
    return token_count > COMPACTION_THRESHOLD


async def compact_conversation(
    messages: list[Any],
    model: str,
) -> dict[str, Any]:
    """Summarize older messages, keeping the most recent ones intact.

    Args:
        messages: List of SessionMessage objects.
        model: Pydantic AI model string for LLM calls.

    Returns:
        Dict with 'messages' (compacted list), 'summary' (str), 'tokens_saved' (int).
    """
    if len(messages) <= KEEP_RECENT_MESSAGES:
        return {"messages": messages, "summary": None, "tokens_saved": 0}

    to_summarize = messages[:-KEEP_RECENT_MESSAGES]
    to_keep = messages[-KEEP_RECENT_MESSAGES:]

    # Format old messages for summarization
    formatted = "\n\n".join(
        f"{m.role.upper()}: {m.content}" for m in to_summarize
    )

    # Estimate tokens before
    tokens_before = sum(estimate_tokens(m.content) for m in messages)

    # Generate summary via LLM
    try:
        agent: Agent[None, str] = Agent(model, system_prompt=SUMMARY_SYSTEM_PROMPT)
        result = await agent.run(
            f"Summarize this mathematical conversation:\n\n{formatted}"
        )
        summary = result.output
    except Exception:
        # Fallback: simple truncated summary
        summary = f"Previous conversation ({len(to_summarize)} messages) about: {to_summarize[0].content[:200]}"

    # Create summary message
    from bourbaki.sessions.manager import SessionMessage
    summary_msg = SessionMessage(
        role="system",
        content=f"[Previous conversation summary]\n{summary}",
    )

    compacted = [summary_msg] + to_keep
    tokens_after = sum(estimate_tokens(m.content) for m in compacted)

    return {
        "messages": compacted,
        "summary": summary,
        "tokens_saved": tokens_before - tokens_after,
    }


def build_context_from_messages(
    messages: list[Any],
    max_tokens: int = 50_000,
) -> str:
    """Build context string from messages, newest first, respecting token limit."""
    parts: list[str] = []
    token_count = 0

    for msg in reversed(messages):
        tokens = estimate_tokens(msg.content)
        if token_count + tokens > max_tokens:
            parts.append("[Earlier messages truncated for context limit]")
            break
        parts.append(f"{msg.role.upper()}: {msg.content}")
        token_count += tokens

    parts.reverse()
    return "\n\n".join(parts)
