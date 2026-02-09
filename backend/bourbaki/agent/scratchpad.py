"""Tool call tracking and deduplication — ported from src/agent/scratchpad.ts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """Record of a completed tool call."""
    tool: str
    args: dict[str, Any]
    result: str


@dataclass
class ToolContext:
    """Full tool context for answer generation."""
    tool: str
    args: dict[str, Any]
    result: str
    summary: str


@dataclass
class Scratchpad:
    """Tracks tool calls, summaries, and limits for a single query."""

    tool_call_limit: int = 3
    similarity_threshold: float = 0.7

    # Internal state
    _call_counts: dict[str, int] = field(default_factory=dict)
    _call_queries: dict[str, list[str]] = field(default_factory=dict)
    _tool_records: list[ToolCallRecord] = field(default_factory=list)
    _tool_contexts: list[ToolContext] = field(default_factory=list)
    _summaries: list[str] = field(default_factory=list)
    _executed_skills: set[str] = field(default_factory=set)

    def can_call_tool(
        self, tool_name: str, query: str | None = None,
    ) -> dict[str, Any]:
        """Check if a tool call is allowed.

        Returns dict with 'allowed', optional 'warning', optional 'blockReason'.
        """
        count = self._call_counts.get(tool_name, 0)

        if count >= self.tool_call_limit:
            return {
                "allowed": False,
                "blockReason": (
                    f"{tool_name} has been called {count} times (limit: {self.tool_call_limit}). "
                    "Try a different approach or answer with available information."
                ),
            }

        result: dict[str, Any] = {"allowed": True}

        # Similarity check
        if query and tool_name in self._call_queries:
            for prev_query in self._call_queries[tool_name]:
                if _jaccard_similarity(query, prev_query) > self.similarity_threshold:
                    result["warning"] = (
                        f"Similar query already sent to {tool_name}. "
                        "Consider a different approach."
                    )
                    break

        # Last attempt warning
        if count == self.tool_call_limit - 1:
            result["warning"] = f"This is the last allowed call to {tool_name}."

        return result

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        summary: str,
        query: str | None = None,
    ) -> None:
        """Record a completed tool call."""
        self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1
        if query:
            self._call_queries.setdefault(tool_name, []).append(query)
        self._tool_records.append(ToolCallRecord(tool=tool_name, args=args, result=result))
        self._tool_contexts.append(ToolContext(tool=tool_name, args=args, result=result, summary=summary))
        self._summaries.append(summary)

    def has_tool_results(self) -> bool:
        return len(self._tool_records) > 0

    def has_executed_skill(self, skill_name: str) -> bool:
        return skill_name in self._executed_skills

    def mark_skill_executed(self, skill_name: str) -> None:
        self._executed_skills.add(skill_name)

    def get_tool_summaries(self) -> list[str]:
        return list(self._summaries)

    def get_tool_records(self) -> list[ToolCallRecord]:
        return list(self._tool_records)

    def get_full_contexts(self) -> list[ToolContext]:
        return list(self._tool_contexts)

    def format_tool_usage_for_prompt(self) -> str | None:
        """Format remaining tool usage for injection into iteration prompts."""
        if not self._call_counts:
            return None

        lines = ["Tool usage:"]
        for tool, count in self._call_counts.items():
            remaining = self.tool_call_limit - count
            lines.append(f"  - {tool}: {count}/{self.tool_call_limit} calls used ({remaining} remaining)")

        if any(c >= self.tool_call_limit for c in self._call_counts.values()):
            lines.append("Some tools have reached their limit. Use available results to answer.")

        return "\n".join(lines)


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
