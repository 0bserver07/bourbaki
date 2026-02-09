"""Multi-turn chat history for the Python backend.

Simpler than the TypeScript InMemoryChatHistory — the Python backend
tracks messages as {role, content} dicts for Pydantic AI message_history.
"""

from __future__ import annotations

from typing import Any


class ChatHistory:
    """In-memory chat history for multi-turn context."""

    def __init__(self) -> None:
        self._messages: list[dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def has_messages(self) -> bool:
        return len(self._messages) > 0

    def load_from_session(self, session_messages: list[dict[str, Any]]) -> None:
        """Load from session manager message format."""
        self.clear()
        for msg in session_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                self._messages.append({"role": role, "content": content})
