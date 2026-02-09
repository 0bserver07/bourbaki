"""Session persistence — ported from src/agent/session-manager.ts."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bourbaki.config import settings
from bourbaki.utils.tokens import estimate_tokens

MAX_SESSIONS = 50


class SessionMessage:
    """A single conversation message."""

    __slots__ = ("role", "content", "timestamp", "tool_calls")

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.tool_calls = tool_calls

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMessage:
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp"),
            tool_calls=data.get("tool_calls"),
        )


class Session:
    """Full session state."""

    def __init__(
        self,
        id: str | None = None,
        model: str = "unknown",
        title: str = "New Session",
        messages: list[SessionMessage] | None = None,
        summary: str | None = None,
        token_count: int = 0,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        self.id = id or uuid.uuid4().hex[:8]
        self.model = model
        self.title = title
        self.messages = messages or []
        self.summary = summary
        self.token_count = token_count
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
            "tokenCount": self.token_count,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(
            id=data["id"],
            model=data.get("model", "unknown"),
            title=data.get("title", "New Session"),
            messages=[SessionMessage.from_dict(m) for m in data.get("messages", [])],
            summary=data.get("summary"),
            token_count=data.get("tokenCount", 0),
            created_at=data.get("createdAt"),
            updated_at=data.get("updatedAt"),
        )


class SessionManager:
    """Manages session persistence to .bourbaki/sessions/."""

    def __init__(self) -> None:
        self._current: Session | None = None
        self._sessions_dir = settings.bourbaki_path / "sessions"

    @property
    def sessions_dir(self) -> Path:
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        return self._sessions_dir

    def start_session(self, model: str) -> Session:
        session = Session(model=model)
        self._current = session
        self._save()
        return session

    def get_current_session(self) -> Session | None:
        return self._current

    def add_user_message(self, content: str) -> None:
        if not self._current:
            return
        msg = SessionMessage(role="user", content=content)
        self._current.messages.append(msg)
        self._current.token_count += estimate_tokens(content)
        # Set title from first user message
        if len([m for m in self._current.messages if m.role == "user"]) == 1:
            self._current.title = content[:60]
        self._current.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()

    def add_assistant_message(
        self, content: str, tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._current:
            return
        msg = SessionMessage(role="assistant", content=content, tool_calls=tool_calls)
        self._current.messages.append(msg)
        self._current.token_count += estimate_tokens(content)
        self._current.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()

    def get_messages_for_context(self, session: Session | None = None) -> list[dict[str, str]]:
        """Get messages formatted for LLM context.

        Args:
            session: Specific session to get messages from. Falls back to _current.
        """
        target = session or self._current
        if not target:
            return []
        return [
            {"role": m.role, "content": m.content}
            for m in target.messages
        ]

    def load_session(self, session_id: str) -> Session | None:
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        session = Session.from_dict(data)
        self._current = session
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        summaries = []
        for path in sorted(self.sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summaries.append({
                    "id": data["id"],
                    "title": data.get("title", "Untitled"),
                    "model": data.get("model", "unknown"),
                    "messageCount": len(data.get("messages", [])),
                    "createdAt": data.get("createdAt"),
                    "updatedAt": data.get("updatedAt"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return summaries

    def delete_session(self, session_id: str) -> bool:
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            return False
        path.unlink()
        if self._current and self._current.id == session_id:
            self._current = None
        return True

    async def check_and_compact(self) -> bool:
        """Check if context needs compaction and compact if needed."""
        if not self._current:
            return False
        from bourbaki.sessions.context_compactor import needs_compaction, compact_conversation
        if not needs_compaction(self._current.token_count):
            return False
        result = await compact_conversation(self._current.messages, self._current.model)
        self._current.messages = result["messages"]
        self._current.summary = result["summary"]
        self._current.token_count -= result["tokens_saved"]
        self._save()
        return True

    def set_summary(self, summary: str) -> None:
        if self._current:
            self._current.summary = summary
            self._save()

    def _save(self) -> None:
        if not self._current:
            return
        path = self.sessions_dir / f"{self._current.id}.json"
        path.write_text(json.dumps(self._current.to_dict(), indent=2), encoding="utf-8")
        self._enforce_max_sessions()

    def _enforce_max_sessions(self) -> None:
        files = sorted(self.sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        while len(files) > MAX_SESSIONS:
            files[0].unlink()
            files.pop(0)


session_manager = SessionManager()
