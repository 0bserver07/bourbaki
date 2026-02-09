"""POST /query → SSE stream — the main agent endpoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from bourbaki.agent.core import run_agent
from bourbaki.events import DoneEvent
from bourbaki.sessions.manager import session_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    model: str | None = None
    model_provider: str | None = None
    session_id: str | None = None
    chat_history: list[dict] | None = None  # Keep for backward compat


def _resolve_model(model: str | None, provider: str | None) -> str | None:
    """Resolve model string to Pydantic AI format ('provider:model')."""
    if not model:
        return None
    # Already in provider:model format
    if ":" in model:
        return model
    # Map provider + model to Pydantic AI format
    if provider:
        provider_map = {
            "anthropic": "anthropic",
            "openai": "openai",
            "google": "google-gla",
            "ollama": "ollama",
            "ollama-cloud": "ollama-cloud",
            "openrouter": "openrouter",
            "xai": "xai",
            "groq": "groq",
        }
        prefix = provider_map.get(provider, provider)
        return f"{prefix}:{model}"
    # Default: assume openai
    return f"openai:{model}"


async def _event_generator(req: QueryRequest) -> AsyncIterator[dict]:
    """Generate SSE events from the agent."""
    model = _resolve_model(req.model, req.model_provider)
    chat_history = req.chat_history

    # Session-based flow: load session and build chat history from it
    session_loaded = False
    if req.session_id:
        session = session_manager.load_session(req.session_id)
        if session:
            session_loaded = True
            # Get context BEFORE adding current query — run_agent receives
            # query separately and will add it as the new user turn.
            chat_history = session_manager.get_messages_for_context(session)
            session_manager.add_user_message(req.query)

    final_answer = ""
    try:
        async for event in run_agent(
            query=req.query,
            model=model,
            chat_history=chat_history,
        ):
            if event.type == "done":
                final_answer = getattr(event, "answer", "")
            yield {
                "event": event.type,
                "data": event.model_dump_json(),
            }
    except Exception as exc:
        logger.exception("Agent error for model=%s", model)
        error_msg = str(exc)
        # Send the error as a done event so the TUI displays it
        err_event = DoneEvent(
            answer=f"**Error:** {error_msg}",
            toolCalls=[],
            iterations=0,
        )
        yield {"event": "done", "data": err_event.model_dump_json()}
        return

    # Persist assistant response and auto-compact if needed.
    # Only write if we successfully loaded this session earlier — avoids
    # writing to a stale _current from a different session.
    if session_loaded and final_answer:
        session_manager.add_assistant_message(final_answer)
        await session_manager.check_and_compact()


@router.post("/query")
async def query(req: QueryRequest) -> EventSourceResponse:
    """Run the agent and stream events via SSE."""
    return EventSourceResponse(_event_generator(req))
