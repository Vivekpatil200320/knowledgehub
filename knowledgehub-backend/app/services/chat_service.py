import asyncio
import json
from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import SessionLocal
from app.models.orm import Message
from app.services.condensation_service import condense
from app.services.llm_service import stream_grounded_answer
from app.services.retrieval_service import above_threshold, retrieve, to_citations

REFUSAL_MESSAGE = (
    "I couldn't find anything in the uploaded documents that answers that. "
    "Try rephrasing, or upload a document that covers it."
)


def _load_history(db: Session, conversation_id: str) -> list[Message]:
    recent = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(settings.chat_history_turns)
        )
    )
    return list(reversed(recent))


def _persist_user_message(db: Session, conversation_id: str, content: str) -> Message:
    message = Message(conversation_id=conversation_id, role="user", content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def _persist_assistant_message(
    db: Session,
    conversation_id: str,
    content: str,
    citations: list[dict] | None,
    condensed_query: str,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=content,
        citations=citations,
        condensed_query=condensed_query,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


async def _prepare_turn(db: Session, conversation_id: str, content: str) -> tuple[str, list[dict]]:
    """Persist the user turn, condense against history, retrieve. Returns (condensed_query, chunks)."""
    history = _load_history(db, conversation_id)
    _persist_user_message(db, conversation_id, content)

    condensed_query = await condense(history, content)
    chunks = above_threshold(retrieve(condensed_query))
    return condensed_query, chunks


def answer_message(db: Session, conversation_id: str, content: str) -> Message:
    async def _run() -> tuple[str, list[dict], str]:
        condensed_query, chunks = await _prepare_turn(db, conversation_id, content)

        # Layer 1 refusal: nothing relevant retrieved, so never call the generation model.
        if not chunks:
            return REFUSAL_MESSAGE, [], condensed_query

        parts = [token async for token in stream_grounded_answer(content, chunks)]
        return "".join(parts), to_citations(chunks), condensed_query

    answer, citations, condensed_query = asyncio.run(_run())
    return _persist_assistant_message(
        db, conversation_id, answer, citations or None, condensed_query
    )


def _sse(event_type: str, data) -> str:
    return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"


async def stream_answer(conversation_id: str, content: str) -> AsyncGenerator[str, None]:
    """SSE variant. Opens its own session — the request-scoped one closes when the response starts."""
    db = SessionLocal()
    try:
        condensed_query, chunks = await _prepare_turn(db, conversation_id, content)
        yield _sse("condensed_query", condensed_query)

        if not chunks:
            yield _sse("token", REFUSAL_MESSAGE)
            _persist_assistant_message(
                db, conversation_id, REFUSAL_MESSAGE, None, condensed_query
            )
            yield _sse("citations", [])
            yield _sse("done", None)
            return

        parts: list[str] = []
        async for token in stream_grounded_answer(content, chunks):
            parts.append(token)
            yield _sse("token", token)

        citations = to_citations(chunks)
        _persist_assistant_message(
            db, conversation_id, "".join(parts), citations, condensed_query
        )
        yield _sse("citations", citations)
        yield _sse("done", None)

    except Exception as exc:
        yield _sse("error", str(exc))
    finally:
        db.close()
