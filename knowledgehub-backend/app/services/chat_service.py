import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import SessionLocal
from app.models.orm import Conversation, Document, Message
from app.services.condensation_service import condense
from app.services.llm_service import stream_grounded_answer
from app.services.retrieval_service import (
    narrative_order,
    retrieve,
    select_citations,
    select_context,
    to_citations,
)

logger = logging.getLogger("knowledgehub.chat")

REFUSAL_MESSAGE = (
    "I couldn't find anything in the uploaded documents that answers that. "
    "Try rephrasing, or upload a document that covers it."
)

# Deliberately narrow: these match "what's in my corpus" questions, not content
# questions that happen to mention the word "document". "which document mentions
# pricing" must NOT match — that needs real retrieval, not a corpus dump.
_LISTING_PATTERNS = [
    re.compile(r"\b(list|show(\s+me)?)\b.{0,20}\b(document|documents|file|files)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+many\b.{0,15}\b(document|documents|file|files)\b", re.IGNORECASE),
    re.compile(r"\bnames?\s+of\b.{0,20}\b(document|documents|file|files)\b", re.IGNORECASE),
    re.compile(
        r"\bwhat\b.{0,10}\b(documents|files)\b.{0,25}"
        r"\b(do i have|have i uploaded|are uploaded|exist|are there|have you got|do you have)\b",
        re.IGNORECASE,
    ),
]


def is_document_listing_question(text: str) -> bool:
    """Detect "what's in my corpus" questions, as distinct from content questions.

    Retrieval has nothing useful to return for these — there's no chunk whose content
    IS "the list of files" — yet the pipeline used to run them through retrieval
    anyway. Whatever scored least-badly still cleared the (top-hit-only) refusal
    threshold often enough, and the model, given irrelevant chunks and no signal that
    its job here was impossible, recited them as if they were a document listing:
    real pricing tables and course names presented, with citations, as "document
    names". This is a hallucination the two-layer refusal guard never covered,
    because retrieval genuinely found *something* — just nothing relevant to the
    actual question.
    """
    return any(p.search(text) for p in _LISTING_PATTERNS)


def _list_ready_document_filenames(db: Session) -> list[str]:
    return list(
        db.scalars(
            select(Document.filename)
            .where(Document.status == "ready")
            .order_by(Document.filename)
        )
    )


def _document_listing_answer(db: Session) -> str:
    filenames = _list_ready_document_filenames(db)
    if not filenames:
        return "No documents have been uploaded yet."
    count = len(filenames)
    noun = "document" if count == 1 else "documents"
    listing = "\n".join(f"- {name}" for name in filenames)
    return f"You have {count} {noun} uploaded:\n\n{listing}"


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


TITLE_MAX_CHARS = 60


def derive_title(content: str) -> str | None:
    """Condense a first message into a sidebar-sized conversation name.

    Deliberately not an LLM call: titling every new conversation would add a round
    trip and a failure mode to the hot path for something a truncation handles fine.

    Returns None when there is nothing nameable. An empty string would be worse than
    None: it is falsy, so the "name it once" guard below would fire again on the next
    turn and silently rename the thread, and the sidebar's `title ?? "Untitled chat"`
    fallback does not catch "" either, leaving a blank row.
    """
    collapsed = " ".join(content.split())
    if not collapsed:
        return None
    if len(collapsed) <= TITLE_MAX_CHARS:
        return collapsed

    clipped = collapsed[:TITLE_MAX_CHARS]
    # Prefer a word boundary, but only if it doesn't leave a stub.
    if " " in clipped[TITLE_MAX_CHARS // 2 :]:
        clipped = clipped[: clipped.rindex(" ")]

    trimmed = clipped.rstrip(" ,.;:")
    # Punctuation-only input can strip back to nothing; fall back to the hard clip.
    return (trimmed or clipped) + "…"


def _persist_user_message(db: Session, conversation_id: str, content: str) -> Message:
    message = Message(conversation_id=conversation_id, role="user", content=content)
    db.add(message)

    # The first user message names the conversation. Later messages never rename it —
    # a thread that renamed itself on every turn would be unfindable in the sidebar.
    conversation = db.get(Conversation, conversation_id)
    if conversation is not None and not conversation.title:
        title = derive_title(content)
        if title:
            conversation.title = title

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


async def _prepare_turn(
    db: Session, conversation_id: str, content: str
) -> tuple[str, list[dict], str | None]:
    """Persist the user turn, condense against history, retrieve.

    Returns (condensed_query, chunks, listing_answer). listing_answer is set (and
    chunks is []) when this is a deterministic corpus-listing question — retrieval
    never runs for that case, both because it would waste an embedding call and,
    more importantly, because running it is what produced the hallucinated "document
    listing" built from irrelevant retrieved chunks. Checked against both the raw
    message and the condensed query, since condensation's job is resolving missing
    context, not preserving intent, and shouldn't be trusted alone for this.

    The condensed query — not the raw message — is what gets generated against: the generation
    call receives no conversation history, so a context-dependent phrasing like "and what about
    Zenith?" reads as unanswerable to it even when the right chunks were retrieved.
    """
    history = _load_history(db, conversation_id)
    _persist_user_message(db, conversation_id, content)

    condensed_query = await condense(history, content)

    if is_document_listing_question(content) or is_document_listing_question(condensed_query):
        return condensed_query, [], _document_listing_answer(db)

    chunks = select_context(retrieve(condensed_query))
    return condensed_query, chunks, None


def answer_message(db: Session, conversation_id: str, content: str) -> Message:
    async def _run() -> tuple[str, list[dict], str]:
        condensed_query, chunks, listing_answer = await _prepare_turn(
            db, conversation_id, content
        )

        if listing_answer is not None:
            return listing_answer, [], condensed_query

        # Layer 1 refusal: nothing relevant retrieved, so never call the generation model.
        if not chunks:
            return REFUSAL_MESSAGE, [], condensed_query

        # Generation reads in document order; citations stay ranked by relevance.
        available_documents = _list_ready_document_filenames(db)
        parts = [
            token
            async for token in stream_grounded_answer(
                condensed_query, narrative_order(chunks), available_documents
            )
        ]
        return "".join(parts), to_citations(select_citations(chunks)), condensed_query

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
        condensed_query, chunks, listing_answer = await _prepare_turn(
            db, conversation_id, content
        )
        yield _sse("condensed_query", condensed_query)

        if listing_answer is not None:
            yield _sse("token", listing_answer)
            _persist_assistant_message(
                db, conversation_id, listing_answer, None, condensed_query
            )
            yield _sse("citations", [])
            yield _sse("done", None)
            return

        if not chunks:
            yield _sse("token", REFUSAL_MESSAGE)
            _persist_assistant_message(
                db, conversation_id, REFUSAL_MESSAGE, None, condensed_query
            )
            yield _sse("citations", [])
            yield _sse("done", None)
            return

        available_documents = _list_ready_document_filenames(db)
        parts: list[str] = []
        async for token in stream_grounded_answer(
            condensed_query, narrative_order(chunks), available_documents
        ):
            parts.append(token)
            yield _sse("token", token)

        citations = to_citations(select_citations(chunks))
        _persist_assistant_message(
            db, conversation_id, "".join(parts), citations, condensed_query
        )
        yield _sse("citations", citations)
        yield _sse("done", None)

    except Exception:
        # str(exc) can carry internal detail (file paths, a provider's raw API error,
        # a DB error string) straight to the browser. Log the real thing server-side
        # and give the client only a generic, safe message.
        logger.exception("Stream error for conversation %s", conversation_id)
        yield _sse("error", "Something went wrong generating a response. Please try again.")
    finally:
        db.close()
