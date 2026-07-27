from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.orm import Conversation, Message
from app.models.schemas import (
    ConversationCreate,
    ConversationOut,
    MessageCreate,
    MessageOut,
)
from app.services.chat_service import answer_message, stream_answer

router = APIRouter(tags=["conversations"])


def _get_conversation(conversation_id: str, db: Session) -> Conversation:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _to_out(conversation: Conversation, last_at, message_count: int) -> ConversationOut:
    return ConversationOut(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        last_message_at=last_at,
        message_count=message_count,
    )


@router.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(
    payload: ConversationCreate | None = None, db: Session = Depends(get_db)
) -> ConversationOut:
    conversation = Conversation(title=payload.title if payload else None)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    # A brand-new thread has no activity yet, so it reads as active at creation.
    return _to_out(conversation, conversation.created_at, 0)


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)) -> list[ConversationOut]:
    """History ordered by real activity, not creation time.

    A thread you replied to an hour ago belongs above one you opened last week, so the
    sidebar sorts on the newest message and falls back to created_at for empty threads.
    """
    last_message_at = func.max(Message.created_at)
    rows = db.execute(
        select(
            Conversation,
            func.coalesce(last_message_at, Conversation.created_at).label("last_message_at"),
            func.count(Message.id).label("message_count"),
        )
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .group_by(Conversation.id)
        .order_by(func.coalesce(last_message_at, Conversation.created_at).desc())
    ).all()

    return [_to_out(conversation, last_at, count) for conversation, last_at, count in rows]


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> Response:
    # Messages go with it via cascade="all, delete-orphan" on the relationship.
    db.delete(_get_conversation(conversation_id, db))
    db.commit()
    return Response(status_code=204)


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(conversation_id: str, db: Session = Depends(get_db)) -> list[Message]:
    _get_conversation(conversation_id, db)
    return list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
    )


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageOut,
    status_code=201,
)
def send_message(
    conversation_id: str,
    payload: MessageCreate,
    db: Session = Depends(get_db),
) -> Message:
    _get_conversation(conversation_id, db)
    return answer_message(db, conversation_id, payload.content)


@router.post("/conversations/{conversation_id}/messages/stream")
def send_message_stream(
    conversation_id: str,
    payload: MessageCreate,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    _get_conversation(conversation_id, db)
    return StreamingResponse(
        stream_answer(conversation_id, payload.content),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
