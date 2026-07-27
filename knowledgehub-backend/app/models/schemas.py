from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    content_type: str
    status: str
    status_detail: str | None = None
    chunk_count: int | None = None
    created_at: datetime


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    snippet: str
    score: float


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    citations: list[Citation] | None = None
    condensed_query: str | None = None
    created_at: datetime


class ConversationCreate(BaseModel):
    title: str | None = None


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str | None = None
    created_at: datetime
    # Derived per-request rather than stored: create_all() cannot add columns to an
    # existing SQLite file, and this project ships without migrations by design.
    last_message_at: datetime
    message_count: int = 0


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class ErrorResponse(BaseModel):
    detail: str
