"""SSE error sanitisation.

chat_service.stream_answer's except block used to `yield _sse("error", str(exc))` —
whatever a provider's client or a DB driver put in the exception message (a
connection string, an internal hostname, an auth failure detail) went straight to
the browser. It must now only ever emit a generic message, with the real detail
going to the server log instead.
"""

import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.services import chat_service


@pytest.fixture
def isolated_session(tmp_path, monkeypatch):
    """stream_answer opens its own session via `from app.core.db import SessionLocal`
    rather than a request-scoped dependency, so it needs to be patched directly to
    point at an isolated test DB."""
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(chat_service, "SessionLocal", TestSession)
    return TestSession


def _collect_events(agen):
    async def _run():
        return [chunk async for chunk in agen]

    frames = asyncio.run(_run())
    return [json.loads(frame.removeprefix("data: ").strip()) for frame in frames]


def test_stream_answer_does_not_leak_the_raw_exception(isolated_session, monkeypatch):
    sensitive = "connection to postgres://admin:s3cr3t@10.0.0.5/prod refused"

    async def failing_condense(history, content):
        raise RuntimeError(sensitive)

    monkeypatch.setattr(chat_service, "condense", failing_condense)

    events = _collect_events(chat_service.stream_answer("conv-1", "hello"))

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert sensitive not in error_events[0]["data"]
    assert error_events[0]["data"] == (
        "Something went wrong generating a response. Please try again."
    )


def test_stream_answer_emits_no_error_event_on_success(isolated_session, monkeypatch):
    """Guards against the fix being too aggressive — a clean run must not also emit
    a spurious error frame."""

    async def passthrough_condense(history, content):
        return content

    async def empty_retrieve(*args, **kwargs):
        return []

    monkeypatch.setattr(chat_service, "condense", passthrough_condense)
    monkeypatch.setattr(chat_service, "retrieve", lambda query: [])

    events = _collect_events(chat_service.stream_answer("conv-1", "hello"))

    assert [e["type"] for e in events] == ["condensed_query", "token", "citations", "done"]
