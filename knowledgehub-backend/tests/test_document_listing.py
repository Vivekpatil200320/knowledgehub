"""Corpus-listing questions ("list my documents") are not content questions.

Retrieval has nothing useful to return for them — no chunk's content IS "the list
of files" — but the pipeline used to run them through retrieval anyway. Whatever
scored least-badly still cleared the (top-hit-only) refusal threshold often enough,
and the model, given irrelevant chunks and no signal its job was impossible, recited
them as a "document listing": real pricing tables and course names, presented with
citations, as if they were filenames. This is the exact failure observed live against
a running instance — verified against the classifier and the short-circuit path.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.models.orm import Document
from app.services import chat_service
from app.services.chat_service import is_document_listing_question


# --- classifier: real observed phrasings must match ------------------------


@pytest.mark.parametrize(
    "question",
    [
        "list my documents",
        "list names of all uploaded documents",  # the exact raw message from the bug report
        "What are the names of all the uploaded documents?",  # its condensed form
        "how many documents do I have",
        "what files have I uploaded",
        "show me all documents",
    ],
)
def test_listing_phrasings_are_detected(question):
    assert is_document_listing_question(question)


# --- classifier: real content questions must NOT match ----------------------
# The false-positive risk: "which document mentions X" looks similar but needs
# actual retrieval, not a corpus dump.


@pytest.mark.parametrize(
    "question",
    [
        "What does the pricing document say about SLAs?",
        "Summarize the key points of candidate profile.",
        "which document mentions pricing",
        "What are Priya Nair's degrees?",
        "What services does Acme Cloud Platform offer?",
    ],
)
def test_content_questions_are_not_misdetected(question):
    assert not is_document_listing_question(question)


# --- the short-circuit itself: no retrieval, no generation, ground truth ----


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(chat_service, "SessionLocal", TestSession)
    return TestSession()


def _make_conversation(db_session) -> str:
    from app.models.orm import Conversation

    conversation = Conversation()
    db_session.add(conversation)
    db_session.commit()
    db_session.refresh(conversation)
    return conversation.id


def _add_document(db_session, filename: str, status: str = "ready") -> None:
    db_session.add(Document(filename=filename, stored_path="/x", content_type="md", status=status))
    db_session.commit()


def test_listing_question_never_calls_retrieve_or_generate(db_session, monkeypatch):
    """The whole point: no embedding call, no LLM call — just the DB."""
    _add_document(db_session, "acme-cloud-platform.md")
    _add_document(db_session, "zenith-analytics-suite.md")
    conversation_id = _make_conversation(db_session)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("retrieval/generation must not run for a listing question")

    monkeypatch.setattr(chat_service, "retrieve", fail_if_called)
    monkeypatch.setattr(chat_service, "stream_grounded_answer", fail_if_called)

    message = chat_service.answer_message(db_session, conversation_id, "list my documents")

    assert "acme-cloud-platform.md" in message.content
    assert "zenith-analytics-suite.md" in message.content
    assert message.citations is None  # not chunk-grounded — nothing to cite


def test_listing_answer_only_names_ready_documents(db_session, monkeypatch):
    """A document stuck on 'processing' or 'failed' isn't searchable yet and must
    not be claimed as available."""
    _add_document(db_session, "ready-doc.md", status="ready")
    _add_document(db_session, "still-processing.md", status="processing")
    _add_document(db_session, "broken.md", status="failed")
    conversation_id = _make_conversation(db_session)
    monkeypatch.setattr(chat_service, "retrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    message = chat_service.answer_message(db_session, conversation_id, "how many documents do I have")

    assert "ready-doc.md" in message.content
    assert "still-processing.md" not in message.content
    assert "broken.md" not in message.content


def test_listing_question_with_no_documents_says_so(db_session):
    conversation_id = _make_conversation(db_session)

    message = chat_service.answer_message(db_session, conversation_id, "list my documents")

    assert "no documents" in message.content.lower()


def test_content_question_still_goes_through_retrieval(db_session, monkeypatch):
    """The fix must not swallow real content questions — only the narrow listing case."""
    _add_document(db_session, "acme-cloud-platform.md")
    conversation_id = _make_conversation(db_session)

    called = {"retrieve": False}

    def fake_retrieve(query):
        called["retrieve"] = True
        return []

    monkeypatch.setattr(chat_service, "retrieve", fake_retrieve)

    message = chat_service.answer_message(
        db_session, conversation_id, "What services does Acme Cloud Platform offer?"
    )

    assert called["retrieve"] is True
    assert message.content == chat_service.REFUSAL_MESSAGE  # empty retrieve() -> refusal, as expected
