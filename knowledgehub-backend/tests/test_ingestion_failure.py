"""Ingestion failure handling: ordering and message sanitisation.

Two things had to hold simultaneously in run_ingestion's except block:
- cleanup failing (a broken Qdrant connection) must not prevent the status update,
  or a document stays stuck on "processing" forever instead of ever reaching "failed".
- the text that reaches status_detail, which the frontend renders verbatim, must be
  safe to show a user — a provider's raw error can carry a URL, an auth failure, or
  other internal detail that a ValueError we raised ourselves never would.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.models.orm import Document
from app.services import ingestion_service


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    """An isolated SQLite DB, with run_ingestion's own SessionLocal pointed at it.

    run_ingestion opens its own session directly via `from app.core.db import
    SessionLocal` rather than through a request-scoped dependency, so there's no
    TestClient/get_db override to piggyback on here — it has to be patched directly.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(ingestion_service, "SessionLocal", TestSession)

    session = TestSession()
    yield session
    session.close()


def _make_document(session) -> str:
    document = Document(
        filename="notes.md", stored_path="/fake/notes.md", content_type="md", status="pending"
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    return document.id


def _raise(exc: Exception):
    def _raiser(*args, **kwargs):
        raise exc

    return _raiser


def test_cleanup_failure_does_not_block_the_status_update(db_session, monkeypatch):
    document_id = _make_document(db_session)

    monkeypatch.setattr(ingestion_service, "extract_text", _raise(RuntimeError("embedding boom")))
    monkeypatch.setattr(
        ingestion_service, "delete_document_vectors", _raise(RuntimeError("qdrant unreachable"))
    )

    ingestion_service.run_ingestion(document_id)  # must not raise

    document = db_session.get(Document, document_id)
    assert document.status == "failed"
    assert document.status_detail  # the update actually happened, not silently skipped


def test_a_deliberate_valueerror_reaches_the_user_verbatim(db_session, monkeypatch):
    """extract_text and split_text raise ValueError on purpose, with messages meant
    to be read (empty file, scanned/image-only PDF, no chunks) — those should pass
    through unchanged."""
    document_id = _make_document(db_session)
    message = "No extractable text found — the file may be scanned or image-based."
    monkeypatch.setattr(ingestion_service, "extract_text", _raise(ValueError(message)))

    ingestion_service.run_ingestion(document_id)

    document = db_session.get(Document, document_id)
    assert document.status == "failed"
    assert document.status_detail == message


def test_a_non_valueerror_becomes_a_generic_message(db_session, monkeypatch):
    """Anything that isn't a ValueError we raised ourselves is a provider/DB failure
    whose raw text isn't safe to hand back — it must not reach status_detail."""
    document_id = _make_document(db_session)
    monkeypatch.setattr(
        ingestion_service,
        "extract_text",
        _raise(RuntimeError("connection to https://internal-proxy.local:8443 refused")),
    )

    ingestion_service.run_ingestion(document_id)

    document = db_session.get(Document, document_id)
    assert document.status == "failed"
    assert "internal-proxy" not in document.status_detail
    assert document.status_detail == "Processing failed. Please try re-uploading the file."


def test_document_deleted_mid_ingestion_is_skipped_not_errored(db_session, monkeypatch):
    """A document removed by the user between upload and background execution should
    be a no-op, not a crash — run_ingestion already guards this; confirm it still does
    after the except-block changes."""
    ingestion_service.run_ingestion("does-not-exist")  # must not raise
