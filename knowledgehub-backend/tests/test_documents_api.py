import io

import pytest

from app.services import ingestion_service
from app.api.routes import documents as documents_route


@pytest.fixture(autouse=True)
def stub_ingestion(monkeypatch):
    """Ingestion hits NVIDIA + Qdrant; the API contract is what's under test here."""
    monkeypatch.setattr(ingestion_service, "run_ingestion", lambda document_id: None)
    monkeypatch.setattr(documents_route, "run_ingestion", lambda document_id: None)
    monkeypatch.setattr(documents_route, "delete_document_vectors", lambda document_id: None)


def upload(client, name="notes.md", body=b"# Notes\n\nSome content."):
    return client.post("/api/documents", files={"file": (name, io.BytesIO(body))})


def test_upload_returns_pending_immediately(client):
    response = upload(client)

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.md"
    assert body["status"] == "pending"
    assert body["chunk_count"] is None


def test_upload_rejects_unsupported_extension(client):
    response = upload(client, name="malware.exe")

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_empty_file(client):
    response = upload(client, body=b"")

    assert response.status_code == 400


def test_list_and_get_document(client):
    document_id = upload(client).json()["id"]

    assert len(client.get("/api/documents").json()) == 1
    assert client.get(f"/api/documents/{document_id}").json()["id"] == document_id


def test_title_is_none_until_the_document_is_ready(client, monkeypatch):
    """The title lookup hits Qdrant; a pending/failed document has nothing there yet.

    Calling it anyway would be a wasted round trip that always misses, so it must be
    gated on status rather than attempted unconditionally.
    """
    lookups = []
    monkeypatch.setattr(
        documents_route,
        "get_document_title",
        lambda document_id: lookups.append(document_id) or "should not be reached",
    )

    upload(client)

    body = client.get("/api/documents").json()[0]
    assert body["status"] == "pending"
    assert body["title"] is None
    assert lookups == []


def test_title_is_looked_up_once_a_document_is_ready(client, monkeypatch):
    document_id = upload(client).json()["id"]
    _mark_ready(document_id)

    monkeypatch.setattr(
        documents_route,
        "get_document_title",
        lambda doc_id: "Priya Nair" if doc_id == document_id else None,
    )

    assert client.get(f"/api/documents/{document_id}").json()["title"] == "Priya Nair"
    assert client.get("/api/documents").json()[0]["title"] == "Priya Nair"


def _mark_ready(document_id: str) -> None:
    """Flip a document to ready using the same session the test client is bound to.

    The client fixture overrides get_db with a per-test engine, so the module-level
    SessionLocal (bound to the app's default engine) would silently miss this row.
    """
    from app.core.db import get_db
    from app.main import app
    from app.models.orm import Document

    db = next(app.dependency_overrides[get_db]())
    document = db.get(Document, document_id)
    document.status = "ready"
    db.commit()


def test_get_unknown_document_returns_404(client):
    response = client.get("/api/documents/does-not-exist")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_delete_document(client):
    document_id = upload(client).json()["id"]

    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert client.get(f"/api/documents/{document_id}").status_code == 404


def test_message_to_unknown_conversation_returns_404(client):
    response = client.post(
        "/api/conversations/missing/messages", json={"content": "hello"}
    )

    assert response.status_code == 404


def test_empty_message_is_rejected(client):
    conversation_id = client.post("/api/conversations", json={}).json()["id"]

    response = client.post(
        f"/api/conversations/{conversation_id}/messages", json={"content": ""}
    )

    assert response.status_code == 422
