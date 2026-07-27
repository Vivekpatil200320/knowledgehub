"""Conversation listing, naming, and deletion.

The RAG chain is stubbed out here — what's under test is the sidebar contract: a
conversation must name itself, report real activity, and be removable.
"""

import pytest

from app.api.routes import conversations as conversations_route
from app.models.orm import Message
from app.services.chat_service import derive_title


@pytest.fixture(autouse=True)
def stub_answer(monkeypatch):
    """Reply without touching NVIDIA/Qdrant, but still persist both turns."""

    def fake_answer(db, conversation_id, content):
        from app.services.chat_service import _persist_user_message

        _persist_user_message(db, conversation_id, content)
        reply = Message(
            conversation_id=conversation_id,
            role="assistant",
            content="stubbed answer",
            citations=None,
            condensed_query=content,
        )
        db.add(reply)
        db.commit()
        db.refresh(reply)
        return reply

    monkeypatch.setattr(conversations_route, "answer_message", fake_answer)


def new_conversation(client):
    return client.post("/api/conversations", json={}).json()["id"]


def send(client, conversation_id, content):
    return client.post(
        f"/api/conversations/{conversation_id}/messages", json={"content": content}
    )


# --- titles -----------------------------------------------------------------


def test_first_message_names_the_conversation(client):
    conversation_id = new_conversation(client)
    send(client, conversation_id, "What is the refund policy?")

    listed = client.get("/api/conversations").json()[0]
    assert listed["title"] == "What is the refund policy?"


def test_later_messages_do_not_rename(client):
    conversation_id = new_conversation(client)
    send(client, conversation_id, "What is the refund policy?")
    send(client, conversation_id, "And what about shipping?")

    listed = client.get("/api/conversations").json()[0]
    assert listed["title"] == "What is the refund policy?"


def test_new_conversation_has_no_title_until_used(client):
    new_conversation(client)

    listed = client.get("/api/conversations").json()[0]
    assert listed["title"] is None
    assert listed["message_count"] == 0


@pytest.mark.parametrize(
    "content, expected",
    [
        ("  spaced   out\nquestion  ", "spaced out question"),
        ("short", "short"),
    ],
)
def test_derive_title_normalises_whitespace(content, expected):
    assert derive_title(content) == expected


def test_derive_title_truncates_on_a_word_boundary():
    long_question = (
        "Can you explain the entire architecture of the retrieval pipeline "
        "including chunking and embeddings?"
    )
    title = derive_title(long_question)

    assert title.endswith("…")
    assert len(title) <= 61  # 60 chars plus the ellipsis
    assert not title.rstrip("…").endswith(" ")  # no dangling separator
    assert long_question.startswith(title.rstrip("…"))


def test_derive_title_handles_a_single_long_word():
    title = derive_title("x" * 200)

    assert title.endswith("…")
    assert len(title) <= 61


# --- activity ordering ------------------------------------------------------


def test_list_orders_by_latest_activity_not_creation(client):
    first = new_conversation(client)
    second = new_conversation(client)

    # Reply in the OLDER thread; it should surface above the newer, empty one.
    send(client, first, "Reviving the older thread")

    listed = client.get("/api/conversations").json()
    assert [c["id"] for c in listed] == [first, second]


def test_list_reports_message_count(client):
    conversation_id = new_conversation(client)
    send(client, conversation_id, "one")
    send(client, conversation_id, "two")

    listed = client.get("/api/conversations").json()[0]
    assert listed["message_count"] == 4  # two user turns + two assistant replies


def test_empty_conversation_falls_back_to_created_at(client):
    new_conversation(client)

    listed = client.get("/api/conversations").json()[0]
    assert listed["last_message_at"] == listed["created_at"]


# --- deletion ---------------------------------------------------------------


def test_delete_removes_conversation_and_messages(client):
    conversation_id = new_conversation(client)
    send(client, conversation_id, "hello")

    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert client.get("/api/conversations").json() == []
    assert client.get(f"/api/conversations/{conversation_id}/messages").status_code == 404


def test_delete_unknown_conversation_is_404(client):
    assert client.delete("/api/conversations/does-not-exist").status_code == 404
