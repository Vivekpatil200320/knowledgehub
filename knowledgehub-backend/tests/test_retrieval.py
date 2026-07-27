import asyncio

import pytest

from app.services import retrieval_service
from app.services.condensation_service import condense, format_history
from app.services.text_splitter import split_text


class FakeMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


def chunk(score, filename="doc.md", index=0, text="body"):
    return {
        "text": text,
        "score": score,
        "metadata": {"document_id": "doc-1", "filename": filename, "chunk_index": index},
    }


def test_split_text_tags_chunks_with_document_metadata():
    chunks = split_text("word " * 400, "doc-1", "doc.md")

    assert len(chunks) > 1
    assert chunks[0]["id"] == "doc-1_chunk_0"
    assert chunks[0]["metadata"] == {
        "document_id": "doc-1",
        "filename": "doc.md",
        "chunk_index": 0,
    }
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))


@pytest.fixture(autouse=True)
def thresholds(monkeypatch):
    monkeypatch.setattr(retrieval_service.settings, "refusal_score_threshold", 0.20)
    monkeypatch.setattr(retrieval_service.settings, "context_score_floor", 0.05)


def test_weak_top_hit_refuses_everything():
    """Nothing in the corpus is close, so the question is unanswerable here."""
    assert retrieval_service.select_context([chunk(0.11), chunk(0.09)]) == []


def test_strong_top_hit_keeps_weaker_supporting_chunks():
    """The regression that motivated splitting the thresholds.

    A resume's education section scores far below its header block. A single bar
    tuned to reject nonsense also rejected the chunk holding the actual answer.
    """
    kept = retrieval_service.select_context([chunk(0.46), chunk(0.20), chunk(0.10)])

    assert [c["score"] for c in kept] == [0.46, 0.20, 0.10]


def test_noise_below_the_floor_is_still_excluded():
    kept = retrieval_service.select_context([chunk(0.46), chunk(0.01)])

    assert [c["score"] for c in kept] == [0.46]


def test_no_hits_at_all_refuses():
    assert retrieval_service.select_context([]) == []


def test_to_citations_shape():
    citations = retrieval_service.to_citations([chunk(0.87654, text="x" * 500)])

    assert citations[0]["document_id"] == "doc-1"
    assert citations[0]["filename"] == "doc.md"
    assert citations[0]["chunk_index"] == 0
    assert citations[0]["score"] == 0.8765
    assert len(citations[0]["snippet"]) == 200


def test_condense_skips_llm_on_first_turn():
    """No history means nothing to resolve — must not spend an LLM call."""
    result = asyncio.run(condense([], "What is the pricing?"))

    assert result == "What is the pricing?"


def test_format_history_labels_roles():
    history = [FakeMessage("user", "Tell me about X"), FakeMessage("assistant", "X is a thing")]

    assert format_history(history) == "User: Tell me about X\nAssistant: X is a thing"


def test_ensure_collection_tolerates_concurrent_creation(monkeypatch):
    """Parallel ingestion tasks race to create the collection; the loser's 409 is fine."""
    from app.services import vector_store

    calls = {"exists": 0}

    class RacingClient:
        def collection_exists(self, name):
            calls["exists"] += 1
            # Absent on the pre-check, present by the time creation fails.
            return calls["exists"] > 1

        def create_collection(self, **kwargs):
            raise RuntimeError("409 Collection already exists")

    monkeypatch.setattr(vector_store, "get_client", lambda: RacingClient())

    vector_store.ensure_collection(2048)  # must not raise


def test_ensure_collection_reraises_real_failures(monkeypatch):
    from app.services import vector_store

    class BrokenClient:
        def collection_exists(self, name):
            return False

        def create_collection(self, **kwargs):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(vector_store, "get_client", lambda: BrokenClient())


    with pytest.raises(RuntimeError, match="connection refused"):
        vector_store.ensure_collection(2048)


def chunk_with(text, score, doc="doc-a", index=0):
    return {
        "text": text,
        "score": score,
        "metadata": {"document_id": doc, "filename": "f.md", "chunk_index": index},
    }


def test_dedupe_collapses_the_same_passage_across_documents():
    """Uploading a file twice yields two document_ids over identical text."""
    hits = [
        chunk_with("Pricing is $45 per seat.", 0.54, doc="doc-a"),
        chunk_with("Pricing is $45 per seat.", 0.54, doc="doc-b"),
        chunk_with("Support answers within a day.", 0.31, doc="doc-a", index=1),
    ]

    kept = retrieval_service.dedupe(hits)

    assert [c["text"] for c in kept] == [
        "Pricing is $45 per seat.",
        "Support answers within a day.",
    ]
    assert kept[0]["metadata"]["document_id"] == "doc-a"  # best-scoring copy first


def test_dedupe_ignores_whitespace_differences():
    hits = [
        chunk_with("Pricing  is\n$45.", 0.5),
        chunk_with("Pricing is $45.", 0.4, doc="doc-b"),
    ]

    assert len(retrieval_service.dedupe(hits)) == 1


def test_dedupe_keeps_genuinely_different_passages():
    hits = [chunk_with("First.", 0.5), chunk_with("Second.", 0.4, index=1)]

    assert len(retrieval_service.dedupe(hits)) == 2
