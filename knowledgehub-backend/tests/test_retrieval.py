import asyncio

import pytest

from app.services import retrieval_service, vector_store
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
    monkeypatch.setattr(retrieval_service.settings, "citation_relative_floor", 0.5)


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


# --- select_citations --------------------------------------------------------
# select_context decides what the MODEL reads; select_citations decides what's
# shown as a "Source" chip. They're deliberately different bars: reusing
# select_context's floor here is the bug this guards against — an unrelated
# document's noise clears that floor easily, and citing it (correctly-answered
# question or not) reads as "why does this cite my resume?" to a user.


def test_noise_from_an_unrelated_document_is_not_cited():
    """Modelled on a live case: a correct answer at 0.52, unrelated-document noise
    at 0.13-0.18 (25-35% of the top hit) riding along because it cleared the much
    more permissive context floor."""
    kept = retrieval_service.select_citations(
        [
            chunk(0.522, filename="correct-doc.md"),
            chunk(0.182, filename="unrelated-resume.pdf"),
            chunk(0.169, filename="unrelated-resume.pdf", index=1),
            chunk(0.130, filename="another-unrelated.md"),
        ]
    )

    assert [c["metadata"]["filename"] for c in kept] == ["correct-doc.md"]


def test_a_same_document_supporting_chunk_is_still_cited():
    """The other side of the same coin: a second chunk from the SAME document,
    scoring close to the top hit, must not be discarded — this is genuine
    supporting evidence, not noise. Modelled on live same-document ratios of
    0.68-0.94 observed across real corpus queries."""
    kept = retrieval_service.select_citations(
        [chunk(0.544, filename="doc.md", index=0), chunk(0.391, filename="doc.md", index=1)]
    )

    assert len(kept) == 2


def test_citation_floor_is_relative_not_absolute():
    """The same absolute score (0.15) must be kept or dropped depending on what the
    top hit was for that question — "relevant" is query-dependent, not a fixed number."""
    weak_top = retrieval_service.select_citations([chunk(0.20), chunk(0.15)])
    strong_top = retrieval_service.select_citations([chunk(0.50), chunk(0.15)])

    assert len(weak_top) == 2  # 0.15 is 75% of 0.20 -> kept
    assert len(strong_top) == 1  # 0.15 is 30% of 0.50 -> dropped; only the top hit remains


def test_select_citations_on_empty_input():
    assert retrieval_service.select_citations([]) == []


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


def test_dedupe_keeps_the_best_scoring_copy_regardless_of_input_order():
    """The invariant the docstring promises, with the duplicates deliberately unsorted.

    Keeping the first occurrence only picks the best copy if the caller happened to
    sort first — a silent dependency on Qdrant's ordering that a second retrieval
    source would break, and `select_context` reads chunks[0] as the top hit.
    """
    hits = [
        chunk_with("Pricing is $45 per seat.", 0.31, doc="doc-weak"),
        chunk_with("Support answers within a day.", 0.20, doc="doc-a", index=1),
        chunk_with("Pricing is $45 per seat.", 0.62, doc="doc-strong"),
    ]

    kept = retrieval_service.dedupe(hits)

    assert [c["score"] for c in kept] == [0.62, 0.20]
    assert kept[0]["metadata"]["document_id"] == "doc-strong"


def test_dedupe_ignores_whitespace_differences():
    hits = [
        chunk_with("Pricing  is\n$45.", 0.5),
        chunk_with("Pricing is $45.", 0.4, doc="doc-b"),
    ]

    assert len(retrieval_service.dedupe(hits)) == 1


def test_dedupe_keeps_genuinely_different_passages():
    hits = [chunk_with("First.", 0.5), chunk_with("Second.", 0.4, index=1)]

    assert len(retrieval_service.dedupe(hits)) == 2


# --- narrative_order --------------------------------------------------------


def test_narrative_order_restores_document_reading_order():
    """The fix for mis-paired résumé degrees: relevance order scrambles sequence."""
    hits = [
        chunk_with("degree line", 0.9, doc="doc-a", index=3),
        chunk_with("header", 0.8, doc="doc-a", index=0),
        chunk_with("university line", 0.7, doc="doc-a", index=1),
    ]

    assert [c["metadata"]["chunk_index"] for c in retrieval_service.narrative_order(hits)] == [
        0,
        1,
        3,
    ]


def test_narrative_order_groups_by_document():
    hits = [
        chunk_with("b1", 0.9, doc="doc-b", index=1),
        chunk_with("a1", 0.8, doc="doc-a", index=1),
        chunk_with("a0", 0.7, doc="doc-a", index=0),
    ]

    ordered = retrieval_service.narrative_order(hits)

    assert [(c["metadata"]["document_id"], c["metadata"]["chunk_index"]) for c in ordered] == [
        ("doc-a", 0),
        ("doc-a", 1),
        ("doc-b", 1),
    ]


def test_narrative_order_survives_a_malformed_point():
    """A point written by an older payload schema must not 500 the whole request."""
    hits = [
        chunk_with("good", 0.9, doc="doc-a", index=1),
        {"text": "legacy", "score": 0.5, "metadata": {"document_id": "doc-a"}},
    ]

    ordered = retrieval_service.narrative_order(hits)

    assert [c["text"] for c in ordered] == ["legacy", "good"]  # missing index sorts as 0


# --- vector_store payload normalisation --------------------------------------


def test_normalise_hit_fills_missing_payload_fields():
    """Guards the whole pipeline: a legacy point must degrade, not raise."""
    normalised = vector_store.normalise_hit({}, 0.42)

    assert normalised["text"] == ""
    assert normalised["score"] == 0.42
    assert normalised["metadata"] == {
        "document_id": "",
        "filename": "unknown",
        "chunk_index": 0,
    }


def test_normalise_hit_coerces_a_string_chunk_index():
    normalised = vector_store.normalise_hit(
        {"text": "t", "document_id": "d", "filename": "f.md", "chunk_index": "3"}, 0.1
    )

    assert normalised["metadata"]["chunk_index"] == 3


def test_normalise_hit_preserves_index_zero():
    """`or`-style defaulting would turn a legitimate 0 into a default."""
    normalised = vector_store.normalise_hit(
        {"text": "t", "document_id": "d", "filename": "f.md", "chunk_index": 0}, 0.1
    )

    assert normalised["metadata"]["chunk_index"] == 0
