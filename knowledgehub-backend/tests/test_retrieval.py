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


def test_refusal_looks_past_a_reranked_top_position_for_the_best_cosine_score():
    """Reproduces a measured false refusal: "describe candidate profile" — a plainly
    answerable, in-corpus question — was refused outright once reranking shipped,
    because `select_context` read `chunks[0]["score"]` and reranking can (and did)
    promote a chunk with a *weaker* cosine score to the front of the list.

    Refusal has to ask "is there a genuinely strong cosine match anywhere in this set",
    not "is whichever chunk currently sits first a strong cosine match" — those are the
    same question only when nothing has reordered the list, which reranking does by
    design. A chunk below `refusal_score_threshold` at position 0, with a clearly
    in-corpus chunk right behind it, must still answer.
    """
    reranked_order = [chunk(0.15), chunk(0.46), chunk(0.30)]

    kept = retrieval_service.select_context(reranked_order)

    assert kept != []
    assert 0.46 in [c["score"] for c in kept]


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


def test_citation_floor_is_relative_for_supporting_chunks():
    """A supporting chunk's fate is query-dependent: 0.30 is kept under a 0.50 top hit
    (60%) and dropped under a 0.90 one (33%), because "relevant" is relative to the best
    match for this question rather than a fixed number. Both are above the absolute
    floor, so the relative bar is what decides here."""
    weak_top = retrieval_service.select_citations([chunk(0.50), chunk(0.30)])
    strong_top = retrieval_service.select_citations([chunk(0.90), chunk(0.30)])

    assert len(weak_top) == 2  # 0.30 is 60% of 0.50 -> kept
    assert len(strong_top) == 1  # 0.30 is 33% of 0.90 -> dropped; only the top hit remains


def test_the_top_hit_is_always_cited_even_below_the_absolute_floor():
    """An answer was generated, so it must show where it came from.

    The refusal threshold in `select_context` already decided this question is
    answerable; letting the citation floor strip the only source would render a
    grounded answer as an ungrounded-looking one — worse than the spurious extra
    citation the floor exists to prevent. Weak supporting chunks still go."""
    kept = retrieval_service.select_citations([chunk(0.22), chunk(0.15)])

    assert [c["score"] for c in kept] == [0.22]


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


def test_citation_absolute_floor_rejects_noise_when_the_top_hit_is_weak():
    """The relative floor alone opens up exactly when the answer is least certain.

    Reproduces the measured case: a genuinely-absent question whose best match scored
    0.36 dropped the relative cutoff to 0.18, admitting unrelated 0.19 chunks as
    confident-looking sources on an answer that said "that isn't in the documents".
    """
    chunks = [
        {"text": "a", "score": 0.36, "metadata": {}},
        {"text": "b", "score": 0.349, "metadata": {}},
        {"text": "c", "score": 0.19, "metadata": {}},   # cross-document noise
        {"text": "d", "score": 0.192, "metadata": {}},  # cross-document noise
    ]
    kept = [c["score"] for c in retrieval_service.select_citations(chunks)]
    assert kept == [0.36, 0.349]


def test_relative_floor_still_governs_when_the_top_hit_is_strong():
    """The absolute floor is a backstop, not a replacement — it must not become the
    binding constraint on a confident answer, or it would start admitting noise that
    the relative bar was correctly rejecting."""
    chunks = [
        {"text": "a", "score": 0.90, "metadata": {}},
        {"text": "b", "score": 0.50, "metadata": {}},  # clears absolute, fails relative (0.45)
        {"text": "c", "score": 0.30, "metadata": {}},
    ]
    kept = [c["score"] for c in retrieval_service.select_citations(chunks)]
    assert kept == [0.90, 0.50]


def _fake_rerank(scores: dict[str, float]):
    """Mimics the real `rerank()` contract: attaches `rerank_score` to every chunk
    and returns them sorted by it, descending."""

    def _rerank(query, chunks):
        for c in chunks:
            c["rerank_score"] = scores[c["text"]]
        return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

    return _rerank


def test_retrieve_reranks_when_enabled(monkeypatch):
    """`retrieve()` must actually call the reranker, not just have one available —
    a wiring test, distinct from `test_rerank_service.py`'s coverage of `rerank()`
    itself. Fetches a wider pool (`rerank_candidate_multiplier`, not the plain `*2`)
    so the cross-encoder has a real candidate set to search, and lets it promote a
    chunk cosine ranked lower to the front of the final order."""
    monkeypatch.setattr(retrieval_service.settings, "rerank_enabled", True)
    monkeypatch.setattr(retrieval_service.settings, "rerank_candidate_multiplier", 3)

    hits = [
        {"text": "cosine-favourite", "score": 0.9, "metadata": {}},
        {"text": "cross-encoder-favourite", "score": 0.5, "metadata": {}},
        {"text": "filler", "score": 0.4, "metadata": {}},
    ]
    monkeypatch.setattr(retrieval_service, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(retrieval_service, "search", lambda emb, limit: hits)
    # Cross-encoder disagrees with cosine about which chunk is best.
    monkeypatch.setattr(
        retrieval_service,
        "rerank",
        _fake_rerank({"cosine-favourite": 1.0, "cross-encoder-favourite": 10.0, "filler": -5.0}),
    )

    out = retrieval_service.retrieve("query", top_k=1)

    assert out[0]["text"] == "cross-encoder-favourite"


def test_retrieve_never_lets_reranking_evict_the_cosine_top_hit(monkeypatch):
    """Reproduces a measured regression: a résumé's EDUCATION chunk was cosine's clear
    #1 match for "where did she study?" (0.46, the best score in the whole corpus) but
    scored -11.36 on the cross-encoder — indistinguishable from unrelated noise — and
    got trimmed out of the final top-k entirely, turning a plainly answerable question
    into a false refusal. The cosine top-k must survive regardless of how badly the
    cross-encoder scores it; reranking may only ADD chunks beyond that guaranteed set."""
    monkeypatch.setattr(retrieval_service.settings, "rerank_enabled", True)
    monkeypatch.setattr(retrieval_service.settings, "rerank_candidate_multiplier", 4)

    hits = [
        {"text": "cosine-best-but-reranker-hates-it", "score": 0.9, "metadata": {}},
        *[{"text": f"noise-{i}", "score": 0.05, "metadata": {}} for i in range(5)],
    ]
    monkeypatch.setattr(retrieval_service, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(retrieval_service, "search", lambda emb, limit: hits)
    scores = {"cosine-best-but-reranker-hates-it": -11.36}
    scores.update({f"noise-{i}": -11.3 + i * 0.01 for i in range(5)})  # all "better" noise
    monkeypatch.setattr(retrieval_service, "rerank", _fake_rerank(scores))

    out = retrieval_service.retrieve("query", top_k=1)

    assert "cosine-best-but-reranker-hates-it" in [c["text"] for c in out]


def test_retrieve_lets_reranking_add_beyond_the_cosine_floor(monkeypatch):
    """The other half of the contract: reranking isn't inert — it can still surface a
    chunk cosine under-ranked, as long as doing so doesn't cost the guaranteed floor."""
    monkeypatch.setattr(retrieval_service.settings, "rerank_enabled", True)
    monkeypatch.setattr(retrieval_service.settings, "rerank_candidate_multiplier", 3)

    hits = [
        {"text": "cosine-top", "score": 0.9, "metadata": {}},
        {"text": "cosine-runner-up-but-actually-best", "score": 0.5, "metadata": {}},
        {"text": "filler", "score": 0.4, "metadata": {}},
    ]
    monkeypatch.setattr(retrieval_service, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(retrieval_service, "search", lambda emb, limit: hits)
    monkeypatch.setattr(
        retrieval_service,
        "rerank",
        _fake_rerank(
            {"cosine-top": 1.0, "cosine-runner-up-but-actually-best": 10.0, "filler": -5.0}
        ),
    )

    out = retrieval_service.retrieve("query", top_k=1)

    texts = [c["text"] for c in out]
    assert "cosine-top" in texts  # the guaranteed floor
    assert "cosine-runner-up-but-actually-best" in texts  # what reranking added
    assert texts[0] == "cosine-runner-up-but-actually-best"  # and it's rightly first


def test_retrieve_skips_reranking_when_disabled(monkeypatch):
    monkeypatch.setattr(retrieval_service.settings, "rerank_enabled", False)
    calls = []
    monkeypatch.setattr(retrieval_service, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(
        retrieval_service,
        "search",
        lambda emb, limit: [{"text": "only", "score": 0.9, "metadata": {}}],
    )
    monkeypatch.setattr(
        retrieval_service, "rerank", lambda query, chunks: calls.append(1) or chunks
    )

    retrieval_service.retrieve("query", top_k=1)

    assert calls == []
