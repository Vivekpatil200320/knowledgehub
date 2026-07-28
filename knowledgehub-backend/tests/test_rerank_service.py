"""Rerank ordering logic, isolated from the real cross-encoder.

Loading `cross-encoder/ms-marco-MiniLM-L-6-v2` costs ~10-15s and needs the model on
disk or a network fetch — a fine one-time cost for the running app (see the lifespan
preload in `app.main`), a bad one to pay on every test run. These monkeypatch
`get_reranker` with a stub whose `predict` is deterministic, so what's under test is
`rerank()`'s own logic (scoring, sorting, leaving `score` alone, prefixing the filename)
rather than the model.
"""

from app.services import rerank_service


class FakeCrossEncoder:
    """predict() returns a fixed score per pair, keyed by the exact passage text sent."""

    def __init__(self, scores_by_passage: dict[str, float]):
        self.scores_by_passage = scores_by_passage
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return [self.scores_by_passage[passage] for _, passage in pairs]


def _chunk(text: str, score: float = 0.5, filename: str = "doc.md") -> dict:
    return {"text": text, "score": score, "metadata": {"filename": filename}}


def _passage(filename: str, text: str) -> str:
    return f"Document: {filename}\n\n{text}"


def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    fake = FakeCrossEncoder(
        {
            _passage("doc.md", "irrelevant"): -5.0,
            _passage("doc.md", "relevant"): 8.0,
            _passage("doc.md", "somewhat"): 1.0,
        }
    )
    monkeypatch.setattr(rerank_service, "get_reranker", lambda: fake)

    chunks = [_chunk("irrelevant"), _chunk("relevant"), _chunk("somewhat")]
    out = rerank_service.rerank("query", chunks)

    assert [c["text"] for c in out] == ["relevant", "somewhat", "irrelevant"]


def test_rerank_does_not_mutate_the_original_cosine_score(monkeypatch):
    """The refusal and citation thresholds are calibrated against cosine similarity —
    reranking must add a score, never overwrite the one those thresholds read."""
    fake = FakeCrossEncoder({_passage("doc.md", "a"): -3.0})
    monkeypatch.setattr(rerank_service, "get_reranker", lambda: fake)

    out = rerank_service.rerank("query", [_chunk("a", score=0.42)])

    assert out[0]["score"] == 0.42
    assert out[0]["rerank_score"] == -3.0


def test_rerank_on_empty_input_does_not_call_the_model(monkeypatch):
    fake = FakeCrossEncoder({})
    monkeypatch.setattr(rerank_service, "get_reranker", lambda: fake)

    assert rerank_service.rerank("query", []) == []
    assert fake.calls == []


def test_rerank_prefixes_each_passage_with_its_document_filename(monkeypatch):
    """Reproduces a measured regression: reranking "describe candidate profile" against
    bare chunk text scored the résumé's own chunks at -11.4 — indistinguishable from
    totally unrelated documents — because the query never uses any word that appears
    in the résumé body. `embedding_service.with_document_context` solves exactly this
    for the bi-encoder by embedding a "Document: {filename}" header; reranking needs
    the same header for the same reason, since it never sees that embedded form, only
    the citation-safe raw text. Prefixing the filename flipped the correct chunk's
    score to +2.9 in the live measurement — this test locks in that the header is
    actually sent, not just that scoring behaves reasonably in the abstract."""
    fake = FakeCrossEncoder({_passage("candidate-profile.pdf", "x"): 1.0})
    monkeypatch.setattr(rerank_service, "get_reranker", lambda: fake)

    rerank_service.rerank("describe candidate profile", [_chunk("x", filename="candidate-profile.pdf")])

    assert fake.calls == [
        [("describe candidate profile", "Document: candidate-profile.pdf\n\nx")]
    ]


def test_rerank_falls_back_to_unknown_for_a_missing_filename(monkeypatch):
    fake = FakeCrossEncoder({_passage("unknown", "x"): 1.0})
    monkeypatch.setattr(rerank_service, "get_reranker", lambda: fake)

    rerank_service.rerank("q", [{"text": "x", "score": 0.5, "metadata": {}}])

    assert fake.calls == [[("q", "Document: unknown\n\nx")]]


def test_get_reranker_is_a_lazy_singleton(monkeypatch):
    """`CrossEncoder` is imported inside `get_reranker`, not at module level (see the
    docstring there for why), so it's patched via `sys.modules` rather than as a
    `rerank_service` attribute."""
    import sys
    import types

    created = []

    class _Stub:
        def __init__(self, model_name):
            created.append(model_name)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _Stub
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(rerank_service, "_reranker", None)

    first = rerank_service.get_reranker()
    second = rerank_service.get_reranker()

    assert first is second
    assert len(created) == 1
