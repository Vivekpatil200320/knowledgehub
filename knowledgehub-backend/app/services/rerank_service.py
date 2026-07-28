from typing import Any

from app.core.config import settings

_reranker: Any = None


def get_reranker() -> Any:
    """Lazy singleton, like every other model client in this codebase
    (`llm_service.get_nvidia_llm`, `embedding_service.get_embedder`).

    The import is deliberately inside the function, not at module level: importing
    `sentence_transformers` alone costs ~2-3s (it pulls in torch), a tax every test run
    would otherwise pay whether or not a test ever touches reranking. Loading the model
    itself costs a further ~10-15s the first time — cheap after that, but not something
    a live chat request should ever pay, which is why `app.main`'s lifespan calls this
    once at startup rather than letting the first real query eat the load time.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(settings.rerank_model)
    return _reranker


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Reorder retrieved chunks by a cross-encoder's judgment of query/passage relevance.

    Bi-encoder cosine similarity — what `vector_store.search` ranks chunks by — embeds
    the query and each passage independently, then compares the two vectors. That's fast
    enough to run over an entire corpus, but blind to interaction between query and
    passage words: "Acme Queue pricing" and "Acme Run pricing" can land close together
    in embedding space even though only one answers the question actually asked. A
    cross-encoder instead reads (query, passage) together in one forward pass, which is
    far more accurate but too slow to run over everything — so it only reranks the small
    candidate set the bi-encoder already narrowed down, never the corpus itself.

    Deliberately does not touch `chunk["score"]`. The refusal and context-floor
    thresholds elsewhere in this pipeline were measured against cosine similarity
    specifically (see `core/config.py`); rescaling them for cross-encoder logits is a
    separate calibration this function doesn't attempt. Reranking's job here is
    narrowing and reordering *among* already-retrieved candidates for better precision —
    not replacing the existence-of-an-answer judgment, which stays on the score it was
    tuned against.

    Scores a `Document: {filename}` header alongside the chunk text, not the bare text
    alone. Measured regression from skipping this: "describe candidate profile" — the
    exact case `embedding_service.with_document_context` exists to handle, a résumé
    findable by the name the user sees but that never uses that phrase internally —
    scored -11.4 for every candidate with bare text, the cross-encoder unable to
    connect a generic query phrase to a specific person's résumé. Prefixing the
    filename flipped the correct chunk's score to +2.9. The bi-encoder gets this signal
    through `embed_text` at embedding time; reranking needs the same hook, since it
    never sees `embed_text` — only the citation-safe `text` that reaches this point.
    """
    if not chunks:
        return chunks
    pairs = [
        (query, f"Document: {c['metadata'].get('filename', 'unknown')}\n\n{c['text']}")
        for c in chunks
    ]
    scores = get_reranker().predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
