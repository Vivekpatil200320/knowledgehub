from app.core.config import settings
from app.services.embedding_service import embed_query
from app.services.rerank_service import rerank
from app.services.vector_store import search


def dedupe(chunks: list[dict]) -> list[dict]:
    """Collapse repeats of the same passage, keeping the best-scoring copy.

    Uploading the same file twice is an ordinary thing to do, and it produces two
    document_ids over identical text. Without this, top-k fills with the same
    passage twice: the model re-reads it, the citation list repeats itself, and
    half the context budget buys nothing. Keyed on the text rather than on
    (document_id, chunk_index) so it catches copies across documents too.

    The result is ordered by score descending. Sorting here rather than relying on
    the caller is what makes "best-scoring copy" true: reading the first occurrence
    only picks the best one if the input happened to arrive sorted, which is a
    silent dependency on Qdrant's ordering that a second retrieval source would break.

    This cosine ordering is provisional, not final: `retrieve` reorders the result by
    cross-encoder score when reranking is enabled, so nothing downstream may assume
    `chunks[0]` is the highest-cosine chunk — `select_context`'s refusal check in
    particular looks at `max()` over the whole set for exactly this reason.
    """
    ranked = sorted(chunks, key=lambda c: c["score"], reverse=True)

    seen: set[str] = set()
    unique = []
    for chunk in ranked:
        key = " ".join(chunk["text"].split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    limit = top_k or settings.retrieval_top_k
    if not settings.rerank_enabled:
        # Over-fetch so that discarding duplicates doesn't shrink context below budget.
        hits = search(embed_query(query), limit * 2)
        return dedupe(hits)[:limit]

    # Over-fetch a wider pool: the point of a cross-encoder is to find better answers
    # than cosine similarity ranked first, which needs a real candidate set to search.
    hits = search(embed_query(query), limit * settings.rerank_candidate_multiplier)
    deduped = dedupe(hits)
    cosine_top = deduped[:limit]

    # Reranking is additive here, never subtractive. Measured regression from letting
    # it fully replace cosine selection: for "where did she study?", a résumé's
    # EDUCATION section — cosine's #1 match at 0.46 — scored -11.36 on the
    # cross-encoder, indistinguishable from unrelated noise, because the chunk's
    # opening line is unrelated job-history text carried over from the chunk-overlap
    # boundary. A small local cross-encoder is exactly the size where this happens;
    # letting it silently drop what cosine confidently found turned a plainly
    # answerable question into a false "I don't have this information."
    #
    # So: the cosine top-k is a guaranteed floor, never evicted. Reranking can only
    # ADD chunks cosine underrated, up to the same budget again — improving precision
    # without being able to regress recall. Final order is by cross-encoder score,
    # since ordering (unlike inclusion) is exactly what reranking is good at.
    reranked = rerank(query, deduped)
    guaranteed_ids = {id(c) for c in cosine_top}
    extra = [c for c in reranked if id(c) not in guaranteed_ids][:limit]
    combined = cosine_top + extra
    return sorted(combined, key=lambda c: c["rerank_score"], reverse=True)


def select_context(chunks: list[dict]) -> list[dict]:
    """Decide what (if anything) to answer from.

    Two separate judgements, deliberately not collapsed into one threshold:
      1. Is this question answerable at all? Judged on the best hit only.
      2. Which chunks are worth putting in the context window? A much lower bar.

    Using a single high threshold for both silently drops the supporting evidence of
    questions that are plainly answerable — a resume's education section scores far
    below its header block, so a bar tuned to reject nonsense also rejected the answer.

    "The best hit" is `max(score)` over the whole set, not `chunks[0]` — deliberately,
    since `chunks[0]` is no longer guaranteed to be the highest-cosine chunk once
    reranking is enabled (`retrieve` reorders by cross-encoder score, which can and does
    promote a chunk with a *lower* cosine score to the front). Measured regression from
    getting this wrong: "describe candidate profile" — a plainly answerable, in-corpus
    question — was refused outright, because the reranked chunks[0] happened to score
    below `refusal_score_threshold` even though a clearly-relevant chunk sat right behind
    it in the reordered list. Refusal is a judgment about the corpus's cosine-similarity
    evidence specifically (that's what `refusal_score_threshold` was measured against),
    so it has to look at all the evidence, not just whichever chunk reranking put first.
    """
    if not chunks or max(c["score"] for c in chunks) < settings.refusal_score_threshold:
        return []
    return [c for c in chunks if c["score"] >= settings.context_score_floor]


def select_citations(chunks: list[dict]) -> list[dict]:
    """Narrow what's SHOWN as a source, separately from what's READ for generation.

    `select_context`'s floor is deliberately permissive — it has to be, to keep a
    resume's low-scoring education section in the model's context. But that same
    permissiveness means an unrelated document's noise (something that merely cleared
    the floor, not something actually relevant) rides along into the context window,
    and showing it as a citation is what erodes trust: a pricing answer citing an
    unrelated resume looks like a bug even when the pricing answer itself is correct.

    `chunks` must already be sorted so `chunks[0]` is the chunk to feature as *the*
    citation — cross-encoder order when reranking is on, cosine order otherwise; either
    way, whichever the pipeline currently considers most relevant. The relative bar is
    computed off that chunk's cosine `score`, because "relevant" is corpus- and
    query-dependent — where a chunk lands relative to the best cosine match for this
    question separates signal from noise better than any fixed number. (Unlike
    `select_context`'s refusal gate, a non-max `score` here isn't a correctness bug: the
    absolute floor below exists precisely to catch a relative bar computed off a
    middling top pick, whatever order produced it.)

    But a purely relative bar inverts under a weak top hit: half of a low score is a very
    low bar, so the moment the corpus barely covers the question — when a spurious source
    is most misleading — the filter opens up instead of tightening. An absolute floor
    backstops that case, and a supporting chunk must clear both.

    The top hit is exempt from the absolute floor. It has already passed the refusal
    threshold in `select_context`, so by the time this runs we are definitely answering;
    applying the floor to it too would let a weak-but-answerable question render an
    answer with no source at all, which reads as ungrounded — a worse failure than the
    spurious-citation one this floor exists to fix. Show where the answer came from, and
    hold everything *else* to a bar that doesn't slacken when the top hit is weak.
    """
    if not chunks:
        return []
    top_score = chunks[0]["score"]
    cutoff = max(
        top_score * settings.citation_relative_floor,
        settings.citation_absolute_floor,
    )
    return [chunks[0]] + [c for c in chunks[1:] if c["score"] >= cutoff]


def narrative_order(chunks: list[dict]) -> list[dict]:
    """Reorder context back into document reading order before generation.

    Retrieval returns chunks by relevance, which scrambles documents whose meaning
    depends on sequence. A resume splits "MIT ADT University, Pune 2024-2026" from the
    degree line that belongs to it across a chunk boundary; presented out of order, the
    model pairs each degree with the wrong university and invents a third. Relevance
    decides what the model sees — the document decides what order it reads them in.
    """
    def position(chunk: dict) -> tuple[str, int]:
        metadata = chunk.get("metadata", {})
        index = metadata.get("chunk_index")
        # Defensive: `vector_store.normalise_hit` guarantees an int for anything that
        # came through retrieval, but this is a public helper and a None here would
        # raise TypeError mid-sort rather than merely ordering badly.
        return (
            str(metadata.get("document_id") or ""),
            index if isinstance(index, int) else 0,
        )

    return sorted(chunks, key=position)


def to_citations(chunks: list[dict]) -> list[dict]:
    return [
        {
            "document_id": c["metadata"]["document_id"],
            "filename": c["metadata"]["filename"],
            "chunk_index": c["metadata"]["chunk_index"],
            "snippet": c["text"][:200],
            "score": round(c["score"], 4),
        }
        for c in chunks
    ]
