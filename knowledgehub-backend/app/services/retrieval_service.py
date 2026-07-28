from app.core.config import settings
from app.services.embedding_service import embed_query
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
    `select_context` also reads `chunks[0]` as the top hit, so that ordering is load-bearing.
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
    # Over-fetch so that discarding duplicates doesn't shrink the context below
    # the configured budget.
    hits = search(embed_query(query), limit * 2)
    return dedupe(hits)[:limit]


def select_context(chunks: list[dict]) -> list[dict]:
    """Decide what (if anything) to answer from.

    Two separate judgements, deliberately not collapsed into one threshold:
      1. Is this question answerable at all? Judged on the best hit only.
      2. Which chunks are worth putting in the context window? A much lower bar.

    Using a single high threshold for both silently drops the supporting evidence of
    questions that are plainly answerable — a resume's education section scores far
    below its header block, so a bar tuned to reject nonsense also rejected the answer.
    """
    if not chunks or chunks[0]["score"] < settings.refusal_score_threshold:
        return []
    return [c for c in chunks if c["score"] >= settings.context_score_floor]


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
