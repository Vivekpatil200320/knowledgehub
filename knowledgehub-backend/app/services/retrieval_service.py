from app.core.config import settings
from app.services.embedding_service import embed_query
from app.services.vector_store import search


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    return search(embed_query(query), top_k or settings.retrieval_top_k)


def above_threshold(chunks: list[dict]) -> list[dict]:
    return [c for c in chunks if c["score"] >= settings.retrieval_score_threshold]


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
