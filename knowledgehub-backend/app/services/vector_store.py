import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import settings

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _client


def ensure_collection(vector_size: int) -> None:
    """Create the collection if absent.

    Concurrent ingestion tasks can both observe an empty store and race to create it;
    the loser gets a 409, which is a success condition here.
    """
    client = get_client()
    if client.collection_exists(settings.qdrant_collection):
        return

    try:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    except Exception:
        if not client.collection_exists(settings.qdrant_collection):
            raise


def store_chunks(
    chunks: list[dict], embeddings: list[list[float]], document_title: str | None = None
) -> None:
    """Store chunks, optionally tagging every point with the document's derived title.

    `document_title` is written to every point rather than only chunk 0: it must
    survive later regardless of which point a filter-based lookup happens to return,
    and it must not vanish if chunk 0 is ever deleted independently of the rest.
    """
    ensure_collection(len(embeddings[0]))
    extra = {"document_title": document_title} if document_title else {}
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])),
            vector=embedding,
            payload={**chunk["metadata"], "text": chunk["text"], **extra},
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    get_client().upsert(collection_name=settings.qdrant_collection, points=points)


def get_document_title(document_id: str) -> str | None:
    """Look up the title derived at ingestion, without re-parsing the source file.

    Uses the vector store rather than a SQL column so this needs no migration: the
    project ships without one by design, and this value is derivable from data the
    store already holds — the same reasoning applied to the conversation list's
    last_message_at/message_count fields.
    """
    client = get_client()
    if not client.collection_exists(settings.qdrant_collection):
        return None

    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
        limit=1,
        with_payload=["document_title"],
    )
    if not points:
        return None
    return points[0].payload.get("document_title") if points[0].payload else None


def search(query_embedding: list[float], top_k: int) -> list[dict]:
    client = get_client()
    if not client.collection_exists(settings.qdrant_collection):
        return []

    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [normalise_hit(hit.payload or {}, hit.score) for hit in hits]


def normalise_hit(payload: dict, score: float) -> dict:
    """Coerce a stored point into the shape the rest of the pipeline assumes.

    The vector store outlives the code that wrote to it: a collection can hold points
    from an earlier payload schema, and orphans survive a database reset. Downstream
    code sorts on `chunk_index` and validates citations against `int`/`str`, so a
    missing key would surface as a TypeError or a ValidationError — a 500 on every
    query that happens to retrieve that point — rather than a slightly worse answer.
    Normalising here keeps that failure contained to the one bad point.
    """
    raw_index = payload.get("chunk_index")
    try:
        chunk_index = int(raw_index)
    except (TypeError, ValueError):
        chunk_index = 0

    return {
        "text": payload.get("text") or "",
        "score": score,
        "metadata": {
            "document_id": str(payload.get("document_id") or ""),
            "filename": str(payload.get("filename") or "unknown"),
            "chunk_index": chunk_index,
        },
    }


def delete_document_vectors(document_id: str) -> None:
    client = get_client()
    if not client.collection_exists(settings.qdrant_collection):
        return
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )
