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


def store_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    ensure_collection(len(embeddings[0]))
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])),
            vector=embedding,
            payload={**chunk["metadata"], "text": chunk["text"]},
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    get_client().upsert(collection_name=settings.qdrant_collection, points=points)


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

    return [
        {
            "text": hit.payload.get("text", ""),
            "score": hit.score,
            "metadata": {
                "document_id": hit.payload.get("document_id"),
                "filename": hit.payload.get("filename"),
                "chunk_index": hit.payload.get("chunk_index"),
            },
        }
        for hit in hits
    ]


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
