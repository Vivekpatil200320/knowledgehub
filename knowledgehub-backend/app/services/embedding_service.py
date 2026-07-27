from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from app.core.config import settings

_embedder: NVIDIAEmbeddings | None = None
EMBEDDING_DIM = 2048


def get_embedder() -> NVIDIAEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = NVIDIAEmbeddings(
            model=settings.nvidia_embedding_model,
            api_key=settings.nvidia_api_key,
        )
    return _embedder


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Single batched call — never loop one chunk per API request."""
    return get_embedder().embed_documents([c["text"] for c in chunks])


def embed_query(query: str) -> list[float]:
    return get_embedder().embed_query(query)
