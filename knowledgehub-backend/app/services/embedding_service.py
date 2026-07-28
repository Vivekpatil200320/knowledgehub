from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from app.core.config import settings

_embedder: NVIDIAEmbeddings | None = None


def get_embedder() -> NVIDIAEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = NVIDIAEmbeddings(
            model=settings.nvidia_embedding_model,
            api_key=settings.nvidia_api_key,
        )
    return _embedder


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """Single batched call — never loop one chunk per API request.

    Embeds `embed_text` when the ingestion pipeline supplied it, falling back to the
    raw chunk. The two differ because what gets embedded is not what gets shown: the
    embedded form carries a document header so the file can be found by name, while
    the payload keeps the original text so citations and snippets stay clean.
    """
    return get_embedder().embed_documents([c.get("embed_text") or c["text"] for c in chunks])


def embed_query(query: str) -> list[float]:
    return get_embedder().embed_query(query)
