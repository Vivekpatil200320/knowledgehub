import logging

from app.core.db import SessionLocal
from app.models.orm import Document
from app.services.embedding_service import embed_chunks
from app.services.file_parser import extract_text
from app.services.text_splitter import split_text
from app.services.vector_store import delete_document_vectors, store_chunks

logger = logging.getLogger("knowledgehub.ingestion")


def run_ingestion(document_id: str) -> None:
    """Background task: parse -> chunk -> embed -> store, updating document status as it goes.

    Opens its own session: the request-scoped session is already closed by the time
    BackgroundTasks runs.
    """
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        if document is None:
            logger.warning("Ingestion skipped, document %s no longer exists", document_id)
            return

        document.status = "processing"
        db.commit()

        text = extract_text(document.stored_path, document.content_type)
        chunks = split_text(text, document.id, document.filename)
        if not chunks:
            raise ValueError("Document produced no chunks after splitting.")

        embeddings = embed_chunks(chunks)
        store_chunks(chunks, embeddings)

        document.status = "ready"
        document.chunk_count = len(chunks)
        document.status_detail = None
        db.commit()
        logger.info("Ingested %s (%d chunks)", document.filename, len(chunks))

    except Exception as exc:
        logger.exception("Ingestion failed for %s", document_id)
        delete_document_vectors(document_id)
        document = db.get(Document, document_id)
        if document is not None:
            document.status = "failed"
            document.status_detail = str(exc)[:500]
            db.commit()
    finally:
        db.close()
