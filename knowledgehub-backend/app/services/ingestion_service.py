import logging

from app.core.db import SessionLocal
from app.models.orm import Document
from app.services.embedding_service import embed_chunks
from app.services.file_parser import extract_text
from app.services.text_splitter import split_text
from app.services.vector_store import delete_document_vectors, store_chunks

logger = logging.getLogger("knowledgehub.ingestion")

TITLE_MAX_CHARS = 80


def derive_document_title(text: str) -> str | None:
    """Take the document's own opening line as its title.

    Filenames are a poor substitute for this: a résumé named "candidate-profile.pdf"
    or "resume-ai.pdf" produces a starter question like "What is candidate profile
    about?" that scores nowhere near the refusal threshold, because "candidate
    profile" never appears anywhere in the document — retrieval correctly refuses a
    query about content that isn't there. In every document seen so far, the true
    subject — a person's name, or a "# Title" heading — is the first non-empty line;
    contact details and body text come after it, not before.
    """
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if not stripped:
            continue
        if len(stripped) <= TITLE_MAX_CHARS:
            return stripped
        clipped = stripped[:TITLE_MAX_CHARS]
        if " " in clipped:
            clipped = clipped[: clipped.rindex(" ")]
        return clipped.rstrip(" ,.;:—-") + "…"
    return None


def with_document_context(chunks: list[dict], filename: str, title: str | None) -> None:
    """Prefix each chunk's *embedded* form with what document it came from.

    Users refer to a document by the name they see in the sidebar, but only its body
    text was ever embedded — so "describe candidate profile" scored 0.09 against a
    résumé whose filename says "candidate profile" and whose text never does, and the
    system refused a question about a document it was holding. Two of the three corpus
    files hid this: "acme-cloud-platform.md" and "zenith-analytics-suite.md" name
    themselves in their own headings, so filename queries scored ~0.58 by accident.

    Sets `embed_text` rather than mutating `text`: the header must reach the embedding
    without leaking into citation snippets or the context the model reads back.
    """
    header = f"Document: {filename}"
    if title and title.lower() != filename.lower():
        header += f"\nTitle: {title}"

    for chunk in chunks:
        chunk["embed_text"] = f"{header}\n\n{chunk['text']}"


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

        title = derive_document_title(text)
        with_document_context(chunks, document.filename, title)

        embeddings = embed_chunks(chunks)
        store_chunks(chunks, embeddings, document_title=title)

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
