from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.orm import Document
from app.models.schemas import DocumentOut
from app.services.file_parser import save_upload
from app.services.ingestion_service import run_ingestion
from app.services.vector_store import delete_document_vectors, get_document_title

router = APIRouter(tags=["documents"])


def _to_out(document: Document) -> DocumentOut:
    # Only ready documents have chunks in Qdrant to look a title up from; asking
    # for any other status would just be a round trip that always misses.
    title = get_document_title(document.id) if document.status == "ready" else None
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        status=document.status,
        status_detail=document.status_detail,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        title=title,
    )


@router.post("/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    db: Session = Depends(get_db),
) -> Document:
    filename, stored_path, content_type = await save_upload(file)

    document = Document(
        filename=filename,
        stored_path=stored_path,
        content_type=content_type,
        status="pending",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    background_tasks.add_task(run_ingestion, document.id)
    # Freshly created: never ready yet, so no title lookup is possible or needed.
    return _to_out(document)


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentOut]:
    documents = db.scalars(select(Document).order_by(Document.created_at.desc()))
    return [_to_out(document) for document in documents]


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentOut:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _to_out(document)


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_document_vectors(document_id)
    db.delete(document)
    db.commit()
