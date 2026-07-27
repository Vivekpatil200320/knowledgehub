import io
import re
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

from app.core.config import settings

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def sanitize_filename(filename: str) -> str:
    name = filename.replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^\w\s\-\.\(\)]", "", name)
    return name[:255]


async def save_upload(file: UploadFile) -> tuple[str, str, str]:
    """Validate the upload and persist it to disk. Returns (filename, stored_path, content_type)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: .pdf, .txt, .md",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max is {settings.max_upload_size_mb}MB.",
        )

    safe_filename = sanitize_filename(file.filename)
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{uuid.uuid4()}{ext}"
    stored_path.write_bytes(file_bytes)

    return safe_filename, str(stored_path), ext.lstrip(".")


def extract_text(stored_path: str, content_type: str) -> str:
    """Extract plain text from a stored file. Raises ValueError when nothing is extractable."""
    data = Path(stored_path).read_bytes()

    if content_type == "pdf":
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = data.decode("utf-8", errors="replace")

    if not text.strip():
        raise ValueError("No extractable text found — the file may be scanned or image-based.")

    return text
