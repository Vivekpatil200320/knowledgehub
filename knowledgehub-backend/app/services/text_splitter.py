from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, document_id: str, filename: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    return [
        {
            "id": f"{document_id}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(chunks)
    ]
