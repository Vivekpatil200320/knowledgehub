from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, document_id: str, filename: str) -> list[dict]:
    # 512/64 (carried over from a prior prose-PDF project) split structured documents
    # mid-record: a resume's "University, City, dates" line landed in one chunk and the
    # degree it belongs to in the next, so retrieving either half lost the pairing.
    # 1024/128 keeps a typical section intact while staying well inside the context
    # budget at top_k=6.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
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
