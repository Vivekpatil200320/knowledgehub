"""Title derivation: the fix for filename-derived starter prompts that refuse.

A résumé named "candidate-profile.pdf" or "resume-ai.pdf" never contains the phrase
"candidate profile" or "resume ai" anywhere in its own text, so a starter question
built from the filename can score below the refusal threshold even though the
document plainly answers a differently-worded question. The fix takes the document's
own opening line — a person's name, or a "# Title" heading — as its title instead.
"""

from app.services.ingestion_service import TITLE_MAX_CHARS, derive_document_title


def test_uses_the_first_line_for_a_resume():
    text = (
        "Priya Nair\n"
        "priya.nair@example.com | +1-555-0142 | Austin, TX\n"
        "SUMMARY\n"
        "Backend engineer focused on data platforms and search."
    )
    assert derive_document_title(text) == "Priya Nair"


def test_strips_markdown_heading_markers():
    text = "# Acme Cloud Platform — Product Overview\n\n## What Acme Cloud Platform is\n\n..."
    assert derive_document_title(text) == "Acme Cloud Platform — Product Overview"


def test_skips_leading_blank_lines():
    assert derive_document_title("\n\n   \nZenith Analytics Suite\nMore text.") == (
        "Zenith Analytics Suite"
    )


def test_truncates_a_long_first_line_on_a_word_boundary():
    long_line = "A very long heading that goes on for quite a while describing the document in detail"
    title = derive_document_title(long_line)

    assert title.endswith("…")
    assert len(title) <= TITLE_MAX_CHARS + 1
    assert not title.rstrip("…").endswith(" ")


def test_returns_none_for_blank_text():
    assert derive_document_title("   \n\n   \n") is None


def test_returns_none_for_empty_string():
    assert derive_document_title("") is None


# --- document context headers ------------------------------------------------
# Users refer to a document by the name they see in the sidebar, but only its body
# text was embedded — so a résumé named "candidate-profile.pdf" could not be found
# by that name, and the system refused a question about a document it was holding.

from app.services.embedding_service import embed_chunks  # noqa: E402
from app.services.ingestion_service import with_document_context  # noqa: E402


def make_chunks():
    return [
        {"id": "d_chunk_0", "text": "Backend engineer.", "metadata": {"chunk_index": 0}},
        {"id": "d_chunk_1", "text": "Master of Science.", "metadata": {"chunk_index": 1}},
    ]


def test_header_names_the_document_and_its_title():
    chunks = make_chunks()

    with_document_context(chunks, "candidate-profile.pdf", "Priya Nair")

    assert chunks[0]["embed_text"] == (
        "Document: candidate-profile.pdf\nTitle: Priya Nair\n\nBackend engineer."
    )
    assert chunks[1]["embed_text"].endswith("Master of Science.")


def test_header_is_applied_to_every_chunk_not_just_the_first():
    """Retrieval can surface any chunk; findability must not depend on which one."""
    chunks = make_chunks()

    with_document_context(chunks, "candidate-profile.pdf", "Priya Nair")

    assert all("Document: candidate-profile.pdf" in c["embed_text"] for c in chunks)


def test_original_text_is_left_untouched():
    """The header must reach the embedding without leaking into citations."""
    chunks = make_chunks()

    with_document_context(chunks, "candidate-profile.pdf", "Priya Nair")

    assert chunks[0]["text"] == "Backend engineer."
    assert "Document:" not in chunks[0]["text"]


def test_title_is_omitted_when_it_only_repeats_the_filename():
    chunks = make_chunks()

    with_document_context(chunks, "Priya Nair", "priya nair")

    assert chunks[0]["embed_text"].startswith("Document: Priya Nair\n\n")


def test_missing_title_still_yields_a_filename_header():
    chunks = make_chunks()

    with_document_context(chunks, "scan001.pdf", None)

    assert chunks[0]["embed_text"].startswith("Document: scan001.pdf\n\n")


def test_embed_chunks_prefers_the_augmented_text(monkeypatch):
    seen = {}

    class FakeEmbedder:
        def embed_documents(self, texts):
            seen["texts"] = texts
            return [[0.0]] * len(texts)

    monkeypatch.setattr(
        "app.services.embedding_service.get_embedder", lambda: FakeEmbedder()
    )
    chunks = make_chunks()
    with_document_context(chunks, "candidate-profile.pdf", "Priya Nair")

    embed_chunks(chunks)

    assert all(t.startswith("Document: candidate-profile.pdf") for t in seen["texts"])


def test_embed_chunks_falls_back_when_no_header_was_applied(monkeypatch):
    """Chunks ingested before this change carry no embed_text."""
    seen = {}

    class FakeEmbedder:
        def embed_documents(self, texts):
            seen["texts"] = texts
            return [[0.0]] * len(texts)

    monkeypatch.setattr(
        "app.services.embedding_service.get_embedder", lambda: FakeEmbedder()
    )

    embed_chunks(make_chunks())

    assert seen["texts"] == ["Backend engineer.", "Master of Science."]
