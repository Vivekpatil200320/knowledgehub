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
