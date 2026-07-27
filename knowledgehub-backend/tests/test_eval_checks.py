"""Teeth for the eval harness.

The old eval suite passed 10/10 while three real bugs shipped, because a
presence-only faithfulness check ("answer contains a keyword") cannot see a
terse answer, a hallucinated fact, or cross-document contamination. These tests
feed the harness's check functions the actual buggy answers observed during
development and assert they are now rejected — so the harness's teeth don't
depend on reproducing an LLM's nondeterministic output.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_pipeline", Path(__file__).resolve().parents[1] / "evals" / "eval_pipeline.py"
)
eval_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_pipeline)

missing_terms = eval_pipeline.missing_terms
present_terms = eval_pipeline.present_terms


# --- the "answer is too small" bug ------------------------------------------
# Real observed answer to "describe him" before the completeness fix.
TERSE_ANSWER = "AI Software Engineer."


def test_terse_answer_fails_completeness():
    expected_all = ["AI Software Engineer", "backend", "deployment"]
    assert missing_terms(TERSE_ANSWER, expected_all)  # non-empty => case fails


def test_complete_answer_passes_completeness():
    full = (
        "He is an AI Software Engineer who owns systems end to end, from backend "
        "architecture through production deployment."
    )
    assert missing_terms(full, ["AI Software Engineer", "backend", "deployment"]) == []


# --- the invented-degree / mis-pairing bug ----------------------------------
# Real observed answer: it invented a third degree and mis-paired universities.
HALLUCINATED_EDUCATION = (
    "Vivek Patil has the following educational qualifications:\n"
    "- Master of Computer Applications from MIT ADT University (2024-2026)\n"
    "- Bachelor of Computer Applications from Parul University (2021-2024)\n"
    "- Master of Computer Applications from Parul University (7.6/10)"
)


def test_forbidden_catches_a_hallucinated_entity():
    # For the fictional Priya corpus, "Parul" is an entity that must never appear.
    assert present_terms(HALLUCINATED_EDUCATION, ["Parul"])  # non-empty => case fails


def test_correct_education_names_both_real_universities():
    correct = (
        "She holds a Master of Science from Ashford Institute of Technology and a "
        "Bachelor of Engineering from Westbrook University."
    )
    assert missing_terms(
        correct, ["Ashford Institute of Technology", "Westbrook University"]
    ) == []
    assert present_terms(correct, ["Parul", "Acme", "Zenith"]) == []


# --- cross-document contamination -------------------------------------------
def test_forbidden_catches_cross_document_bleed():
    contaminated = "Priya uses Python and also works on Acme Run and Acme Queue."
    assert present_terms(contaminated, ["Acme", "Zenith"])  # non-empty => case fails


def test_clean_answer_has_no_contamination():
    clean = "Priya uses Python, Go, and SQL."
    assert present_terms(clean, ["Acme", "Zenith", "Snowflake"]) == []


# --- matching is case-insensitive, so phrasing casing can't sneak past ------
def test_checks_are_case_insensitive():
    assert missing_terms("PYTHON and go", ["Python", "Go"]) == []
    assert present_terms("mentions acme run", ["Acme Run"])
