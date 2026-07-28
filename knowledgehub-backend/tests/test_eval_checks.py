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


# --- terms match on word boundaries, not raw substrings ---------------------
# Naive `in` made short terms nearly unfalsifiable, which is how a completeness
# check could pass on an answer that never contained the fact.


def test_substring_collisions_do_not_satisfy_a_term():
    # The résumé fixture literally contains "PostgreSQL", so a naive "SQL" check was
    # satisfied by the corpus regardless of what the model said.
    assert missing_terms("She works on PostgreSQL pipelines.", ["SQL"]) == ["SQL"]
    assert missing_terms("She is going to lead it.", ["Go"]) == ["Go"]
    assert missing_terms("The documents note that.", ["not"]) == ["not"]


def test_the_exact_false_pass_this_replaced():
    answer = "Priya uses Python. She works on PostgreSQL pipelines and is going to lead the migration."

    # Previously returned [] — a passing completeness check on an answer that never
    # lists Go or SQL as languages.
    assert sorted(missing_terms(answer, ["Python", "Go", "SQL"])) == ["Go", "SQL"]


def test_real_occurrences_still_match():
    assert missing_terms("Languages: Python, Go, SQL.", ["Python", "Go", "SQL"]) == []


def test_numeric_terms_are_bounded_but_tolerate_adjacent_punctuation():
    assert present_terms("guarantees 99.95% uptime", ["99.95"]) == ["99.95"]
    assert present_terms("costs $45 per seat", ["45"]) == ["45"]
    assert present_terms("a GPA of 3.8/4.0", ["3.8"]) == ["3.8"]
    # A longer number that merely contains the digits must not satisfy it.
    assert present_terms("in the year 1945", ["45"]) == []
    assert present_terms("priced at 199.95", ["99.95"]) == []


def test_multi_word_and_apostrophe_terms_match():
    assert present_terms("I couldn't find that here.", ["couldn't find"])
    assert present_terms("billed per seat monthly", ["per seat"])
