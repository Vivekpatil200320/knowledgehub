"""Deterministic eval harness for KnowledgeHub.

Run against a live server with the eval corpus ingested:

    python evals/eval_pipeline.py

Assertions are deterministic term matches rather than an LLM judge: they are fast,
free, and reproducible. The tradeoff is brittleness to phrasing, so each case
declares its checks explicitly:

  expected_source      the document a correct answer must cite (None = no citation expected)
  expected_all         every term must appear (completeness — guards terse answers)
  expected_any         at least one must appear (phrasing tolerance, e.g. refusals)
  forbidden            no term may appear (no hallucination, no cross-doc bleed)
  expect_no_citations  the answer must cite nothing (the structural refusal signal)

The all/any/forbidden checks exist because a presence-only faithfulness metric
passed while real bugs shipped: a one-line answer satisfies "contains a keyword",
and a hallucinated or cross-contaminated fact is never a keyword you were checking
*for*.

Terms match on word boundaries, not raw substrings. Naive `in` made short terms
nearly unfalsifiable: "SQL" was satisfied by "PostgreSQL", "Go" by "going", and
"not" by "note" — so a completeness check could pass on an answer that never
contained the fact, which is the exact failure this harness exists to catch.
"""

import json
import re
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
CORPUS_DIR = Path(__file__).parent / "corpus"
RESULTS_PATH = Path(__file__).parent / "results.json"

CASES = [
    # --- single-turn, single-fact ---------------------------------------------
    {
        "name": "single-turn: specific figure",
        "turns": ["What uptime does Acme Run guarantee?"],
        "expected_source": "acme-cloud-platform.md",
        "expected_all": ["99.95"],
    },
    {
        "name": "single-turn: second document",
        "turns": ["Which warehouses does Zenith Explore support?"],
        "expected_source": "zenith-analytics-suite.md",
        "expected_all": ["Snowflake", "BigQuery", "Postgres"],
    },
    # --- completeness: a terse answer must fail (the 'AI Software Engineer.' bug)
    {
        "name": "completeness: all three services, not just one",
        "turns": ["What services does Acme Cloud Platform offer?"],
        "expected_source": "acme-cloud-platform.md",
        "expected_all": ["Acme Run", "Acme Queue", "Acme Vault"],
    },
    # --- PDF extraction path (the corpus was previously markdown-only) ---------
    {
        "name": "pdf: extraction and a specific fact",
        "turns": ["Where does Priya Nair currently work?"],
        "expected_source": "candidate-profile.pdf",
        "expected_all": ["Meridian Freight"],
        "forbidden": ["Acme", "Zenith"],
    },
    # --- structured PDF: multi-fact pairing across a chunk boundary ------------
    # The regression that motivated narrative ordering. A correct answer names both
    # degrees AND both universities AND both GPAs. If ordering breaks, the model
    # mis-pairs or invents an entry and drops one of these exact tokens.
    {
        "name": "pdf: education pairing (two degrees, two universities)",
        "turns": ["What are Priya Nair's degrees and where did she study?"],
        "expected_source": "candidate-profile.pdf",
        "expected_all": [
            "Ashford Institute of Technology",
            "Westbrook University",
            "Master",
            "Bachelor",
            "3.8",
            "3.6",
        ],
        "forbidden": ["Acme", "Zenith", "Parul"],
    },
    # --- findable by the name the user actually sees --------------------------
    # The corpus hid this for a long time: "acme-cloud-platform.md" and
    # "zenith-analytics-suite.md" name themselves in their own headings, so querying
    # them by filename scored ~0.58 by accident. "candidate-profile.pdf" is a résumé
    # that never says "candidate profile", and scored 0.09 — refused. Only the
    # filename/content mismatch exercises this, so only this case guards it.
    {
        "name": "findability: document referenced by its filename, not its content",
        "turns": ["describe candidate profile"],
        "expected_source": "candidate-profile.pdf",
        "expected_any": ["Priya", "engineer"],
        "forbidden": ["couldn't find", "could not find"],
    },
    # --- no cross-document contamination (negative assertion) -----------------
    {
        "name": "no contamination: skills answer stays within the resume",
        "turns": ["What programming languages does Priya Nair use?"],
        "expected_source": "candidate-profile.pdf",
        "expected_all": ["Python", "Go", "SQL"],
        "forbidden": ["Acme", "Zenith", "Snowflake", "BigQuery"],
    },
    # --- follow-up: memory must be applied (bare pronoun-style) ----------------
    {
        "name": "follow-up: bare pronoun-style follow-up",
        "turns": ["What services does Acme Cloud Platform offer?", "What about pricing?"],
        "expected_source": "acme-cloud-platform.md",
        # A complete pricing answer prices all three services, not just one.
        "expected_all": ["Acme Run", "Acme Queue", "Acme Vault"],
        "expected_any": ["0.000024", "vCPU-second"],
    },
    # --- follow-up: memory must be displaced (topic switch) -------------------
    {
        "name": "follow-up: topic switch to the other document",
        "turns": [
            "What services does Acme Cloud Platform offer?",
            "And what about Zenith? How much is it?",
        ],
        "expected_source": "zenith-analytics-suite.md",
        "expected_any": ["45", "600", "per seat"],
        "forbidden": ["Acme Run", "Acme Queue"],
    },
    # --- follow-up spanning documents: memory + PDF ---------------------------
    {
        "name": "follow-up: switch to the resume person, then ask education",
        "turns": [
            "Tell me about Priya Nair.",
            "Where did she study?",
        ],
        "expected_source": "candidate-profile.pdf",
        "expected_all": ["Ashford Institute of Technology", "Westbrook University"],
        "forbidden": ["Acme", "Zenith"],
    },
    # --- refusal --------------------------------------------------------------
    {
        "name": "refusal: out of corpus",
        "turns": ["What is the capital of France?"],
        "expected_source": None,
        # Citing nothing is the structural signal: the refusal short-circuit returns
        # before the generation call, so a refusal cannot carry sources. Asserting the
        # wording alone would let a regression that answers *with* citations pass, as
        # long as the text happened to contain a hedging word.
        "expect_no_citations": True,
        "expected_any": [
            "couldn't find",
            "could not find",
            "don't have",
            "do not have",
            "cannot",
            "can't",
            "unable",
            "no information",
        ],
        "forbidden": ["Paris"],
    },
]


def ingest_corpus(client: httpx.Client) -> None:
    """Bring the server's corpus up to date with `evals/corpus/`.

    A document is only treated as present if it actually reached `ready`. Skipping on
    filename alone meant a document left `failed` by a transient ingestion error (an
    embedding rate limit, a Qdrant blip) was never retried, so the suite silently ran
    against an incomplete corpus and reported retrieval failures that pointed at the
    retrieval logic instead of at the missing file.
    """
    documents = client.get(f"{BASE_URL}/api/documents").json()
    usable = {d["filename"] for d in documents if d["status"] in {"ready", "pending", "processing"}}
    stale = [d for d in documents if d["status"] == "failed"]

    # Remove the failed rows first: re-uploading without deleting would leave a
    # duplicate filename in the document list and orphan the failed row forever.
    for document in stale:
        print(f"Re-ingesting failed document: {document['filename']}")
        client.delete(f"{BASE_URL}/api/documents/{document['id']}")

    pending = []
    for path in sorted(list(CORPUS_DIR.glob("*.md")) + list(CORPUS_DIR.glob("*.pdf"))):
        if path.name in usable:
            continue
        with path.open("rb") as handle:
            response = client.post(
                f"{BASE_URL}/api/documents", files={"file": (path.name, handle)}
            )
        response.raise_for_status()
        pending.append(response.json()["id"])

    if not pending:
        return

    print(f"Ingesting {len(pending)} document(s)…")
    statuses: dict[str, str] = {}
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        statuses = {
            d["id"]: d["status"] for d in client.get(f"{BASE_URL}/api/documents").json()
        }
        if all(statuses.get(doc_id) in {"ready", "failed"} for doc_id in pending):
            break
        time.sleep(2)
    else:
        print("WARNING: ingestion did not settle within 120s; results may be unreliable.")

    failed = [doc_id for doc_id in pending if statuses.get(doc_id) == "failed"]
    if failed:
        print(f"WARNING: {len(failed)} document(s) failed to ingest; cases needing them will fail.")


def check_retrieval_hit(citations: list[dict], expected_source) -> bool | None:
    """Whether the expected document was cited. None when the case names no source.

    A case with no `expected_source` is not asserting an absence here — use
    `expect_no_citations` for that. This only reports "not applicable".
    """
    if expected_source is None:
        return None
    return any(expected_source in c["filename"] for c in citations)


def term_matches(answer: str, term: str) -> bool:
    """Match a term on alphanumeric boundaries.

    `\\b` is wrong here because several terms are numeric or punctuated ("99.95",
    "0.000024", "3.8"): `\\b` would anchor against the trailing digit and let "199.95"
    satisfy "99.95". Asserting the neighbours are not alphanumeric instead means
    "$45." and "45%" match "45" while "1945" does not, and "PostgreSQL" no longer
    satisfies "SQL".
    """
    pattern = rf"(?<![0-9A-Za-z]){re.escape(term)}(?![0-9A-Za-z])"
    return re.search(pattern, answer, re.IGNORECASE) is not None


def missing_terms(answer: str, terms: list[str]) -> list[str]:
    return [t for t in terms if not term_matches(answer, t)]


def present_terms(answer: str, terms: list[str]) -> list[str]:
    return [t for t in terms if term_matches(answer, t)]


def run_case(client: httpx.Client, turns: list[str]) -> dict:
    conversation_id = client.post(f"{BASE_URL}/api/conversations", json={}).json()["id"]

    reply = None
    started = time.perf_counter()
    for turn in turns:
        response = client.post(
            f"{BASE_URL}/api/conversations/{conversation_id}/messages",
            json={"content": turn},
        )
        response.raise_for_status()
        reply = response.json()

    return {
        "answer": reply["content"],
        "citations": reply["citations"] or [],
        "condensed_query": reply["condensed_query"],
        "latency_s": round(time.perf_counter() - started, 2),
    }


def main() -> int:
    with httpx.Client(timeout=180.0) as client:
        try:
            client.get(f"{BASE_URL}/health").raise_for_status()
        except Exception:
            print(f"Server not reachable at {BASE_URL}. Start the backend first.")
            return 1

        ingest_corpus(client)

        results = []
        for case in CASES:
            outcome = run_case(client, case["turns"])
            answer = outcome["answer"]

            cited_files = sorted({c["filename"] for c in outcome["citations"]})

            retrieval_hit = check_retrieval_hit(outcome["citations"], case["expected_source"])
            missing = missing_terms(answer, case.get("expected_all", []))
            any_terms = case.get("expected_any", [])
            any_ok = bool(present_terms(answer, any_terms)) if any_terms else True
            leaked = present_terms(answer, case.get("forbidden", []))
            citations_ok = not (
                case.get("expect_no_citations", False) and outcome["citations"]
            )

            passed = (
                retrieval_hit is not False
                and not missing
                and any_ok
                and not leaked
                and citations_ok
            )

            results.append(
                {
                    "case": case["name"],
                    "turns": case["turns"],
                    "condensed_query": outcome["condensed_query"],
                    "retrieval_hit": retrieval_hit,
                    "complete": not missing,
                    "missing_terms": missing,
                    "any_ok": any_ok,
                    "contamination_free": not leaked,
                    "leaked_terms": leaked,
                    "citations_ok": citations_ok,
                    "passed": passed,
                    "latency_s": outcome["latency_s"],
                    "answer": answer,
                    "cited_files": cited_files,
                }
            )

    print(
        f"\n{'case':<52} {'retr':<6} {'compl':<6} {'clean':<6} {'pass':<5}"
    )
    print("-" * 82)
    for r in results:
        retr = "n/a" if r["retrieval_hit"] is None else ("ok" if r["retrieval_hit"] else "FAIL")
        print(
            f"{r['case']:<52} {retr:<6} "
            f"{'ok' if r['complete'] and r['any_ok'] else 'FAIL':<6} "
            f"{'ok' if r['contamination_free'] else 'FAIL':<6} "
            f"{'PASS' if r['passed'] else 'FAIL':<5}"
        )
        if not r["passed"]:
            if r["missing_terms"]:
                print(f"      missing: {r['missing_terms']}")
            if r["leaked_terms"]:
                print(f"      leaked:  {r['leaked_terms']}")
            if not r["citations_ok"]:
                print(f"      expected no citations, got: {r['cited_files']}")

    applicable = [r for r in results if r["retrieval_hit"] is not None]
    # Guard the degenerate all-refusal suite rather than dividing by zero.
    precision = (
        sum(r["retrieval_hit"] for r in applicable) / len(applicable) if applicable else 1.0
    )
    completeness = sum(r["complete"] and r["any_ok"] for r in results) / len(results)
    clean = sum(r["contamination_free"] for r in results) / len(results)
    passed = sum(r["passed"] for r in results)

    print("-" * 82)
    print(f"Retrieval precision: {precision:.0%}  ({len(applicable)} applicable)")
    print(f"Completeness:        {completeness:.0%}  ({len(results)} cases)")
    print(f"Contamination-free:  {clean:.0%}  ({len(results)} cases)")
    print(f"Passed:              {passed}/{len(results)}")

    print("\nCondensed queries for multi-turn cases (the memory mechanism, made visible):")
    for r in results:
        if len(r["turns"]) > 1:
            print(f'  "{r["turns"][-1]}"  ->  "{r["condensed_query"]}"')

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "retrieval_precision": precision,
                "completeness": completeness,
                "contamination_free": clean,
                "passed": passed,
                "total": len(results),
                "cases": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {RESULTS_PATH}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
