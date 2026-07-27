"""Deterministic eval harness for KnowledgeHub.

Run against a live server with the eval corpus ingested:

    python evals/eval_pipeline.py

Assertions are plain substring checks rather than an LLM judge: they are fast,
free, and reproducible. The tradeoff is brittleness to phrasing, so expected
keywords are pipe-separated alternatives rather than exact sentences.
"""

import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
CORPUS_DIR = Path(__file__).parent / "corpus"
RESULTS_PATH = Path(__file__).parent / "results.json"

# (name, turns, expected_source_filename_or_None, expected_keywords_pipe_separated)
# Only the FINAL turn is asserted on; earlier turns exist to build conversation history.
CASES = [
    (
        "single-turn: services",
        ["What services does Acme Cloud Platform offer?"],
        "acme-cloud-platform.md",
        "Acme Run|Acme Queue|Acme Vault",
    ),
    (
        "single-turn: specific figure",
        ["What uptime does Acme Run guarantee?"],
        "acme-cloud-platform.md",
        "99.95",
    ),
    (
        "single-turn: second document",
        ["Which warehouses does Zenith Explore support?"],
        "zenith-analytics-suite.md",
        "Snowflake|BigQuery|Postgres",
    ),
    (
        # The core memory test: "pricing" alone is ambiguous across both documents,
        # so this only passes if turn 1 is carried into the retrieval query.
        "follow-up: bare pronoun-style follow-up",
        ["What services does Acme Cloud Platform offer?", "What about pricing?"],
        "acme-cloud-platform.md",
        "0.000024|vCPU-second|per-service",
    ),
    (
        # Memory must NOT over-apply: the new subject should displace the old one.
        "follow-up: topic switch to the other document",
        [
            "What services does Acme Cloud Platform offer?",
            "And what about Zenith? How much is it?",
        ],
        "zenith-analytics-suite.md",
        "45|600|per seat",
    ),
    (
        "refusal: out of corpus",
        ["What is the capital of France?"],
        None,
        "couldn't find|not|don't|cannot|unable|outside",
    ),
]


def ingest_corpus(client: httpx.Client) -> None:
    existing = {d["filename"] for d in client.get(f"{BASE_URL}/api/documents").json()}
    pending = []

    for path in sorted(CORPUS_DIR.glob("*.md")):
        if path.name in existing:
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
    for _ in range(60):
        statuses = {
            d["id"]: d["status"] for d in client.get(f"{BASE_URL}/api/documents").json()
        }
        if all(statuses.get(doc_id) in {"ready", "failed"} for doc_id in pending):
            break
        time.sleep(2)


def check_retrieval_hit(citations: list[dict], expected_source: str | None) -> bool | None:
    if expected_source is None:
        return None  # not applicable — refusal cases cite nothing
    return any(expected_source in c["filename"] for c in citations)


def check_faithfulness(answer: str, expected_keywords: str) -> bool:
    lowered = answer.lower()
    return any(keyword.lower() in lowered for keyword in expected_keywords.split("|"))


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
        for name, turns, expected_source, expected_keywords in CASES:
            outcome = run_case(client, turns)
            retrieval_hit = check_retrieval_hit(outcome["citations"], expected_source)
            faithful = check_faithfulness(outcome["answer"], expected_keywords)

            results.append(
                {
                    "case": name,
                    "turns": turns,
                    "condensed_query": outcome["condensed_query"],
                    "retrieval_hit": retrieval_hit,
                    "faithful": faithful,
                    "passed": faithful and retrieval_hit is not False,
                    "latency_s": outcome["latency_s"],
                    "answer": outcome["answer"],
                    "cited_files": sorted({c["filename"] for c in outcome["citations"]}),
                }
            )

    print(f"\n{'case':<48} {'retrieval':<10} {'faithful':<9} {'latency':<8}")
    print("-" * 78)
    for r in results:
        retrieval = "n/a" if r["retrieval_hit"] is None else ("PASS" if r["retrieval_hit"] else "FAIL")
        print(
            f"{r['case']:<48} {retrieval:<10} "
            f"{'PASS' if r['faithful'] else 'FAIL':<9} {r['latency_s']:<8}"
        )

    applicable = [r for r in results if r["retrieval_hit"] is not None]
    precision = sum(r["retrieval_hit"] for r in applicable) / len(applicable)
    faithfulness = sum(r["faithful"] for r in results) / len(results)
    passed = sum(r["passed"] for r in results)

    print("-" * 78)
    print(f"Retrieval precision: {precision:.0%}  ({len(applicable)} applicable cases)")
    print(f"Faithfulness:        {faithfulness:.0%}  ({len(results)} cases)")
    print(f"Passed:              {passed}/{len(results)}")

    print("\nCondensed queries for multi-turn cases (the memory mechanism, made visible):")
    for r in results:
        if len(r["turns"]) > 1:
            print(f'  "{r["turns"][-1]}"  ->  "{r["condensed_query"]}"')

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "retrieval_precision": precision,
                "faithfulness": faithfulness,
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
