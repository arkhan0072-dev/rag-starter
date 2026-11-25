"""Measure retrieval quality on a fixed question set and compare pipeline configurations.

A question counts as answered at rank r when the r-th retrieved chunk comes from the
expected document and contains the expected answer phrase.

    python -m eval.run_eval            # prints a markdown table, writes eval/results.md
"""

import json
from pathlib import Path

import psycopg

from app.config import settings
from app.db import configure, init_schema
from app.ingest import ingest
from app.retrieval import Hit, search

HERE = Path(__file__).parent
TOP_K = 5

CONFIGS = [
    ("Fixed-size chunks, vector search (baseline)", "fixed", "vector"),
    ("Structure-aware chunks, vector search", "structure", "vector"),
    ("Structure-aware chunks, hybrid search (RRF)", "structure", "hybrid"),
    ("Structure-aware chunks, hybrid + reranker", "structure", "hybrid_rerank"),
]


def first_relevant_rank(hits: list[Hit], q: dict) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.source == q["source"] and q["answer"].lower() in h.content.lower():
            return rank
    return None


def score(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@5": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def main() -> None:
    lines = (HERE / "questions.jsonl").read_text().splitlines()
    questions = [json.loads(line) for line in lines]
    docs = sorted((HERE / "corpus").glob("*.md"))

    init_schema()
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        configure(conn)
        for chunker in {c for _, c, _ in CONFIGS}:
            for doc in docs:
                ingest(
                    conn, doc.name, doc.read_text(), collection=f"eval-{chunker}", chunker=chunker
                )

        rows = []
        for label, chunker, mode in CONFIGS:
            ranks = []
            for q in questions:
                hits = search(
                    conn, q["question"], collection=f"eval-{chunker}", top_k=TOP_K, mode=mode
                )
                ranks.append(first_relevant_rank(hits, q))
            rows.append((label, score(ranks)))

    table = [
        f"Questions: {len(questions)}, documents: {len(docs)}, top_k: {TOP_K}",
        "",
        "| Pipeline | Hit@1 | Hit@5 | MRR |",
        "|---|---|---|---|",
    ]
    table += [
        f"| {label} | {s['hit@1']:.0%} | {s['hit@5']:.0%} | {s['mrr']:.2f} |" for label, s in rows
    ]
    report = "\n".join(table)
    print(report)
    (HERE / "results.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
