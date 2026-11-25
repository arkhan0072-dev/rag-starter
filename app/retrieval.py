from dataclasses import dataclass, replace
from typing import Literal

import psycopg

from .models import embed_query, rerank_scores

Mode = Literal["vector", "hybrid", "hybrid_rerank"]
CANDIDATES = 20  # how many results each search contributes before fusion/reranking


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    source: str
    heading: str
    content: str
    score: float


def _hits(rows) -> list[Hit]:
    return [Hit(*row) for row in rows]


def vector_search(conn: psycopg.Connection, collection: str, query: str, k: int) -> list[Hit]:
    qvec = embed_query(query)
    rows = conn.execute(
        """
        SELECT c.id, d.source, c.heading, c.content, 1 - (c.embedding <=> %(q)s) AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.collection = %(col)s
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": qvec, "col": collection, "k": k},
    ).fetchall()
    return _hits(rows)


def keyword_search(conn: psycopg.Connection, collection: str, query: str, k: int) -> list[Hit]:
    # plainto_tsquery joins every word with AND, which almost never matches a full
    # natural-language question, so the terms are OR-ed and ts_rank_cd does the ranking.
    rows = conn.execute(
        """
        WITH q AS (
            SELECT replace(plainto_tsquery('english', %(text)s)::text, '&', '|')::tsquery AS tsq
        )
        SELECT c.id, d.source, c.heading, c.content, ts_rank_cd(c.tsv, q.tsq) AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id, q
        WHERE d.collection = %(col)s AND c.tsv @@ q.tsq
        ORDER BY score DESC
        LIMIT %(k)s
        """,
        {"text": query, "col": collection, "k": k},
    ).fetchall()
    return _hits(rows)


def reciprocal_rank_fusion(*rankings: list[Hit], k: int = 60) -> list[Hit]:
    """Merge ranked lists by position (1 / (k + rank)), so raw scores of different scales
    (cosine similarity vs. ts_rank) never have to be compared."""
    scores: dict[int, float] = {}
    hits: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            hits.setdefault(hit.chunk_id, hit)
    fused = [replace(hits[cid], score=s) for cid, s in scores.items()]
    return sorted(fused, key=lambda h: h.score, reverse=True)


def search(
    conn: psycopg.Connection,
    query: str,
    *,
    collection: str = "default",
    top_k: int = 5,
    mode: Mode = "hybrid_rerank",
) -> list[Hit]:
    if mode == "vector":
        return vector_search(conn, collection, query, top_k)

    fused = reciprocal_rank_fusion(
        vector_search(conn, collection, query, CANDIDATES),
        keyword_search(conn, collection, query, CANDIDATES),
    )[:CANDIDATES]
    if mode == "hybrid" or not fused:
        return fused[:top_k]

    scores = rerank_scores(query, [f"{h.heading}\n{h.content}" for h in fused])
    reranked = sorted(zip(fused, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [replace(hit, score=score) for hit, score in reranked[:top_k]]
