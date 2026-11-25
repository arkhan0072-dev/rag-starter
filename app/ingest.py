import io
from typing import Literal

import psycopg
from pypdf import PdfReader

from .chunking import by_structure, fixed_size
from .config import settings
from .models import embed_passages

Chunker = Literal["structure", "fixed"]


def read_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        pages = PdfReader(io.BytesIO(data)).pages
        return "\n\n".join(page.extract_text() or "" for page in pages)
    return data.decode("utf-8", errors="replace")


def ingest(
    conn: psycopg.Connection,
    source: str,
    text: str,
    *,
    collection: str = "default",
    chunker: Chunker = "structure",
) -> int:
    """Chunk, embed and store a document. Re-ingesting the same source replaces it."""
    if chunker == "structure":
        chunks = by_structure(text, settings.chunk_max_chars)
    else:
        chunks = fixed_size(text)
    vectors = embed_passages([f"{c.heading}\n{c.content}".strip() for c in chunks])

    with conn.transaction():
        conn.execute(
            "DELETE FROM documents WHERE collection = %s AND source = %s", (collection, source)
        )
        doc_id = conn.execute(
            "INSERT INTO documents (collection, source) VALUES (%s, %s) RETURNING id",
            (collection, source),
        ).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, ord, heading, content, embedding)"
                " VALUES (%s, %s, %s, %s, %s)",
                [
                    (doc_id, i, c.heading, c.content, v)
                    for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )
    return len(chunks)
