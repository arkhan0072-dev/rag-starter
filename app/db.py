import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import settings

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          serial PRIMARY KEY,
    collection  text NOT NULL DEFAULT 'default',
    source      text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (collection, source)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           serial PRIMARY KEY,
    document_id  int NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord          int NOT NULL,
    heading      text NOT NULL DEFAULT '',
    content      text NOT NULL,
    embedding    vector({settings.embedding_dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (
                     to_tsvector('english', heading || ' ' || content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def init_schema() -> None:
    # The vector type must exist before connections can register it.
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(SCHEMA)


def make_pool() -> ConnectionPool:
    return ConnectionPool(
        settings.database_url,
        kwargs={"autocommit": True},
        configure=register_vector,
        open=True,
    )
