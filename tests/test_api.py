"""End-to-end test against a real PostgreSQL + pgvector database.

Skipped when the database is not reachable (CI provides one as a service).
"""

import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.db import configure
from app.ingest import ingest
from app.llm import NO_RESULTS
from app.main import app
from tests.test_ingest import make_pdf


def database_available() -> bool:
    try:
        psycopg.connect(settings.database_url, connect_timeout=2).close()
        return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not database_available(), reason="PostgreSQL not available")

DOC = b"""# Travel policy

## Flights

Economy class is required for flights shorter than six hours.

## Hotels

The nightly hotel limit is 180 USD in most cities and 250 USD in London and New York.
"""


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_upload_and_query_returns_cited_answer(client):
    collection = f"test-{uuid.uuid4().hex[:8]}"
    up = client.post(
        "/documents",
        files={"file": ("travel.md", DOC, "text/markdown")},
        data={"collection": collection},
    )
    assert up.status_code == 201
    assert up.json()["chunks"] == 2

    res = client.post(
        "/query",
        json={"question": "What is the hotel budget in London?", "collection": collection},
    )
    assert res.status_code == 200
    body = res.json()
    top = body["citations"][0]
    assert top["source"] == "travel.md"
    assert top["heading"] == "Travel policy > Hotels"
    assert "250 USD" in body["answer"]


def test_reingesting_a_source_replaces_it(client):
    collection = f"test-{uuid.uuid4().hex[:8]}"
    for _ in range(2):
        client.post(
            "/documents",
            files={"file": ("travel.md", DOC, "text/markdown")},
            data={"collection": collection},
        )
    with psycopg.connect(settings.database_url) as conn:
        count = conn.execute(
            "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.collection = %s",
            (collection,),
        ).fetchone()[0]
    assert count == 2


def test_rejects_unsupported_file_type(client):
    res = client.post("/documents", files={"file": ("image.png", b"\x89PNG", "image/png")})
    assert res.status_code == 415


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_rejects_files_over_the_size_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    res = client.post("/documents", files={"file": ("big.txt", b"x" * 11, "text/plain")})
    assert res.status_code == 413


def test_rejects_files_without_text(client):
    res = client.post("/documents", files={"file": ("empty.txt", b"  \n\n ", "text/plain")})
    assert res.status_code == 422


def test_rejects_too_short_question(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 422


def test_query_on_empty_collection_says_nothing_found(client):
    res = client.post("/query", json={"question": "anything at all?", "collection": "nope-empty"})
    assert res.json() == {"answer": NO_RESULTS, "citations": []}


def test_pdf_upload_is_searchable(client):
    collection = f"test-{uuid.uuid4().hex[:8]}"
    pdf = make_pdf("The night shift allowance is 40 USD")
    up = client.post(
        "/documents",
        files={"file": ("shifts.pdf", pdf, "application/pdf")},
        data={"collection": collection},
    )
    assert up.status_code == 201
    res = client.post(
        "/query",
        json={"question": "What is the night shift allowance?", "collection": collection},
    )
    assert "40 USD" in res.json()["answer"]


@pytest.mark.parametrize("mode", ["vector", "hybrid", "hybrid_rerank"])
def test_every_search_mode_finds_the_right_section(client, mode):
    collection = f"test-{uuid.uuid4().hex[:8]}"
    client.post(
        "/documents",
        files={"file": ("travel.md", DOC, "text/markdown")},
        data={"collection": collection},
    )
    res = client.post(
        "/query",
        json={"question": "flight class rules", "collection": collection, "mode": mode},
    )
    assert res.json()["citations"][0]["heading"] == "Travel policy > Flights"


def test_fixed_size_chunker_stores_chunks_without_headings():
    collection = f"test-{uuid.uuid4().hex[:8]}"
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        configure(conn)
        count = ingest(conn, "travel.md", DOC.decode(), collection=collection, chunker="fixed")
        headings = conn.execute(
            "SELECT DISTINCT c.heading FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.collection = %s",
            (collection,),
        ).fetchall()
    assert count >= 1
    assert headings == [("",)]
