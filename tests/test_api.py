"""End-to-end test against a real PostgreSQL + pgvector database.

Skipped when the database is not reachable (CI provides one as a service).
"""

import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


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
