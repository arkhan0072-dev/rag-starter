from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import settings
from .db import init_schema, make_pool
from .ingest import ingest, read_text
from .llm import answer
from .models import check_embedding_dim
from .retrieval import Mode, search

ALLOWED = (".pdf", ".md", ".markdown", ".txt")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_embedding_dim(settings.embedding_dim)  # before the schema, so no table gets a wrong size
    init_schema()
    app.state.pool = make_pool()
    yield
    app.state.pool.close()


app = FastAPI(title="RAG Starter", version="0.1.0", lifespan=lifespan)


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Mode = "hybrid_rerank"
    collection: str = "default"


class Citation(BaseModel):
    source: str
    heading: str
    snippet: str
    score: float


class QueryOut(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/health")
def health() -> dict:
    with app.state.pool.connection() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/documents", status_code=201)
def upload_document(file: UploadFile, collection: str = Form("default")) -> dict:
    name = file.filename or ""
    if not name.lower().endswith(ALLOWED):
        raise HTTPException(415, f"Supported file types: {', '.join(ALLOWED)}")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 10 MB")
    text = read_text(name, data)
    if not text.strip():
        raise HTTPException(422, "No text could be extracted from the file")
    with app.state.pool.connection() as conn:
        count = ingest(conn, name, text, collection=collection)
    return {"source": name, "collection": collection, "chunks": count}


@app.post("/query")
def query(body: QueryIn) -> QueryOut:
    with app.state.pool.connection() as conn:
        hits = search(
            conn, body.question, collection=body.collection, top_k=body.top_k, mode=body.mode
        )
    return QueryOut(
        answer=answer(body.question, hits),
        citations=[
            Citation(source=h.source, heading=h.heading, snippet=h.content[:300], score=h.score)
            for h in hits
        ],
    )
