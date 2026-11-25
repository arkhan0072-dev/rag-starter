"""Local embedding and reranking models (ONNX via fastembed), loaded on first use."""

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from .config import settings


@lru_cache
def _embedder() -> TextEmbedding:
    return TextEmbedding(settings.embedding_model)


@lru_cache
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(settings.rerank_model)


def embed_passages(texts: list[str]) -> list[np.ndarray]:
    return list(_embedder().passage_embed(texts))


def embed_query(text: str) -> np.ndarray:
    # BGE models expect a different prefix for queries; query_embed adds it.
    return next(iter(_embedder().query_embed(text)))


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    return [float(s) for s in _reranker().rerank(query, texts)]
