import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import models
from app.config import settings
from app.main import app


def test_matching_dimension_passes(monkeypatch):
    monkeypatch.setattr(models, "embed_query", lambda text: np.zeros(384))
    models.check_embedding_dim(384)


def test_mismatched_dimension_fails_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(models, "embed_query", lambda text: np.zeros(768))
    with pytest.raises(RuntimeError, match="EMBEDDING_DIM=384.*768"):
        models.check_embedding_dim(384)


def test_app_refuses_to_start_with_wrong_dimension(monkeypatch):
    # Fails before touching the database, so no table is created with the wrong size.
    monkeypatch.setattr(settings, "embedding_dim", 999)
    with pytest.raises(RuntimeError, match="EMBEDDING_DIM=999"), TestClient(app):
        pass
