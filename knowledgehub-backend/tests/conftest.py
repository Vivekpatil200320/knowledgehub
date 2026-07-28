import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.db import Base, get_db
from app.main import app


@pytest.fixture(autouse=True)
def _no_real_reranker_load(monkeypatch):
    """`app.main`'s lifespan preloads the cross-encoder at startup, and the `client`
    fixture below spins up that real lifespan for every test that uses it. Without
    this, every such test would download/load an actual model from HuggingFace —
    a network dependency and ~10s tax the rest of this suite doesn't have and CI
    shouldn't need. `test_rerank_service.py` and `test_retrieval.py`'s reranking tests
    cover the real behaviour at the unit level and re-enable this per-test as needed.
    """
    monkeypatch.setattr(settings, "rerank_enabled", False)


@pytest.fixture
def client(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
