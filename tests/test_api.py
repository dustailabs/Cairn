from fastapi.testclient import TestClient

from cairn.api.main import _store, app

client = TestClient(app)


def setup_function():
    # Reset shared in-memory store between tests since the app module is
    # imported once for the whole test session.
    _store._documents.clear()
    _store._chunks.clear()


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_returns_chunk_counts():
    resp = client.post("/ingest", json={"text": "Kafka replicates partitions. " * 30, "source": "doc-a"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunk_count"] >= 1
    assert body["total_chunks"] == body["chunk_count"]


def test_query_without_documents_returns_400():
    resp = client.post("/query", json={"query": "anything"})
    assert resp.status_code == 400


def test_query_without_api_key_returns_503(monkeypatch):
    client.post("/ingest", json={"text": "Some content here.", "source": "doc-a"})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post("/query", json={"query": "anything"})
    assert resp.status_code == 503


def test_ingest_rejects_empty_text():
    resp = client.post("/ingest", json={"text": "", "source": "doc-a"})
    assert resp.status_code == 422
