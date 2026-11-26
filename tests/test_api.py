"""Tests for the in-memory graph store and the REST API. No Neo4j or Ollama required."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app  # noqa: E402
from memory_store import MemoryGraphStore  # noqa: E402

DOC = app.DEMO_DOCUMENTS["CloudEngine Specification"]
EXTRACTION = {
    "nodes": [
        {"name": "CloudEngine", "type": "PRODUCT", "summary": "Automated deployment framework."},
        {"name": "Knowledge Graph Labs", "type": "ORGANIZATION", "summary": "Builds developer tools."},
        {"name": "Dhruv Mittal", "type": "PERSON", "summary": "Architect of CloudEngine."},
    ],
    "edges": [
        {"source": "Knowledge Graph Labs", "target": "CloudEngine", "type": "designed", "context": "designed by"},
        {"source": "Dhruv Mittal", "target": "CloudEngine", "type": "ARCHITECTED", "context": "architected by"},
        {"source": "CloudEngine", "target": "Neo4j", "type": "INTEGRATES_WITH", "context": "integrates with Neo4j"},
    ],
}


class FakeLLM:
    def __init__(self, answer="CloudEngine was architected by Dhruv Mittal [1][2]. [9]"):
        self.answer = answer

    def chat(self, model, messages, **kwargs):
        if kwargs.get("format") == "json":
            return {"message": {"content": json.dumps(EXTRACTION)}}
        return {"message": {"content": self.answer}}

    def list(self):
        return {"models": []}


@pytest.fixture
def store():
    s = MemoryGraphStore()
    nodes, edges = app.build_graph_elements(EXTRACTION, DOC, "CloudEngine Specification")
    s.ingest(nodes, edges, DOC)
    return s


def test_ingest_creates_missing_edge_endpoints(store):
    names = {n["name"] for n in store.export_graph()["nodes"]}
    assert "Neo4j" in names  # only mentioned in an edge
    assert store.graph_size() == (4, 3)


def test_ingest_merges_same_entity_across_documents(store):
    nodes, edges = app.build_graph_elements(EXTRACTION, DOC, "Second Doc")
    store.ingest(nodes, edges, DOC)
    cloud = next(n for n in store.export_graph()["nodes"] if n["name"] == "CloudEngine")
    assert cloud["documents"] == ["CloudEngine Specification", "Second Doc"]
    assert store.graph_size() == (4, 3)


def test_find_entry_nodes_prefers_name_matches(store):
    hits = store.find_entry_nodes("Who architected CloudEngine?")
    assert hits[0]["name"] == "CloudEngine"
    assert all(h["name_match"] for h in hits)


def test_fetch_hops_returns_shortest_paths(store):
    root = app.make_node_id("Dhruv Mittal")
    by_name = {r["name"]: r for r in store.fetch_hops_context(root, depth=2)}
    assert by_name["CloudEngine"]["hops"] == 1
    assert by_name["Neo4j"]["hops"] == 2
    assert by_name["Neo4j"]["path_names"] == ["Dhruv Mittal", "CloudEngine", "Neo4j"]
    assert store.fetch_hops_context(root, depth=1) and all(r["hops"] == 1 for r in store.fetch_hops_context(root, depth=1))


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    import api

    fake = FakeLLM()
    monkeypatch.setattr(app, "inference_client", fake)
    monkeypatch.setattr(api, "inference_client", fake)
    api.store.reset_database()
    api.documents.clear()
    api.citation_stats.update(cited=0, valid=0)
    return TestClient(api.app)


def test_api_ingest_and_overview(client):
    r = client.post("/api/documents", json={"name": "CloudEngine Specification", "text": DOC})
    assert r.status_code == 200 and r.json()["entities"] == 3
    ov = client.get("/api/overview").json()
    assert ov["stats"]["entities"] == 4 and ov["stats"]["relationships"] == 3
    assert ov["documents"][0]["name"] == "CloudEngine Specification"


def test_api_ask_validates_citations(client):
    client.post("/api/documents", json={"name": "CloudEngine Specification", "text": DOC})
    res = client.post("/api/ask", json={"question": "Who architected CloudEngine?"}).json()
    assert [s["n"] for s in res["sources"]] == [1, 2]
    assert res["invalidCitations"] == [9]
    assert set(res["citedIds"]) <= set(res["contextIds"])
    stats = client.get("/api/overview").json()["stats"]
    assert stats["citationValidity"] == round(100 * 2 / 3, 1)


def test_api_rejects_short_documents(client):
    assert client.post("/api/documents", json={"name": "x", "text": "too short"}).status_code == 422


def test_reset_reseeds_sample_graph(client):
    import api
    r = client.delete("/api/graph")
    assert r.status_code == 200
    stats = client.get("/api/overview").json()["stats"]
    if (api.WEB_DIR / "demo-data.json").is_file():
        assert stats["entities"] > 0 and stats["documents"] == len(app.DEMO_DOCUMENTS)


def test_ask_is_rate_limited(client, monkeypatch):
    import api
    monkeypatch.setitem(api.RATE_LIMITS, "ask", 1)
    api._hits.clear()
    assert client.post("/api/ask", json={"question": "Who architected CloudEngine?"}).status_code == 200
    assert client.post("/api/ask", json={"question": "Who architected CloudEngine?"}).status_code == 429


def test_pick_latest_flash_prefers_newest_stable():
    from llm import pick_latest_flash

    class Lister:
        def list(self):
            ids = ["models/gemini-2.5-flash", "models/gemini-3.8-flash", "models/gemini-3.8-flash-lite",
                   "models/gemini-3.10-pro", "models/gemini-3.1-flash"]
            return {"models": [{"model": i} for i in ids]}

    class Broken:
        def list(self):
            raise RuntimeError("offline")

    assert pick_latest_flash(Lister(), "fallback") == "gemini-3.8-flash"
    assert pick_latest_flash(Broken(), "fallback") == "fallback"


def test_client_falls_back_when_model_overloaded(monkeypatch):
    import httpx
    import llm

    calls = []

    def handler(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "busy-model":
            return httpx.Response(503, json={"error": "high demand"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    client = llm.OpenAICompatClient("k", "https://example.test/v1", "Test")
    client._http = httpx.Client(base_url="https://example.test/v1/", transport=httpx.MockTransport(handler))
    client.fallback_models = ["busy-model", "spare-model"]
    assert client.chat("busy-model", [{"role": "user", "content": "hi"}])["message"]["content"] == "ok"
    assert calls == ["busy-model", "busy-model", "spare-model"]
