"""End-to-end API tests against the real index, using the no-LLM provider (deterministic, no key)."""
import json

import pytest
from fastapi.testclient import TestClient

from conftest import requires_index

pytestmark = requires_index


@pytest.fixture(scope="module")
def client():
    from api import index

    index.RATE_LIMIT_PER_MINUTE = 0  # disabled except in the rate-limit test
    return TestClient(index.app)


def ask(client, question, **extra):
    res = client.post("/api/ask", json={"question": question, "provider": "extractive", **extra})
    assert res.status_code == 200, res.text
    return res.json()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["chunks"] > 0


def test_providers_always_include_no_key_option(client):
    providers = {p["id"]: p for p in client.get("/api/providers").json()["providers"]}
    assert providers["extractive"]["ready"] is True


def test_answer_cites_the_right_lecture(client):
    body = ask(client, "How does Dijkstra's algorithm find the shortest path?")
    assert body["sources"][0]["source"].startswith("L19.")
    assert "[L19." in body["answer"]
    assert body["abstained"] is False


def test_off_topic_question_abstains(client):
    body = ask(client, "Who won the FIFA World Cup in 2018?")
    assert body["abstained"] is True
    assert body["provider"] is None  # no LLM was called


def test_follow_up_uses_conversation(client):
    history = [{"role": "user", "content": "How does Dijkstra's algorithm work?"},
               {"role": "assistant", "content": "It repeatedly picks the closest unvisited vertex."}]
    body = ask(client, "What is its time complexity?", history=history)
    assert "Dijkstra" in body["search_query"]
    assert any(s["source"].startswith("L19.") for s in body["sources"])


def test_stream_sends_meta_then_done(client):
    res = client.post("/api/ask/stream", json={"question": "What is a circular queue?", "provider": "extractive"})
    events = [json.loads(line) for line in res.text.splitlines() if line.strip()]
    assert [e["type"] for e in events] == ["meta", "done"]
    assert events[0]["sources"] and events[1]["answer"]


def test_unknown_provider_without_key_falls_back(client):
    res = client.post("/api/ask", json={"question": "What is a heap?", "provider": "groq"})
    body = res.json()
    assert res.status_code == 200
    assert body["note"] and "No API key" in body["note"]


def test_validation_rejects_empty_question(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_rate_limit(client):
    from api import index

    index.RATE_LIMIT_PER_MINUTE = 2
    index._requests.clear()
    try:
        codes = [client.post("/api/ask", json={"question": "What is a stack?", "provider": "extractive"}).status_code
                 for _ in range(3)]
        assert codes == [200, 200, 429]
    finally:
        index.RATE_LIMIT_PER_MINUTE = 0
        index._requests.clear()
