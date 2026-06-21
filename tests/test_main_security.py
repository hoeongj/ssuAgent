"""
Tests for the FastAPI hardening: opt-in /agent API-key gate and open /health.

The graph/DB are never touched: _stream_graph is monkeypatched to a dummy async
generator, and TestClient is instantiated WITHOUT a context manager so the
lifespan (which opens a real Postgres pool) does not run.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ssu_agent import config, main


async def _fake_stream_graph(input_data, config):  # noqa: A002 - mirrors prod signature
    """Stand-in for _stream_graph: one dummy SSE line, no LLM/DB."""
    yield 'data: {"type": "done"}\n\n'


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Endpoints resolve _stream_graph as a module global at call time, so this
    # monkeypatch takes effect without rebuilding the app.
    monkeypatch.setattr(main, "_stream_graph", _fake_stream_graph)
    # Bare TestClient: no `with`, so lifespan/Postgres pool is never opened.
    return TestClient(main.app)


def _post_stream(client: TestClient, headers: dict | None = None):
    return client.post(
        "/agent/stream",
        json={"message": "hi", "thread_id": "t1"},
        headers=headers or {},
    )


# ── No key configured → gate is a no-op (prod behavior preserved) ───────────────


def test_stream_open_when_no_api_key(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(config, "AGENT_API_KEY", "")
    resp = _post_stream(client)
    assert resp.status_code == 200
    assert "done" in resp.text


def test_health_open(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "UP"


# ── Key configured → header required ────────────────────────────────────────────


def test_stream_401_without_header(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(config, "AGENT_API_KEY", "s3cret")
    resp = _post_stream(client)
    assert resp.status_code == 401


def test_stream_401_with_wrong_header(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(config, "AGENT_API_KEY", "s3cret")
    resp = _post_stream(client, headers={"X-Agent-Key": "nope"})
    assert resp.status_code == 401


def test_stream_passes_with_correct_header(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(config, "AGENT_API_KEY", "s3cret")
    resp = _post_stream(client, headers={"X-Agent-Key": "s3cret"})
    assert resp.status_code == 200
    assert "done" in resp.text


def test_health_open_even_with_api_key(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(config, "AGENT_API_KEY", "s3cret")
    resp = client.get("/health")
    assert resp.status_code == 200
