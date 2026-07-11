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

# ADR 0011: thread_owners rows are (owner, owner_kind) pairs — owner_kind is
# "principal" (stable subject), "session" (mcp_session_id, ADR 0010 legacy
# behavior), or None alongside owner=None for an anonymous thread.
OwnerRow = tuple[str | None, str | None]


async def _fake_stream_graph(input_data, config):  # noqa: A002 - mirrors prod signature
    """Stand-in for _stream_graph: one dummy SSE line, no LLM/DB."""
    yield 'data: {"type": "done"}\n\n'


class _FakeOwnerCursor:
    def __init__(self, owners: dict[str, OwnerRow]):
        self.owners = owners
        self._row: OwnerRow | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def execute(self, query: str, params: tuple | None = None):
        normalized = " ".join(query.split()).upper()
        if normalized.startswith("INSERT INTO THREAD_OWNERS"):
            thread_id, owner, owner_kind = params
            self.owners.setdefault(thread_id, (owner, owner_kind))
            self._row = None
            return
        if normalized.startswith("SELECT OWNER, OWNER_KIND FROM THREAD_OWNERS"):
            (thread_id,) = params
            self._row = self.owners.get(thread_id)
            return
        if normalized.startswith("UPDATE THREAD_OWNERS"):
            owner, thread_id = params
            self.owners[thread_id] = (owner, "principal")
            self._row = None
            return
        if normalized.startswith("CREATE TABLE IF NOT EXISTS THREAD_OWNERS"):
            self._row = None
            return
        if normalized.startswith("ALTER TABLE THREAD_OWNERS"):
            self._row = None
            return
        raise AssertionError(f"unexpected query: {query}")

    async def fetchone(self):
        return self._row


class _FakeOwnerConnection:
    def __init__(self, owners: dict[str, OwnerRow]):
        self.owners = owners

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def cursor(self):
        return _FakeOwnerCursor(self.owners)


class _FakeOwnerPool:
    def __init__(self):
        self.owners: dict[str, OwnerRow] = {}

    def connection(self):
        return _FakeOwnerConnection(self.owners)


@pytest.fixture
def owner_pool(monkeypatch: pytest.MonkeyPatch) -> _FakeOwnerPool:
    pool = _FakeOwnerPool()
    monkeypatch.setattr(main, "_pool", pool)
    return pool


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, owner_pool: _FakeOwnerPool) -> TestClient:
    # Endpoints resolve _stream_graph as a module global at call time, so this
    # monkeypatch takes effect without rebuilding the app.
    monkeypatch.setattr(main, "_stream_graph", _fake_stream_graph)
    # Disable per-IP rate limiting by default so functional tests are not
    # throttled; the dedicated rate-limit test re-enables it.
    monkeypatch.setattr(main.limiter, "enabled", False)
    # Bare TestClient: no `with`, so lifespan/Postgres pool is never opened.
    return TestClient(main.app)


def _post_stream(client: TestClient, headers: dict | None = None):
    return client.post(
        "/agent/stream",
        json={"message": "hi", "thread_id": "t1"},
        headers=headers or {},
    )


class _FakeResumeGraph:
    def __init__(self):
        self.update_calls: list[tuple[dict, dict]] = []

    async def aupdate_state(self, config: dict, values: dict):
        self.update_calls.append((config, values))
        return {
            "configurable": {
                **config["configurable"],
                "checkpoint_id": "fresh-checkpoint",
            }
        }


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


def test_deep_health_reports_mcp_up(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    class FakeMCPClient:
        def __init__(self):
            self.calls = 0

        async def get_tools(self):
            self.calls += 1
            return []

    fake_mcp_client = FakeMCPClient()
    captured: dict[str, float | None] = {}

    def fake_create_mcp_client(*, timeout_seconds: float | None = None):
        captured["timeout_seconds"] = timeout_seconds
        return fake_mcp_client

    monkeypatch.setattr(main, "create_mcp_client", fake_create_mcp_client)

    resp = client.get("/healthz/deep")

    assert resp.status_code == 200
    assert resp.json() == {"status": "UP", "mcp": "UP"}
    assert fake_mcp_client.calls == 1
    assert captured["timeout_seconds"] == main._DEEP_HEALTH_MCP_TIMEOUT_SECONDS


def test_deep_health_reports_mcp_down(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    class FailingMCPClient:
        async def get_tools(self):
            raise RuntimeError("ssuMCP unavailable")

    monkeypatch.setattr(
        main,
        "create_mcp_client",
        lambda *, timeout_seconds=None: FailingMCPClient(),
    )

    resp = client.get("/healthz/deep")

    assert resp.status_code == 503
    assert resp.json() == {"status": "DEGRADED", "mcp": "DOWN"}


def test_agent_request_models_default_and_accept_library_connected():
    assert main.AgentRequest(message="hi").library_connected is False
    assert main.AgentRequest(message="hi", library_connected=True).library_connected is True

    assert main.ResumeRequest(thread_id="t1", approved=True).library_connected is False
    assert (
        main.ResumeRequest(thread_id="t1", approved=True, library_connected=True).library_connected
        is True
    )


def test_stream_initial_state_includes_library_connected(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
):
    captured: dict[str, object] = {}

    async def capture_stream_graph(input_data, config):  # noqa: A002 - mirrors prod signature
        captured["input_data"] = input_data
        yield 'data: {"type": "done"}\n\n'

    monkeypatch.setattr(main, "_stream_graph", capture_stream_graph)

    resp = client.post(
        "/agent/stream",
        json={
            "message": "도서관 좌석 알려줘",
            "thread_id": "library-connected-stream",
            "library_connected": True,
        },
    )

    assert resp.status_code == 200
    assert captured["input_data"]["library_connected"] is True


def test_resume_updates_fresh_session_state_before_resume_stream(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
):
    from langgraph.types import Command

    fake_graph = _FakeResumeGraph()
    captured: dict[str, object] = {}

    async def capture_stream_graph(input_data, config):  # noqa: A002 - mirrors prod signature
        captured["input_data"] = input_data
        captured["config"] = config
        yield 'data: {"type": "done"}\n\n'

    monkeypatch.setattr(main, "_graph", fake_graph)
    monkeypatch.setattr(main, "_stream_graph", capture_stream_graph)

    resp = client.post(
        "/agent/resume",
        json={
            "thread_id": "library-resume-fresh-api",
            "approved": True,
            "action_id": 100,
            "mcp_session_id": "fresh-session",
            "library_connected": True,
        },
    )

    assert resp.status_code == 200
    assert fake_graph.update_calls == [
        (
            {"configurable": {"thread_id": "library-resume-fresh-api"}},
            {"mcp_session_id": "fresh-session", "library_connected": True},
        )
    ]
    assert captured["config"]["configurable"]["checkpoint_id"] == "fresh-checkpoint"
    assert isinstance(captured["input_data"], Command)
    assert captured["input_data"].resume["mcp_session_id"] == "fresh-session"
    assert captured["input_data"].resume["library_connected"] is True


# ── Thread ownership binding ──────────────────────────────────────────────────


def test_stream_binds_new_thread_and_allows_same_owner(
    client: TestClient,
    owner_pool: _FakeOwnerPool,
):
    resp = client.post(
        "/agent/stream",
        json={"message": "hi", "thread_id": "owned-t1", "mcp_session_id": "mcp-a"},
    )
    assert resp.status_code == 200
    assert owner_pool.owners["owned-t1"] == ("mcp-a", "session")

    resp = client.post(
        "/agent/stream",
        json={"message": "again", "thread_id": "owned-t1", "mcp_session_id": "mcp-a"},
    )
    assert resp.status_code == 200


def test_stream_rejects_different_owner(client: TestClient):
    resp = client.post(
        "/agent/stream",
        json={"message": "hi", "thread_id": "owned-t2", "mcp_session_id": "mcp-a"},
    )
    assert resp.status_code == 200

    resp = client.post(
        "/agent/stream",
        json={"message": "steal", "thread_id": "owned-t2", "mcp_session_id": "mcp-b"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "이 대화는 현재 세션의 소유가 아닙니다."


def test_stream_allows_anonymous_thread(client: TestClient, owner_pool: _FakeOwnerPool):
    resp = client.post(
        "/agent/stream",
        json={"message": "hi", "thread_id": "anon-t1"},
    )
    assert resp.status_code == 200
    assert owner_pool.owners["anon-t1"] == (None, None)

    resp = client.post(
        "/agent/stream",
        json={"message": "again", "thread_id": "anon-t1", "mcp_session_id": "mcp-a"},
    )
    assert resp.status_code == 200
    assert owner_pool.owners["anon-t1"] == (None, None)


def test_resume_rejects_different_owner(client: TestClient):
    resp = client.post(
        "/agent/stream",
        json={"message": "hi", "thread_id": "resume-t1", "mcp_session_id": "mcp-a"},
    )
    assert resp.status_code == 200

    resp = client.post(
        "/agent/resume",
        json={
            "thread_id": "resume-t1",
            "approved": True,
            "action_id": 1,
            "mcp_session_id": "mcp-b",
        },
    )
    assert resp.status_code == 403


# ── ADR 0011: stable-principal thread ownership ─────────────────────────────────


def test_stream_same_principal_across_sessions_sees_same_thread(
    client: TestClient,
    owner_pool: _FakeOwnerPool,
):
    """Re-login (new mcp_session_id) with the same stable principal must still
    resolve to the thread the principal created — the whole point of ADR 0011."""
    resp = client.post(
        "/agent/stream",
        json={
            "message": "hi",
            "thread_id": "principal-t1",
            "mcp_session_id": "mcp-device-a",
            "principal": "student-123",
        },
    )
    assert resp.status_code == 200
    assert owner_pool.owners["principal-t1"][1] == "principal"

    # Different device/session (e.g. re-login issued a new mcp_session_id), same
    # principal — must be treated as the same owner, not rejected.
    resp = client.post(
        "/agent/stream",
        json={
            "message": "again from another device",
            "thread_id": "principal-t1",
            "mcp_session_id": "mcp-device-b",
            "principal": "student-123",
        },
    )
    assert resp.status_code == 200


def test_stream_rejects_different_principal(client: TestClient):
    resp = client.post(
        "/agent/stream",
        json={
            "message": "hi",
            "thread_id": "principal-t2",
            "mcp_session_id": "mcp-a",
            "principal": "student-A",
        },
    )
    assert resp.status_code == 200

    resp = client.post(
        "/agent/stream",
        json={
            "message": "steal",
            "thread_id": "principal-t2",
            "mcp_session_id": "mcp-b",
            "principal": "student-B",
        },
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "이 대화는 현재 세션의 소유가 아닙니다."


def test_stream_anonymous_flow_unchanged_when_no_principal_ever_sent(
    client: TestClient,
    owner_pool: _FakeOwnerPool,
):
    """No caller today sends `principal` (ADR 0011 is ssuAgent-side prep only) —
    the entire existing session-bound / anonymous behavior must be untouched."""
    resp = client.post(
        "/agent/stream",
        json={"message": "hi", "thread_id": "legacy-anon"},
    )
    assert resp.status_code == 200
    assert owner_pool.owners["legacy-anon"] == (None, None)

    resp = client.post(
        "/agent/stream",
        json={"message": "hi", "thread_id": "legacy-session", "mcp_session_id": "mcp-x"},
    )
    assert resp.status_code == 200
    assert owner_pool.owners["legacy-session"] == ("mcp-x", "session")

    resp = client.post(
        "/agent/stream",
        json={
            "message": "steal",
            "thread_id": "legacy-session",
            "mcp_session_id": "mcp-y",
        },
    )
    assert resp.status_code == 403


def test_lazy_migration_rebinds_session_owned_thread_to_principal_once(
    client: TestClient,
    owner_pool: _FakeOwnerPool,
):
    """A thread created before any caller sent `principal` (session-owned, ADR
    0010 shape) must be lazily upgraded to principal-owned the first time its
    rightful session presents one — then a different session with that same
    principal must find it (rotation survived), while the upgrade must not
    silently re-run / re-key on every subsequent call."""
    # 1) Legacy session-only claim (no principal yet — mirrors a thread that
    #    predates this frontend rollout).
    resp = client.post(
        "/agent/stream",
        json={"message": "hi", "thread_id": "migrate-t1", "mcp_session_id": "mcp-orig"},
    )
    assert resp.status_code == 200
    assert owner_pool.owners["migrate-t1"] == ("mcp-orig", "session")

    # 2) The rightful session now starts sending a principal -> lazy rebind.
    resp = client.post(
        "/agent/stream",
        json={
            "message": "now authenticated",
            "thread_id": "migrate-t1",
            "mcp_session_id": "mcp-orig",
            "principal": "student-123",
        },
    )
    assert resp.status_code == 200
    assert owner_pool.owners["migrate-t1"][1] == "principal"
    migrated_owner = owner_pool.owners["migrate-t1"][0]

    # 3) Re-login: brand new mcp_session_id, same principal -> same thread found
    #    (this is what ADR 0010 alone could never do).
    resp = client.post(
        "/agent/stream",
        json={
            "message": "after re-login",
            "thread_id": "migrate-t1",
            "mcp_session_id": "mcp-new-device",
            "principal": "student-123",
        },
    )
    assert resp.status_code == 200
    # Runs at most once: the stored owner/kind is stable across further calls,
    # not re-derived or re-written on every request.
    assert owner_pool.owners["migrate-t1"] == (migrated_owner, "principal")

    # 4) The original session, now stale for this thread, no longer matches on
    #    its own (session-only auth is no longer sufficient once a thread is
    #    principal-owned) unless it also presents the principal.
    resp = client.post(
        "/agent/stream",
        json={
            "message": "old session without principal",
            "thread_id": "migrate-t1",
            "mcp_session_id": "mcp-orig",
        },
    )
    assert resp.status_code == 403


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


# ── Edge hardening: rate limit, payload cap, error non-disclosure ───────────────


def test_stream_rate_limited_over_limit(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    # Limit is read per-request (callable), so a low override takes effect.
    monkeypatch.setattr(config, "AGENT_API_KEY", "")
    monkeypatch.setattr(config, "AGENT_RATE_LIMIT", "3/minute")
    monkeypatch.setattr(main.limiter, "enabled", True)
    statuses = [_post_stream(client).status_code for _ in range(5)]
    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:]


def test_stream_rejects_oversized_message(monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr(config, "AGENT_API_KEY", "")
    huge = "x" * (config.AGENT_MAX_MESSAGE_CHARS + 1)
    resp = client.post("/agent/stream", json={"message": huge, "thread_id": "t1"})
    assert resp.status_code == 422


async def test_stream_graph_hides_exception_detail(monkeypatch: pytest.MonkeyPatch):
    """The error SSE must not leak internal exception detail to the client."""

    class _Boom:
        def astream_events(self, *args, **kwargs):
            raise RuntimeError("internal dsn postgres://secret leaked")

    monkeypatch.setattr(main, "_graph", _Boom())
    chunks = [
        chunk
        async for chunk in main._stream_graph(
            {"messages": []}, {"configurable": {"thread_id": "t1"}}
        )
    ]
    joined = "".join(chunks)
    assert "postgres://secret" not in joined
    assert '"type": "error"' in joined
