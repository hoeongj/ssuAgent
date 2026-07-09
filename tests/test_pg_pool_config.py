from __future__ import annotations

import importlib
import os

import pytest

from ssu_agent import config, main


def _pool_size_for_env(value: str | None) -> int:
    original = os.environ.get("AGENT_PG_POOL_MAX_SIZE")
    try:
        if value is None:
            os.environ.pop("AGENT_PG_POOL_MAX_SIZE", None)
        else:
            os.environ["AGENT_PG_POOL_MAX_SIZE"] = value
        return importlib.reload(config).AGENT_PG_POOL_MAX_SIZE
    finally:
        if original is None:
            os.environ.pop("AGENT_PG_POOL_MAX_SIZE", None)
        else:
            os.environ["AGENT_PG_POOL_MAX_SIZE"] = original
        importlib.reload(config)


def test_pg_pool_max_size_default_is_five() -> None:
    assert _pool_size_for_env(None) == 5


def test_pg_pool_max_size_env_override_honored() -> None:
    assert _pool_size_for_env("12") == 12


@pytest.mark.asyncio
async def test_lifespan_uses_configured_pg_pool_max_size(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class FakePool:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            created.append(kwargs)

        async def __aenter__(self) -> FakePool:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    class FakeSaver:
        def __init__(self, pool: FakePool) -> None:
            self.pool = pool

        async def setup(self) -> None:
            return None

    async def fake_setup_thread_owners(pool: FakePool) -> None:
        assert isinstance(pool, FakePool)

    async def fake_build_supervisor_graph(*, checkpointer: FakeSaver) -> object:
        assert isinstance(checkpointer, FakeSaver)
        return object()

    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://unit-test")
    monkeypatch.setattr(config, "AGENT_PG_POOL_MAX_SIZE", 9)
    monkeypatch.setattr(main, "AsyncConnectionPool", FakePool)
    monkeypatch.setattr(main, "AsyncPostgresSaver", FakeSaver)
    monkeypatch.setattr(main, "_setup_thread_owners", fake_setup_thread_owners)
    monkeypatch.setattr(main, "build_supervisor_graph", fake_build_supervisor_graph)

    async with main._lifespan(main.app):
        assert main._pool is not None

    assert created[0]["max_size"] == 9
