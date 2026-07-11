from __future__ import annotations

import httpx
import pytest
from langchain_core.tools import StructuredTool, tool

from ssu_agent.mcp_client import wrap_mcp_tool_for_retry
from ssu_agent.supervisor.graph import categorise_tools


@pytest.mark.asyncio
async def test_retry_wrapper_retries_transport_exception_once():
    calls = 0

    async def flaky_tool_impl() -> tuple[str, dict[str, str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connection dropped")
        return "ok", {"raw": "ok"}

    flaky_tool = StructuredTool.from_function(
        coroutine=flaky_tool_impl,
        name="flaky_tool",
        description="Flaky MCP tool.",
        response_format="content_and_artifact",
    )

    wrapped = wrap_mcp_tool_for_retry(flaky_tool, retry_backoff_seconds=0)

    result = await wrapped.ainvoke({})

    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_retry_wrapper_does_not_retry_non_transport_exception():
    calls = 0

    @tool
    async def bug_tool() -> str:
        """Tool with a non-transport bug."""
        nonlocal calls
        calls += 1
        raise ValueError("bad input")

    wrapped = wrap_mcp_tool_for_retry(bug_tool, retry_backoff_seconds=0)

    with pytest.raises(ValueError, match="bad input"):
        await wrapped.ainvoke({})

    assert calls == 1


def test_retry_wrapper_preserves_tool_identity_for_categorisation():
    @tool
    async def get_library_available_seats() -> str:
        """Library seat availability."""
        return "available"

    wrapped = wrap_mcp_tool_for_retry(get_library_available_seats)

    assert wrapped.name == get_library_available_seats.name
    assert wrapped.description == get_library_available_seats.description
    assert wrapped.args_schema is get_library_available_seats.args_schema

    cats = categorise_tools([wrapped])
    assert cats["library"] == [wrapped]
