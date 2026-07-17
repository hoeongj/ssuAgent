from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from ssu_agent.agents import auth_guard
from ssu_agent.agents.auth_guard import (
    ProviderLinkState,
    auth_denial_status,
    check_provider_link,
    contains_internal_auth_guidance,
    sanitize_messages_for_model,
    sanitize_tool_result_for_model,
    tools_for_model,
)


@pytest.mark.asyncio
async def test_session_bound_tool_hides_schema_and_injects_session():
    calls: list[tuple[str, str]] = []

    @tool
    async def private_lookup(query: str, mcp_session_id: str) -> str:
        """Look up private data with mcp_session_id."""
        calls.append((query, mcp_session_id))
        return '{"status":"OK","mcpSessionId":"secret","data":{"value":1}}'

    [bound] = tools_for_model([private_lookup], "secret-session")

    assert "mcp_session_id" not in bound.tool_call_schema.model_json_schema()["properties"]
    await bound.ainvoke({"query": "grades"})
    assert calls == [("grades", "secret-session")]


@pytest.mark.asyncio
async def test_session_bound_tool_supports_mcp_adapter_dict_schema():
    class DictSchemaTool(BaseTool):
        def _run(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:
            return kwargs

    original = DictSchemaTool(
        name="private_lookup",
        description="Lookup requiring mcp_session_id.",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mcp_session_id": {"type": "string"},
            },
            "required": ["query", "mcp_session_id"],
            "additionalProperties": False,
        },
    )

    [bound] = tools_for_model([original], "secret-session")
    result = await bound.ainvoke({"query": "grades"})

    assert bound.tool_call_schema["required"] == ["query"]
    assert "mcp_session_id" not in bound.tool_call_schema["properties"]
    assert result["mcp_session_id"] == "secret-session"


def test_no_session_removes_private_and_auth_lifecycle_tools():
    @tool
    def private_lookup(mcp_session_id: str) -> str:
        """Private lookup."""
        return "private"

    @tool
    def get_auth_status(mcp_session_id: str | None = None) -> str:
        """Auth status."""
        return "status"

    @tool
    def public_lookup(query: str) -> str:
        """Public lookup."""
        return query

    assert [
        tool.name
        for tool in tools_for_model([private_lookup, get_auth_status, public_lookup], None)
    ] == ["public_lookup"]


def test_structured_result_removes_auth_secrets_but_preserves_data():
    safe = sanitize_tool_result_for_model(
        json.dumps(
            {
                "status": "OK",
                "mcpSessionId": "secret-session",
                "loginUrl": "https://secret.example",
                "developerMessage": "retry with mcp_session_id",
                "data": {"remainingCredits": 12},
            }
        )
    )

    assert json.loads(safe) == {"status": "OK", "data": {"remainingCredits": 12}}


def test_unstructured_result_redacts_current_session_and_auth_start_url():
    safe = sanitize_tool_result_for_model(
        "session=secret-session-value "
        "https://ssumcp.duckdns.org/api/mcp/auth/saint/start?state=secret-state",
        "secret-session-value",
    )

    assert "secret-session-value" not in safe
    assert "/api/mcp/auth/" not in safe


def test_unstructured_history_redacts_a_previous_session_handle():
    old_session = "0f1e2d3c-4b5a-6978-90ab-cdef12345678"

    safe = sanitize_tool_result_for_model(
        f"backend rejected session {old_session}",
        "new-session-handle",
    )

    assert old_session not in safe
    assert "backend rejected" in safe


def test_history_redaction_is_non_mutating_and_preserves_domain_data():
    session_id = "library-session-secret"
    auth_url = "https://ssumcp.duckdns.org/api/mcp/auth/library/start?state=private"
    assistant = AIMessage(
        content=f"로그인: {auth_url}",
        tool_calls=[
            {
                "id": "library-1",
                "name": "get_my_library_seat",
                "args": {"mcp_session_id": session_id},
                "type": "tool_call",
            }
        ],
    )
    result = ToolMessage(
        content=json.dumps(
            {
                "status": "OK",
                "mcpSessionId": session_id,
                "loginUrl": auth_url,
                "data": {"floor": 5, "available": 8},
            }
        ),
        tool_call_id="library-1",
    )

    safe = sanitize_messages_for_model([assistant, result], session_id)
    serialized = repr(safe)

    assert session_id not in serialized
    assert auth_url not in serialized
    assert "mcp_session_id" not in safe[0].tool_calls[0]["args"]
    assert json.loads(safe[1].content)["data"] == {"floor": 5, "available": 8}
    assert assistant.tool_calls[0]["args"]["mcp_session_id"] == session_id


def test_model_history_redacts_lms_download_capability_but_keeps_link_label():
    capability_url = (
        "https://ssumcp.duckdns.org/api/lms/exports/job-1/download?token=secret-download-token"
    )
    assistant = AIMessage(content=f"[강의 파일 다운로드]({capability_url})")

    [safe] = sanitize_messages_for_model([assistant])

    assert "강의 파일 다운로드" in safe.content
    assert capability_url not in safe.content
    assert "secret-download-token" not in safe.content
    assert capability_url in assistant.content


def test_current_tool_result_keeps_lms_capability_for_local_terminal_formatter():
    capability_url = (
        "https://ssumcp.duckdns.org/api/lms/exports/job-1/download?token=secret-download-token"
    )
    content = json.dumps({"status": "OK", "data": {"downloadUrl": capability_url}})

    safe = sanitize_tool_result_for_model(content)

    assert json.loads(safe)["data"]["downloadUrl"] == capability_url


def test_model_history_redacts_lms_capability_from_tool_messages():
    capability_url = (
        "https://ssumcp.duckdns.org/api/lms/exports/job-1/download?token=secret-download-token"
    )
    result = ToolMessage(
        content=json.dumps({"status": "OK", "data": {"downloadUrl": capability_url}}),
        tool_call_id="confirm-1",
    )

    [safe] = sanitize_messages_for_model([result])

    assert capability_url not in safe.content
    assert "secret-download-token" not in safe.content
    assert "download capability redacted" in safe.content
    assert capability_url in result.content


def test_internal_auth_guidance_detects_raw_auth_url():
    assert contains_internal_auth_guidance("/api/mcp/auth/lms/start?state=private-state")


@pytest.mark.parametrize(
    "status",
    ["AUTH_REQUIRED", "NO_SESSION", "INVALID_SESSION", "SESSION_MISMATCH"],
)
def test_auth_denial_status_uses_exact_json_status(status: str):
    assert auth_denial_status(json.dumps({"status": status})) == status


def test_auth_denial_status_ignores_status_name_inside_normal_data():
    content = json.dumps({"status": "OK", "data": {"note": "AUTH_REQUIRED is documented here"}})

    assert auth_denial_status(content) is None


@pytest.mark.asyncio
async def test_provider_link_check_uses_structured_provider_state():
    @tool
    def get_auth_status(mcp_session_id: str) -> str:
        """Auth status."""
        return json.dumps(
            {
                "status": "OK",
                "mcpSessionId": mcp_session_id,
                "providers": [
                    {"provider": "SAINT", "linked": True, "health": "VALID"},
                    {"provider": "LMS", "linked": False, "health": "UNKNOWN"},
                ],
            }
        )

    assert (
        await check_provider_link([get_auth_status], "secret", "SAINT", {})
        is ProviderLinkState.CONNECTED
    )
    assert (
        await check_provider_link([get_auth_status], "secret", "LMS", {})
        is ProviderLinkState.DISCONNECTED
    )


@pytest.mark.asyncio
async def test_provider_link_check_missing_contract_is_unsupported():
    assert await check_provider_link([], "secret", "SAINT", {}) is ProviderLinkState.UNSUPPORTED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not-json", ProviderLinkState.UNAVAILABLE),
        (
            json.dumps(
                {
                    "status": "OK",
                    "providers": [{"provider": "SAINT", "linked": True, "health": "EXPIRED"}],
                }
            ),
            ProviderLinkState.DISCONNECTED,
        ),
        (
            json.dumps(
                {
                    "status": "OK",
                    "providers": [{"provider": "SAINT", "linked": True, "health": "ERROR"}],
                }
            ),
            ProviderLinkState.UNAVAILABLE,
        ),
    ],
)
async def test_provider_link_check_handles_unhealthy_payloads(
    payload: str,
    expected: ProviderLinkState,
):
    @tool("get_auth_status")
    def status_fixture(mcp_session_id: str) -> str:
        """Provider status fixture."""
        return payload

    assert await check_provider_link([status_fixture], "secret", "SAINT", {}) is expected


@pytest.mark.asyncio
async def test_provider_link_check_times_out(monkeypatch: pytest.MonkeyPatch):
    @tool("get_auth_status")
    async def slow_status(mcp_session_id: str) -> str:
        """Slow provider status fixture."""
        await asyncio.sleep(0.1)
        return '{"status":"OK","providers":[]}'

    monkeypatch.setattr(auth_guard, "_AUTH_STATUS_TIMEOUT_SECONDS", 0.01)

    assert (
        await check_provider_link([slow_status], "secret", "SAINT", {})
        is ProviderLinkState.UNAVAILABLE
    )
