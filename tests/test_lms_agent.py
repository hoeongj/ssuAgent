from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import Field

from ssu_agent import config
from ssu_agent.agents.lms import (
    _LMS_LOGIN_MESSAGE,
    _LMS_SERVICE_UNAVAILABLE_MESSAGE,
    _LMS_STATUS_UNAVAILABLE_MESSAGE,
    _build_lms_prompt,
    _format_lms_export_confirmation,
    build_lms_agent,
)
from ssu_agent.supervisor.state import SsuAgentState


class _SpyLmsLLM(FakeMessagesListChatModel):
    bind_tools_calls: int = 0
    visible_properties: list[set[str]] = []

    def bind_tools(self, tools, **kwargs):
        self.bind_tools_calls += 1
        self.visible_properties = [
            set(tool.tool_call_schema.model_json_schema().get("properties", {})) for tool in tools
        ]
        return self


class _CapturingLmsLLM(_SpyLmsLLM):
    seen_inputs: list[list] = Field(default_factory=list)

    async def ainvoke(self, input, config=None, **kwargs):
        self.seen_inputs.append(list(input) if isinstance(input, list) else input)
        return await super().ainvoke(input, config=config, **kwargs)


@pytest.fixture(autouse=True)
def _trusted_ssumcp_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SSUMCP_URL", "https://ssumcp.example/mcp")


@tool
def get_my_assignments(mcp_session_id: str) -> str:
    """LMS assignments lookup."""
    return '{"status":"OK","mcpSessionId":"secret","data":[]}'


@tool("get_auth_status")
def disconnected_lms_status(mcp_session_id: str) -> str:
    """Disconnected LMS provider status."""
    return (
        '{"status":"OK","mcpSessionId":"secret",'
        '"providers":[{"provider":"LMS","linked":false,"health":"UNKNOWN"}]}'
    )


@tool("get_auth_status")
def connected_lms_status(mcp_session_id: str) -> str:
    """Connected LMS provider status."""
    return (
        '{"status":"OK","mcpSessionId":"secret",'
        '"providers":[{"provider":"LMS","linked":true,"health":"VALID"}]}'
    )


@tool("get_auth_status")
def degraded_lms_status(mcp_session_id: str) -> str:
    """Linked LMS provider whose last upstream call failed."""
    return (
        '{"status":"OK","mcpSessionId":"secret",'
        '"providers":[{"provider":"LMS","linked":true,"health":"ERROR"}]}'
    )


@tool("confirm_lms_material_export")
def confirm_lms_export(mcp_session_id: str) -> str:
    """Confirm the prepared LMS material export."""
    return (
        '{"status":"OK","mcpSessionId":"secret","data":{"jobId":"job-1",'
        '"fileCount":74,"estimatedBytes":327155712,'
        '"downloadUrl":"https://ssumcp.example/api/lms/exports/job-1/'
        'download?token=test-token"}}'
    )


def _state(session_id: str | None) -> SsuAgentState:
    return {
        "messages": [HumanMessage(content="이번 학기 과제 보여줘")],
        "mcp_session_id": session_id,
        "library_connected": False,
        "active_agent": "lms",
    }


@pytest.mark.asyncio
async def test_lms_request_without_session_skips_llm():
    llm = _SpyLmsLLM(responses=[AIMessage(content="사용하면 안 되는 응답")])
    graph = build_lms_agent([get_my_assignments], llm=llm).compile()

    result = await graph.ainvoke(_state(None))

    assert result["messages"][-1].content == f"[LMS 에이전트] {_LMS_LOGIN_MESSAGE}"
    assert llm.bind_tools_calls == 0


@pytest.mark.asyncio
async def test_lms_missing_status_contract_fails_safe_without_llm():
    llm = _SpyLmsLLM(responses=[AIMessage(content="사용하면 안 되는 응답")])
    graph = build_lms_agent([get_my_assignments], llm=llm).compile()

    result = await graph.ainvoke(_state("lms-session"))

    assert result["messages"][-1].content == (f"[LMS 에이전트] {_LMS_STATUS_UNAVAILABLE_MESSAGE}")
    assert llm.bind_tools_calls == 0


@pytest.mark.asyncio
async def test_lms_provider_preflight_blocks_disconnected_session():
    llm = _SpyLmsLLM(responses=[AIMessage(content="MCP session ID를 알려주세요.")])
    graph = build_lms_agent(
        [disconnected_lms_status, get_my_assignments],
        llm=llm,
    ).compile()

    result = await graph.ainvoke(_state("saint-only-session"))

    assert result["messages"][-1].content == f"[LMS 에이전트] {_LMS_LOGIN_MESSAGE}"
    assert llm.bind_tools_calls == 0


@pytest.mark.asyncio
async def test_connected_lms_tools_hide_session_argument_from_model():
    llm = _SpyLmsLLM(responses=[AIMessage(content="과제 조회 결과입니다.")])
    graph = build_lms_agent(
        [connected_lms_status, get_my_assignments],
        llm=llm,
    ).compile()

    result = await graph.ainvoke(_state("lms-session"))

    assert result["messages"][-1].content == "[LMS 에이전트] 과제 조회 결과입니다."
    assert llm.visible_properties == [set()]


@pytest.mark.asyncio
async def test_degraded_lms_tool_uses_bounded_invocation_then_custom_unavailable_response():
    calls = 0

    @tool("get_my_assignments")
    def failing_assignments(mcp_session_id: str) -> str:
        """LMS bounded-invocation fixture that still fails upstream."""
        nonlocal calls
        calls += 1
        raise RuntimeError("sensitive LMS upstream failure")

    llm = _CapturingLmsLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_my_assignments",
                        "args": {},
                        "id": "assignments-bounded-1",
                    }
                ],
            ),
            AIMessage(content="일반적인 과제 안내를 대신 제공하겠습니다."),
        ]
    )
    graph = build_lms_agent(
        [degraded_lms_status, failing_assignments],
        llm=llm,
    ).compile()

    result = await graph.ainvoke(_state("degraded-lms-session"))

    assert calls == 1
    assert len(llm.seen_inputs) == 1
    assert result["messages"][-1].content == (f"[LMS 에이전트] {_LMS_SERVICE_UNAVAILABLE_MESSAGE}")
    assert "일반적인 과제 안내" not in result["messages"][-1].content
    assert "sensitive LMS upstream failure" not in repr(result)


@pytest.mark.asyncio
async def test_lms_dashboard_legacy_ok_envelope_is_an_operational_failure():
    calls = 0

    @tool("get_lms_dashboard")
    def unavailable_dashboard(mcp_session_id: str) -> str:
        """Mirror the legacy ssuMCP dashboard error envelope."""
        nonlocal calls
        calls += 1
        return (
            '{"status":"OK","mcpSessionId":"lms-secret","data":'
            '"LMS API 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. '
            '(sensitive Canvas implementation detail)"}'
        )

    llm = _CapturingLmsLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_lms_dashboard",
                        "args": {},
                        "id": "dashboard-legacy-failure-1",
                    }
                ],
            ),
            AIMessage(content="일반적인 LMS 안내를 대신 제공하겠습니다."),
        ]
    )
    graph = build_lms_agent(
        [degraded_lms_status, unavailable_dashboard],
        llm=llm,
    ).compile()

    result = await graph.ainvoke(_state("degraded-lms-session"))

    assert calls == 1
    assert len(llm.seen_inputs) == 1
    assert result["messages"][-1].content == (f"[LMS 에이전트] {_LMS_SERVICE_UNAVAILABLE_MESSAGE}")
    assert "일반적인 LMS 안내" not in result["messages"][-1].content
    assert "lms-secret" not in repr(result)
    assert "sensitive Canvas implementation detail" not in repr(result)


@pytest.mark.asyncio
async def test_degraded_lms_rejects_batched_private_calls_before_execution():
    calls = 0

    @tool("get_my_assignments")
    def assignments(mcp_session_id: str) -> str:
        """Count private LMS invocations."""
        nonlocal calls
        calls += 1
        return '{"status":"OK","data":[]}'

    llm = _CapturingLmsLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_my_assignments", "args": {}, "id": "batch-1"},
                    {"name": "get_my_assignments", "args": {}, "id": "batch-2"},
                ],
            )
        ]
    )
    graph = build_lms_agent([degraded_lms_status, assignments], llm=llm).compile()

    result = await graph.ainvoke(_state("degraded-lms-session"))

    assert calls == 0
    assert len(llm.seen_inputs) == 1
    assert result["messages"][-1].content == (f"[LMS 에이전트] {_LMS_SERVICE_UNAVAILABLE_MESSAGE}")


@pytest.mark.asyncio
async def test_degraded_lms_rejects_model_driven_private_recall():
    calls = 0

    @tool("get_my_assignments")
    def assignments(mcp_session_id: str) -> str:
        """Return one successful bounded LMS result."""
        nonlocal calls
        calls += 1
        return '{"status":"OK","data":[]}'

    llm = _CapturingLmsLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "get_my_assignments", "args": {}, "id": "bounded-1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_my_assignments", "args": {}, "id": "recall-2"}],
            ),
        ]
    )
    graph = build_lms_agent([degraded_lms_status, assignments], llm=llm).compile()

    result = await graph.ainvoke(_state("degraded-lms-session"))

    assert calls == 1
    assert len(llm.seen_inputs) == 2
    assert result["messages"][-1].content == (f"[LMS 에이전트] {_LMS_SERVICE_UNAVAILABLE_MESSAGE}")


def test_lms_prompt_uses_the_all_materials_shortcut() -> None:
    prompt = _build_lms_prompt(True)

    assert "export_all_lms_materials" in prompt
    assert "전체 과목" in prompt
    assert "prepare/export와 confirm을 같은" in prompt


def test_confirm_result_becomes_an_immediate_download_link() -> None:
    content = (
        '{"status":"OK","data":{"jobId":"job-1","fileCount":74,'
        '"estimatedBytes":327155712,'
        '"downloadUrl":"https://ssumcp.example/api/lms/exports/job-1/'
        'download?token=test-token"}}'
    )

    answer = _format_lms_export_confirmation("confirm_lms_material_export", content)

    assert answer is not None
    assert "파일 74개" in answer
    assert "약 312 MB" in answer
    assert "[강의 파일 다운로드]" in answer
    assert "https://ssumcp.example/api/lms/exports/job-1/download?token=test-token" in answer


@pytest.mark.parametrize(
    "tool_name,content",
    [
        ("get_my_lms_courses", '{"downloadUrl":"https://example.com"}'),
        ("confirm_lms_material_export", '{"status":"UPSTREAM_UNAVAILABLE"}'),
        (
            "confirm_lms_material_export",
            '{"data":{"downloadUrl":"https://ssumcp.example/api/lms/exports/job/'
            'download?token=test"}}',
        ),
        (
            "confirm_lms_material_export",
            '{"status":200,"data":{"downloadUrl":"https://ssumcp.example/api/lms/'
            'exports/job/download?token=test"}}',
        ),
        (
            "confirm_lms_material_export",
            '{"status":"OK","downloadUrl":"https://ssumcp.example/api/lms/exports/job/'
            'download?token=test"}',
        ),
        ("confirm_lms_material_export", '{"status":"OK","data":[]}'),
        (
            "confirm_lms_material_export",
            '{"status":"OK","data":{"downloadUrl":"javascript:alert(1)"}}',
        ),
        (
            "confirm_lms_material_export",
            '{"status":"OK","data":{"downloadUrl":"https://ssumcp.example/'
            'not-an-export?token=test"}}',
        ),
        (
            "confirm_lms_material_export",
            '{"status":"OK","data":{"downloadUrl":"https://evil.example/api/lms/'
            'exports/job/download?token=test"}}',
        ),
        (
            "confirm_lms_material_export",
            '{"status":"OK","data":{"downloadUrl":"http://ssumcp.example/api/lms/'
            'exports/job/download?token=test"}}',
        ),
        (
            "confirm_lms_material_export",
            '{"status":"OK","data":{"downloadUrl":"https://ssumcp.example/api/lms/'
            'exports/job/download?token=x)"}}',
        ),
        (
            "confirm_lms_material_export",
            '{"status":"OK","data":{"downloadUrl":"https://user:pass@ssumcp.example/'
            'api/lms/exports/job/download?token=test"}}',
        ),
    ],
)
def test_only_valid_lms_confirm_downloads_are_terminal(tool_name: str, content: str) -> None:
    assert _format_lms_export_confirmation(tool_name, content) is None


@pytest.mark.asyncio
async def test_lms_agent_checkpoints_confirm_link_without_final_model_round() -> None:
    llm = _SpyLmsLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "confirm_lms_material_export",
                        "args": {},
                        "id": "confirm-export-1",
                    }
                ],
            )
        ]
    )
    graph = build_lms_agent(
        [connected_lms_status, confirm_lms_export],
        llm=llm,
    ).compile()

    result = await graph.ainvoke(_state("lms-session"))

    answer = result["messages"][-1]
    assert answer.name == "lms_agent"
    assert "[강의 파일 다운로드]" in answer.content
    assert "token=test-token" in answer.content
    assert llm.bind_tools_calls == 1


@pytest.mark.asyncio
async def test_lms_agent_defers_confirm_when_model_batches_it_with_export() -> None:
    calls = {"export": 0, "confirm": 0}

    @tool("export_all_lms_materials")
    def export_all(mcp_session_id: str) -> str:
        """Prepare all current-term LMS materials for export."""
        calls["export"] += 1
        return '{"status":"OK","data":{"actionId":"action-1","fileCount":74}}'

    @tool("confirm_lms_material_export")
    def confirm_after_export(mcp_session_id: str) -> str:
        """Confirm a prepared LMS material export."""
        calls["confirm"] += 1
        return (
            '{"status":"OK","data":{"jobId":"job-1","fileCount":74,'
            '"downloadUrl":"https://ssumcp.example/api/lms/exports/job-1/'
            'download?token=test-token"}}'
        )

    llm = _SpyLmsLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "export_all_lms_materials", "args": {}, "id": "export-1"},
                    {"name": "confirm_lms_material_export", "args": {}, "id": "confirm-early"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "confirm_lms_material_export", "args": {}, "id": "confirm-2"}],
            ),
        ]
    )
    graph = build_lms_agent(
        [connected_lms_status, export_all, confirm_after_export],
        llm=llm,
    ).compile()

    result = await graph.ainvoke(_state("lms-session"))

    assert calls == {"export": 1, "confirm": 1}
    assert "[강의 파일 다운로드]" in result["messages"][-1].content
