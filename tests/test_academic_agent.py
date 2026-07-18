"""
Tests for the Academic sub-agent.

Academic agent uses a manual bind_tools loop — tests verify graph builds and tags responses.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import Field

from ssu_agent.agents.academic import (
    _ACADEMIC_LOGIN_MESSAGE,
    _ACADEMIC_SERVICE_UNAVAILABLE_MESSAGE,
    _ACADEMIC_STATUS_UNAVAILABLE_MESSAGE,
    _build_academic_prompt,
    _requires_private_academic_context,
    _requires_private_academic_data,
    build_academic_agent,
)
from ssu_agent.supervisor.state import SsuAgentState


@tool
def get_my_grades(mcp_session_id: str) -> str:
    """성적 조회"""
    return '{"terms": []}'


@tool
def search_academic_policy_sources(query: str) -> str:
    """학칙 검색"""
    return '{"sources": [{"title": "학사규정", "content": "졸업학점 130학점"}]}'


ACADEMIC_TOOLS = [get_my_grades, search_academic_policy_sources]


class _MockAcademicLLM(FakeMessagesListChatModel):
    """Fake chat model that returns a fixed academic answer. bind_tools returns self."""

    def bind_tools(self, tools, **kwargs):
        return self


class _SpyAcademicLLM(FakeMessagesListChatModel):
    bind_tools_calls: int = 0
    visible_tool_names: list[str] = []

    def bind_tools(self, tools, **kwargs):
        self.bind_tools_calls += 1
        self.visible_tool_names = [tool.name for tool in tools]
        return self


class _CapturingAcademicLLM(_SpyAcademicLLM):
    seen_inputs: list[list] = Field(default_factory=list)

    async def ainvoke(self, input, config=None, **kwargs):
        self.seen_inputs.append(list(input) if isinstance(input, list) else input)
        return await super().ainvoke(input, config=config, **kwargs)


def _make_academic_llm() -> _MockAcademicLLM:
    return _MockAcademicLLM(
        responses=[AIMessage(content="졸업 학점은 130학점입니다. (출처: 학사규정)")]
    )


def test_academic_agent_builds():
    graph = build_academic_agent(ACADEMIC_TOOLS, llm=_make_academic_llm())
    compiled = graph.compile()
    assert compiled is not None


@pytest.mark.asyncio
async def test_academic_agent_tags_response():
    graph = build_academic_agent(ACADEMIC_TOOLS, llm=_make_academic_llm()).compile()

    state: SsuAgentState = {
        "messages": [HumanMessage(content="졸업 학점이 몇 점이야?")],
        "mcp_session_id": None,
        "active_agent": "academic",
    }
    result = await graph.ainvoke(state)

    last_msg = result["messages"][-1]
    assert "[학사 에이전트]" in last_msg.content
    assert result["active_agent"] is None  # cleared on return


@pytest.mark.parametrize(
    "query",
    [
        "졸업까지 어떤 조건들이 남았어?",
        "내 졸업요건 알려줘",
        "채플 몇 번 남았어?",
        "현재 성적 조회해줘",
        "성적 보여줘",
        "이번 학기 시간표 알려줘",
        "제 졸업요건 알려줘",
        "GPA 계산해줘",
        "졸업 사정표 보여줘",
        "취득학점 알려줘",
        "채플 얼마나 더 들어야 돼?",
        "장학금 받은 거 알려줘",
        "졸업할 수 있어?",
        "지난학기 성적은?",
        "전공 학점 몇 학점 남았어?",
    ],
)
def test_private_academic_intent_detection(query: str):
    assert _requires_private_academic_data(query)


@pytest.mark.parametrize(
    "query",
    [
        "조기졸업 요건이 뭐야?",
        "조기졸업 요건을 충족하려면?",
        "일반 졸업 기준 알려줘",
        "GPA 계산법 알려줘",
        "채플 이수 기준",
        "졸업 판정 기준이 뭐야",
        "내일 학사일정 알려줘",
        "졸업 학점이 몇 점이야?",
        "전공 몇 학점 들어야 해?",
    ],
)
def test_public_academic_policy_questions_stay_public(query: str):
    assert not _requires_private_academic_data(query)


@pytest.mark.parametrize("follow_up", ["지난학기요", "그 성적은?"])
def test_ambiguous_followup_inherits_immediately_previous_private_academic_turn(
    follow_up: str,
):
    messages = [
        HumanMessage(content="내 성적 알려줘"),
        AIMessage(content="어느 학기를 볼까요?", name="academic_agent"),
        HumanMessage(content=follow_up),
    ]

    assert _requires_private_academic_context(messages)


@pytest.mark.parametrize("query", ["일반 졸업 기준 알려줘", "내일 학사일정 알려줘"])
def test_explicit_public_request_does_not_inherit_private_academic_turn(query: str):
    messages = [
        HumanMessage(content="내 성적 알려줘"),
        AIMessage(content="어느 학기를 볼까요?", name="academic_agent"),
        HumanMessage(content=query),
    ]

    assert not _requires_private_academic_context(messages)


def test_authenticated_prompt_never_contains_raw_session_value():
    prompt = _build_academic_prompt(authenticated=True)

    assert "secret-session-value" not in prompt
    assert "mcp_session_id" not in prompt


@pytest.mark.asyncio
async def test_private_academic_request_without_session_skips_llm():
    llm = _SpyAcademicLLM(responses=[AIMessage(content="사용하면 안 되는 응답")])
    graph = build_academic_agent(ACADEMIC_TOOLS, llm=llm).compile()
    state: SsuAgentState = {
        "messages": [HumanMessage(content="졸업까지 어떤 조건들이 남았어?")],
        "mcp_session_id": None,
        "library_connected": False,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert result["messages"][-1].content == f"[학사 에이전트] {_ACADEMIC_LOGIN_MESSAGE}"
    assert result["active_agent"] is None
    assert llm.bind_tools_calls == 0
    assert "MCP" not in result["messages"][-1].content
    assert "세션 ID" not in result["messages"][-1].content
    assert "버튼" not in result["messages"][-1].content


@pytest.mark.asyncio
async def test_public_policy_question_without_session_still_uses_llm():
    llm = _SpyAcademicLLM(responses=[AIMessage(content="조기졸업은 공식 학칙을 따릅니다.")])
    graph = build_academic_agent(ACADEMIC_TOOLS, llm=llm).compile()
    state: SsuAgentState = {
        "messages": [HumanMessage(content="조기졸업 요건이 뭐야?")],
        "mcp_session_id": None,
        "library_connected": False,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert result["messages"][-1].content == "[학사 에이전트] 조기졸업은 공식 학칙을 따릅니다."
    assert llm.bind_tools_calls == 1


@tool
def check_graduation_requirements(mcp_session_id: str) -> str:
    """개인 졸업 요건 조회 (provider mismatch fixture)"""
    return '{"status":"AUTH_REQUIRED","provider":"SAINT","mcpSessionId":"internal"}'


@tool("get_auth_status")
def disconnected_auth_status(mcp_session_id: str) -> str:
    """Provider status fixture with only a library connection."""
    return (
        '{"status":"OK","mcpSessionId":"internal",'
        '"providers":[{"provider":"SAINT","linked":false,"health":"UNKNOWN"},'
        '{"provider":"LIBRARY","linked":true,"health":"VALID"}]}'
    )


@tool("get_auth_status")
def connected_auth_status(mcp_session_id: str) -> str:
    """Provider status fixture with a valid SAINT connection."""
    return (
        '{"status":"OK","mcpSessionId":"internal",'
        '"providers":[{"provider":"SAINT","linked":true,"health":"VALID"}]}'
    )


@pytest.mark.asyncio
async def test_missing_status_contract_fails_safe_without_llm():
    llm = _SpyAcademicLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "graduation-1",
                        "name": "check_graduation_requirements",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="MCP session ID를 알려주세요. 아래 버튼을 누르세요."),
        ]
    )
    graph = build_academic_agent([check_graduation_requirements], llm=llm).compile()
    state: SsuAgentState = {
        "messages": [HumanMessage(content="내 졸업요건 알려줘")],
        "mcp_session_id": "library-only-session",
        "library_connected": True,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert result["messages"][-1].content == (
        f"[학사 에이전트] {_ACADEMIC_STATUS_UNAVAILABLE_MESSAGE}"
    )
    assert llm.bind_tools_calls == 0
    assert "MCP" not in result["messages"][-1].content
    assert "세션 ID" not in result["messages"][-1].content
    assert "버튼" not in result["messages"][-1].content


@pytest.mark.asyncio
async def test_provider_preflight_blocks_library_only_session_before_llm():
    llm = _SpyAcademicLLM(
        responses=[AIMessage(content="MCP session ID를 알려주세요. 아래 버튼을 누르세요.")]
    )
    graph = build_academic_agent(
        [disconnected_auth_status, check_graduation_requirements],
        llm=llm,
    ).compile()
    state: SsuAgentState = {
        "messages": [HumanMessage(content="내 졸업요건 알려줘")],
        "mcp_session_id": "library-only-session",
        "library_connected": True,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert result["messages"][-1].content == f"[학사 에이전트] {_ACADEMIC_LOGIN_MESSAGE}"
    assert llm.bind_tools_calls == 0


@pytest.mark.asyncio
async def test_private_academic_followup_keeps_private_tools_after_preflight():
    llm = _SpyAcademicLLM(responses=[AIMessage(content="지난 학기 성적입니다.")])
    graph = build_academic_agent(
        [connected_auth_status, get_my_grades, search_academic_policy_sources],
        llm=llm,
    ).compile()
    state: SsuAgentState = {
        "messages": [
            HumanMessage(content="내 성적 알려줘"),
            AIMessage(content="어느 학기를 볼까요?", name="academic_agent"),
            HumanMessage(content="지난학기요"),
        ],
        "mcp_session_id": "saint-session",
        "library_connected": False,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert result["messages"][-1].name == "academic_agent"
    assert "get_my_grades" in llm.visible_tool_names
    assert "get_auth_status" not in llm.visible_tool_names


@pytest.mark.asyncio
async def test_connected_private_tool_failure_skips_model_synthesis():
    @tool("check_graduation_requirements")
    def failing_graduation_lookup(mcp_session_id: str) -> str:
        """Personal graduation lookup fixture that fails after auth preflight."""
        raise RuntimeError("sensitive upstream failure")

    llm = _CapturingAcademicLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "graduation-failure-1",
                        "name": "check_graduation_requirements",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="일반적인 졸업 요건을 대신 안내하겠습니다."),
        ]
    )
    graph = build_academic_agent(
        [connected_auth_status, failing_graduation_lookup],
        llm=llm,
    ).compile()
    state: SsuAgentState = {
        "messages": [HumanMessage(content="졸업까지 어떤 조건들이 남았어?")],
        "mcp_session_id": "saint-session",
        "library_connected": False,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert len(llm.seen_inputs) == 1
    assert result["messages"][-1].content == (
        f"[학사 에이전트] {_ACADEMIC_SERVICE_UNAVAILABLE_MESSAGE}"
    )
    assert "일반적인 졸업 요건" not in result["messages"][-1].content
    assert "sensitive upstream failure" not in repr(result)


@pytest.mark.asyncio
async def test_structured_academic_upstream_failure_skips_model_synthesis():
    @tool("check_graduation_requirements")
    def unavailable_graduation_lookup(mcp_session_id: str) -> str:
        """Personal lookup returning the ssuMCP non-OK private envelope."""
        return (
            '{"status":"UPSTREAM_UNAVAILABLE","code":"UPSTREAM_UNAVAILABLE",'
            '"retryable":true,"mcpSessionId":"academic-secret",'
            '"userMessage":"일시적 오류",'
            '"developerMessage":"sensitive academic implementation detail"}'
        )

    llm = _CapturingAcademicLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "graduation-structured-failure-1",
                        "name": "check_graduation_requirements",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="일반적인 졸업 요건을 대신 안내하겠습니다."),
        ]
    )
    graph = build_academic_agent(
        [connected_auth_status, unavailable_graduation_lookup],
        llm=llm,
    ).compile()
    state: SsuAgentState = {
        "messages": [HumanMessage(content="졸업까지 어떤 조건들이 남았어?")],
        "mcp_session_id": "saint-session",
        "library_connected": False,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert len(llm.seen_inputs) == 1
    assert result["messages"][-1].content == (
        f"[학사 에이전트] {_ACADEMIC_SERVICE_UNAVAILABLE_MESSAGE}"
    )
    assert "일반적인 졸업 요건" not in result["messages"][-1].content
    assert "academic-secret" not in repr(result)
    assert "sensitive academic implementation detail" not in repr(result)


@pytest.mark.asyncio
async def test_public_policy_with_unrelated_session_uses_only_public_tools():
    llm = _SpyAcademicLLM(responses=[AIMessage(content="공개 졸업 기준 답변입니다.")])
    graph = build_academic_agent(
        [disconnected_auth_status, check_graduation_requirements, search_academic_policy_sources],
        llm=llm,
    ).compile()
    state: SsuAgentState = {
        "messages": [HumanMessage(content="조기졸업 요건이 뭐야?")],
        "mcp_session_id": "library-only-session",
        "library_connected": True,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert result["messages"][-1].content == "[학사 에이전트] 공개 졸업 기준 답변입니다."
    assert llm.visible_tool_names == ["search_academic_policy_sources"]


@pytest.mark.asyncio
async def test_connected_provider_still_replaces_hallucinated_internal_auth_guidance():
    llm = _SpyAcademicLLM(
        responses=[AIMessage(content="MCP session ID를 알려주세요. 로그인 버튼을 누르세요.")]
    )
    graph = build_academic_agent(
        [connected_auth_status, check_graduation_requirements],
        llm=llm,
    ).compile()
    state: SsuAgentState = {
        "messages": [HumanMessage(content="내 졸업요건 알려줘")],
        "mcp_session_id": "saint-session",
        "library_connected": False,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert result["messages"][-1].content == f"[학사 에이전트] {_ACADEMIC_LOGIN_MESSAGE}"
    assert llm.bind_tools_calls == 1


@pytest.mark.asyncio
async def test_connected_provider_replaces_hallucinated_auth_url_only():
    llm = _SpyAcademicLLM(
        responses=[
            AIMessage(
                content=("https://ssumcp.duckdns.org/api/mcp/auth/saint/start?state=secret-state")
            )
        ]
    )
    graph = build_academic_agent(
        [connected_auth_status, check_graduation_requirements],
        llm=llm,
    ).compile()
    state: SsuAgentState = {
        "messages": [HumanMessage(content="내 졸업요건 알려줘")],
        "mcp_session_id": "saint-session",
        "library_connected": False,
        "active_agent": "academic",
    }

    result = await graph.ainvoke(state)

    assert result["messages"][-1].content == f"[학사 에이전트] {_ACADEMIC_LOGIN_MESSAGE}"
