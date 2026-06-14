"""
Tests for the Academic sub-agent.

Academic agent uses create_react_agent — tests verify graph builds and tags responses.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from ssu_agent.agents.academic import build_academic_agent
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
        "pending_action": None,
    }
    result = await graph.ainvoke(state)

    last_msg = result["messages"][-1]
    assert "[학사 에이전트]" in last_msg.content
    assert result["active_agent"] is None  # cleared on return
