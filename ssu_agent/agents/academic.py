"""
Academic sub-agent — u-SAINT + official policy RAG.

Uses create_react_agent because all tools are read-only.
No HITL needed (no write actions).

Covers: grades, GPA simulation, graduation requirements, academic calendar,
chapel attendance, scholarships, and ssuMCP's embedded academic policy RAG
(classify_academic_question, search/get/evaluate policy sources).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from ssu_agent import config
from ssu_agent.supervisor.state import SsuAgentState

_SYSTEM_PROMPT = """당신은 숭실대학교 학사 정보 전문 AI 어시스턴트입니다.

담당 영역:
- 성적 조회 및 GPA 시뮬레이션
- 졸업 요건 확인 및 정책 근거 RAG 검색
- 학사 일정 조회
- 채플 이수 현황
- 장학금 조회 및 장학 정책 검색
- 공식 학칙·졸업·장학 출처 기반 근거 제시

정책 답변 시 반드시 공식 출처(search_academic_policy_sources 또는
get_academic_policy_brief)를 조회하여 근거를 포함하세요.
추측이나 일반 지식으로 답하지 마세요.

mcp_session_id가 있다면 private 도구 호출 시 항상 포함하세요.
"""


def build_academic_agent(
    academic_tools: list[BaseTool],
    llm: ChatGoogleGenerativeAI | None = None,
) -> StateGraph:
    """Build the Academic sub-agent graph."""
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=config.GOOGLE_API_KEY,
        )

    inner_agent = create_react_agent(llm, academic_tools, prompt=_SYSTEM_PROMPT)

    def agent_node(state: SsuAgentState) -> dict:
        result = inner_agent.invoke({"messages": state["messages"]})
        last = result["messages"][-1]
        # Tag response so supervisor can identify sub-agent completion
        from langchain_core.messages import AIMessage
        tagged = AIMessage(content=f"[학사 에이전트] {last.content}")
        return {"messages": [tagged], "active_agent": None}

    graph = StateGraph(SsuAgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    return graph
