"""
LMS sub-agent — lecture lists, transcripts, and assignments.

Uses create_react_agent because all tools are read-only.
Supports optional term_id for semester selection (LMS term bug fix, PR #61).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from ssu_agent import config
from ssu_agent.supervisor.state import SsuAgentState

_SYSTEM_PROMPT = """당신은 숭실대학교 LMS(Canvas) 전문 AI 어시스턴트입니다.

담당 영역:
- 강의 목록 조회 (get_my_lecture_list): term_id로 특정 학기 강의 선택 가능
- 강의 자막/STT 전사 (get_lecture_transcript): 특정 강의의 자막 또는 음성 텍스트
- 과제 목록 조회 (get_my_assignments): compact=true 옵션으로 요약 제공
- LMS 학기 목록 (get_my_lms_terms): 학기 선택 시 먼저 이 도구로 학기 ID를 확인하세요.

학기 관련 주의: Canvas API는 6월에 여름학기를 기본(default)으로 반환하므로,
1학기 강의나 과제를 조회할 때는 get_my_lms_terms로 학기 목록을 먼저 조회하고
올바른 term_id를 사용하세요.

mcp_session_id가 있다면 항상 포함하세요.
"""


def build_lms_agent(
    lms_tools: list[BaseTool],
    llm: ChatGoogleGenerativeAI | None = None,
) -> StateGraph:
    """Build the LMS sub-agent graph."""
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=config.GOOGLE_API_KEY,
        )

    inner_agent = create_react_agent(llm, lms_tools, prompt=_SYSTEM_PROMPT)

    def agent_node(state: SsuAgentState) -> dict:
        result = inner_agent.invoke({"messages": state["messages"]})
        last = result["messages"][-1]
        from langchain_core.messages import AIMessage

        tagged = AIMessage(content=f"[LMS 에이전트] {last.content}")
        return {"messages": [tagged], "active_agent": None}

    graph = StateGraph(SsuAgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    return graph
