"""
Academic sub-agent — u-SAINT + official policy RAG.

Uses direct bind_tools loop (not create_react_agent) to avoid turn-2 looping
and enable per-provider fallback. No HITL needed (no write actions).

Covers: grades, GPA simulation, graduation requirements, academic calendar,
chapel attendance, scholarships, and ssuMCP's embedded academic policy RAG
(classify_academic_question, search/get/evaluate policy sources).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from ssu_agent.agents.react_loop import run_react_loop
from ssu_agent.llm_factory import create_llm, get_llm_sequence
from ssu_agent.supervisor.state import SsuAgentState

_SYSTEM_PROMPT_BASE = """당신은 숭실대학교 학사 정보 전문 AI 어시스턴트입니다.

담당 영역:
- 성적 조회 및 GPA 시뮬레이션
- 졸업 요건 확인 및 정책 근거 RAG 검색
- 학사 일정 조회
- 채플 이수 현황
- 장학금 조회 및 장학 정책 검색
- 공식 학칙·졸업·장학 출처 기반 근거 제시

정책 답변 시 반드시 공식 출처(search_academic_policy_sources 또는
get_academic_policy_brief)를 조회하여 근거를 포함하세요.
추측이나 일반 지식으로 답하지 마세요."""


def _build_academic_prompt(mcp_session_id: str | None) -> str:
    prompt = _SYSTEM_PROMPT_BASE
    if mcp_session_id:
        prompt += (
            f'\n\n[인증 세션] mcp_session_id = "{mcp_session_id}"\n'
            "get_my_grades, check_graduation_requirements, simulate_gpa, get_my_schedule, "
            "get_my_chapel_info, get_my_scholarships 등 인증이 필요한 도구 호출 시 "
            "이 값을 mcp_session_id 파라미터로 반드시 포함하세요."
        )
    return prompt


def build_academic_agent(
    academic_tools: list[BaseTool],
    llm: BaseChatModel | None = None,
) -> StateGraph:
    """Build the Academic sub-agent graph."""
    llm_seq = [llm] if llm is not None else get_llm_sequence()
    if not llm_seq:
        llm_seq = [create_llm()]

    async def agent_node(state: SsuAgentState, config: RunnableConfig) -> dict:
        prompt = _build_academic_prompt(state.get("mcp_session_id"))
        return await run_react_loop(
            llm_seq, academic_tools, prompt, "학사 에이전트", state, config
        )

    graph = StateGraph(SsuAgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    return graph
