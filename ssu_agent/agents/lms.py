"""
LMS sub-agent — assignments and LMS terms.

Uses direct bind_tools loop (not create_react_agent) to avoid turn-2 looping
and enable per-provider fallback.
Supports optional term_id for semester selection (LMS term bug fix, PR #61).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from ssu_agent.agents.react_loop import run_react_loop
from ssu_agent.llm_factory import create_llm, get_llm_sequence
from ssu_agent.supervisor.state import SsuAgentState

_SYSTEM_PROMPT_BASE = """당신은 숭실대학교 LMS(Canvas) 전문 AI 어시스턴트입니다.

담당 영역:
- 과제 목록 조회 (get_my_assignments): compact=true 옵션으로 요약 제공
- LMS 학기 목록 (get_my_lms_terms): 학기 선택 시 먼저 이 도구로 학기 ID를 확인하세요.
- get_lms_dashboard: 미제출 과제·퀴즈 마감, 학사일정(시험·수강신청), 진행 중 공지를 한 번에 조회.
  선택 파라미터: term_id (생략 시 현재 학기 자동).
  항상 mcp_session_id 포함.
- 강의자료 내보내기 플로우:
  1. get_my_lms_courses: 수강 과목 목록 조회 (term_id 선택, 생략 시 현재 학기)
  2. get_my_lms_materials: 선택 과목의 비영상 자료 목록 조회 (영상·오디오 자동 제외)
  3. prepare_lms_material_export: 내보낼 자료 검증 및 확인 요청 생성
  4. confirm_lms_material_export: 확인 후 ZIP 내보내기 시작 → 다운로드 링크 반환 (20분 유효)
  항상 mcp_session_id 포함. 영상·오디오 파일은 절대 포함되지 않음.

학기 관련 주의: Canvas API는 6월에 여름학기를 기본(default)으로 반환하므로,
1학기 과제를 조회할 때는 get_my_lms_terms로 학기 목록을 먼저 조회하고
올바른 term_id를 사용하세요.

효율 규칙(중요): 질문에 맞는 도구를 최소 횟수로 호출하세요. 필요한 도구가
여러 개면 한 번에 함께 호출하세요(병렬 실행됨). 같은 도구를 반복 호출하거나
결과가 나온 뒤 불필요하게 재확인하지 마세요."""


def _build_lms_prompt(mcp_session_id: str | None) -> str:
    prompt = _SYSTEM_PROMPT_BASE
    if mcp_session_id:
        prompt += (
            "\n\n[인증 완료] 사용자는 이미 로그인되어 있습니다. "
            f'mcp_session_id = "{mcp_session_id}".\n'
            "절대 로그인을 다시 요청하지 마세요. 과제·강의자료·대시보드 질문에는 해당 도구"
            "(get_my_assignments, get_my_lms_terms, get_lms_dashboard, get_my_lms_courses, "
            "get_my_lms_materials, prepare_lms_material_export, confirm_lms_material_export)를 "
            "위 mcp_session_id 값을 파라미터로 넣어 지금 즉시 호출하고, 그 결과로 답하세요."
        )
    else:
        # No auth session: every LMS tool needs the session. Answer with a login
        # nudge instead of burning turns on calls that can only fail.
        prompt += (
            "\n\n[인증 세션 없음] LMS 도구는 모두 인증이 필요합니다. 도구를 "
            "호출하지 말고 'LMS 연결(로그인) 후 확인할 수 있다'고 안내만 하세요."
        )
    return prompt


def build_lms_agent(
    lms_tools: list[BaseTool],
    llm: BaseChatModel | None = None,
) -> StateGraph:
    """Build the LMS sub-agent graph."""
    llm_seq = [llm] if llm is not None else get_llm_sequence()
    if not llm_seq:
        llm_seq = [create_llm()]

    async def agent_node(state: SsuAgentState, config: RunnableConfig) -> dict:
        prompt = _build_lms_prompt(state.get("mcp_session_id"))
        return await run_react_loop(llm_seq, lms_tools, prompt, "LMS 에이전트", state, config)

    graph = StateGraph(SsuAgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    return graph
