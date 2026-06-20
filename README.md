# ssuAgent

숭실대학교 MCP 서버([ssuMCP](https://github.com/hoeongj/ssuMCP))에 연결하는 LangGraph 기반 **멀티에이전트** 캠퍼스 AI 에이전트. [ssuAI](https://github.com/hoeongj/ssuAI) 웹 채팅 UI에 SSE 스트리밍으로 연동된다.

## Architecture

```
User Query
    │
    ▼
Supervisor (LangGraph StateGraph) ── 질문 분류 → 도메인 라우팅
    ├── academic agent   (학사/성적/졸업/장학 + LlamaIndex RAG)
    ├── library agent    (좌석 추천·예약, prepare/confirm HITL)
    └── lms agent        (강의/과제/자료 내보내기)
    │  Streamable HTTP (MCP 2025-03-26)
    ▼
ssuMCP Server (Spring Boot 4)
    ├── Pyxis (도서관)
    ├── u-SAINT (학사/성적)
    └── LMS (강의/과제)
```

- **멀티 프로바이더 LLM 폴백**: `llm_factory.get_llm_sequence()`가 Groq(llama-3.3-70b, 무료 14,400 req/day) → Gemini → OpenRouter 순으로 폴백(단일 장애점 제거). Groq는 `ChatOpenAI` 래퍼 대신 `ChatGroq`를 쓴다 — 제네릭 래퍼가 assistant content를 list로 직렬화해 2번째 tool call에서 Groq가 400을 내기 때문.
- **공식 출처 RAG**: `rag/academic_rag.py`의 `AcademicRagEngine`(LlamaIndex `SimpleVectorStore` + RelevancyEvaluator)로 학칙·졸업·장학 답변 근거를 검색·평가한다.
- **상태 영속화**: LangGraph Postgres checkpointer로 대화 상태를 저장한다.
- **HITL 안전장치**: 도서관 write action은 `prepare_*` → 사용자 승인 → `confirm_action` 2단계로만 실행된다.

### 주요 구성요소

| 구성요소 | 파일 | 역할 |
|---|---|---|
| Supervisor | `supervisor/graph.py` | 질문 분류 → `ROUTE_TO:X` 마커로 도메인 라우팅. LangGraph 1.2.4의 `create_react_agent`가 도구 반환 `Command`를 상위 그래프로 전파하지 않아, 라우팅 도구가 마커 문자열을 반환하고 `post_supervisor` 노드가 스캔해 `Command(goto=X)`를 내는 패턴으로 우회(ADR 0001) |
| 도메인 에이전트 | `agents/{academic,library,lms}.py` | 도메인별 MCP 도구 묶음 + 수동 `bind_tools` 폴백 루프(프로바이더 장애점 제거) |
| MCP 클라이언트 | `mcp_client.py` | ssuMCP에 Streamable HTTP(MCP 2025-03-26)로 연결, 도구 동적 로드 |
| RAG 엔진 | `rag/academic_rag.py` | LlamaIndex `SimpleVectorStore`(인메모리) + OpenAI `text-embedding-3-small`(1536-dim) + `RelevancyEvaluator`. `llm=None`이면 retrieval-only(CI에서 API 키 불필요), `MockEmbedding`으로 테스트 |
| LLM 팩토리 | `llm_factory.py` | `get_llm_sequence()` — Groq→Gemini→OpenRouter 우선순위 폴백 |
| 체크포인터 | LangGraph Postgres | 대화 상태(turn 간) 영속 |

## Why LangGraph?

| 방식 | 이유 |
|------|------|
| LangChain LCEL | 단순 체인에 적합. 상태·루프·분기 표현 어려움 |
| 직접 function calling | 오케스트레이션 코드 직접 관리. 멀티스텝 복잡도 증가 |
| **LangGraph** (채택) | StateGraph로 상태·분기·루프를 명시적 그래프로 표현. Phase 2 멀티에이전트 확장 용이 |

## Setup

```bash
pip install uv
uv sync --extra dev
```

## Run

```bash
export GOOGLE_API_KEY=<your-gemini-key>
export SSUMCP_URL=https://ssumcp.duckdns.org/mcp  # optional, this is the default
uv run python -c "
import asyncio
from ssu_agent.graph import run_query
print(asyncio.run(run_query('오늘 학식 알려줘')))
"
```

## Test

```bash
uv run pytest
```

## Phase Roadmap

| Phase | 범위 | 상태 |
|-------|------|------|
| 1 | ReAct 단일 에이전트, 공개 도구 3종 (식단/도서관/공지) | ✅ 완료 |
| 2 | 도메인별 supervisor 멀티에이전트, 도서관 예약 인증 도구(HITL), 스트리밍 응답 | ✅ 완료 |
| 3 | ssuAI 프론트엔드 연동 (웹 UI 채팅, SSE) | ✅ 완료 |
| 4 | LlamaIndex 공식 출처 RAG + RelevancyEvaluator 평가 | ✅ 완료 |

> 구현 메모: `create_react_agent`의 루핑 이슈로 도메인 에이전트는 수동 `bind_tools` 폴백 루프로 전환했다(단일 프로바이더 장애점 제거). 근거·대안은 `docs/adr/` 참조.
