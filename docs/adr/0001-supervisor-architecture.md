# ADR 0001 — 멀티에이전트 수퍼바이저 아키텍처 (EPIC 6 Phase 2)

작성일: 2026-06-14  
상태: 확정(구현 완료)

> **갱신 (2026-07-02)**: §3·§6·§8이 기술하는 SQLite(`SqliteSaver`) 체크포인터는 이후 [ADR 003](003-postgres-checkpointer.md)에서 `AsyncPostgresSaver`(Postgres)로 대체됐다(현행 코드 `main.py`). 현행 체크포인터는 ADR 003을 참조. 나머지 아키텍처 결정은 그대로 유효하다.

---

## 1. 배경 및 문제

ssuAgent는 숭실대학교 AI 어시스턴트로, 도서관·학사·LMS 세 도메인의 MCP 도구를 갖는다. Phase 1은 단일 ReAct 에이전트로 모든 도구를 묶었으나 다음 문제가 발생했다:

1. **Context window 오염**: 도메인과 무관한 도구 수십 개가 항상 프롬프트에 포함됨 → 라우팅 실수, 토큰 낭비
2. **HITL 불가**: 도서관 좌석 예약은 `prepare_reserve_library_seat` → 사용자 확인 → `confirm_action` 순서가 필요한데, 단일 ReAct 루프에는 중간 인터럽트 지점이 없음
3. **도메인 분리 불가**: 학사 정책 RAG와 도서관 좌석 도구가 동일 에이전트에서 섞이면 도구 선택 품질이 저하됨

---

## 2. 고려한 대안

### 대안 A: `langgraph-supervisor` 패키지 사용
- **내용**: Anthropic이 제공하는 고수준 수퍼바이저 패키지. 핸드오프를 `Command`를 반환하는 도구로 처리
- **거부 이유**: LangGraph 1.2.4에서 `create_react_agent`는 도구 함수가 `Command`를 반환해도 상위 그래프로 전파하지 않음 (verify: `inspect.getsource(create_react_agent)` — tool return value는 `ToolMessage`로 변환될 뿐 `Command`로 처리 안 됨). 즉 이 버전에서 핸드오프 도구 방식은 동작하지 않음

### 대안 B: Supervisor가 조건부 엣지로 직접 라우팅
- **내용**: 수퍼바이저 LLM 출력의 마지막 메시지를 파싱하여 `add_conditional_edges`로 분기
- **거부 이유**: LLM 자유 텍스트 파싱은 취약함 (구분자 변경, 다국어 혼용 등). 구조화 출력(Pydantic)으로 해결 가능하지만 추가 LLM 호출이 필요

### 대안 C (채택): "Route Marker + Post-Router" 패턴
- **내용**:
  1. 수퍼바이저 ReAct 에이전트에 경량 라우팅 도구(`transfer_to_*`) 제공. 이 도구들은 실제 작업 없이 `"ROUTE_TO:library_agent"` 같은 문자열 마커만 반환함
  2. `post_supervisor` 노드가 최근 8개 메시지를 정규식으로 스캔하여 마커 발견 시 `Command(goto=target)` 반환
  3. 서브에이전트는 별도 `StateGraph`로 컴파일하여 부모 그래프의 노드로 임베딩

- **채택 이유**:
  - LangGraph 1.2.4 제약(Command from tools 미지원) 내에서 안정적으로 동작
  - 라우팅 도구가 LLM 출력을 캡처 → 타이핑 명확, 파싱 안전
  - `post_supervisor` 정규식 스캔 비용은 O(1), 추가 LLM 호출 없음

---

## 3. HITL 설계 (도서관 예약 승인 게이트)

### 핵심 제약: `interrupt()` 위치
LangGraph는 노드 경계에서만 상태를 체크포인트한다. `interrupt()`를 `add_conditional_edges`의 라우터 함수 안에서 호출하면 상태 저장 없이 조용히 실패한다.

**→ `interrupt()`는 반드시 Graph Node 함수 내부에서 호출해야 한다.**

### 채택한 HITL 흐름 (Library 서브그래프)

```
agent_node  
    └─ 수동 bind_tools 루프 (prepare_*만 포함, confirm_action 제외)
       ※ create_react_agent는 turn-2에서 prepare_*를 중복 호출(actionId 2개 → 승인 게이트가
         stale action을 물어 오작동)해 폐기. 근거는 agents/library.py 모듈 docstring 참조.
    └─ 루프 실행 → prepare_reserve 결과의 ToolMessage 포함, actionId 발견 즉시 break
router (_has_pending_action)  ← 순수 함수, interrupt 없음
    └─ actionId 발견 → check_approval_node
    └─ 없음 → done_node
check_approval_node  ← 여기서만 interrupt() 호출
    └─ interrupt() 호출 → LangGraph가 체크포인트(prod=Postgres) 저장 후 실행 일시정지
    └─ FastAPI astream_events → on_chain_stream 청크의 __interrupt__ 감지 → SSE {type:"interrupt"} 전송
       (⚠️ on_interrupt 이벤트가 아님 — 아래 "실제 동작" 정정 참조)
    └─ 클라이언트가 POST /agent/resume 전송
    └─ 재개 후 승인이면 confirm_action 실행, 거부이면 취소 메시지
```

### LangGraph 1.2.4 interrupt() 실제 동작 (2026-07-05 실측 정정 ⚠️)
- `ainvoke()`: `GraphInterrupt`를 raise하지 않음. 결과 dict에 `__interrupt__` 키로 반환
  ```python
  result = {"messages": [...], "__interrupt__": [Interrupt(value={...}, id="...")]}
  ```
- `astream_events(version="v2")`: **`on_interrupt` 이벤트를 내지 않는다.** (이 ADR의 이전 판이 "on_interrupt 이벤트로 노출됨(검증됨)"이라 단언했으나 **틀렸다** — 그 미검증 가정 위에 `_stream_graph`가 `elif etype == "on_interrupt"` 죽은 분기를 만들어 **도서관 HITL 승인 카드가 브라우저에 한 번도 안 뜨는 종단 파손**을 낳았다.)
  - **실측**: 설치된 langgraph 1.2.4로 prod 토폴로지(부모 그래프+임베드 서브그래프+`interrupt()`)를 실행해 이벤트 전수 출력 → 나온 타입은 `on_chain_start/stream/end`뿐. 인터럽트는 **`on_chain_stream` 청크**에 `{"__interrupt__": (Interrupt(value=...),)}`로 실려 나온다.
  - **처리**: `_stream_graph`가 `on_chain_stream` 청크에서 `__interrupt__`를 감지(`_extract_interrupt`)해 **첫 `Interrupt.value`만** SSE로 포워드(주변 청크의 raw state=`mcp_session_id` 유출 방지). `commit e32079c`, 회귀 테스트 `tests/test_stream_interrupt.py`(실물 langgraph 그래프 구동).
  - **교훈**: 외부 라이브러리의 이벤트 계약은 "검증됨"이라 적기 전에 **설치된 버전으로 실제 실행해** 확인할 것. mock이 없는 이벤트를 심으면 테스트는 green이면서 prod는 죽는다. 상세 = `ssuMCP/TROUBLESHOOTING.md` 2026-07-05 (1).

---

## 4. 서브에이전트 임베딩 방식

**선택**: 서브에이전트를 컴파일된 서브그래프(Subgraph)로 만들어 부모 그래프의 노드로 추가

```python
library_subgraph = build_library_agent(cats["library"], llm).compile()
builder.add_node("library_agent", library_subgraph)
```

**거부된 방식**: 서브에이전트를 도구 함수 내부에서 `invoke` 호출

```python
# 이 방식은 HITL에서 동작 안 함
@tool
def library_tool(query: str) -> str:
    return library_agent.invoke({"messages": [...]})["messages"][-1].content
```

**이유**: 도구 함수 내부 `invoke`는 별도 실행 컨텍스트다. 내부에서 `interrupt()`를 호출해도 체크포인트가 부모 그래프 레벨에서 저장되지 않아 재개 시 상태를 찾지 못함.

---

## 5. 공유 상태 설계 (Parent-Child State Isolation)

```python
class SsuAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # merge channel (reducer)
    mcp_session_id: str | None   # 단순 override (마지막 쓰기가 승리)
    active_agent: str | None     # 단순 override
    pending_action: dict | None  # 단순 override
```

| 필드 | 채널 유형 | 소유자 | 설명 |
|------|-----------|--------|------|
| `messages` | `add_messages` reducer | 전체 | 모든 에이전트가 append, ID 기반 중복 제거 |
| `mcp_session_id` | plain override | FastAPI (요청마다 갱신) | ssuMCP 사설 auth 토큰 |
| `active_agent` | plain override | post_supervisor (set), 서브에이전트 (clear) | 현재 활성 에이전트 추적 |
| `pending_action` | plain override | check_approval_node (clear on completion) | 미사용 (향후 다중 동시 prepare 대응용) |

> **갱신 (2026-07-02)**: 위 `pending_action` 필드는 어디서도 읽히지 않는 예약 필드였고(쓰기도 `None`뿐), dead code로 상태 스키마에서 제거됐다 — HITL 감지는 `messages` 스캔(`_extract_action_id`)으로 동작한다.

**`add_messages` reducer가 필요한 이유**: 수퍼바이저와 세 서브에이전트 모두 `messages`에 쓴다. 단순 override(`messages: list[BaseMessage]`)를 쓰면 나중에 실행된 에이전트가 이전 대화 기록을 덮어쓴다.

---

## 6. thread_id ↔ mcp_session_id 라이프사이클

두 ID는 의도적으로 분리되어 있다.

| | `thread_id` | `mcp_session_id` |
|---|---|---|
| 역할 | LangGraph SQLite 체크포인트 키 | ssuMCP 사설 도구 인증 토큰 |
| 생명주기 | 클라이언트 세션 전체 (대화 지속성) | ssuMCP 로그인 세션 (최대 7일, 재발급 가능) |
| 저장 위치 | `configurable.thread_id` | `SsuAgentState.mcp_session_id` |
| 교체 시나리오 | 새 대화 시작 | 사용자가 재로그인하거나 토큰 만료 후 refresh |

FastAPI 요청마다 최신 `mcp_session_id`를 state에 주입하므로, 동일한 thread_id를 유지하면서 mcp_session_id만 교체 가능하다.

**SQLite 체크포인터 라이프사이클**:
- `SqliteSaver.from_conn_string(path)` → context manager를 반환 (SqliteSaver 인스턴스가 아님)
- FastAPI lifespan에서 `with` 블록으로 앱 전체 생애 동안 연결 유지
- 연결이 닫히면 HITL 재개 시 체크포인트 로드 실패 → 반드시 lifespan 안에서 관리

---

## 7. SSE 스트리밍 이벤트 필터링 (Gemini 제안 반영)

`astream_events(version="v2")`로 다음 이벤트만 필터링:

| LangGraph 이벤트 | SSE 페이로드 | 용도 |
|---|---|---|
| `on_chat_model_stream` | `{type:"text", content:"..."}` | 토큰별 LLM 출력 |
| `on_tool_start` (name: `transfer_to_*`) | `{type:"handoff", agent:"library", message:"도서관 에이전트로 전환 중..."}` | 에이전트 전환 UX |
| `on_tool_start` (other) | `{type:"tool", name:"..."}` | 디버그용 도구 호출 |
| `on_chain_stream` 청크의 `__interrupt__` | `{type:"interrupt", data:{...}}` | HITL 승인 요청 (⚠️ `on_interrupt` 이벤트 아님 — 위 3절 정정 참조) |

나머지 `on_chain_*`(인터럽트 없는 청크), `on_retriever_*` 이벤트는 필터링 제외 — SSE 노이즈 방지.

---

## 8. 구현 결과

| 항목 | 내용 |
|------|------|
| 테스트 | 19/19 통과 (`pytest tests/ -v`) |
| 핵심 파일 | `ssu_agent/supervisor/graph.py`, `agents/library.py`, `main.py` |
| LangGraph 버전 | 1.2.4 (확인: `importlib.metadata.version('langgraph')`) |
| 체크포인터 | `langgraph-checkpoint-sqlite 3.1.0` (SQLite, 앱 lifespan 관리) |

---

## 9. 예상 면접 질문

1. **"수퍼바이저 패턴에서 핸드오프를 어떻게 구현했나요? Command를 반환하는 도구를 왜 안 썼나요?"**  
   → LangGraph 1.2.4의 `create_react_agent`는 도구 반환값을 `Command`로 처리하지 않는다. 라우팅 도구가 `"ROUTE_TO:X"` 마커 문자열을 반환하고, `post_supervisor` 노드가 이를 스캔해 `Command(goto=X)`를 반환하는 패턴을 직접 설계했다.

2. **"HITL을 구현할 때 `interrupt()`를 어디에 놓아야 하는지, 그리고 왜 그 위치여야 하는지 설명해주세요."**  
   → LangGraph는 노드 경계에서 상태를 체크포인트한다. 라우터 함수(conditional edge)는 체크포인트 경계가 아니므로 거기서 `interrupt()`를 호출하면 상태 저장 없이 실패한다. `interrupt()`는 반드시 `add_node`로 등록된 노드 함수 안에서 호출해야 한다.

3. **"thread_id와 mcp_session_id를 분리한 이유는 무엇인가요?"**  
   → `thread_id`는 LangGraph의 대화 지속성 키로 클라이언트당 고정된다. `mcp_session_id`는 외부 시스템(ssuMCP)의 인증 토큰으로 만료·재발급이 가능하다. 동일 대화(thread)에서 재로그인 후 세션만 갱신하는 시나리오를 지원하려면 두 값을 독립적으로 관리해야 한다.
