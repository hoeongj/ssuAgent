# ADR 0016 - MCP content-block 언랩으로 도서관 예약 HITL 승인 복구

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-12 |
| 상태 | Accepted |
| 범위 | `ssu_agent/tool_results.py` (신규), `ssu_agent/agents/library.py`, `ssu_agent/agents/react_loop.py`, `ssu_agent/llm_factory.py`, `ssu_agent/main.py` |
| 관련 | [ADR 0001](0001-supervisor-architecture.md), [ADR 0013](0013-library-reservation-preauth-gate.md), [ADR 0015](0015-optional-anthropic-provider.md) |

## 배경

프로덕션에서 `prepare_reserve_library_seat`가 ssuMCP 쪽에서는 정상적으로 성공(액션
생성)했는데도, 승인 카드(HITL interrupt)가 한 번도 뜨지 않고 스트림이 그냥
`[done]`으로 끝나는 사건이 있었다. 사용자 입장에서는 요청이 조용히 사라진 것처럼
보였고, 서버에 남은 PENDING 액션은 TTL 이후 만료됐다.

### 틀린 가설들

1. **PENDING 액션 잔존/충돌** — 이전에 처리되지 않은 PENDING 액션이 남아 있어서
   confirm 대상이 꼬였을 것이라는 가설. ssuMCP `ActionService`/`ConfirmActionMcpTool`
   코드를 확인한 결과 액션 생성과 조회 로직 자체는 정상이었고, 애초에
   `check_approval_node`가 interrupt에 도달하지 못해 confirm 호출 자체가 일어나지
   않고 있었다 — 액션 잔존과는 무관했다.
2. **SSE emission 누락** — `main._stream_graph`가 `on_chain_stream` 청크의
   `__interrupt__`를 못 잡는 것 아니냐는 가설(ADR 0009/0014 계열 이슈와 유사 패턴).
   `test_stream_interrupt.py`가 이 경로를 이미 실제 langgraph interrupt로
   회귀 테스트하고 있었고, 재현 결과 SSE 레이어는 정상이었다 — 애초에
   `check_approval_node`로 라우팅되지 않아 interrupt() 자체가 호출되지 않고 있었다.

### 실제 원인

`langchain_mcp_adapters`가 만드는 MCP tool은 전부
`StructuredTool(response_format="content_and_artifact")`다. 그런데
`agents/library.py`의 agent_node와 `agents/react_loop.py`의 `_run_tool_call`은 tool을
`matched.ainvoke(tc.get("args", {}), config=config)`처럼 **순수 args dict**로
호출한다 — LangChain의 `"type": "tool_call"` 래퍼(및 그 안의 `tool_call_id`)가 없는
호출이다. `tool_call_id`가 없으면 `StructuredTool._format_output`이 artifact를 붙일
`ToolMessage`가 없어, content/artifact 튜플 대신 **content-block 리스트를 그대로
반환**한다:

```python
[{"type": "text", "text": "{\"status\":\"OK\",...,\"data\":{\"actionId\":42}}"}]
```

기존 코드는 `content = result if isinstance(result, str) else json.dumps(result,
ensure_ascii=False)`로 처리했는데, `result`가 이미 위 리스트이므로 `json.dumps`가
**리스트 자체를 문자열화**해 다음과 같은 이중 이스케이프 문자열을 만든다:

```
'[{"type": "text", "text": "{\\"status\\": \\"OK\\", ...}"}]'
```

`_extract_action_id`와 agent_node의 `hitl_triggered` 체크는 둘 다
`json.loads(content)` 결과가 dict인지 확인하는데, 위 문자열을 파싱하면 **list**가
나온다. `isinstance(data, dict)`가 항상 False가 되어 `interrupt()`가 있는
`check_approval_node`로 라우팅되지 않고 `done_node`로 빠진다 — 서버 쪽 prepare는
성공했는데 클라이언트는 승인 요청을 영영 못 받는 정확한 그 증상이다.

기존 테스트가 이를 못 잡은 이유: `tests/test_library_agent.py` 등 모든 테스트가
`@tool` 데코레이터로 만든 plain 함수를 mock으로 썼고, 이런 tool은
`response_format="content_and_artifact"`가 아니라 그냥 문자열을 반환하므로 위 버그
경로를 전혀 타지 않는다. (`ssu_agent/tool_results.py` 도입 스크립트로 실제
`StructuredTool(response_format="content_and_artifact")`를 bare-args로 invoke해
raw list가 돌아오는 것과, 언랩 전/후 `hitl_triggered` 값이 각각 False/True로 갈리는
것을 직접 확인했다 — `tests/test_library_agent.py`의
`test_library_agent_interrupt_on_real_mcp_content_block_shape`가 이 실제 shape를
재현하는 회귀 테스트다.)

## 결정

### 대안 A: 모든 호출부에서 완전한 ToolCall dict로 invoke

`matched.ainvoke({"name": ..., "args": ..., "id": ..., "type": "tool_call"})`처럼
호출하면 LangChain이 `tool_call_id`를 갖게 되어 `_format_output`이 정상적으로
`ToolMessage`를 만들고 artifact를 붙인다. 이론적으로는 가장 "정석"에 가까운
수정이다.

**기각 사유**: agent_node/react_loop 양쪽의 tool-call 처리 루프, 특히 tool
not-found 분기의 수동 `ToolMessage` 생성, 그리고 향후 다른 서브에이전트가 같은
패턴을 재사용할 가능성까지 모두 건드려야 한다 — id 배관(plumbing)이 넓게 퍼져 있어
blast radius가 크고, 이번 장애의 본질(문자열이어야 할 자리에 list가 들어옴)과 직접
관련 없는 부분까지 리스크에 노출시킨다.

### 대안 B (채택): tool-result 경계에서 언랩

`tool_result_to_text(result)` 헬�퍼를 하나 만들어 모든 tool-invoke 직후, 결과를
`ToolMessage.content`나 `json.loads` 대상으로 쓰기 **직전**에 통과시킨다. list면
`type == "text"`인 블록들의 `text`를 이어붙이고(dict 블록과 `.text` 속성을 가진
객체 모두 허용), 텍스트를 하나도 못 뽑으면 `json.dumps(result)`로 폴백한다. str/dict
등 나머지 타입은 기존과 동일하게 처리한다.

**선택 이유**: 손대는 경계(seam)가 하나뿐이다 — id 배관을 바꿀 필요가 없고, 버그의
실제 위치(list가 문자열이어야 할 자리로 새어 들어옴)에서 정확히 막는다. 부수 효과로
LLM에게 보이는 `ToolMessage.content`도 이스케이프된 wrapper 노이즈 없이 실제 JSON
페이로드만 남아 컨텍스트가 더 깨끗해진다. 적용 지점: `agents/library.py` agent_node
tool-call 루프, `agents/react_loop.py`의 `_run_tool_call`, 그리고 방어적으로
`_extract_action_id`(오래된 checkpoint에 언랩 전 list가 남아 있을 경우 대비).

## 부수 수정: confirm_action 강화

`check_approval_node`가 `confirm_tool.ainvoke({"mcp_session_id": ...})`만 호출하고
`action_id`를 넘기지 않고 있었다. ssuMCP `ConfirmActionMcpTool.confirmAction`의
실제 파라미터를 소스로 확인한 결과 인자명은 `action_id`(`Long`, optional)이고, 생략
시 "현재 대기 중인 단일 액션"을 확정한다 — 여러 PENDING 액션이 동시에 존재하면(예:
서로 다른 좌석을 연속 prepare) 모호해진다. 서버가 이미 추출해 승인 페이로드에 실은
`action["action_id"]`(클라이언트 resume payload가 아니라)를 그대로 confirm에
전달하도록 고쳤다 — 승인 당시 서버가 확정했던 그 액션만 confirm되도록 보장한다.

또한 기존 코드는 confirm 결과를 검사하지 않고 항상 `"예약 확정 완료: {result}"`를
보여줬다. `McpPrivateToolResponse` 구조를 확인하니 **confirm_action은 실행 여부와
무관하게 항상 `status == "OK"`를 반환한다** — "대기 중인 액션이 없습니다.",
"확정 대기 중인 액션이 여러 개입니다...", "액션이 만료됐습니다..." 같은 미실행
notice조차 `status: "OK"` + 문자열 `data`로 온다. 즉 `status == "OK"` 하나로는
"진짜 confirm이 실행됐는지"를 구분할 수 없고, `data`의 실제 문구만이 유일한 신호다.
(참고: 처음에는 "실행된 confirm은 구조화된 payload를 반환한다"고 가정했으나,
`ConfirmActionMcpTool` 소스 확인 결과 모든 분기 — 성공/미실행/부분 실패 — 가
`data`에 사람이 읽는 **문자열** 메시지를 담는다는 것이 드러나 이 가정과 달랐다.
그래서 구조화 payload 유무가 아니라, 백엔드의 알려진 미실행 notice —
"대기 중인 액션이 없습니다", "확정 대기 중인 액션이 여러 개입니다",
"지정한 action_id에 해당하는 대기 액션이 없습니다", "액션이 만료됐습니다",
"지원하지 않는 대기 액션" — 와의 문자열 매칭으로 실행 여부를 가른다.)
`_confirm_result_message`는 3단계로 판별한다:

1. `data`가 미실행 notice 중 하나와 매치 → 그 문구를 그대로 보여준다(아무것도
   실행되지 않음).
2. `data`가 **비동기 접수(accepted-async)** 문구와 매치("접수했습니다" 또는
   "intentId=") → 백엔드 원문을 그대로 보여주고 절대 "예약 확정 완료"라고 하지
   않는다. `ConfirmActionMcpTool.acceptedReservationResponse`(ADR 0086/C1)에 따라
   **좌석 예약 confirm은 intent 큐에 접수만 하고 즉시 반환**하며, 비동기 worker가
   이후에 실패할 수 있다(좌석 선점, upstream 타임아웃 등). 접수 시점에 "확정
   완료"라고 말하면 worker 실패 시 거짓말이 된다. 백엔드 접수 문구가 이미
   intentId와 `get_library_wait_status`로 최종 결과를 확인하는 방법을 안내하므로
   원문 그대로가 가장 정직하다. (동기 실행인 반납/이석의 완료 문구는 기존대로
   "예약 확정 완료:" 라벨을 유지한다.)
3. 그 외 → 동기 실행 완료(반납/이석)로 보고 `"예약 확정 완료: {data}"`.

`status != "OK"`(예: prepare와 confirm 사이에 세션이 만료돼 AUTH_REQUIRED가 낀
경우)면 raw JSON을 덤프하는 대신 `McpPrivateToolResponse`의 `userMessage` 필드를
우선 노출한다(AUTH_REQUIRED의 `userMessage`는 loginUrl을 이미 본문에 포함하며,
포함되지 않은 경우에만 `loginUrl`을 덧붙인다). `userMessage`가 없으면 기존 raw-text
폴백을 유지한다. 파싱 실패 시에도 백엔드 원문을 그대로 노출해 절대 거짓으로
"완료"라고 말하지 않는다.

## 부수 수정: actionId=0 no-op sentinel은 HITL 게이트 제외

ssuMCP의 prepare 3종은 `LibraryPrepareResult(0L, message)` — **actionId=0** — 를
명시적 no-op sentinel로 반환한다(소스 확인: `LibraryReservationMcpTool` "이미 ...
좌석 예약 중입니다", `LibraryCancelMcpTool` "현재 예약된 좌석이 없습니다.",
`LibrarySwapMcpTool` "현재 예약된 좌석이 없습니다. prepare_reserve...를
사용하세요."). 이때 PENDING 액션은 생성되지 않는다.

그런데 두 HITL 게이트(`_extract_action_id`와 agent_node의 인라인 `hitl_triggered`
체크)는 모두 키 **존재**(`"actionId" in inner`)만 검사했다. actionId=0에도 승인
카드가 잘못 떠서, 사용자가 승인하면 `confirm_action(action_id=0)` → "지정한
action_id에 해당하는 대기 액션이 없습니다"가 되고, 정작 유용한 안내 문구("자리를
바꾸려면 prepare_swap...을 사용하세요")는 승인 플로우에 먹혀 사라졌다.

수정: `_pending_action_id(value)` 헬퍼로 두 게이트를 통일 — `isinstance(value,
bool)`이 아니고(`bool`은 `int`의 서브클래스라 `True == 1`이 통과해버림)
`isinstance(value, int)`이며 `value > 0`일 때만 PENDING 액션으로 취급한다.
actionId=0이면 게이트가 열리지 않고, sentinel의 ToolMessage는 history에 그대로
남아 LLM이 안내 문구를 사용자에게 전달한다(이 PR 이전과 동일한 동작).

## 부수 수정: LLM provider fallback 로깅/재시도/메시지

- `agents/library.py`의 provider fallback 루프는 `last_exc`만 갱신하고 실패를
  로깅하지 않아, prod에서 quota/스키마 오류를 진단할 때 앞선(우선순위 높은)
  provider가 왜 실패했는지 알 수 없었다. `agents/react_loop.py:196-201`의 기존
  로깅 스타일(`provider=%s failed: %s: %s`)을 그대로 따라 `logger.warning`을
  추가했다.
- `llm_factory.py`의 `ChatAnthropic`은 `max_retries=1`이었다. 저가 유료 키가 평소
  트래픽에서도 429를 자주 반환하는 상황이라 재시도 여지를 `max_retries=3`으로
  늘렸다 — Anthropic SDK가 재시도 사이 `Retry-After`를 존중하므로 일시적 rate
  limit를 무료 provider로 즉시 넘어가지 않고 자체적으로 흡수할 수 있다.
- `main.py`의 capacity-error 메시지가 `"지금 무료 AI 사용량이 잠시 초과됐어요..."`로
  고정돼 있어, ADR 0015로 유료 Anthropic 키가 첫 번째 provider로 활성화된
  상태에서도 "무료"라고 잘못 말하는 문제가 있었다. provider tier와 무관하게 항상
  참인 `"지금 AI 요청이 많아 잠시 처리가 어려워요. 잠시 후 다시 시도해 주세요."`로
  교체했다.

## 검증

`tests/test_tool_results.py`가 `tool_result_to_text`의 str/list(단일·다중
block)/빈 리스트 폴백/dict/`.text` 속성 객체 케이스를 단위 테스트한다.
`tests/test_library_agent.py`의
`test_library_agent_interrupt_on_real_mcp_content_block_shape`는 실제
`StructuredTool(response_format="content_and_artifact")` shape로 HITL이 트리거되는
것을 검증하며, 수정 전 코드 경로에서는 통과할 수 없다는 것을 스크립트로 직접 재현해
확인했다(리버트 없이 — 언랩 전/후 로직을 나란히 돌려 `hitl_triggered`가
False→True로 바뀌는 것을 관찰). `test_check_approval_confirm_called_with_action_id`와
`test_check_approval_non_executed_result_not_reported_as_complete`가 confirm 호출
인자와 결과 메시지 정직성을 각각 검증한다.
actionId=0 sentinel은 `test_extract_action_id_ignores_zero_noop_sentinel`(단위,
bool/문자열/음수 거부 포함)과
`test_library_agent_no_interrupt_on_zero_noop_sentinel`(통합: interrupt 미발생 +
sentinel ToolMessage가 history에 잔존 + LLM 전달 답변이 최종 메시지)이 커버한다.
비동기 접수는 `test_confirm_result_message_async_accept_not_labeled_complete`가,
non-OK `userMessage` 노출은
`test_confirm_result_message_non_ok_surfaces_user_message` /
`test_confirm_result_message_non_ok_appends_login_url_when_missing`이 커버한다.
