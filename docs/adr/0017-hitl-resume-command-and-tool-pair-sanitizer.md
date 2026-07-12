# ADR 0017 - HITL resume는 Command로 원자 처리하고 tool pair는 호출 경계에서 정리

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-12 |
| 상태 | Accepted |
| 범위 | `ssu_agent/main.py`, `ssu_agent/agents/library.py`, `ssu_agent/agents/react_loop.py`, `ssu_agent/supervisor/graph.py`, `ssu_agent/tool_results.py` |
| 관련 | [ADR 0013](0013-library-reservation-preauth-gate.md), [ADR 0016](0016-mcp-content-block-hitl-unwrap.md) |

## 배경

운영에서 사용자가 도서관 예약 승인 카드를 승인했지만 `/agent/resume`은 200을 반환한
뒤 추가 로그 없이 종료됐다. `confirm_action`은 ssuMCP까지 도달하지 않았고,
`ActionAudit` row는 PENDING 상태로 남았다가 TTL 만료로 사라졌다. 브라우저로도 confirm
결과나 취소/완료 SSE가 렌더링되지 않았다.

같은 시간대에 Anthropic provider가 다음 형태의 400을 반환하는 문제도 관찰됐다.
assistant tool call과 tool result의 짝이 multi-turn history 안에서 분리되어,
이전 assistant message에 없는 `tool_use_id`를 가진 tool result가 전달된 것이다. llama
계열 provider는 이를 느슨하게 받아들였기 때문에 유료 provider를 붙이기 전까지 잠복해
있었다.

## 원인

`/agent/resume`은 기존에 다음 순서로 동작했다.

1. `config = {"configurable": {"thread_id": ...}}`
2. `await _graph.aupdate_state(config, {"mcp_session_id": ..., "library_connected": ...})`
3. 반환된 config로 `_stream_graph(Command(resume=payload), config)` 실행

LangGraph에서 interrupt로 멈춘 thread에 `update_state`를 먼저 호출하면 새 checkpoint가
fork되고 `next`가 다시 계산된다. 그 결과 원래 pending 상태였던 interrupted task가
resume 대상에서 빠지고, 이어지는 `Command(resume=...)`은 재개할 작업을 찾지 못한 채
조용히 끝난다.

이 문제는 ADR 0016의 content-block 언랩 수정 전에는 드러나지 않았다. 당시에는
`prepare_*` 결과에서 `actionId`를 추출하지 못해 `interrupt()` 자체에 도달하지 못했다.
ADR 0016 수정으로 interrupt가 실제로 발생하기 시작하면서, 그 다음 단계인 resume 경로의
checkpoint fork 문제가 처음 운영 경로에 노출됐다.

tool pair 문제는 checkpoint history가 여러 provider 호출에 재사용되는 동안 message
filtering이나 과거 state shape 때문에 assistant tool call과 ToolMessage가 항상
인접한 쌍으로 남는다는 보장이 없었던 것이 원인이다. Anthropic은 이 불변식을 엄격히
검증한다.

## 결정

`/agent/resume`에서는 resume 전에 `aupdate_state`를 절대 호출하지 않는다. endpoint와
회귀 테스트가 같은 코드를 쓰도록 `build_resume_command(req)`를 두고, 다음 형태의
Command만 스트리밍한다.

```python
Command(
    resume=resume_payload,
    update={
        "mcp_session_id": req.mcp_session_id,
        "library_connected": req.library_connected,
    },
)
```

설치된 LangGraph의 `Command`는 `graph`, `update`, `resume`, `goto` 필드를 지원한다.
회귀 테스트 결과 `Command(update=..., resume=...)`는 interrupted subgraph task를
정상적으로 재개한다. 다만 top-level graph에 보낸 `update`가 이미 멈춰 있는 child
subgraph의 local checkpoint를 다시 쓰지는 않는다. 그래서 `check_approval_node`는
FastAPI에서 검증된 resume payload의 `mcp_session_id`를 우선 사용하고, 없으면 기존
checkpoint state로 fallback한다. `action_id`는 계속 client payload가 아니라 서버가
prepare 결과에서 추출한 pending action을 사용한다.

또한 confirm/cancel처럼 코드가 생성한 approval node 응답은 chat model stream event를
발생시키지 않는다. `_stream_graph`는 `agent` node뿐 아니라 `check_approval` node의
새 assistant message도 SSE text로 내보내도록 했다.

tool pair 정리는 checkpoint를 수정하지 않고 provider 호출 직전에만 수행한다.
`sanitize_tool_pairing(messages)`는 다음 규칙을 보장한다.

1. contiguous ToolMessage group의 각 `tool_call_id`는 바로 앞 assistant message의
   `tool_calls` 안에 있어야 한다. 없으면 해당 ToolMessage를 drop한다.
2. assistant message가 `tool_calls`를 선언하면 다음 non-tool message가 나오기 전에
   모든 tool call result가 있어야 한다. 없는 call은 assistant message의 copy에서
   제거한다.
3. tool call을 제거한 뒤 content도 비어 있으면 그 assistant message는 drop한다.
4. 입력 list와 message object는 mutate하지 않는다. 수정이 필요한 message만 copy한다.

적용 위치는 provider invocation boundary다. library agent, shared react loop, supervisor
prebuilt react invocation 모두 같은 sanitizer를 통과한다. checkpoint history 자체를
정리하지 않는 이유는 저장된 대화 상태를 사후 변경하면 resume/debug 재현성이 떨어지기
때문이다.

## window slicing 점검

`state["messages"][-N:]` 형태의 고정 window가 tool pair를 자를 가능성을 점검했다.
발견된 slice는 두 곳이다.

- `ssu_agent/supervisor/graph.py`: `_post_supervisor`가 최근 8개 message에서
  `ROUTE_TO:*` marker를 찾는다.
- `ssu_agent/agents/library.py`: `_extract_action_id`가 최근 10개 ToolMessage에서
  pending `actionId`를 찾는다.

둘 다 provider history를 만드는 window가 아니라 라우팅/승인 대상 탐색용 scan이다.
따라서 pair-aware window extension은 적용하지 않았고, provider 호출 경계의 sanitizer로
엄격 provider 불변식을 맞춘다.

## 결과

- resume 전에 checkpoint를 fork하지 않으므로 HITL 승인/거절이 실제 interrupted task로
  돌아간다.
- 승인 경로는 `confirm_action`을 호출하고, 거절 경로는 confirm 없이 취소 문구를 SSE로
  전달한다.
- Anthropic에 전달되는 history는 orphan ToolMessage와 dangling tool call을 포함하지
  않는다.
- llama 계열 provider의 느슨한 허용에 기대지 않고 provider boundary에서 동일한 message
  invariant를 강제한다.
