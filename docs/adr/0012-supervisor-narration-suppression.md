# ADR 0012 — Supervisor narration suppression by runnable tags

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-11 |
| 상태 | Accepted |
| 범위 | `ssu_agent/supervisor/graph.py`, `ssu_agent/main.py`, `ssu_agent/agents/react_loop.py`, 도메인 에이전트 빈 응답 fallback |
| 관련 | [ADR 0001](0001-supervisor-architecture.md) |

## 배경

수퍼바이저는 부모 그래프의 `supervisor` 노드 안에서 다시
`create_react_agent(...).ainvoke(...)`를 실행한다. 브라우저로 나가는 SSE는
`graph.astream_events(version="v2")`에서 직접 만들어지므로, 수퍼바이저 내부
ReAct 그래프의 채팅 모델 스트림도 그대로 관측된다.

이 구조에서 라우팅 도구(`transfer_to_library_agent` 등)를 호출하는 동안
수퍼바이저 모델이 "도서관 에이전트에게 전달했습니다" 같은 진행 문장을 같이
생성할 수 있다. 이 문장은 사용자가 볼 최종 답변이 아니라 라우팅 부산물이다.
기존 구현은 `metadata.langgraph_node == "supervisor"`인 채팅 스트림만 보류했지만,
실제 이벤트에서 수퍼바이저 내부 모델의 노드명은 `supervisor`가 아니라 내부 ReAct
그래프의 `agent`였다. 그래서 mock 기반 테스트는 통과해도 운영에서는 문장이
그대로 사용자에게 흘러나갔다.

같은 문장이 공유 `messages` 상태에도 남아 있었다. 서브에이전트가 이 문장을
이전 assistant 답변으로 읽으면 이미 응답이 끝난 것으로 판단하고 도구 호출 없이
빈 답변을 내는 경우가 생긴다.

## 관측

이번 변경에 `scripts/dump_stream_metadata.py`를 추가해 실제
`build_supervisor_graph` 결과물을 fake streaming LLM으로 실행했다. 이 스크립트는
mock 이벤트를 만들지 않고, 컴파일된 그래프의 `astream_events(version="v2")`를
그대로 순회한다.

대표 이벤트는 다음 모양이었다. `checkpoint_ns`의 UUID 부분은 실행마다 달라진다.

```text
event=on_chat_model_stream name=_StreamingMessagesListChatModel tags=['seq:step:2', 'supervisor_llm'] langgraph_node=agent checkpoint_ns=supervisor:34639177-196e-1ade-fab6-8d91d4fc6bf1
event=on_tool_start name=transfer_to_library_agent tags=['seq:step:1', 'supervisor_llm'] langgraph_node=tools checkpoint_ns=supervisor:34639177-196e-1ade-fab6-8d91d4fc6bf1
event=on_chat_model_stream name=_StreamingMessagesListChatModel tags=['seq:step:1'] langgraph_node=agent checkpoint_ns=library_agent:6d2bc589-aac5-3e50-e973-7e274811ec52
```

결론:

- `langgraph_node`는 수퍼바이저와 서브에이전트 모두 내부 모델 호출에서 `agent`가
  될 수 있어 식별자로 쓸 수 없다.
- `config.tags`로 넘긴 `supervisor_llm`은 수퍼바이저 내부 모델 스트림과 라우팅
  도구 이벤트까지 전파된다.
- 서브에이전트 내부 모델 스트림은 같은 `langgraph_node=agent`라도
  `supervisor_llm` 태그가 없다.

## 결정

수퍼바이저 노드는 내부 ReAct agent를 호출할 때 기존 runnable config를 보존하면서
`tags=["supervisor_llm"]`를 추가한다. SSE 스트림 필터는 채팅 모델 이벤트의
`metadata.langgraph_node` 대신 top-level `event["tags"]`에
`supervisor_llm`이 있는지 확인한다.

수퍼바이저가 직접 답한 경우에는 기존처럼 보류한 텍스트를 마지막에 내보낸다.
라우팅 도구가 시작된 경우에는 보류 버퍼를 비우고, 그 뒤에 들어오는
수퍼바이저-tagged 텍스트도 최종 flush 대상에서 제외한다. 실제 ReAct 실행에서는
도구 호출 뒤에도 수퍼바이저 모델이 한 번 더 호출될 수 있기 때문에, 이 routed
상태가 없으면 뒤늦은 "전달했습니다" 문장이 끝에 붙어 나간다.

그래프 상태에서는 수퍼바이저가 새로 만든 `AIMessage`의 `name`을 `"supervisor"`로
설정한다. 서브에이전트 앞단의 `drop_routing_messages`는 기존
`transfer_to_*` tool call 및 `ROUTE_TO:*` ToolMessage 제거에 더해,
`name == "supervisor"`인 `AIMessage`를 제거한다. 스트리밍 식별에는 runnable tag를
쓰고, 저장된 대화 정리에는 메시지 name을 쓰는 식으로 두 경로를 분리했다.

## 빈 응답 fallback

빈 응답은 "수퍼바이저 narration을 보고 이미 끝났다고 판단함"이라는 한 원인만으로
발생하지 않는다. 모델이 도구 결과를 해석하지 못했거나, provider가 빈 assistant
메시지를 반환했거나, 약한 모델이 tool call 없이 공백을 낼 수도 있다.

따라서 fallback은 원인 추정이 아니라 최종 assistant 메시지의 content가 비어
있는지로만 판단한다. 도구 호출용 `AIMessage`는 content가 비어 있어도 정상적인
중간 단계이므로 fallback을 적용하지 않는다. 최종 assistant 답변이 빈 문자열 또는
공백뿐일 때만 고정 문구로 교체한다.

## 거부한 대안

### `metadata.langgraph_node` 유지

실제 이벤트에서 수퍼바이저 내부 모델도 `agent`, 서브에이전트 내부 모델도 `agent`로
나온다. 부모 노드 이름이 아니라 내부 ReAct 그래프의 노드 이름이 노출되므로 같은
실수를 반복하게 된다.

### `checkpoint_ns` 문자열 파싱

수퍼바이저 이벤트의 `checkpoint_ns`가 `supervisor:...` 형태라는 점은 관측됐지만,
이 값은 체크포인터/서브그래프 네임스페이스 세부 구현에 가깝다. runnable config
tag는 LangChain/LangGraph가 이벤트에 전파하도록 제공하는 공개 식별 수단이므로
태그가 더 직접적인 계약이다.

### `parent_ids`나 이벤트 계층 탐색

부모 chain을 거슬러 올라가면 수퍼바이저 호출인지 추론할 수 있지만, 스트리밍
핫패스에서 매 이벤트마다 계층을 해석해야 하고 테스트 fixture도 복잡해진다.
명시 태그 하나가 같은 정보를 더 작고 안정적으로 전달한다.

### 특정 no-op 원인 감지

서브에이전트 빈 응답의 원인은 여러 가지이고 provider마다 표현도 다르다. "앞에
수퍼바이저 narration이 있었다" 같은 조건을 감지하려 하면 실제 사용자에게 빈 SSE가
가는 다른 케이스를 놓친다. 사용자가 볼 수 없는 빈 assistant 답변은 항상 실패한
응답이므로 content-empty fallback이 더 안전하다.
