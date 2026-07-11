# ADR 0014 - MCP tool-call retry and deep health endpoint

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-11 |
| 상태 | Accepted |
| 범위 | `ssu_agent/mcp_client.py`, `ssu_agent/supervisor/graph.py`, `ssu_agent/main.py` |
| 관련 | [ADR 0009](0009-agent-edge-hardening.md) |

## 배경

ssuMCP 재배포 직후 ssuAgent 챗봇 경로가 한동안 502를 반환하다가, ssuAgent를
수동으로 `kubectl rollout restart`하면 회복되는 사건이 있었다.

초기 가설은 "ssuAgent가 오래 사는 MCP transport session을 들고 있고, ssuMCP의 기존
pod가 죽으면 그 session이 깨진다"였다. 코드 확인 결과 이 가설은 맞지 않다.
`ssu_agent/mcp_client.py`는 `MultiServerMCPClient`와 `transport: streamable_http`를
사용하고, 설치된 `langchain_mcp_adapters` 0.3.0의 `get_tools()`가 만드는 LangChain
tool은 tool 호출마다 `create_session(...)`으로 새 MCP session을 연다. ssuAgent가
startup에서 가져와 장기 보관하는 것은 tool definition 목록이지, MCP HTTP session이
아니다.

따라서 이 ADR은 502의 root cause를 확정하지 않는다. 다음 재발 시 ingress 로그와
ssuAgent 로그를 함께 캡처해서 502가 agent에 도달하기 전 ingress에서 생기는지, agent
event loop/streaming 경로가 멈춘 것인지, 아니면 MCP tool call 중 발생한 예외가 전파된
것인지 구분해야 한다.

## 결정

MCP에서 로드한 각 LangChain `BaseTool`을 얇은 retry wrapper로 감싼다. wrapper는
`.name`, `.description`, `.args_schema`를 원본과 동일하게 유지하므로
`categorise_tools()`의 prefix/exact-name routing은 그대로 동작한다. async tool 호출에서
transport/session 계열 예외가 발생하면 WARNING 로그를 남기고 짧게 대기한 뒤 정확히 한
번만 다시 호출한다.

재시도 대상은 설치된 adapter와 MCP SDK에서 실제로 관찰되는 경로로 제한한다.
`httpx` 연결/읽기/protocol/timeout 계열 transport exception, MCP SDK가 broken stream을
JSON-RPC error로 변환해 올리는 `McpError(CONNECTION_CLOSED)`, request timeout, 그리고
streamable-http session 404 계열인 `Session terminated`/`Session not found`/expired
session 메시지만 재시도한다. tool validation error, adapter의 MCP tool execution error
(`CallToolResult(isError=True)`), ssuMCP의 `AUTH_REQUIRED` 같은 business response는
재시도하지 않는다.

`GET /healthz/deep`를 추가한다. 이 endpoint는 짧은 timeout으로 `client.get_tools()`를
한 번 수행해 ssuMCP와의 lightweight round trip을 확인한다. 성공하면
`{"status": "UP", "mcp": "UP"}`를 200으로 반환하고, 실패하면 예외를 밖으로 내보내지
않고 `{"status": "DEGRADED", "mcp": "DOWN"}`를 503으로 반환한다. 기존 shallow
`GET /health`는 process liveness 신호로 그대로 둔다.

## probe 배선 결정

이번 변경에서는 Helm values와 deployment template을 수정하지 않는다. 현재 chart의
readiness와 liveness는 모두 shallow `/health`를 가리키며, 그대로 유지한다.

liveness를 ssuMCP 같은 downstream dependency에 연결하는 것은 cascading failure
anti-pattern이다. ssuMCP의 짧은 blip이 agent pod 재시작과 crash loop를 만들 수 있고,
이는 원래 장애보다 더 큰 outage로 번진다. liveness는 process 자체가 살아 있는지만
반영해야 한다.

readiness를 deep check에 연결하는 것은 replica가 2개 이상일 때 의미가 있다. 한 pod가
unready가 되어 load balancer에서 빠져도 다른 healthy pod가 트래픽을 받을 수 있기
때문이다. 현재 배포는 single pod(`replicaCount: 1`)이므로 readiness-gating은 부분 502를
전체 503(no endpoints)로 바꿀 가능성이 높고 개선이 아니다.

결론: `/healthz/deep`는 지금 manual diagnosis와 미래 readiness hook으로 제공한다.
HPA 또는 multi-replica 배포가 들어오기 전에는 probe wiring을 바꾸지 않는다.

## 거부한 대안

### persistent reconnecting MCP session 직접 구현

설치된 adapter는 tool invocation마다 fresh session을 여는 설계다. ssuAgent에서 별도
persistent reconnecting session을 유지하려면 library의 per-call lifecycle과 싸우는
재작성에 가깝고, 이번 사건의 root cause도 아직 확정되지 않았다. 따라서 작고 국소적인
retry-once wrapper가 더 낮은 위험의 방어책이다.

### 여러 번 또는 긴 backoff로 재시도

실제 ssuMCP outage 중에 agent가 많은 tool call을 여러 번 재시도하면 downstream load를
증폭한다. rollout 직후의 짧은 transient failure를 흡수하는 목적에 맞게 한 번만
재시도한다.
