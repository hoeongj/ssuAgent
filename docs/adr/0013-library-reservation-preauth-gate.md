# ADR 0013 - 도서관 좌석 예약 사전 인증 게이트와 resume 세션 최신화

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-11 |
| 상태 | Accepted |
| 범위 | `ssu_agent/agents/library.py`, `ssu_agent/main.py`, `ssu_agent/supervisor/state.py` |
| 관련 | [ADR 0012](0012-supervisor-narration-suppression.md) |

## 배경

도서관 좌석 예약은 ssuMCP의 private tool이 최종적으로 인증을 검사한다. 기존
ssuAgent도 `prepare_*` tool 결과가 `AUTH_REQUIRED`이면 고정 로그인 안내를 반환하는
결정적 guard를 갖고 있었다.

하지만 약한 LLM이 예약 요청에서 tool call 자체를 만들지 못하면 이 guard까지
도달하지 못한다. 그 결과 사용자는 실제 예약이 되지 않았는데도 "전달했습니다" 같은
라우팅 문장, 빈 응답, 또는 모호한 안내를 받을 수 있었다.

프론트엔드 ssuAI는 이제 `/agent/stream`과 `/agent/resume` 요청에
`library_connected` boolean을 보낸다. 이 값은 프론트엔드
`useLibraryAuth().isConnected` 상태를 반영한 client-asserted hint다. 도서관 세션
토큰은 브라우저로 전달되지 않으므로, ssuAgent가 LLM 실행 전에 사용할 수 있는 가장
이른 신호는 이 boolean뿐이다.

별도로 HITL 승인 resume에서는 원래 `/agent/stream` 시점의 `mcp_session_id`가
checkpoint state에 저장된다. 사용자가 prepare와 confirm 사이에 다시 로그인하면
resume 요청의 `mcp_session_id`가 더 최신인데도, `check_approval_node`가 stale state를
읽어 confirm을 호출할 수 있었다.

## 결정

도서관 에이전트는 LLM 호출 전에 마지막 `HumanMessage`에서 좌석 예약 의도를 정규식으로
검사한다. 예약 의도이고 `mcp_session_id`가 없거나 `library_connected`가 false이면
LLM과 tool을 호출하지 않고 고정 문구로 도서관 로그인을 안내한다.

이 사전 게이트는 보안 경계가 아니다. `library_connected`는 서버가 검증한 값이 아니라
클라이언트가 주장한 UX hint이기 때문이다. 클라이언트가 오래된 값을 보내거나 잘못된
값을 보내도 실제 예약 권한은 ssuMCP가 private tool에서 검사한다.

따라서 기존 `AUTH_REQUIRED` guard는 유지한다. 사전 게이트는 흔한 미로그인 예약 요청을
LLM 전에 끊어 사용자 경험을 안정화하고 비용을 줄이는 역할만 한다. 서버-side enforcement
backstop은 계속 ssuMCP의 `AUTH_REQUIRED` 결과와 ssuAgent의 후처리 guard다.

`/agent/resume`은 `Command(resume=...)`를 스트리밍하기 전에 LangGraph
`aupdate_state(config, {"mcp_session_id": ..., "library_connected": ...})`를 호출한다.
그리고 `aupdate_state`가 반환한 config를 이어지는 resume stream에 사용한다. 이렇게
하면 승인 노드가 원래 prepare 시점의 stale session이 아니라 resume 요청에 담긴 최신
세션 상태를 읽는다.

## 결과

- 미로그인 또는 도서관 미연결 상태에서 좌석 예약 의도가 분명하면 LLM을 호출하지 않는다.
- 좌석 현황, 도서 검색, 위치 질문 같은 비예약 요청은 기존 도서관 에이전트 흐름을 탄다.
- `library_connected`가 틀렸거나 오래된 경우에도 ssuMCP `AUTH_REQUIRED`가 실제 권한
  검사를 담당한다.
- prepare와 confirm 사이에 `mcp_session_id`가 회전해도 resume 요청의 최신 값으로
  confirm을 호출한다.

## 추가 기록: 미로그인 공개 조회 prompt 보강

운영에서 미로그인 사용자의 공개 좌석 현황(빈자리) 질문이 기존 "가능한 범위에서"
지시 아래 로그인 안내로 빠지는 사례가 관찰됐다. 예약 경로는 위 사전 게이트가 이미
결정적으로 처리하므로, 이 수정은 코드 로직 변경 없이 미로그인 prompt만 보강했다.
좌석 현황·도서 검색·시설/학사일정/공지 같은 공개 조회는 반드시 공개 읽기 도구를
호출해 실제 결과로 답하고, 로그인 안내로 돌리지 말라고 명시했다.

무료 LLM rate limit 또는 provider fallback은 관찰을 혼동할 수 있으므로, 무료 LLM이
rate-limited되지 않은 상태에서 재검증한다.

## 거부한 대안

### `library_connected`를 인증 근거로 사용

이 값은 클라이언트가 보낸 boolean일 뿐이다. 서버가 검증한 도서관 세션 토큰이 아니므로
예약 허용 여부를 결정하는 근거로 쓰면 안 된다.

### LLM prompt만 보강

prompt 보강은 tool call을 만들지 못하는 약한 모델 실패를 막지 못한다. 이번 문제는
LLM이 실행되기 전에 결정적으로 알 수 있는 실패 조건이 있으므로, 코드 레벨 게이트가 더
작고 재현 가능하다.

### resume payload만 확장

`interrupt()` 이후 `check_approval_node`는 resume payload가 아니라 checkpointed state의
`mcp_session_id`를 읽어 confirm tool에 전달한다. 따라서 payload에 최신 값을 넣는 것만으로
는 부족하고, resume 전에 graph state 자체를 업데이트해야 한다.

## 후속 수정 (2026-07-11)

같은 날 운영에서 도서관에는 로그인된 사용자가 u-SAINT 점검일에 사전 게이트에 걸려
"도서관 탭에서 로그인"하라는 오안내를 받는 사고가 있었다. 당시 `library_connected`는
true였지만, ssuAI의 SAINT-gated 세션 발급이 u-SAINT 점검으로 막혀 `mcp_session_id`가
없었다.

이 ADR은 `mcp_session_id`의 유일한 클라이언트 발급 경로가 SAINT 인증에 결합되어 있다는
숨은 결합을 놓쳤다. 구체적으로 ssuAI ChatPanel의 세션 발급과 ssuMCP web-session JWT가
필수 경로였기 때문에, 도서관 로그인 자체는 살아 있어도 채팅 세션 연결이 만들어지지
않을 수 있었다.

수정 범위는 세 서비스로 나눴다. ssuMCP는 web-session JWT를 선택화하고, ssuAI는
library-only 세션 발급과 재발급을 처리하며, ssuAgent는 이 문서의 사전 게이트 문구를
결핍 신호별로 분리한다. 도서관 로그인이 없으면 기존 도서관 탭 로그인 안내를 유지하고,
도서관 로그인은 확인됐지만 `mcp_session_id`만 없으면 잠시 후 재시도 또는 새로고침을
안내한다.

게이트는 유지한다. 흔한 미로그인 좌석 예약 요청을 LLM 전에 결정적으로 차단해 약한
모델이 tool call 없이 성공한 것처럼 답하는 실패를 줄이는 목적은 여전히 유효하다. 또한
이 게이트는 계속 UX 안정화 장치일 뿐이고, 실제 권한 검사는 ssuMCP private tool과
`AUTH_REQUIRED` 후처리 guard가 담당한다.

검토했지만 기각한 대안은 게이트에서 `mcp_session_id` 조건을 제거하는 것이다. 세션이
없으면 private tool 인증이 불가능하므로, LLM을 통과시켜도 `start_auth` 중복 로그인을
강제하게 된다. 이는 도서관 로그인 사용자를 더 정확히 안내하지 못하고, 불필요한 tool
호출과 중복 인증 흐름만 만든다.
