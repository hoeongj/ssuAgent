# ADR 0020 - 개인 학사 조회의 인증 게이트와 내부 세션 은닉

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-16 |
| 상태 | Accepted |
| 범위 | `auth_guard.py`, 학사·LMS·도서관 에이전트, supervisor, SSE stream |
| 관련 | [ADR 0013](0013-library-reservation-preauth-gate.md), [ADR 0014](0014-mcp-connection-resilience.md) |

## 배경과 장애 증상

사용자가 로그인하지 않은 상태에서 "졸업까지 어떤 조건들이 남았어?"라고 물었을 때 학사
에이전트가 내부 MCP 세션 ID를 직접 알려 달라고 요청하고, 실제 채팅 UI에 없는 로그인 버튼을
안내했다. `mcp_session_id`가 없는데도 요청을 LLM에 전달했고, 기존 prompt는 개인 도구를
호출하지 말라는 제약만 두었을 뿐 사용자가 따라야 할 실제 UI 경로를 결정적으로 정하지 않았다.

또한 하나의 MCP 세션에 도서관만 연결된 경우처럼 세션 ID는 있지만 SAINT provider가 없는
상태도 가능하다. 이때 모델이 private tool을 호출하지 않으면 tool 결과 후처리에도 도달하지
않으며, 호출하더라도 `AUTH_REQUIRED`, `INVALID_SESSION` 같은 developer message를 다시 LLM에
주면 로그인 URL이나 세션 값을 사용자에게 노출하는 문장을 합성할 수 있었다. 세션 ID는 서비스
사이에서만 전달해야 하는 내부 인증 값이며 사용자가 채팅에 복사해 넣는 제품 계약이 아니다.

## 결정

명확한 개인 학사 현황 요청은 LLM 호출 전에 분류한다. 졸업·성적·학점·채플·장학·시간표 등
학사 영역이면서 개인 주어 또는 "남은", "현재 성적", "시간표 조회" 같은 현황 표현이 있는 요청에
`mcp_session_id`가 없으면 고정 안내를 즉시 반환한다. 안내는 화면 상단의 연결 패널에서
u-SAINT를 연결한 뒤 같은 질문을 다시 보내도록 하며, 학번·비밀번호와 세션 ID를 채팅에
입력하지 않아도 된다고 명시한다.

"일반 졸업 기준"이나 "조기졸업 요건" 같은 공개 정책 질문은 기존 LLM·공개 정책 도구 흐름을
유지한다. 인증이 없어도 답할 수 있는 질문까지 막지 않기 위해서다.

세션 ID가 있어도 실제 provider 권한은 ssuMCP가 판정한다. 명확한 개인 학사 요청과 모든 LMS
요청은 LLM 전에 `get_auth_status`를 코드에서 직접 호출해 SAINT/LMS provider의 `linked`와
health를 확인한다. library-only 세션이나 만료 상태면 LLM을 실행하지 않는다. 상태 조회 자체가
실패하면 로그인 실패로 단정하지 않고 잠시 후 재시도하라는 별도 고정 문구를 반환한다.
link가 남은 `ERROR`는 인증 만료가 아니라 직전 upstream 실패이므로 현재 사용자 요청에
bounded private invocation 정책을 적용한다. 하나의 요청에서 session-bound tool은 최대 1회만
실행한다. 모델이 같은 turn에 private call을 여러 개 배치하면 실행 전에 거부하고,
한 번 실행한 뒤 추가 private call을 만들어도 재호출하지 않는다. 기존 MCP transport
wrapper의 단일 retry는 하나의 논리적 호출 내에서 그대로 유지한다. 예외, 구조화된
operational failure, 예산 초과 중 하나라도 발생하면 공유 ReAct loop가 고정 서비스
장애 안내를 반환한다. 성공하면 ssuMCP가 health를 `VALID`로 갱신하며 다음 사용자
요청은 provider preflight를 새로 수행한다.

private tool의 `mcp_session_id` 인자는 model-visible schema에서 제거하고 실행 wrapper가 실제
호출 직전에 주입한다. prompt에도 raw 값을 넣지 않는다. 정상 tool 결과의 `mcpSessionId`와 인증
URL·developer 지침은 다음 LLM turn 전에 제거한다. `start_auth`, `logout_*`, `get_auth_status`
같은 인증 lifecycle tool도 모든 에이전트와 supervisor의 모델 도구 목록에서 제외하며 정식
로그인 UX는 ssuAI 상단 연결 패널만 사용한다.

공유 checkpoint의 과거 도서관 turn에는 이전 버전이 남긴 session tool argument, ToolMessage,
로그인 URL이 있을 수 있다. 모든 LLM 경계는 원본 state를 수정하지 않는 복사본에서 인증 lifecycle
tool pair를 제거하고, 세션 인자·결과·URL을 redaction한 뒤 모델에 전달한다. 현재 요청의 handle과
다른 회전 전 값도 “session <value>” 문맥으로 제거한다. 도구 예외 원문도 세션을 포함할 수
있으므로 로그에는 예외 type만 남긴다. 학사·LMS 공유 ReAct loop에서는 예외를
`ToolMessage.status=error`로 표시하고, 이 형식화된 실패 신호를 확인하는 즉시 도메인별 고정 서비스
장애 안내를 반환한다. masked 오류 문자열조차 다음 LLM turn에 전달하지 않으므로 모델이 일반적인
졸업 기준이나 임의 복구 절차를 합성할 수 없다. 예외 원문을 노출하지 않는 경계는 도서관 custom
loop와 supervisor에도 동일하게 적용한다.

checkpoint 전체를 specialist에게 전달하지 않는다. 최신 사용자 요청만 기본 입력으로 쓰고,
직전 완료 답변이 같은 specialist의 답변일 때만 한 턴을 문맥으로 보존한다. specialist가 남기는
`AIMessage.name`을 출처의 기준으로 사용하고, 배포 전 checkpoint 호환에 한해서 기존 표시 prefix를
fallback으로 읽는다. 표시 문구 자체에만 의존하지 않으므로 도서관 로그인 뒤 “로그인했어”라고
답해도 직전 예약 요청을 잃지 않는다. supervisor는 명확한 새 도메인 요청에서 이전 턴을 제거하고,
“자료구조요”, “지난학기요” 같은 짧은 후속 질문에만 직전 완료 턴을 함께 받는다. 과거 sub-agent
답변을 무조건 요약하라는 supervisor 지시는 제거한다.

학사 게이트도 현재 문장만 분류하지 않는다. “지난학기요”, “그 성적은?”처럼 짧고 지시적인
후속문장은 직전 학사 턴이 개인 조회였을 때만 private 의도를 승계한다. “일반 졸업 기준”과
“학사일정”처럼 독립적인 공개 요청은 승계하지 않으며, 일반적인 졸업 학점 질문도 개인 현황으로
오분류하지 않는다.

학사 또는 LMS private tool 결과의 top-level JSON `status`가 `AUTH_REQUIRED`, `NO_SESSION`,
`INVALID_SESSION`, `SESSION_MISMATCH` 중 하나면 공유 ReAct loop가 다음 LLM 합성 단계를 실행하지
않고 고정 연결 안내를 반환한다. 단순 문자열 포함 검사가 아니라 구조화된 status만 검사하므로
공개 정책 본문에 상태명이 등장해도 오탐하지 않는다. 모델의 최종 문장에 내부 세션 ID나
`start_auth`, 존재하지 않는 로그인 버튼 지시, `/api/mcp/auth/.../start` URL이 나타나도 같은
고정 안내로 대체한다. SSE는 모델 token을 보류하는 기존 구조를 이용해 graph가 안전한 고정
안내로 교체한 경우 이전 raw model buffer를 폐기한다. graph 결과만 안전하고 stream 끝에서 원문
URL이 다시 flush되는 경로도 함께 차단한다.

ssuMCP private response의 top-level `status`가 `OK`가 아니면 `retryable=true` 또는
`status`/`code`가 `UPSTREAM_`으로 시작하는 경우만 operational failure로 분류한다.
`NO_PENDING_ACTION`같은 non-retryable domain outcome은 모델이나 terminal formatter가 처리하도록
통과시킨다. 기존 LMS 구현 중 `get_lms_dashboard`, 강의·자료 조회 및 내보내기 준비
도구는 이전 버전에서 API 실패를 `status=OK`와 string `data`로 반환했다. companion ssuMCP
변경은 이를 top-level `UPSTREAM_UNAVAILABLE` 또는 `UPSTREAM_PROTOCOL_CHANGED`로 이전한다.
rolling deployment 하위 호환 경로는 해당 도구 이름과 구버전 서버가 사용하는 정확한 한국어
접두어가 모두 맞을 때만 failure로 분류한다.
일반 `data`, 정책 본문, 사용자 메시지 안의 `UPSTREAM_UNAVAILABLE` 단어는 스캔하지 않는다.

공개 정책 질문은 현재 MCP 세션이 있더라도 private tool과 인증 prompt를 노출하지 않는다.
개인 학사 의도로 분류되고 provider preflight가 `CONNECTED` 또는 `DEGRADED`일 때만 session-bound
private tool을 만든다. `DEGRADED`에는 위 bounded invocation을 적용한다. 상태 계약 누락,
malformed 응답, timeout, non-OK status는 fail-safe로 고정된 상태 확인
실패 안내를 반환한다. unlinked 또는 `EXPIRED` provider는 재연결 안내를 반환한다.

## 기각한 대안

### Prompt만 보강

약한 모델이나 provider fallback이 지시를 어길 수 있고, 이미 관찰된 응답을 결정적으로 막지
못한다. 인증 부재와 `AUTH_REQUIRED`는 코드에서 판정 가능한 상태이므로 실행 경계에서 차단한다.

### `saint_connected` 상태를 새 필수 계약으로 추가

서버가 확인한 provider 상태가 더 풍부한 신호인 것은 맞지만, 현재 ssuAgent를 직접 호출하는
기존 클라이언트는 이 필드를 보내지 않는다. 이번 결함은 기존 `get_auth_status`를 모델 밖에서
호출하는 provider preflight와 구조화된 인증 거부 후처리로 요청 계약을 깨지 않고 닫을 수 있다.
별도 provider 상태를 추가한다면 ssuAI·ssuAgent 요청 계약을 함께 버전 관리하는 후속 결정으로
다룬다.

### 인증 URL 또는 세션 ID를 채팅 본문에 노출

ssuAI에는 상단 연결 패널이라는 정식 인증 경로가 있다. 내부 값 복사를 요구하면 피싱에 취약한
사용 습관을 만들고, 세션 값이 대화 기록에 남으며, 실제 UI 계약과도 어긋난다.

## 검증과 회귀 방지

- 문제 문장 그대로 무세션 요청 시 LLM을 한 번도 호출하지 않고 고정 안내를 반환한다.
- 공개 학사 정책 질문은 무세션에서도 LLM 흐름을 유지한다.
- library-only 세션은 provider preflight에서 LLM을 한 번도 호출하지 않는다.
- private tool schema와 prompt에서 세션 인자를 제거하고, wrapper가 실제 호출에만 주입한다.
- 정상 tool 결과에서도 세션 ID·로그인 URL·developer 지침을 제거한 뒤 모델에 전달한다.
- 과거 도서관 tool call/result와 supervisor 입력에서도 session/auth lifecycle 흔적을 제거한다.
- 새 도메인 요청은 과거 다른 도메인 턴을 model 입력에서 제거하고, 같은 agent의 연속 후속만
  직전 한 턴을 유지한다.
- 도서관 로그인 후속은 메시지 출처로 직전 예약 요청을 유지하고, 학사 후속은 직전 private
  의도만 승계하며 명시적 공개 요청은 승계하지 않는다.
- 학사·LMS 도구 예외는 형식화된 실패 신호로 차단하며 원문과 masked 오류 모두 다음 모델
  turn·checkpoint·SSE에 남기지 않는다.
- `AUTH_REQUIRED`, `NO_SESSION`, `INVALID_SESSION`, `SESSION_MISMATCH`는 구조화된 status로
  차단하고, 정상 데이터 안의 같은 문자열은 차단하지 않는다.
- retryable top-level non-OK와 정확한 legacy LMS `status=OK`/string `data` 오류는 고정
  장애 안내로 차단하고, non-retryable domain outcome과 정상 데이터 속 상태명은 통과시킨다.
- preflight의 누락·malformed·non-OK·timeout 계약은 unavailable, unlinked·expired는 disconnected,
  linked `ERROR`는 요청당 private 호출 예산·배치 차단·재호출 차단으로 검증한다.
- 최종 안내와 실제 SSE에 `MCP`, 세션 ID, 인증 start URL, 존재하지 않는 버튼 표현이 없는지
  검증한다.

남은 위험은 자연어 분류의 경계 사례다. 오분류가 발생해도 실제 개인 데이터 권한은 ssuMCP가
계속 강제하며, provider preflight와 구조화된 인증 거부 후처리 guard가 내부 인증 안내 생성을
차단한다.
