# ADR 0019 - HITL 턴 스코프, 도서관 태그 라우팅, fallback dedup 회귀 수정

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-12 |
| 상태 | Accepted |
| 범위 | `ssu_agent/agents/library.py`, `ssu_agent/supervisor/graph.py`, `ssu_agent/agents/react_loop.py` |
| 관련 | [ADR 0012](0012-supervisor-narration-suppression.md), [ADR 0016](0016-mcp-content-block-hitl-unwrap.md), [ADR 0017](0017-hitl-resume-command-and-tool-pair-sanitizer.md), [ADR 0018](0018-deterministic-library-routing-and-wait-followthrough.md), PR #44 (`5e6a2a8`, `738a1b3`, `ecb07e7`) |

## 배경

예약 파이프라인을 반영한 뒤 코드 재감사(self code review)와 회귀 점검에서 세 가지
정확성 문제가 확인됐다. 모두 정상 단일 턴에서는 잘 드러나지 않지만, 누적 대화 상태,
짧은 후속 답변, SSE id-dedup처럼 경계 조건이 겹칠 때 사용자에게 잘못된 승인 카드나
빈 응답을 만들 수 있는 문제였다.

이 ADR은 PR #44에 포함된 세 수정의 의사결정을 기록한다. 공통 원칙은 저장된 대화 전체를
느슨하게 재해석하지 않고, 실제로 현재 턴에서 발생한 신호와 이미 스트리밍된 메시지
경계를 더 엄격히 구분하는 것이다.

## 결정 1 - HITL 승인 게이트를 현재 턴으로 스코프

### 배경

HITL 승인 라우터 게이트는 `_has_pending_action`에서 `_extract_action_id`로 이어진다.
기존 `_extract_action_id`는 `state["messages"][-10:]`를 턴 경계 없이 역스캔하고, 첫
번째 양수 `actionId`를 가진 `ToolMessage`를 승인 대상으로 반환했다. 그 결과 이전 턴의
PENDING 예약 액션이 무관한 이후 턴에서 스퓨리어스 승인 카드로 다시 나타날 수 있었다.
읽기 전용 도서관 후속 질문이나 `actionId=0` no-op sentinel 턴이 대표적인 재현 경로다.

최악의 경우 사용자가 이전 턴 예약을 거절했는데도, `check_approval`의 deny 분기가
백엔드 액션을 취소하지 않아 액션이 TTL까지 PENDING으로 남는다. 이후 다른 턴에서 낡은
`actionId`로 열린 가짜 카드를 승인하면 사용자가 거절했던 예약이 실행될 수 있다.

인라인 `agent_node` 게이트는 현재 턴에서 방금 생성된 tool 결과만 검사하므로 이미
턴-스코프였다. 그러나 그 뒤에 도는 라우터 게이트는 누적 상태를 다시 스캔해 두 게이트의
불변식이 달랐다. 10차 no-op sentinel 수정 `80b9545`가 `actionId <= 0`을 skip하게
만들면서, sentinel을 지나 더 오래된 양수 `actionId`까지 도달하는 경로가 노출됐다.

### 대안 검토

소비된 action id를 상태에 추적하는 방법을 검토했다. 승인 또는 거절이 끝난 id를 별도
상태 필드에 저장하고 이후 스캔에서 제외하면 정확하지만, 상태 스키마 확장과 checkpoint
호환성 부담이 생긴다. 이번 문제는 현재 턴 경계만으로 차단할 수 있으므로 범위가 과했다.

선택한 방법은 스캔을 현재 턴으로 한정하는 것이다. 마지막 `HumanMessage`를 만나면
역스캔을 멈추고, 그 이후에 생성된 `prepare_*` `ToolMessage`만 승인 게이트를 열 수 있게
한다. 인라인 게이트의 의미와 일치하고 상태 스키마를 바꾸지 않는다.

### 결정

`_extract_action_id`는 `reversed(messages)`를 순회하다가 `HumanMessage`를 만나면
`break`한다. 기존 `[-10:]` 슬라이스는 제거했다. 스캔 범위는 고정 개수가 아니라
마지막 사용자 입력 이후의 현재 턴으로 제한된다.

### 작동 방식

정상 HITL 흐름에서는 `prepare_*` 뒤 `interrupt()`가 발생하고, `/agent/resume`은 새
`HumanMessage`를 추가하지 않는다. 따라서 resume 시점의 `actionId` `ToolMessage`는
여전히 마지막 `HumanMessage` 뒤에 있고 기존처럼 `check_approval`로 라우팅된다.

새 턴이 시작되면 새 `HumanMessage`가 누적 상태 뒤에 붙는다. 이전 턴의 pending action은
그 경계 앞에 있으므로 무시된다. 회귀 테스트는 `[H1, A1, TM(actionId=5), A2, H2,
읽기전용 TM]` 상태가 `"done"`으로 라우팅되고, 단일 턴의 `actionId=5`는 계속
`"check_approval"`로 라우팅되는 것을 확인한다.

## 결정 2 - 도서관 연속 라우팅은 도서관 태그가 붙은 clarification에만 적용

### 배경

`_deterministic_route`의 continuation 분기는 20자 이하의 짧은 후속 답변을 도서관
서브그래프로 보낼 수 있다. 기존 조건인 `_is_library_reservation_clarification`은
`어디`, `어느`, `원하`, `선호`, `괜찮`, `몇층` 같은 일반 신호와 `?`, `까요`, `세요`
어미만 보았고, 도서관 특정 용어를 요구하지 않았다.

반면 `_strip_library_agent_prefix`는 리터럴 `[도서관 에이전트]` 태그만 제거했다. 즉
도서관 태그가 없어도 supervisor의 "어느 시설을 찾으세요?", `[LMS 에이전트]`의 과목
질문, `[학사 에이전트]`의 학기 질문 같은 일반 clarification이 도서관 예약 확인 질문처럼
해석될 수 있었다. 사용자가 짧게 답하면 도서관 서브그래프가 후속 턴을 하이재킹했다.

### 대안 검토

clarification cue 목록 자체를 도서관 특정 용어로 좁히는 방법을 검토했다. 그러나 실제
예약 확인 질문은 "몇 층 좌석을 원하세요?", "선호 좌석이 있나요?"처럼 짧고 일반적인
문장으로 표현될 수 있어, 문구 변형에 취약하다.

선택한 방법은 마지막 assistant 메시지가 실제 도서관 에이전트가 낸 일반 clarification일
때만 그 일반 신호를 신뢰하는 것이다. 도서관 로그인 게이트는 문구 자체가 도서관 특정
텍스트이므로 태그와 무관하게 유지한다.

### 결정

`_is_library_awaiting_user_input(raw_text)`는 prefix를 제거한 본문만이 아니라 태그를
포함한 원문도 받는다. 로그인 필요 안내는 기존처럼 무조건 도서관 연속 흐름으로 본다.
일반 clarification은 `raw_text.strip().startswith("[도서관 에이전트]")`가 참일 때만
도서관 연속 흐름으로 인정한다.

### 작동 방식

`[도서관 에이전트] 몇 층 좌석을 원하세요?` 다음의 `"2층"`은 계속 도서관
서브그래프로 간다. 반대로 supervisor의 `"어느 시설을 찾으세요?"`, `[LMS 에이전트]`
과목 질문, `[학사 에이전트]` 학기 질문에 대한 짧은 후속은 도서관으로 강제 라우팅되지
않는다. 이 경우 기존 supervisor 라우팅이 다시 판단한다.

## 결정 3 - 빈 응답 fallback이 SSE id-dedup에 삼켜지지 않게 처리

### 배경

10차 커밋 `ef0dff4`는 태그가 붙은 최종 `AIMessage`에 `id=last_ai.id`를 재사용했다.
이미 토큰 스트리밍으로 전달된 답변의 태그 사본을 SSE 계층에서 dedup하기 위한 변경이다.

그러나 모델이 공백-only 응답을 내면 `apply_empty_response_fallback`이 그 내용을
`EMPTY_RESPONSE_FALLBACK` 문구로 교체한다. 공백 chunk는 이미 라이브 스트리밍되어 같은
id가 streamed set에 들어가 있는데, fallback 문구를 담은 태그 메시지가 같은 id를
재사용하면 SSE id-dedup에 걸린다. 결과적으로 클라이언트는 fallback 문구를 받지 못하고
빈 말풍선만 남는다.

### 대안 검토

SSE dedup을 id가 아니라 content 비교 기반으로 바꾸는 방법을 검토했다. 하지만
스트리밍 조각과 최종 chain message의 표현이 항상 동일하다는 보장이 없고, 정상 dedup
경로까지 넓게 흔들 수 있다.

선택한 방법은 fallback이 적용되어 최종 태그 내용이 이미 스트리밍된 내용과 달라진 경우에만
새 id를 쓰는 것이다. 정상 답변의 id 재사용 최적화는 유지하고, graceful degradation
문구만 dedup 대상에서 분리한다.

### 결정

shared `react_loop.py`는 `last_ai`가 있고 `content_to_text(last_ai.content).strip()`이
`EMPTY_RESPONSE_FALLBACK`과 같으면 `fallback_applied=True`로 본다. 이때 태그 메시지는
`id=None`으로 생성해 fresh id를 받게 한다. fallback이 적용되지 않은 정상 답변은 기존처럼
`last_ai.id`를 재사용한다. `library.py`의 병렬 fallback 경로도 같은 규칙으로 빈 응답
fallback 메시지의 id를 `None`으로 둔다.

### 작동 방식

정상 비공백 답변은 token stream과 tagged chain message가 같은 id를 공유하므로 SSE
dedup이 계속 중복 표시를 막는다. 공백-only 답변은 fallback 문구로 내용이 바뀐 최종
메시지가 fresh id를 받으므로 dedup에 걸리지 않고 클라이언트에 전달된다.

## 결과

- HITL 승인 라우터는 현재 턴에서 생성된 pending action만 승인 카드로 올린다.
- 도서관 결정적 연속 라우팅은 도서관 에이전트의 clarification에만 적용되고, 다른
  도메인의 짧은 후속 답변을 가로채지 않는다.
- 빈 응답 fallback은 정상 답변 dedup 최적화를 유지하면서도 사용자에게 실제 fallback
  문구를 전달한다.
- 각 수정은 PR #44의 단위/회귀 테스트로 고정됐다.
