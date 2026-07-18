# 운영 장애 기록

실제로 확인한 장애만 기록한다. 운영 로그가 없는 원인은 추정으로 명시하고 검증하지 않은 결과는
성공으로 적지 않는다.

## 2026-07-18 — 개인 도구 실패 뒤 일반 답변 합성 및 ERROR 상태 고착

### 기대와 영향

연결된 u-SAINT나 LMS 도구가 실패하면 에이전트는 현재 개인 데이터를 가져오지 못했다고 명확히
알려야 한다. 실제로 졸업요건 도구가 실패한 뒤에는 일반적인 졸업 기준을 대신 생성했고, LMS는
credential grant가 남아 있어도 직전 `ERROR` health 때문에 실제 복구 호출을 시도하지 않았다.
사용자는 로그인 문제와 학교 시스템의 일시 장애를 구분할 수 없었다.

### 증거와 원인

- 학사 요청은 `get_auth_status`를 통과해 `check_graduation_requirements` 실행까지 진행한 뒤 실패했다.
- 공유 ReAct loop는 예외 원문을 노출하지 않기 위해 일반 tool error 문자열로 바꿨지만, 그 문자열을
  다음 모델 turn에 다시 전달했다. 모델은 개인 데이터 근거가 없는 일반 안내를 합성할 수 있었다.
- ssuMCP의 private response가 예외 대신 top-level `UPSTREAM_UNAVAILABLE` envelope를
  반환하면 기존 loop는 이를 성공 ToolMessage로 다루었다. 또한 일부 legacy LMS
  도구는 API 실패를 `status=OK`인 response의 string `data`에 넣어 반환했다.
- auth guard는 linked provider의 `ERROR`를 `UNAVAILABLE`로 처리했다. `ERROR`는 직전 upstream
  실패이고 grant 취소나 `EXPIRED`와 같지 않으므로, 일시 장애가 끝나도 다음 요청이 실제 도구를
  호출해 health를 `VALID`로 되돌릴 경로가 없었다.
- 당시 사용자 요청과 일치하는 trace가 없어 최초 upstream 실패가 credential, 네트워크, 학교 포털
  응답 변경 중 무엇이었는지는 확정하지 않는다.

### 해결과 대안

도구 invocation 예외를 `ToolMessage.status=error`로 표시하고 공유 loop가 즉시 도메인별 고정 서비스
장애 안내를 반환하도록 했다. masked 오류도 모델, checkpoint, SSE로 넘기지 않아 일반 졸업요건이나
임의 복구 절차를 만들 수 없다. top-level non-OK은 `retryable=true` 또는 `UPSTREAM_` status/code일
때만 operational failure로 분류하고, legacy LMS 호환 경로는 정확한 도구
이름과 오류 접두어로만 분류한다. 정상 data의 같은 단어와 non-retryable domain outcome은
통과시킨다.

linked `ERROR`는 명시적인 다음 사용자 요청에서 private tool 실행 예산을 1회로
제한한다. 같은 turn의 private batch는 실행 전에 거부하고, 예산을 사용한 뒤 모델이
추가 private call을 만들어도 재호출하지 않는다. 기존 transport wrapper의 한 번 retry는
하나의 논리적 호출 안에서 유지한다. 성공하면 ssuMCP가 health를 `VALID`로 갱신하고,
다음 사용자 요청이 새 provider preflight를 수행한다.

`ERROR`마다 재로그인을 강제하는 방식은 일시 장애와 credential 만료를 혼동해 제외했다. 반대로
모델에게 tool error를 설명하게 하는 방식은 근거 없는 fallback을 다시 허용하므로 제외했다. 누락,
malformed, non-OK auth status와 알 수 없는 health는 계속 fail-closed하고, unlinked와 `EXPIRED`만
재연결 경로로 보낸다.

### 검증과 남은 위험

- 전체 pytest 305개 통과
- Ruff check와 format check 통과
- 학사와 LMS 도구 예외가 모델의 두 번째 turn 없이 고정 안내로 끝나는 회귀 테스트 통과
- 학사 top-level operational envelope와 LMS legacy `status=OK`/string `data` 오류가 모델 합성
  전에 차단되고, 정상 data의 상태명과 non-retryable outcome은 통과하는 회귀 테스트 통과
- linked `ERROR`가 degraded 경로로 들어가고, private batch와 모델 재호출이 실행 전에
  차단되는 계약 테스트 통과

실제 학교 시스템의 일시 장애는 이 서비스가 제거할 수 없다. 배포 뒤에는 동일 요청이 성공할 때
health가 `VALID`로 회복되고, 계속 실패할 때 고정 안내와 프론트 degraded 표시가 함께 유지되는지
실계정으로 확인해야 한다. companion ssuMCP 변경은 LMS 목록·대시보드·내보내기 오류를 top-level
non-OK 계약으로 이전한다. 정확한 legacy 접두어 guard는 backend-first rolling deployment 동안만
구버전 응답을 보완하며, 두 서비스가 모두 배포된 뒤 제거할 수 있다.
