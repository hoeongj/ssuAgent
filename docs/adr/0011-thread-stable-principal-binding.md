# ADR 0011 — 회전 세션 ID 대신 안정 principal로 thread 소유권 재바인딩

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-09 |
| 상태 | Accepted — ssuAgent 측 구현 완료, 종단(end-to-end) 활성화는 ssuAI 후속 필요 |
| 범위 | `ssu_agent/main.py`(`thread_owners` 스키마, `claim_or_verify_thread_owner`, `AgentRequest`/`ResumeRequest`) · `tests/test_main_security.py` |
| 연관 | [ADR 0010](0010-agent-thread-ownership-binding.md)(회전 세션 ID를 처음 도입한 IDOR 방어. 이 ADR이 그 기반 위에서 "회전"이라는 남은 결함을 고침) |

---

## 배경 — 무엇이 문제인가 (4-service 감사 B1)

ADR 0010은 `thread_owners(thread_id, owner, created_at)` 테이블로 IDOR(다른 사람의 `thread_id`를 알면 대화를 훔쳐볼 수 있는 문제)는 막았지만, `owner`로 저장한 값이 **`mcp_session_id`** 였다. 그런데 `mcp_session_id`는 ssuAI 로그인 흐름에서 **재로그인마다 새로 발급되는 회전 값**이다(`McpWebSessionController.create`가 매 호출마다 `mcpAuthService.createSession()`으로 **무조건 새 세션**을 만든다 — 같은 학번이어도 재사용하지 않음, 확인 완료).

결과:
1. 사용자가 로그아웃 후 재로그인하면 `mcp_session_id`가 바뀌어 이전 대화 thread의 owner와 더 이상 일치하지 않는다 → 대화 히스토리가 **영구적으로 orphan** 된다(ADR 0010이 완화책으로 로그아웃 시 `sessionStorage["ssuagent_thread_id"]`를 지우게 했지만, 이는 403 에러를 피할 뿐 — 새 빈 thread로 시작하게 만들어 **히스토리 접근 자체를 포기**시키는 것이다).
2. 같은 사용자가 다른 기기에서 접속하면 별도 `mcp_session_id`가 발급되므로 대화를 공유할 수 없다.
3. 익명(비로그인) 경로는 ADR 0010에서 이미 "owner NULL → 아무나 접근 가능"으로 설계되어 있는데, 이건 의도된 것이지만 감사 관점에서는 여전히 "누구든 읽을 수 있는 IDOR 표면"으로 다시 지적될 수 있다 — 이번 작업에서 그 계약을 문서로 명확히 하고 회귀 테스트로 고정한다.

## 탐색 — ssuAgent가 스스로 안정 principal을 구할 수 있는가?

구현 전에 코드를 직접 확인했다 (아래는 모두 실제 코드 근거 있음, 추측 아님):

- **ssuMCP `get_auth_status`**: `McpAuthMcpTools.getAuthStatus` 및 `McpAuthStatusResponse`의 Javadoc이 명시적으로 **"Student id / principalKey is never included"** 라고 선언한다. 응답은 `status`/`mcpSessionId`(입력과 동일한 값의 echo)/`providers`(연동 여부 불리언)뿐이다. 즉 ssuAgent가 이 도구를 직접 호출해도(그래프 밖에서 `MultiServerMCPClient` 툴을 직접 `ainvoke`하는 패턴은 `library.py`의 `confirm_tool.ainvoke`에 이미 있어 기술적으로는 가능) **얻을 수 있는 안정 식별자가 없다** — 이건 버그가 아니라 ssuMCP의 의도된 프라이버시 경계다.
- **ssuAI → ssuAgent 프록시**: `ssuAI/lib/server/agentProxy.ts`는 `{message, thread_id, mcp_session_id}`만 그대로 전달하고, 서비스 간 공유 크리덴셜(`X-Agent-Key`) 외에 **사용자 JWT/식별자는 일절 전달하지 않는다**. ssuAI 자체는 SmartID SSO로 발급한 JWT에 `studentId`를 담아 갖고 있지만(`lib/api/auth.ts`), 그 값이 ssuAgent에 도달하는 배선은 현재 존재하지 않는다.

결론: **ssuAgent 저장소만으로는 안정 principal을 자체적으로 만들어낼 방법이 없다.** 이걸 만들려면 ssuMCP의 응답 스키마(프라이버시 경계 변경) 또는 ssuAI의 프록시 배선(신규 필드) 중 하나를 반드시 바꿔야 하는데, 둘 다 이번 작업의 범위(ssuAgent 저장소 단독) 밖이다.

## 대안 비교

### A. `get_auth_status` 호출로 학번(u-SAINT principalKey)을 얻어 캐시

세션 claim 시점에 ssuMCP `get_auth_status(mcp_session_id)`를 직접 호출해 학번을 얻고, 그 값으로 바인딩.

**기각.** 위 탐색에서 확인했듯 `get_auth_status`는 학번/principalKey를 **의도적으로** 반환하지 않는다(프라이버시 설계, ssuMCP 자체 문서화됨). 이를 되돌리려면 ssuMCP DTO를 바꿔야 하는데, 이는 "새 인증 인프라를 만들지 않는다"는 이번 작업의 제약과 "ssuAgent 저장소 단독 작업" 범위 둘 다를 위반한다. 설령 바꾸더라도 매 `/agent/stream` 호출마다 ssuMCP로 왕복 호출이 추가되어(가용성 의존 + 지연 추가) A의 "실패 시 세션 바인딩으로 폴백" 요구사항 자체가 새로운 장애 모드를 만든다.

### B. 프론트엔드가 안정 subject(JWT sub/학번)를 함께 보내는 구조를 ssuAgent가 수용

ssuAI가 이미 보유한 안정 식별자(`studentId`, SmartID JWT 기반)를 요청 바디의 새 필드로 실어 보내고, ssuAgent는 그 값이 있으면 우선 사용하고 없으면 기존 세션 바인딩으로 폴백.

**채택.** ssuAI 쪽 전달 배선은 아직 없지만(확인됨), 그 값 자체는 이미 ssuAI 안에 존재하는 기존 인증 결과다 — 즉 "새 인증 인프라"가 아니라 "이미 있는 안정 식별자를 한 단계 더 전달"하는 배선 문제다. ADR 0010도 정확히 이런 2-repo 협업 패턴이었다(ssuAgent 테이블+검증 / ssuAI 로그아웃 시 정리, 별도 PR). 이번 유닛은 그 전례를 따라 **ssuAgent 측 절반**을 구현한다: 필드를 선제적으로 수용하고, 절대 필수로 만들지 않으며(부재 시 채팅이 절대 깨지지 않음), 저장 시 해시(아래 참고)해 원문을 보관하지 않는다. ssuAI가 이 필드를 채워 보내기 시작하는 순간 즉시 활성화된다 — 별도 ssuAgent 배포 불필요.

### C. 하이브리드(인증 시 principal, 익명 시 세션 유지)

**B에 이미 포함.** principal이 오면 그것으로, 없으면 세션으로, 세션도 없으면 익명(NULL, 누구나 접근 — ADR 0010 계약 유지)으로 3단 폴백하는 것 자체가 B의 설계이지 별도 대안이 아니다. 감사가 "대안 3안"으로 나열했지만 코드를 보면 C는 B를 구현하기 위한 방식이지, A/B와 대등한 제3의 선택지가 아니었다.

## 결정

`thread_owners`에 `owner_kind TEXT` 컬럼을 추가(`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, ADR 0010의 기존 prod 테이블에 대해 additive)하고, `claim_or_verify_thread_owner(thread_id, mcp_session_id, principal=None)`을 다음 우선순위로 재구현한다:

1. `principal` 있음 → `owner = sha256(principal)`, `owner_kind = 'principal'`. 다른 `mcp_session_id`에서 와도 같은 principal이면 통과, 다른 principal이면 403.
2. `principal` 없고 `mcp_session_id` 있음 → `owner = mcp_session_id`, `owner_kind = 'session'` (ADR 0010과 동일 동작).
3. 둘 다 없음 → `owner = NULL` (익명, 누구나 접근 — ADR 0010과 동일 계약).

`AgentRequest`/`ResumeRequest`에 `principal: str | None = None` 필드를 추가했다. 오늘 시점에는 이 필드를 채워 보내는 호출자가 없다(확인됨) — 이 유닛은 **선제적 수용(forward-compatible) 구현**이고, ssuAI가 안정 subject를 실어 보내는 후속 변경이 있어야 실제 사용자에게 효과가 생긴다.

## 마이그레이션 규칙 — 왜 lazy가 batch보다 나은가

배치 마이그레이션(기존 `thread_owners` 행을 한 번에 스캔해 `owner_kind='session'`인 행을 `principal`로 다시 채우는 잡)은 **원천적으로 불가능하다** — 서버가 채워 넣을 principal 값 자체가 아직 어디서도 오지 않기 때문이다(오늘은 어떤 호출자도 이 필드를 보내지 않는다). 배치가 할 수 있는 일이 없다.

대신 lazy 재바인딩을 택했다: `owner_kind='session'`(또는 ADR 0011 배포 이전에 적힌 `owner_kind IS NULL` 레거시 행)인 스레드를, **그 스레드의 정당한 소유 세션이 검증을 통과한 바로 그 요청에서 `principal`을 처음 실어 보내는 순간** `owner_kind='principal'`로 승격한다. 즉:

- 트래픽이 실제로 있는 스레드만 마이그레이션 비용을 낸다(콜드 스레드를 훑는 전체 스캔 없음).
- ssuAI가 언제 이 필드를 채우기 시작하든 서버 재배포 없이 그 시점부터 자연히 작동한다.
- 정확히 한 번만 일어난다: 승격 후에는 `owner_kind == 'principal'` 분기가 항상 먼저 매치되므로 같은 UPDATE가 다시 실행되지 않는다(`test_lazy_migration_rebinds_session_owned_thread_to_principal_once`로 고정).

## 구현 선택

- **`owner`를 원문이 아닌 `sha256(principal)`로 저장**: 오늘은 ssuAgent가 principal 원문을 받을 계획조차 없지만, 언젠가 학번 같은 값이 실려 올 경우를 대비해 저장 시점부터 해시한다. `thread_owners`는 대화 내용이 아니라 소유권 인덱스일 뿐이라 원문이 굳이 필요 없고(동등 비교만 하면 됨), ssuMCP가 `get_auth_status`에서 학번을 노출하지 않기로 한 것과 동일한 프라이버시 원칙을 ssuAgent 저장소에도 일관되게 적용한 것이다.
- **`owner_kind` 별도 컬럼(값 재해석 아님)**: `owner` 하나로 세션값과 principal 해시값을 구분 없이 섞으면(예: prefix 문자열로 구분) 레거시 데이터와의 충돌 위험이나 파싱 버그가 생긴다. 명시적 컬럼이 `ALTER TABLE ADD COLUMN IF NOT EXISTS`로 기존 prod 테이블에 안전하게 additive하고, 레거시 행(`owner_kind IS NULL`)을 `'session'`과 동일하게 취급하는 분기 하나로 하위호환이 끝난다.
- **principal 승격 후 세션 단독 인증 거부**: 한 번 `principal`로 승격된 스레드는, 원래 세션이 `principal` 없이(예: 프론트가 그 필드를 빼먹은 호출) 다시 접근하면 403을 받는다. 세션이 정당하다고 해서 언제까지나 신뢰하면 애초에 이 ADR이 고치려는 "회전 값 신뢰" 문제로 되돌아가기 때문이다. 트레이드오프는 아래 참고.
- **A안과 달리 실패 모드가 없음**: principal은 요청 바디의 정적 필드일 뿐 ssuAgent가 매 호출마다 외부(ssuMCP)를 조회하지 않는다. 그래서 "auth-status 조회 실패 시 폴백" 같은 별도 방어 로직이 필요 없다 — 필드가 없으면 그냥 세션 바인딩(2번 규칙)으로 자연히 떨어진다.

## 신뢰 모델 — principal은 클라이언트가 주장할 수 있는 값이어서는 절대 안 된다

`principal` 필드가 요청 바디에 있다는 사실 자체가 오해를 부를 수 있어 명시한다: **이 값은 브라우저(최종 사용자)가 채워 보내는 값이 아니다.** 규칙:

1. **주입 주체는 ssuAI의 서버측 프록시(route handler)뿐이다.** 프록시가 자체 검증한 세션/토큰(SmartID JWT)에서 principal을 도출해 주입하고, 브라우저가 보낸 요청에 `principal` 필드가 들어 있으면 **무조건 폐기(strip)** 한다. 클라이언트 주장값이 서버 주입값으로 둔갑할 경로를 원천 차단한다.
2. **ssuAgent는 신뢰된 프록시로부터만, 클러스터 내부에서만 도달 가능해야 한다.** 공개 노출 금지. 인터넷에서 ssuAgent에 직접 POST할 수 있으면 누구나 임의 principal을 주장할 수 있으므로 이 설계 전체가 무너진다.
3. **`sha256(principal)` 저장은 프라이버시 조치일 뿐 위조 방어가 아니다.** 학번은 열거 가능한(enumerable) 값이라 공격자가 임의 학번을 넣고 해시가 일치하도록 만드는 건 자명하다. 위조 방어는 오직 ①(프록시만 주입)과 ②(내부 전용 도달)에서 나온다 — 해시는 DB 유출 시 원문 노출을 막는 별개 목적이다.
4. **lazy 재바인딩이 "정당한 세션 + principal 동시 제시"를 요구하는 이유도 이것이다.** 기존 세션 소유 스레드의 승격은 저장된 `mcp_session_id` 검증을 먼저 통과한 요청에서만 일어나므로, principal 스푸핑 단독으로는 기존 스레드를 탈취할 수 없다(§마이그레이션 규칙의 UPDATE는 세션 일치 분기 안에서만 실행됨).

동반 ssuAI 변경(프록시에서 principal 도출·주입 + 브라우저 필드 strip)은 별도 유닛으로 예정되어 있으며, **그 변경이 배포되기 전까지 이 변경은 완전히 비활성(inert)이다** — 어떤 호출자도 principal을 보내지 않으므로 모든 요청이 기존 ADR 0010 경로(2·3번 규칙)로만 흐른다.

## 트레이드오프

- **얻는 것**: principal이 실제로 도착하기 시작하면 재로그인·멀티기기에서도 히스토리가 보존되고, ADR 0010의 세션-회전 결함이 사라진다. 원문 미저장으로 DB 유출 시 실제 신원 노출이 없다.
- **잃는 것 / 남는 리스크**:
  - **오늘은 아무것도 바뀌지 않는다** — ssuAI가 `principal`을 보내기 시작하기 전까지는 여전히 ADR 0010의 세션-회전 한계가 그대로다. 이 ADR은 "절반의 배선"이고, 실제 효과는 ssuAI 후속 PR에 달려 있다(ADR 0010의 "ssuAI 짝꿍" 선례와 동일 패턴).
  - **드랍 위험**: principal로 승격된 스레드는 그 필드가 실려오지 않는 호출을 전부 거부한다. ChatGPT가 `mcp_session_id`를 턴 경계에서 종종 드랍하는 것과 같은 계층의 위험이 `principal`에도 그대로 적용된다(ADR 0036 참고) — 프론트가 매 요청 principal을 빠짐없이 동봉해야 하는 책임이 새로 생긴다.
  - **해시 비교라 사람이 읽는 감사 로그가 약해짐**: DB에서 `owner` 컬럼만 봐서는 어떤 학번인지 알 수 없다(의도된 트레이드오프이지만 운영 중 "이 thread 누구 거야?"를 즉시 답하기 어려워진다 — 필요하면 별도 조회 경로를 principal 원문을 아는 상위 계층에서 만들어야 한다).

## 예상 면접 질문

1. "principal을 왜 원문으로 저장하지 않고 해시했나요? 무엇을 얻고 무엇을 잃나요?"
2. "ssuAgent 혼자서는 안정 principal을 만들 수 없다고 판단한 근거는 뭔가요? (get_auth_status 응답에 학번이 없다는 걸 어떻게 확인했나요?)"
3. "아직 아무도 안 보내는 필드를 왜 지금 서버에 먼저 넣었나요? 절반만 배포된 상태에서 안전성을 어떻게 보장하나요?"
4. "배치 마이그레이션이 왜 불가능하다고 판단했나요? lazy 방식이 정확히 한 번만 실행됨을 어떻게 보장/테스트했나요?"
