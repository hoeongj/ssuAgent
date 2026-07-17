# ADR 0022 - LMS 전체 자료 내보내기의 결정적 다운로드 응답

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-17 |
| 상태 | Accepted |
| 범위 | LMS agent 라우팅, 전체 자료 내보내기, 다운로드 링크 응답 |
| 관련 | [ADR 0001](0001-supervisor-architecture.md), [ADR 005](005-router-bug-fix-prepare-collision.md) |

## 배경과 실제 증상

사용자가 "지금 수강중인 수업 모든 강의 파일을 다운받고 싶어"라고 요청한 운영 대화에서
LMS handoff와 인증 확인, `get_my_lms_courses`, `prepare_lms_material_export`,
`confirm_lms_material_export` 상태까지 표시된 뒤 최종 링크가 나오지 않았다. 이어서
"링크 띄워줘"라고 보내자 수퍼바이저는 직전 LMS 작업을 이어가지 않고 일반 안내를 반환했다.
같은 전체 내보내기 요청을 다시 보냈을 때는 링크가 생성됐지만, 프런트는 Markdown 링크를
일반 텍스트로 표시해 사용자가 긴 capability URL을 복사해야 했다.

ssuMCP의 confirm 경로는 ZIP 완성을 기다리지 않고 job과 20분 기본 TTL의 capability URL을 즉시
반환한다. 해당 URL의 브라우저 페이지도 BUILDING 상태를 polling하고 준비 완료 시 다운로드
버튼과 자동 다운로드를 제공한다. 따라서 ZIP worker가 링크 발급을 막은 것이 아니라, 링크를 받은
뒤 사용자 답변으로 확정하는 orchestration과 렌더링 경계가 문제였다.

## 원인과 증거

1. ssuMCP에 이미 있는 `export_all_lms_materials`가 ssuAgent의 `_LMS_NAMES`에 없었다. 이 도구는
   public/supervisor 범주로 잘못 분류되어 LMS agent가 사용할 수 없었고, 전체 요청도 조회·prepare·
   confirm의 세 단계 도구 흐름을 거쳤다. 전용 도구라면 export·confirm 두 단계면 충분하다.
2. 공용 ReAct 루프는 URL을 받은 뒤에도 이를 사용자 문장으로 만드는 추가 모델 턴에 의존했다.
   transcript는 confirm 상태 직후 답변이 사라졌고, 같은 요청을 다시 보냈을 때는 정상 완료됐다.
   이는 ZIP worker 실패보다는 60초 SSE 경계 부근에서 최종 합성 턴이 끝나지 못한 증상과 일치한다.
   네 도구가 필요한 변형에서는 공용 루프의 최대 도구 턴 한도에도 닿을 수 있지만, 제공된
   transcript 자체는 세 도구였다.
3. 공용 루프는 중간 ToolMessage를 체크포인트에 남기지 않고 최종 AIMessage 하나만 반환했다.
   최종 답변 전에 스트림이 종료되면 후속 턴이 복원할 URL도 없다. capability URL을 모델 history에
   계속 노출하는 것도 불필요하므로, 사용자 응답을 로컬에서 확정한 뒤 모델 경계에서는 가린다.

## 결정

명확한 LMS 요청은 LLM 수퍼바이저 전에 `lms_agent`로 보수적으로 라우팅한다. 대상은 LMS를 명시하거나
강의·수업 파일/자료 다운로드처럼 도메인이 분명한 표현이다. `과제`만 있는 문장은 공부 조언이나 다른
도메인과 섞일 수 있어 기존 수퍼바이저에 남긴다.

`export_all_lms_materials`를 LMS 도구 목록과 prompt, 진행 상태 label에 추가한다. 사용자가 전체
과목 또는 모든 자료를 명시하면 `export_all_lms_materials → confirm_lms_material_export` 두 단계로
처리한다. 특정 과목은 기존 조회·prepare·confirm 경로를 유지한다.

공용 ReAct 루프에는 선택적 `terminal_tool_result_formatter`를 추가한다. LMS agent는 성공한
`confirm_lms_material_export` 결과만 처리하고 다음 조건을 모두 검증한다.

- 응답에 명시적인 `status: OK`와 객체형 `data`가 있어야 한다.
- URL scheme은 HTTP 또는 HTTPS이고 userinfo가 없어야 한다.
- URL origin은 설정된 ssuMCP origin과 정확히 일치해야 한다.
- path는 `/api/lms/exports/{jobId}/download` 형식이고 비어 있지 않은 `token` query가 있어야 한다.

검증에 성공하면 파일 수와 예상 용량, Markdown 다운로드 링크를 결정적으로 포맷해 즉시 최종
AIMessage로 체크포인트한다. 추가 모델 호출과 모델의 URL 재작성에 의존하지 않는다. 이전 terminal
응답이 후속 요청의 history로 들어갈 때는 capability URL을 모델 입력 경계에서 redaction하되, 현재
tool 결과는 로컬 formatter가 읽을 수 있도록 보존한다. 오류나 계약이 다른 응답은 terminal 처리하지
않고 기존 모델 흐름에 남긴다. formatter는 URL이나 tool 원문을 로그하지 않는다.

`confirm_lms_material_export`는 독립 tool turn에서만 실행한다. 모델이 export/prepare와 confirm을
같은 묶음으로 내면 선행 호출만 실행하고 confirm에는 `INVALID_TOOL_SEQUENCE` 결과를 돌려준다. 모델은
선행 결과를 받은 다음 턴에 confirm을 다시 호출하므로 preview 생성 전 확정 race를 코드 경계에서 막는다.

ssuAI는 assistant 메시지의 HTTP(S) Markdown 링크와 bare URL만 안전한 anchor로 변환한다. 설정된
ssuMCP origin과 export path, token이 모두 일치하는 링크만 `강의 파일 다운로드` action으로 표시하고,
다른 origin의 일반 링크에는 hostname을 노출한다. 사용자 메시지와 `javascript:` 같은 scheme은
활성화하지 않는다. 링크는 새 탭에서 `noopener noreferrer`로 열리고 새 탭 동작을 접근 가능한 이름에
포함하며, 긴 token 값은 화면 텍스트에 노출하지 않는다.

## 검토한 대안

### 모델 턴 한도나 Vercel 실행 시간만 늘리기

전체 요청의 불필요한 도구 왕복과 URL 유실 가능성을 남긴다. 배포 플랜에 따라 프록시 최대 시간이
다르고 외부 LMS 지연도 변하므로 시간 예산 증가는 근본 해결이 아니다.

### confirm tool output 전체를 새 SSE event로 전달하기

typed download event는 장기적으로 명확하지만 raw tool output을 SSE allowlist에 추가하면 세션과
개발자 메시지 누출 위험이 생긴다. 현재 text 계약에서 필요한 필드만 결정적으로 확정하고 프런트가
안전한 링크를 렌더링하는 변경으로 충분해 보류했다. 추후 event를 추가한다면 URL·파일 수·용량만
새 DTO로 allowlist해야 한다.

### 전체 Markdown renderer 의존성 추가

현재 요구는 링크 action이며 raw HTML이나 광범위한 Markdown 해석은 필요하지 않다. 작은
assistant-only parser로 지원 범위를 HTTP(S) 링크에 한정해 번들·공격 표면·업데이트 비용을 늘리지
않았다.

## 검증과 재발 방지

- 전체 내보내기 도구가 LMS 범주에 있고 public 범주에는 없음을 계약 테스트로 고정한다.
- 명확한 과제·LMS·강의 파일 요청이 `lms_agent`로 직접 라우팅됨을 테스트한다.
- terminal formatter가 confirm tool을 한 번만 실행한 뒤 추가 모델 턴 없이 링크를 체크포인트함을
  공용 루프와 LMS graph 수준에서 각각 검증한다.
- 오류 status, 다른 tool, `javascript:` URL, 다른 origin, export가 아닌 path는 terminal 응답으로
  채택하지 않는다.
- 성공 envelope는 명시적인 `status: OK`와 객체형 `data`를 모두 요구한다.
- 같은 모델 턴의 export/prepare와 confirm은 선행 호출만 실행되고 confirm은 다음 턴으로 지연된다.
- 체크포인트된 capability URL은 후속 모델 history에서 가리고 현재 tool 결과는 formatter에 보존한다.
- 프런트 컴포넌트와 SSE 정착 테스트가 assistant 링크의 `href`, 새 탭 보안 속성, action label,
  신뢰 origin, 사용자 입력·unsafe scheme 비활성화와 새 탭 접근성 안내를 검증한다.

실계정 export는 write 성격의 학교 시스템 작업이므로 자동 테스트에서 실행하지 않는다. 운영 확인은
사용자가 전체 강의자료 요청을 한 번 보내고, 두 단계 상태 뒤 다운로드 action이 나타나는지와 해당
action이 ssuMCP의 polling 페이지를 여는지만 확인한다.

## 남은 위험과 면접 질문

외부 LMS course/material 수집 자체가 60초를 넘으면 confirm 이전에 프록시가 종료될 수 있다. 이번
결정은 불필요한 수퍼바이저·도구·최종 모델 왕복을 제거하지만 upstream의 절대 지연을 없애지는 않는다.
이 경우 장기 대안은 export 요청 접수 자체를 짧은 비동기 API로 분리하고 job 상태를 별도 event로
보내는 것이다.

- capability URL을 LLM 최종 문장에 맡기지 않은 이유는 무엇인가?
- 전체 내보내기만 단축 도구로 보내고 특정 과목 흐름은 유지한 이유는 무엇인가?
- raw tool output을 SSE로 전달하지 않고 terminal formatter를 둔 보안상 이유는 무엇인가?
