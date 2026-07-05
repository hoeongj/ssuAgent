# ADR 0009 — ssuAgent /agent/* 엣지 하드닝 (rate-limit · payload cap · error 비노출 · CORS)

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-06-23 |
| 상태 | Accepted — 구현(브랜치 `fix/agent-edge-hardening`) |
| 범위 | `ssu_agent/main.py` · `ssu_agent/config.py` · `pyproject.toml`/`uv.lock`(slowapi) · `tests/test_main_security.py` |
| 연관 | ssuMCP [ADR 0061](https://github.com/ghdtjdwn/ssuMCP/blob/main/docs/adr/0061-per-ip-rate-limit-input-caps.md)(포팅 원본) |

---

## 배경 — 무슨 문제

2026-06-23 4종 AI 분석(S1, High): ssuAgent `/agent/*`가 prod에서 사실상 공개였다 — `AGENT_API_KEY` 미설정 시 인증 no-op + **per-IP rate-limit 없음 + message 크기 무제한 + CORS `allow_origins=*`**. `/agent/stream`은 LLM(Gemini/Groq/OpenRouter, 유료)으로 fan-out하는 **플래그십 경로**라, 인터넷의 누구나 POST를 퍼부어 **토큰 비용 소진 / 프롬프트 DoS**가 가능했다. ssuMCP 코어는 동일 위험을 ADR 0061(per-IP rate-limit + 입력 상한)로 막았는데 **그 통제가 형제 서비스(ssuAgent)엔 미이식**이었다.

추가로 `_stream_graph`의 예외 핸들러가 `str(exc)`를 SSE `error`로 **클라이언트에 그대로 반사** → 내부 스택/DB 컨텍스트 누출.

## 결정 (사용자 확정: "코드 하드닝만")

ssuMCP ADR 0061의 통제를 ssuAgent로 포팅한다. **API 키 인증 활성화 + Next 프록시(S1-b)는 별도 후속**(브라우저가 `/agent/*`를 직접 호출 → 비밀 헤더를 클라에 두면 노출되므로 서버사이드 프록시 필요, 더 큰 아키텍처 변경).

- **per-IP rate-limit**: `slowapi` `Limiter`를 `/agent/stream`·`/agent/resume`에 적용(기본 `30/minute`, `AGENT_RATE_LIMIT`로 튜닝). 키 함수는 **X-Forwarded-For 좌측 홉**(k3s ingress 뒤에서 모든 요청이 ingress IP를 공유하므로) → ssuMCP `ClientIpResolver`와 동일 전략. 초과 시 429.
- **payload cap**: `AgentRequest.message`에 pydantic `Field(max_length=AGENT_MAX_MESSAGE_CHARS)`(기본 8000) → 초과 시 422.
- **error 비노출**: `_stream_graph` 예외 시 정적 메시지만 SSE로, 전체 traceback은 서버 로그(`logger.exception`).
- **CORS**: `allow_methods`를 `["GET","POST"]`로 축소(API는 POST `/agent/*` + GET `/health`만). 오리진은 env(`ALLOWED_ORIGINS`)로 prod에서 vercel 도메인으로 좁힘(configmap).

## 대안과 기각 이유

- **API 키를 지금 강제** — 브라우저가 ssuAgent를 직접 호출하므로 클라에 키가 노출됨. 안전하려면 Next Route Handler 프록시(서버사이드 키 주입)가 필요 → ssuAI 아키텍처 변경 + 양쪽 동시 배포. 비용/DoS의 즉시 위험은 rate-limit가 더 작은 diff로 막으므로 **키+프록시는 후속**(사용자 확정).
- **`limits`/직접 미들웨어 자작** — slowapi가 FastAPI 표준(decorator + key_func + 429 핸들러)이고 검증돼 있어 자작보다 유지보수성↑. (2026 best-practice 검색 일치.)
- **Redis 분산 카운터** — 현재 replica=1이라 in-memory로 충분(per-pod 한계는 ssuMCP와 동일하게 문서화). 스케일아웃 시 도입.

## 동작 방식 / 검증

- `limiter`는 `lambda: config.AGENT_RATE_LIMIT`(per-request 평가)로 한도를 읽어 런타임 튜닝 + 테스트 오버라이드 가능. `app.state.limiter` + `RateLimitExceeded` 핸들러 등록.
- 의존성: `slowapi>=0.1.9` 추가 → `uv.lock` 재생성(Dockerfile이 `uv sync --frozen`이라 lock 동기 필수). `limits` 전이 의존성 포함.
- 테스트(`test_main_security.py`): 율제한 초과 429(저한도 오버라이드) · 과대 message 422 · `_stream_graph`가 예외 detail 비노출(내부 DSN 미포함, `type:error`만). 기존 6개(키 게이트/health) 보존(기본 fixture가 limiter 비활성).

## 예상 면접 질문

1. **"폴리글랏 MSA에서 보안 통제를 어떻게 일관 적용했나?"** — ssuMCP(Java)에 ADR 0061로 적용한 per-IP rate-limit·입력 상한·에러 비노출을 ssuAgent(Python)에 동일 의도로 포팅. "코어엔 했는데 형제 서비스엔 안 한" 갭을 분석으로 잡아 메웠고, before/after를 코드로 보여줄 수 있다.
2. **"ingress 뒤에서 per-IP rate-limit를 어떻게 정확히 하나?"** — `request.client.host`는 ingress IP라 전역 1버킷이 된다. X-Forwarded-For 좌측 홉(실클라이언트)을 키로 써야 IP별 분리. ssuMCP `ClientIpResolver`와 동일.
3. **"왜 API 키 인증을 지금 안 켰나?"** — 브라우저 직접 호출이라 클라 키 노출. 서버사이드 프록시가 선결이라 별도 후속. 비용/DoS 즉시 위험은 rate-limit로 더 작은 변경으로 차단(우선순위 판단).

---

## 갱신 (2026-07-02) — 후속(S1-b: API 키 인증 활성화 + Next 프록시) 완료

"별도 후속"으로 남겼던 API 키 인증 활성화는 **2026-06-30 완료**됐다:

- prod k3s secret `ssuagent-secrets`에 `AGENT_API_KEY` 주입 → `/agent/*` 키 게이트 활성화(미설정 no-op → 강제).
- ssuAI가 서버 전용 프록시(`lib/server/agentProxy.ts`, Vercel)에서 `X-Agent-Key` 헤더를 주입 — 브라우저는 same-origin `/api/agent/*`만 호출하므로 키가 클라이언트에 노출되지 않는다.
- 3중 검증: ① 키 없는 직접 호출 → 401 ② 올바른 키 직접 호출 → 422(게이트 통과, payload 검증 도달) ③ Vercel 프록시 경유 → 422(프록시 키 주입 정상).

현행 구성은 README의 "`/agent` 엔드포인트 인증" 절 참조.
