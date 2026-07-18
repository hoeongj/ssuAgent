# ssuAgent

[![CI](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/ci.yml)
[![Security](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/security.yml/badge.svg)](https://github.com/ghdtjdwn/ssuAgent/actions/workflows/security.yml)

**한국어** · [English](README.en.md)

ssuAI의 자연어 요청을 도메인별 MCP 도구로 라우팅하고, 대화 상태·SSE 스트림·사용자 승인 흐름을
관리하는 FastAPI/LangGraph 오케스트레이션 서비스다.

[라이브 챗봇](https://ssuai.vercel.app/chat) ·
[플랫폼 사례 연구](https://seongju.vercel.app/projects/ssu-platform/) · [문서 지도](docs/README.md)

## 플랫폼에서 맡는 역할

| 서비스 | 책임 | 저장소 |
| --- | --- | --- |
| ssuAI | 사용자 화면, same-origin BFF, 인증 상태와 SSE/HITL UX | [ghdtjdwn/ssuAI](https://github.com/ghdtjdwn/ssuAI) |
| **ssuAgent** | **의도 라우팅, 도구 조합, 대화 checkpoint, HITL 오케스트레이션** | 현재 저장소 |
| ssuMCP | 캠퍼스 도메인 로직, MCP/REST 계약, 인증과 상태 변경 | [ghdtjdwn/ssuMCP](https://github.com/ghdtjdwn/ssuMCP) |
| ssu-ai-service | 격리된 임베딩 요청 게이트웨이 | [ghdtjdwn/ssu-ai-service](https://github.com/ghdtjdwn/ssu-ai-service) |

이 서비스는 학교 시스템을 직접 호출하지 않는다. 도메인 기능은 MCP로만 소비하며, 브라우저 인증은
ssuAI BFF가 검증한 뒤 최소한의 principal과 MCP session context만 전달한다.

## 아키텍처

![ssuAgent 오케스트레이션 아키텍처 — 신뢰 경계, LangGraph 라우팅, 체크포인트, MCP와 LLM 폴백](docs/assets/architecture.svg)

```text
browser
  → ssuAI same-origin agent proxy
  → FastAPI stream/resume boundary
  → LangGraph supervisor
  → academic · library · LMS specialist
  → ssuMCP tools over Streamable HTTP
```

- Supervisor는 현재 사용자 turn을 도메인 specialist로 라우팅하고, specialist는 필요한 MCP 도구만
  동적으로 로드한다.
- LangGraph PostgreSQL checkpointer가 대화와 interrupt 상태를 저장한다. 검증된 principal의 hash와
  `thread_id`를 결합해 다른 사용자의 checkpoint 읽기·resume을 차단한다.
- 도서관 write는 `prepare_*` 결과에서 graph를 interrupt하고, ssuAI의 명시적 승인 이후에만
  `confirm_action`으로 재개한다.
- 설정된 provider만 Anthropic → Groq → Gemini → OpenRouter 순서로 시도한다. 각 agent의 수동
  `bind_tools` loop가 provider별 tool-call 차이와 fallback을 통제한다. Groq는 tool-call turn의
  content 호환성 때문에 범용 `ChatOpenAI` 대신 `ChatGroq`를 사용한다. 가격·모델 제공 여부·조직별
  한도는 외부 runtime constraint이며, 이 순서는 비용이나 정확도 우위를 뜻하지 않는다.

요청·신뢰·상태 경계와 현재 단일 replica 제약은 [아키텍처 문서](docs/architecture.md)에 정리했다.

## 엔지니어링 근거

| 문제 | 구현과 검증 근거 |
| --- | --- |
| 다른 사용자가 기존 대화를 resume할 위험 | stable principal hash와 thread owner binding — [ADR 0010](docs/adr/0010-agent-thread-ownership-binding.md) · [security tests](tests/test_main_security.py) |
| stream 재연결 또는 interrupt 뒤 resume 계약 오류 | stable thread와 명시적 `resume` event ordering — [stream contract tests](tests/test_stream_interrupt.py) |
| 명확한 LMS export가 추가 LLM turn에서 timeout되는 문제 | 보수적 직접 라우팅과 결정적 링크 생성 — [ADR 0022](docs/adr/0022-deterministic-lms-export-download.md) · [LMS tests](tests/test_lms_agent.py) |
| provider별 tool-call 형식 차이와 장애 전파 | 설정 기반 provider sequence와 agent-local fallback — [factory tests](tests/test_llm_factory.py) · [ADR 0004](docs/adr/004-multi-provider-llm-fallback.md) |
| handoff가 답변을 중복하거나 잘못된 도구를 고르는 문제 | routing/safety 평가 세트 — [routing eval](tests/test_eval_routing.py) · [safety eval](tests/test_eval_safety.py) |
| 검증되지 않은 이미지의 자동 배포 | Ruff·format·pytest 뒤 ARM64 image publish — [CI workflow](.github/workflows/ci.yml) · [deployment guide](docs/deploy.md) |

주요 스택은 Python 3.12, FastAPI, LangGraph, LangChain, PostgreSQL checkpointer, MCP Streamable
HTTP, SSE, uv, Ruff, pytest, Docker, Helm과 ArgoCD다.

## 로컬 실행과 검증

PostgreSQL과 한 개 이상의 LLM provider key가 필요하다. 실제 값은 `.env`에만 두고 커밋하지 않는다.

```bash
git clone https://github.com/ghdtjdwn/ssuAgent.git
cd ssuAgent
cp .env.example .env
set -a && source .env && set +a

uv sync --extra dev
uv run uvicorn ssu_agent.main:app --host 0.0.0.0 --port 8000
```

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests/test_eval_routing.py
```

환경 변수의 보안 의미와 운영 차이는 [설정 문서](docs/configuration.md)에 있다.

Supervisor 라우팅 계약은 `evals/routing_contract.v1.json`의 버전 고정 corpus와
실패 분류를 사용한다. 이 평가는 모델이 도구를 선택한 뒤의 실제 라우팅 도구,
marker parser, graph 목적지를 검증하며 외부 LLM을 호출하지 않는다. 따라서 결과를
live-model 라우팅 정확도로 해석하지 않는다. 경계와 재현 방법은
`evals/README.md`에 기록했다.

corpus는 9개 질의(도메인 라우팅 6, 직접 응답 3)와 네 가지 실패 유형을 포함한다.
외부 프로바이더나 로컬 모델을 이용한 정확도 평가는 실행하지 않았으므로, 결과는 LLM 선택
정확도가 아니라 결정론적 후속 라우팅 계약의 회귀 근거다.

## 문서

- [문서 지도](docs/README.md)
- [아키텍처와 신뢰 경계](docs/architecture.md)
- [설정과 환경 변수](docs/configuration.md)
- [GitOps 배포와 운영 검증](docs/deploy.md)
- [운영 장애 기록](docs/troubleshooting.md)
- [ADR 목록](docs/adr/)

## 범위와 제약

- 현재 production은 단일 replica이고 inbound rate limit은 process-local이다. scale-out 전에 shared
  limiter와 checkpoint 동시성 검증이 필요하다.
- CI의 stream/HITL 단위 테스트는 `MemorySaver`를 사용한다. 실제 PostgreSQL을 재시작한 뒤 resume하는
  container integration test는 현재 gate에 포함되지 않는다.
- LLM과 학교 시스템의 가용성에 의존한다. 도구 실패 시 응답을 추측하지 않고 제한이나 연결 필요 상태를
  반환한다.
- 직접 `/agent/*` 호출은 운영에서 server-to-server API key가 필요하며, 공개 사용자는 ssuAI를 통해
  접근한다.

## 라이선스

[MIT](LICENSE)
