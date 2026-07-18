# ADR-004: 멀티 LLM 프로바이더 자동 Fallback

## Status
Accepted (2026-06-15)

## 배경

ssuAgent는 초기에 `ChatGoogleGenerativeAI(model="gemini-2.5-flash")` 단일 모델만 사용했다.
Gemini Free Tier는 RPD(일별 요청수), RPM(분당 요청수), TPM(분당 토큰) 세 가지 쿼타가 독립적으로 관리되며, 집중 개발/테스트 기간에 쿼타가 모두 소진됐다.

쿼타 소진 시 `LangChain tenacity` retry가 기본 설정(최대 6회, 지수 백오프)으로 동작하여 `/agent/stream` 엔드포인트가 최대 수분간 응답 없이 hang 상태가 됐다. 프론트엔드에서는 "network error"로 표시.

이전 대응은 `kubectl set env GEMINI_MODEL=gemini-2.5-flash-lite` 등으로 수동으로 모델을 교체하는 것이었는데, 개발자가 항상 모니터링할 수 없고 교체 후에도 다른 모델의 쿼타도 소진될 수 있었다.

## 고려한 대안

### 대안 A: Gemini 모델 목록 내에서 순환 (기존 방식)
- `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-2.5-flash-lite` 등 Google 모델 내에서 수동 교체
- **문제**: 모든 Gemini 모델이 같은 Google Cloud 프로젝트의 Free Tier 쿼타를 공유하므로, 한 모델이 소진되면 다른 모델도 곧 소진됨. 결국 모든 Gemini 모델이 동시에 429를 반환하는 상황이 반복됨

### 대안 B: 유료 Gemini API 티어로 전환
- 쿼타 걱정 없이 사용 가능
- **문제**: 포트폴리오 프로젝트에서 LLM API 비용이 예측 불가능하게 발생할 수 있음. 또한 이미 다른 프로바이더의 API 키(Groq, OpenRouter 등)가 ssuMCP에 등록되어 있는데 활용하지 않는 것은 비효율

### 대안 C (채택): LangChain `.with_fallbacks()` 로 멀티 프로바이더 체인
- LangChain의 `RunnableWithFallbacks` 패턴으로 primary LLM 실패 시 자동으로 다음 프로바이더로 전환
- 각 LLM에 `max_retries=1` 설정으로 tenacity retry 최소화 → 쿼타 소진 시 빠르게 다음 프로바이더로 이동
- **선택 이유**:
  - ssuMCP의 `ssuai-backend-secrets`에 이미 Groq, OpenRouter, Cerebras, Fireworks 등 다수 API 키 등록되어 있음
  - LangChain 표준 API라 코드 변경 최소화 (`primary.with_fallbacks([groq_llm, openrouter_llm])`)
  - 완전 자동 — 운영자 개입 없이 쿼타 소진 시 즉시 전환

## 결정

**Fallback 체인: Gemini 2.5 Flash → Groq Llama 3.3 70B → OpenRouter Llama 3.3 70B**

```
create_llm() 반환값:
  ChatGoogleGenerativeAI(gemini-2.5-flash, max_retries=1)
    .with_fallbacks([
      ChatOpenAI(groq, llama-3.3-70b-versatile, max_retries=1),
      ChatOpenAI(openrouter, meta-llama/llama-3.3-70b-instruct:free, max_retries=1)
    ])
```

### 프로바이더 선택 근거

| 프로바이더 | 모델 | 선택 이유 |
|---|---|---|
| Gemini 2.5 Flash | Primary | 한국어 품질 최우수, 기존 사용 모델 |
| Groq Llama 3.3 70B | 1st fallback | 무료 티어, 매우 빠름(추론 속도 1위권), `langchain-openai`로 연결 가능 |
| OpenRouter Llama 3.3 70B | 2nd fallback | 다수 오픈모델 접근 가능한 aggregator, 무료 모델 존재 |

### 구현 포인트

- `ssu_agent/llm_factory.py` 신설: 환경 변수 존재 여부에 따라 동적으로 fallback chain 구성
- `GROQ_API_KEY`, `OPENROUTER_API_KEY` env var 없으면 해당 프로바이더 건너뜀 (Graceful degradation)
- `supervisor/graph.py`, `agents/library.py`, `agents/academic.py`, `agents/lms.py` 모두 `create_llm()` 사용
- `pyproject.toml`에 `langchain-openai` 추가 (OpenAI-compatible API 클라이언트)
- k8s secret `ssuagent-secrets`에 `GROQ_API_KEY`, `OPENROUTER_API_KEY` 추가

## 동작 방식

1. 요청이 들어오면 primary(Gemini)로 LLM 호출
2. Gemini가 429(쿼타 소진) 또는 기타 예외 발생 시 `RunnableWithFallbacks`가 자동으로 Groq으로 재시도
3. Groq도 실패하면 OpenRouter로 재시도
4. 세 곳 모두 실패하면 최종 예외 raise

`max_retries=1` 설정으로 각 프로바이더에서 1회 재시도 후 빠르게 다음 프로바이더로 이동 (기존 기본값 6회 대비 응답 속도 개선).

## 검증

- 19개 unit test 통과 (mock LLM으로 fallback chain 구성 검증)
- pod 내부에서 `gemini-2.5-flash` 직접 호출 → 2.5초 내 응답 확인
- `GROQ_API_KEY`, `OPENROUTER_API_KEY` pod 환경 변수 주입 확인

## 관련 파일 및 커밋

- `ssu_agent/llm_factory.py` (신규)
- `ssu_agent/config.py` (`GROQ_API_KEY`, `OPENROUTER_API_KEY` 추가)
- `pyproject.toml`, `uv.lock` (`langchain-openai` 추가)
- 커밋: `2541aa8` (feat), `5e36c10` (lint fix)

## 예상 면접 질문

1. LangChain `.with_fallbacks()`의 동작 원리는? 어떤 예외가 발생했을 때 fallback이 트리거되나요?
2. `max_retries=1`로 설정한 이유는? 기본값과 비교해 트레이드오프는?
3. 여러 LLM 프로바이더를 사용할 때 응답 포맷(특히 tool calling) 차이를 어떻게 처리하나요?

## 갱신 (2026-07-02) — 실제 출하 구현과의 차이

위 원문은 채택 시점(2026-06-15)의 설계를 기록한 역사적 문서로 보존한다. 이후 구현 과정에서 세 가지가 바뀌었고, 최종 출하 상태는 다음과 같다 (`ssu_agent/llm_factory.py` 기준):

1. **메커니즘: `RunnableWithFallbacks` → `get_llm_sequence()` + 에이전트별 수동 retry 루프**
   - langchain_core 1.4.x의 `RunnableWithFallbacks`는 `bind_tools`를 지원하지 않아, `create_react_agent`가 내부에서 `model.bind_tools()`를 호출하는 순간 `.with_fallbacks()` 체인이 깨진다 (`llm_factory.py` 상단 NOTE).
   - 따라서 `create_llm()`이 폴백 체인을 반환하는 대신, `get_llm_sequence()`가 우선순위 순 LLM 리스트를 반환하고 각 에이전트가 수동 retry 루프에서 순서대로 시도한다.

2. **폴백 순서: Gemini-first → Groq-first**
   - 최종 순서: **Groq(llama-3.3-70b-versatile) → Gemini(`GEMINI_MODEL`, 기본 gemini-2.5-flash) → OpenRouter(meta-llama/llama-3.3-70b-instruct:free)**
   - 이유: 당시 무료 구간의 요청 여유가 더 크다는 운영 가정과 빠른 추론 속도를 근거로 Groq를 먼저 두었다. 이 비교는 모델·조직·시점에 따라 바뀌는 외부 조건이며 현재 보장값이 아니다. Gemini는 한국어 품질을 고려해 2순위로 유지하고, OpenRouter는 catch-all aggregator로 최후순위에 두었다 (`llm_factory.py` docstring, README, 아래 2026-07-18 갱신).

3. **Groq 클라이언트: `ChatOpenAI`(base_url) → `ChatGroq`**
   - 제네릭 `ChatOpenAI` 래퍼는 assistant content를 content-block 리스트로 직렬화하는데, Groq API가 두 번째 tool call turn에서 이를 400으로 거부한다. `ChatGroq`는 string-content 변환을 내부에서 처리한다 (`fix/chatgroq-message-format`).

프로바이더 순서 회귀는 `tests/test_llm_factory.py`의 provider-order 테스트로 고정된다.

## 갱신 (2026-07-18) — 외부 쿼터를 설계 계약에서 분리

위 2026-07-02 갱신의 무료 쿼터 비교는 당시의 운영 판단 기록이며 현재 보장값이 아니다.
Groq를 포함한 프로바이더의 모델별 한도, 가격, 제공 여부는 조직과 시점에 따라 바뀌고
정확한 값은 계정 콘솔이 소유한다. 따라서 README와 `llm_factory.py`에서는 고정 쿼터를
제거하고, 우선순위·키별 opt-in·실패 시 다음 provider로 이동하는 코드 계약만 남겼다.

현재 순서는 tool-call 호환성과 기존 운영 선택을 보존하기 위한 명시적 정책이다. 비용
또는 정확도 우위를 뜻하지 않는다. 배포 전에는 공식 provider 문서와 계정 한도를 다시
확인하고, 모델/버전/데이터셋/비용 경계가 기록된 별도 평가 없이 품질 우위를 주장하지
않는다.
