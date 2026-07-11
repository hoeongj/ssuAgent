# ADR 0015 - Optional Anthropic Claude provider for temporary dev testing

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-07-11 |
| 상태 | Accepted |
| 범위 | `ssu_agent/config.py`, `ssu_agent/llm_factory.py`, `tests/test_llm_factory.py` |
| 관련 | [ADR 004](004-multi-provider-llm-fallback.md) |

## 배경

프로덕션에서 Groq, Gemini, OpenRouter로 구성된 무료 LLM 체인이 쿼터 소진 상태에 자주
도달하면서 예약 등 LLM 의존 액션이 `All LLM providers exhausted`로 실패했다. 사용자는
이 장애를 개발 및 테스트 중에 재현하고 우회하기 위해 임시 유료 Claude API 키를
제공했다.

이 키는 장기 운영 전환이 아니라 임시 dev/test 용도다. 따라서 키가 없을 때는 기존 무료
체인의 동작을 그대로 유지하고, 키 제거만으로 즉시 원상복구할 수 있어야 한다.

## 결정

`ANTHROPIC_API_KEY`가 존재할 때만 Claude Haiku 4.5를 LLM provider 시퀀스 최상단에
prepend한다. 모델 기본값은 `claude-haiku-4-5`이며, 필요하면 `ANTHROPIC_MODEL`로
오버라이드할 수 있다.

무료 provider 체인인 Groq → Gemini → OpenRouter는 그대로 fallback으로 유지한다. 키가
없으면 Anthropic provider는 생성하지 않으므로 기존 무료 체인으로 바로 복귀한다. 이
구성은 secret에서 `ANTHROPIC_API_KEY`를 제거하고 프로세스를 재시작하는 것만으로
되돌릴 수 있다.

## 검토한 대안과 기각 사유

### 무료 provider만 유지

무료 쿼터 소진이 반복되어 프로덕션에서 예약 등 핵심 LLM 의존 액션이 계속 실패한다.
동일한 체인만 유지하면 재발을 막을 수 없어 기각했다.

### `with_fallbacks` 사용

기존 ADR과 `llm_factory.py` 주석대로 `RunnableWithFallbacks`는 `bind_tools`를 지원하지
않아 `create_react_agent`의 tool binding 단계에서 깨진다. 현재의
`get_llm_sequence()`와 에이전트별 retry loop 구조를 유지한다.

### 특정 에이전트만 Claude 사용

예약 실패처럼 특정 workflow에서 관찰된 문제가 있더라도, provider gate를 에이전트마다
나누면 설정과 되돌리기 절차가 복잡해진다. 임시 dev/test 목적에 맞게 시퀀스 최상단에
일괄 적용하는 방식으로 단순화했다.

## 동작 방식

`get_llm_sequence()`는 `ANTHROPIC_API_KEY`가 비어 있지 않을 때만
`ChatAnthropic(model="claude-haiku-4-5", max_tokens=2048, max_retries=1)`를 첫 번째
항목으로 추가한다. 이후 기존 순서인 Groq, Gemini, OpenRouter를 그대로 추가한다.

에이전트들은 기존처럼 이 리스트를 순서대로 시도한다. Claude 호출이 실패하면 기존 무료
provider들이 fallback으로 사용되고, 키가 없으면 리스트 구성은 기존과 동일하다.

## 비용 및 되돌리기

Claude Haiku 4.5 비용은 MTok 기준 입력 $1, 출력 $5로 본다. 이 변경은 임시 개발 및
테스트용이며, 장기 운영 정책 결정은 별도 ADR에서 다룬다.

되돌리기는 `ANTHROPIC_API_KEY`를 secret에서 제거하고 ssuAgent를 재시작하면 된다. 코드
되돌림 없이도 무료 provider 체인으로 즉시 복귀한다.
