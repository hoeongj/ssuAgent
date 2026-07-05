# ADR-005: LMS 강의자료 내보내기 도구 라우팅 및 수퍼바이저 라우터 충돌 버그 수정

## Status
Accepted (2026-06-16)

## 배경

Phase C에서 LMS 비영상 자료 ZIP 내보내기 기능이 구현되면서, 다음 4종의 신규 LMS MCP 도구가 도입되었다:
1. `get_my_lms_courses` (수강 과목 목록 조회)
2. `get_my_lms_materials` (선택 과목의 비영상 자료 목록 조회)
3. `prepare_lms_material_export` (내보낼 자료 검증 및 확인 요청 생성)
4. `confirm_lms_material_export` (확인 후 ZIP 내보내기 시작)

이 과정에서 수퍼바이저 라우터의 심각한 라우팅 버그가 발견되었다.
수퍼바이저 그래프(`ssu_agent/supervisor/graph.py`)의 `categorise_tools()` 함수는 도서관(Library) 도구를 분류하기 위해 `_LIBRARY_PREFIXES` 내의 접두사를 기반으로 검사한다.
기존 `_LIBRARY_PREFIXES`에는 `"prepare_"` 접두사가 포함되어 있었는데, `categorise_tools()` 내의 `if/elif` 분기 체인상에서 Library 분류가 LMS 분류보다 먼저 위치하고 있었다.
이로 인해 신규 LMS 도구인 `prepare_lms_material_export`가 `"prepare_"` 접두사 매칭에 걸려 LMS 에이전트가 아닌 Library 에이전트로 잘못 분류(Collision)되어 LMS 비영상 자료 내보내기 흐름이 오작동하는 버그가 발생했다.

## 고려한 대안

### 대안 A (채택): Library 도구의 `prepare_*` 명칭만 exact match 검사
- `_LIBRARY_PREFIXES`에서 `"prepare_"` 접두사를 제거한다.
- 실제 도서관 준비 도구들의 정확한 전체 명칭(`prepare_reserve_library_seat`, `prepare_swap_library_seat`, `prepare_cancel_library_seat`, `confirm_action`)을 포함하는 `_LIBRARY_NAMES` exact match set을 정의한다.
- `categorise_tools()`에서는 `any(name.startswith(p) for p in _LIBRARY_PREFIXES) or name in _LIBRARY_NAMES` 형태로 검사한다.
- **선택 이유**:
  - 접두사 검출 범위를 필요 최소한으로 좁히는 가장 안전하고 근본적인 해결책이다.
  - 향후 다른 도메인(LMS, Academic 등)에서 `prepare_`로 시작하는 새로운 도구가 추가되더라도 충돌할 우려가 영구히 제거된다.

### 대안 B: `categorise_tools()` 내의 `if/elif` 분기 순서 조정
- LMS 도구 목록 검사(`elif name in _LMS_NAMES`)를 Library 도구 접두사 검사(`elif any(name.startswith(p) ... )`)보다 위로 배치한다.
- **거부 이유**:
  - 간단하게 구현할 수 있으나, 여전히 `_LIBRARY_PREFIXES`가 광범위한 `"prepare_"` 접두사를 삼키는 근본적인 위험은 잔존한다.
  - 만약 LMS 도구 외에 다른 영역(예: Academic)에서 `prepare_*` 형태의 도구를 도입할 경우 유사한 충돌 버그가 다시 발생할 수 있어 근본적인 아키텍처 방어막으로서는 취약하다.

## 결정

대안 A를 채택하여, `ssu_agent/supervisor/graph.py`의 `_LIBRARY_PREFIXES`에서 `"prepare_"` 및 `"confirm_action"`을 제거하고, 정확한 라이브러리 도구 세트 `_LIBRARY_NAMES`를 정의하여 매칭하도록 리팩토링하였다.

## 구현 상세

1. **`ssu_agent/supervisor/graph.py`**:
   - `_LIBRARY_PREFIXES`에서 `"prepare_"` 및 `"confirm_action"` 제거
   - `_LIBRARY_NAMES`에 `prepare_reserve_library_seat`, `prepare_swap_library_seat`, `prepare_cancel_library_seat`, `confirm_action` 등록
   - `_LMS_NAMES`에 신규 4종 도구 추가
   - `categorise_tools()`의 Library 에이전트 분류 로직을 `_LIBRARY_NAMES` exact match 검사 추가로 변경
   - `transfer_to_lms_agent` docstring 및 `_SUPERVISOR_PROMPT`에 LMS 자료 내보내기 플로우 가이드 보완
2. **`ssu_agent/agents/lms.py`**:
   - `_SYSTEM_PROMPT_BASE` 및 `_build_lms_prompt()`에 신규 4개 도구 설명 및 `mcp_session_id` 주입 명세 반영
3. **`tests/test_supervisor.py`**:
   - 신규 4개 LMS 도구를 Mock 도구로 등록
   - `test_categorise_splits_lms_tools()`에 `prepare_lms_material_export`가 `library_names`에 속하지 않고 `lms_names`에 정상적으로 들어감을 보장하는 회귀 테스트 추가

## 검증

- `ssu_agent` 테스트 스위트 (`.venv\Scripts\python -m pytest tests/ -v`) 19개 항목 전원 통과 확인.

## 예상 면접 질문

1. **"에이전트 라우팅 시 도구 접두사 매칭 충돌 버그가 발생했을 때 어떻게 대처했나요?"**
   - 기존의 광범위한 접두사 매칭(`prepare_`)이 새로 추가된 도메인의 도구명과 겹치면서 오분류가 발생했습니다. 단순한 조건 분기 순서 변경에 그치지 않고, 충돌 가능성이 있는 도구들의 명칭을 정확한 집합(Exact Match Set)으로 변환하고 접두사 집합을 정밀하게 축소함으로써 버그의 재발 가능성을 차단했습니다.
