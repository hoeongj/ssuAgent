# ssuAgent 문서 지도

이 저장소의 현재 코드, 테스트, CI와 배포 manifest가 최종 기준이다. 문서와 충돌하면 해당 구현을
먼저 확인하고 문서를 함께 수정한다.

## 시작하기

- [프로젝트 개요와 로컬 실행](../README.md)
- [설정과 환경 변수](configuration.md)
- [아키텍처와 신뢰 경계](architecture.md)

## 설계 결정

- [ADR 디렉터리](adr/)
- [Supervisor architecture](adr/0001-supervisor-architecture.md)
- [PostgreSQL checkpointer](adr/003-postgres-checkpointer.md)
- [Agent edge hardening](adr/0009-agent-edge-hardening.md)
- [Thread ownership binding](adr/0010-agent-thread-ownership-binding.md)
- [Stable principal binding](adr/0011-thread-stable-principal-binding.md)
- [Optional Anthropic provider](adr/0015-optional-anthropic-provider.md)
- [Deterministic LMS export](adr/0022-deterministic-lms-export-download.md)

## 운영

- [GitOps 배포와 검증](deploy.md)
- [운영 장애 기록](troubleshooting.md)
- [GitHub Actions CI](../.github/workflows/ci.yml)
- [Helm chart](../deploy/charts/ssu-agent/)
- [ArgoCD Application](../deploy/argocd/application-ssu-agent.yaml)

## 검증 책임

- Python 코드: Ruff check/format과 pytest가 기준이다.
- Stream/HITL 계약: `tests/test_stream_interrupt.py`, `tests/test_supervisor.py`가 기준이다.
- 인증과 thread 소유권: `tests/test_main_security.py`, `tests/test_auth_guard.py`가 기준이다.
- 운영 배포: GitHub Actions image job, Image Updater write-back, ArgoCD 상태, running image SHA와
  `/healthz/deep`을 함께 확인한다.
