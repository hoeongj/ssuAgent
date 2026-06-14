# ssuAgent Containerization & k3s Deployment Guide

이 문서는 ssuAgent의 컨테이너화(Containerization) 및 k3s 클러스터 배포 환경 구축에 대해 다룹니다.

## 배경 (Background)

- **Phase 2 로컬 실행**: 기존에는 로컬 환경에서 `python ssu_agent/main.py` 형태로 직접 프로세스를 띄워 실행하였으며, 대화 상태 보존(HITL interrupt 및 대화 메모리)을 위해 `SqliteSaver`를 사용한 파일 형태의 로컬 DB를 참조했습니다.
- **문제점**: Kubernetes 배포 시 파드(Pod)가 재시작되면 로컬 파일 시스템에 저장된 SQLite 파일이 유실되어 대화 상태 및 승인 대기 상태가 초기화되는 문제가 있습니다.
- **Phase 3 목표**: ssuAgent 서비스를 Docker 이미지로 패키징하여 GitHub Container Registry (GHCR)에 푸시하고, k3s 클러스터 내에서 ArgoCD 및 ArgoCD Image Updater를 통해 지속적인 통합/배포(CI/CD)를 구축합니다.

---

## 체크포인터 의사결정 (Checkpointer Decision)

LangGraph의 대화 상태를 저장하기 위한 체크포인터로 아래의 대안들을 평가했습니다.

### 대안 비교

1. **후보 A: SqliteSaver (기존)**
   - **장점**: 별도의 DB 인프라 구축 없이 로컬 파일 형태로 빠르게 구축 가능.
   - **단점**: 파드 재시작 시 상태가 소멸하므로 Persistent Volume Claim (PVC) 마운트가 필요하며, 멀티 레플리카 환경(ReadWriteOnce 제한)에서 다중 파드 간 SQLite 파일 공유가 불가능합니다.
   - **결정**: **탈락**

2. **후보 B: langgraph-checkpoint-postgres (채택)**
   - **장점**: 기존 k3s 클러스터 내 구동 중인 PostgreSQL 데이터베이스(`postgres-service:5432/ssuai`)를 그대로 활용 가능합니다. 파드 재시작 후에도 대화 재개 및 HITL interrupt 영속성이 완벽히 보장되며, 멀티 레플리카 스케일아웃이 자유롭습니다. ACID 트랜잭션을 지원하여 신뢰성이 높습니다.
   - **결정**: **채택**

### 세부 구현 설정

- **AsyncConnectionPool 사용**: PostgreSQL 연결 효율을 극대화하기 위해 `psycopg_pool`의 `AsyncConnectionPool`을 백엔드로 사용합니다. (`max_size=5` 풀 크기 지정)
- **autocommit=True 필수**: LangGraph Postgres 체크포인터는 자체적으로 트랜잭션 내에서 `savepoint`를 정의하고 관리하므로, DB 커넥션 풀 초기 설정 시 `autocommit=True`가 강제됩니다.
- **prepare_threshold=0 설정**: psycopg3의 prepared statement 기능을 비활성화합니다. LangGraph 내부 쿼리 엔진과 prepared statement 캐싱이 충돌을 유발하여 에러가 발생할 수 있기 때문에 안정성을 위해 비활성화 설정을 적용합니다.
- **migration 자동화**: FastAPI `lifespan` 시작 시 `checkpointer.setup()`이 호출되며 필요한 테이블(`checkpoint_blobs`, `checkpoint_writes` 등)이 없으면 자동으로 생성(마이그레이션)합니다.

---

## Docker 멀티스테이지 빌드 (Dockerfile Design)

Oracle Ampere A1 (ARM64 아키텍처) 기반의 k3s 환경을 타겟으로 멀티스테이지 빌드를 설계했습니다.

1. **Stage 1 (Builder)**: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
   - 초고속 Python 패키지 인스톨러인 `uv` 빌더 이미지를 베이스로 삼고, `pyproject.toml` 및 `uv.lock`을 복사해 가상환경(`.venv`)을 구성합니다.
   - `--no-dev --frozen` 플래그로 락파일의 개발용 의존성을 제외하고 프로덕션 의존성만 고정 설치합니다.
2. **Stage 2 (Runtime)**: `python:3.12-slim`
   - 불필요한 빌드 도구와 캐시가 제거된 경량 런타임 이미지입니다.
   - 1단계 빌더에서 준비된 가상환경(`.venv`)과 `ssu_agent` 소스 코드만 복사하여 최종 이미지 크기를 극대화하여 줄입니다.
   - 컨테이너 보안을 위해 비루트(non-root) 실행 환경을 권장하도록 구성합니다.

---

## GitHub Actions CI 파이프라인 (CI/CD Workflow)

`.github/workflows/ci.yml`을 통해 자동화된 검증 및 이미지 빌드를 수행합니다.

1. **Test Job**:
   - `ruff check .` 및 `ruff format --check .` 린터 검사 수행.
   - `pytest`를 통한 단위 테스트 검증.
2. **Docker Job (needs: test)**:
   - `main` 브랜치에 푸시되었을 때만 실행됩니다.
   - `setup-qemu-action` 및 `setup-buildx-action`을 사용하여 `linux/arm64` 멀티플랫폼 이미지 크로스컴파일을 지원합니다.
   - GitHub Token을 기반으로 GHCR에 로그인한 후, `ghcr.io/ghdtjdwn/ssuagent:sha-<commit_sha>` 태그로 빌드 및 푸시합니다.
   - `type=gha` 캐시를 활용하여 레이어 캐싱으로 빌드 시간을 획기적으로 단축합니다.

---

## ArgoCD Image Updater 배포 파이프라인

k3s 클러스터 내 ArgoCD가 아래와 같이 ssuAgent 서비스를 자동 동기화 및 롤아웃합니다.

```mermaid
graph TD
    A[Local Code Push] -->|Git Commit| B(GitHub Repository)
    B -->|Trigger CI/CD| C[GitHub Actions]
    C -->|Ruff & Pytest| D{Tests Pass?}
    D -->|Yes| E[Build & Push Docker Image]
    E -->|ghcr.io/ghdtjdwn/ssuagent:sha-hash| F(GitHub Container Registry)
    F -->|Detect New Build Tag| G[ArgoCD Image Updater]
    G -->|Write Back git: image.tag| B
    B -->|Webhook Trigger| H[ArgoCD Sync]
    H -->|Helm Apply| I[k3s Cluster ssuai-prod]
```

1. **태그 감지**: ArgoCD Image Updater가 `newest-build` 전략을 바탕으로 `sha-<hash>` 정규식 태그 형태의 최신 이미지를 주기적으로 감지합니다.
2. **Write-back**: 이미지 업데이트가 확인되면, ssuAgent 레포지토리의 `deploy/charts/ssu-agent/values.yaml` 내 `image.tag` 값을 새 커밋 해시로 자동 커밋 및 푸시(`write-back`)합니다.
3. **Helm 배포**: ArgoCD가 values.yaml의 변경점을 인식하고 동기화(Sync)를 수행하여 k3s 클러스터에 Rolling Update 배포를 수행합니다.

---

## k3s 클러스터 수동 설정 사항 (Manual Setup)

배포 전, 클러스터 어드민이 수동으로 세팅해야 하는 인프라 항목입니다.

### 1. Kubernetes Secret 생성
파드 구동에 절대적으로 필요한 API Key와 데이터베이스 접속용 DSN은 보안을 위해 Secret 객체로 분리하여 관리하며, Helm Chart는 이를 환경변수(`envFrom`)로 주입받습니다.

```bash
kubectl create secret generic ssuagent-secrets \
  --namespace ssuai-prod \
  --from-literal=GOOGLE_API_KEY="your-gemini-api-key" \
  --from-literal=DATABASE_URL="postgresql://ssuai:your-db-password@postgres-service:5432/ssuai"
```
*주의: `DATABASE_URL` 내의 DB 패스워드는 `ssuai-backend-secrets`에 설정된 `SSUAI_DB_PASSWORD` 값과 일치해야 합니다.*

### 2. DNS A 레코드 등록
`ssuagent.duckdns.org` 도메인 명의 트래픽이 k3s 트래픽을 처리하는 Public Gateway IP(`168.110.104.199`)로 도달할 수 있도록 DuckDNS 혹은 해당 네임서버에 A 레코드를 매핑해야 합니다.
Ingress 컨트롤러(Traefik)는 이 호스트 요청을 감지하고, `letsencrypt-prod` ClusterIssuer를 통해 자동으로 TLS 인증서를 갱신 및 오프로딩합니다.
