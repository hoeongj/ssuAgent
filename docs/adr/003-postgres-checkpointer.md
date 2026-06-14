# ADR-003: LangGraph 체크포인터 마이그레이션 (SqliteSaver -> AsyncPostgresSaver)

## Status
Accepted (2026-06-14)

## Context (배경 및 맥락)
- ssuAgent의 캠퍼스 비서 서비스는 사용자와의 대화 히스토리 및 Human-in-the-loop (HITL) interrupt 승인 대기 상태를 영속적으로 보존하기 위해 LangGraph 체크포인터를 활용합니다.
- 기존 Phase 2 단계에서는 `SqliteSaver`를 사용하여 로컬 파일 시스템 내 `ssu_agent_checkpoints.db`에 상태를 기록하고 있었습니다.
- 그러나 Kubernetes (k3s) 배포 환경으로 전환함에 따라 파드(Pod)가 예기치 않게 재시작되거나, 트래픽 분산을 위해 복수 개의 레플리카(Replica)로 스케일 아웃될 때 다음과 같은 치명적인 한계점이 노출되었습니다.
  1. 컨테이너 재시작 시 파일 시스템이 초기화되어 진행 중이던 대화 세션 및 결제/예약 등의 승인 대기 상태가 유실됨.
  2. SQLite 파일은 단일 파일 잠금 방식이므로 여러 파드 인스턴스가 동시 접근하여 체크포인트를 기록할 수 없어 분산 환경에 부적합함.

## Decision (결정 사항)
- SQLite 기반의 기존 체크포인터를 PostgreSQL 기반의 `langgraph-checkpoint-postgres` (`AsyncPostgresSaver`)로 대체합니다.
- 비동기 DB 커넥션 관리를 위해 `psycopg_pool` 패키지의 `AsyncConnectionPool`을 백엔드로 연동합니다.
- Kubernetes 배포 환경에서 환경변수 `DATABASE_URL`을 주입받아 기존 구동 중인 `postgres-service` 데이터베이스 서비스를 공유하도록 구성합니다.

## Alternatives Considered (고려한 대안들)

### 1. SqliteSaver + Persistent Volume Claim (PVC)
- **설명**: 기존 SQLite 체크포인터를 유지하되, 컨테이너 볼륨을 k3s의 Persistent Volume에 마운트하여 데이터 유실을 방지하는 방안.
- **기각 이유**:
  - `ReadWriteOnce` PV 마운트의 경우, 파드를 2개 이상 띄우는 스케일 아웃(ReplicaCount >= 2)이 불가능하여 고가용성(HA) 구성에 제약이 생김.
  - `ReadWriteMany` PV를 쓰더라도 SQLite 파일 자체의 다중 프로세스 쓰기 잠금 충돌 위험이 상존함.

### 2. Redis 체크포인터 (langgraph-checkpoint-redis)
- **설명**: Redis 인메모리 저장소를 체크포인터 저장소로 사용하는 방안.
- **기각 이유**:
  - Redis 인메모리 특성상 영속성(Persistence) 보장을 위해 별도의 RDB/AOF 설정이 요구됨.
  - 기존 인프라에 PostgreSQL이 이미 안정적으로 구축되어 운영 중이므로, 오직 체크포인트를 위해 추가적인 Redis 인프라 클러스터를 구축하고 관리하는 것은 오버헤드가 큼.

### 3. PostgreSQL 체크포인터 (AsyncPostgresSaver) - 채택
- **설명**: PostgreSQL 데이터베이스에 `checkpoint_blobs` 등의 테이블을 생성해 체크포인트를 영속화하는 방안.
- **채택 이유**:
  - 이미 k3s 내부에 서비스 중인 PostgreSQL(`postgres-service`)을 그대로 활용하므로 추가 인프라 유지 보수 비용이 들지 않음.
  - 완전한 ACID 트랜잭션을 지원하여 여러 파드가 동시에 여러 스레드의 체크포인트를 저장해도 데이터 일관성을 유지함.
  - 대화 스레드 단위의 분산 접근 처리에 최적화되어 있어 무중단 배포 및 HPA(Horizontal Pod Autoscaler) 환경에 완벽 대응이 가능함.

## Consequences (의사결정 결과)
- **DB 마이그레이션**: `AsyncPostgresSaver` 인스턴스 초기화 직후 `checkpointer.setup()` 메소드를 호출함으로써, 최초 시작 시 필요한 테이블 구조가 존재하지 않으면 DDL 쿼리가 실행되어 자동으로 테이블이 구성됩니다. 수동 DB 마이그레이션 비용이 제거됩니다.
- **환경 설정**: 어플리케이션 환경변수에 `DATABASE_URL` DSN 정보가 주입되어야 합니다. 로컬 개발 환경은 `postgresql://ssuai:dev@localhost:5432/ssuai`를 기본값으로 사용해 개발자 편의성을 유지합니다.
- **의존성 확장**: `langgraph-checkpoint-postgres`와 `psycopg[binary,pool]`가 신규 런타임 의존성으로 추가되었습니다.
