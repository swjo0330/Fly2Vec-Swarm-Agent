# Fly2Vec 상세 구현 설계안

**작성일**: 2026-05-02 | **상태**: 구현 진행 중

---

## 1. 구현 단계 (Phase 분류)

### Phase 1: 인프라 ✅ 완료
- **목표**: Docker + pyproject.toml + .env + 프로젝트 골격
- **산출물**: `docker/`, `pyproject.toml`, `.env.example`, `RULES.md`, `README.md`
- **의존성**: 없음

### Phase 2: Base Layer ✅ 완료
- **목표**: LangGraph 에이전트 기반 + A2A 통신 + LLM Factory
- **산출물**:
  - `src/base/fly2vec_base_agent.py` — Fly2VecBaseAgent
  - `src/base/fly2vec_state.py` — Fly2VecState, MissionPlannerState
  - `src/base/fly2vec_a2a_interface.py` — Fly2VecA2AAgent
  - `src/a2a/fly2vec_client.py` — Fly2VecClientManager
  - `src/a2a/fly2vec_executor.py` — Fly2VecExecutor
  - `src/llm/fly2vec_llm_factory.py` — build_fly2vec_llm()
- **의존성**: Phase 1

### Phase 3: Qdrant + RAG ✅ 완료
- **목표**: 벡터DB 매니저 + 컬렉션 정의
- **산출물**:
  - `src/qdrant/collections.py` — 상수 (128d, COSINE)
  - `src/qdrant/fly2vec_qdrant.py` — Fly2VecQdrantManager
- **의존성**: Phase 1

### Phase 4: Graph Embedding MCP ✅ 완료
- **목표**: embed_mission, find_similar, anomaly_score 도구 서버
- **산출물**:
  - `src/mcp_servers/fly2vec_mcp_base.py` — Fly2VecMCPBase
  - `src/mcp_servers/graph_embedding_mcp.py` — GraphEmbeddingMCP
  - `src/embedding/node2vec_encoder.py` — Node2VecEncoder
  - `src/embedding/gcn_encoder.py` — GCNEncoder
- **의존성**: Phase 2, 3

### Phase 5: Fly2Vec-Swarm MCP ✅ 완료
- **목표**: analyze_swarm, detect_conflict, optimize_assignment 도구 서버
- **산출물**:
  - `src/mcp_servers/swarm_analysis_mcp.py` — SwarmAnalysisMCP
  - `src/embedding/rgat_encoder.py` — RGATEncoder
- **의존성**: Phase 4

### Phase 6: MissionPlannerAgent ✅ 완료
- **목표**: 검증-수정 루프 에이전트
- **산출물**: `src/agents/mission_planner_agent.py` — MissionPlannerAgent
- **의존성**: Phase 4, 5

### Phase 7: 기존 실험 데이터 통합 ⬜ TODO
- **목표**: 기존 run_*.py 결과(CSV, 모델)를 Qdrant에 초기 로드
- **산출물**:
  - `src/data/loader.py` — 기존 CSV → PG JSON → 임베딩 → Qdrant 벌크 로드
  - `data/` 디렉토리 (기존 데이터 심볼릭 링크 또는 복사)
- **기존 파일 연결**:
  - `proposal/data/dataset_mission_plan.csv` (2,769건)
  - `proposal/data/dataset_wp_spot.csv` (47,266 WP)
  - `proposal/results/anomaly_results_full.csv` (1,477건 앙상블 점수)
  - `proposal/results/cluster_summary.csv`
  - `proposal/results/enriched_routes.csv` (MapBox 보강)
- **의존성**: Phase 3, 4

### Phase 8: 통합 테스트 ⬜ TODO
- **목표**: E2E 파이프라인 검증
- **산출물**: `tests/`, `docker compose up` 성공
- **의존성**: Phase 7

---

## 2. 클래스 다이어그램

```
Fly2VecBaseAgent (LangGraph 기반)
  ├── MissionPlannerAgent (검증-수정 루프)
  │     ├── uses: GraphEmbeddingMCP (via MCP tools)
  │     ├── uses: SwarmAnalysisMCP (via MCP tools)
  │     └── state: MissionPlannerState
  └── (향후) AnalysisAgent (배치 분석)

Fly2VecMCPBase (FastMCP 기반)
  ├── GraphEmbeddingMCP
  │     ├── uses: Node2VecEncoder
  │     ├── uses: GCNEncoder
  │     └── uses: Fly2VecQdrantManager
  └── SwarmAnalysisMCP
        ├── uses: RGATEncoder
        └── uses: Fly2VecQdrantManager

Fly2VecA2AAgent (A2A 인터페이스)
  └── Fly2VecExecutor → MissionPlannerAgent

Fly2VecClientManager → 외부 에이전트 호출
```

---

## 3. 데이터 흐름 상세

```
[입력]
  자연어: "서울역→인천공항 50m" 또는 PG JSON 직접 입력
    │
    ▼
[MissionPlannerAgent]
  parse_input → messages에서 의도 추출
    │
    ▼
  generate_pg → (MapBox MCP 호출) → PG JSON 생성
    │
    ▼
  ┌─── 검증-수정 루프 (max 3회) ───┐
  │                                 │
  │  embed_and_check:               │
  │    ① PG → Node2VecEncoder.build_graph() → NetworkX
  │    ② Node2VecEncoder.encode_mission() → 128d 벡터
  │    ③ GraphEmbeddingMCP.anomaly_score(vector)
  │    ④ SwarmAnalysisMCP.detect_conflict(missions)
  │                                 │
  │  판단: anomaly > 0.5 OR conflict > 0?
  │    YES → modify_pg (LLM 수정) → 루프 재실행
  │    NO  → finalize            │
  └─────────────────────────────────┘
    │
    ▼
[출력]
  최종 PG JSON + anomaly_score + similar_missions + swarm_analysis
```

---

## 4. MCP 도구 상세 명세

### 4.1 Graph Embedding MCP (port 8050)

| 도구 | 입력 | 출력 | 에러 처리 |
|------|------|------|----------|
| `embed_mission` | `pg_json: dict`, `store: bool=True` | `{vector: [128d], point_id: str}` | PG 파싱 실패 → Fly2VecErrorResponse |
| `find_similar` | `vector: list[float]`, `top_k: int=5` | `{similar_missions: [{score, payload}], count}` | Qdrant 미연결 → 빈 리스트 |
| `anomaly_score` | `vector: list[float]` | `{score: 0~1, is_anomalous: bool, breakdown: {...}}` | 데이터 부족 → score=1.0 |

### 4.2 Swarm Analysis MCP (port 8051)

| 도구 | 입력 | 출력 | 에러 처리 |
|------|------|------|----------|
| `analyze_swarm` | `missions: list[dict]` | `{pattern, anomaly_avg, drone_scores, edge_stats}` | 미션 < 2 → cooperative_search |
| `detect_conflict` | `missions: list[dict]`, `proximity_threshold_m: float=500` | `{conflict_count, conflicts: [{drone_a, drone_b, distance_m}], has_conflict}` | 빈 미션 → 0건 |
| `optimize_assignment` | `missions: list[dict]`, `drones: list[dict]` | `{assignments, coverage_score, conflict_risk}` | 드론 0대 → 에러 |

---

## 5. Qdrant 컬렉션 스키마

### fly2vec_mission_paths (단일 경로)

| 필드 | 타입 | 용도 |
|------|------|------|
| vector | float[128] | Node2Vec/GCN 미션 임베딩 |
| mission_type | str | waypoint_sequence, grid_search, circular_flight, linear_flight |
| wp_count | int | 웨이포인트 수 |
| region | str | 지역명 (MapBox reverse_geocode) |
| anomaly_score | float | 저장 시점 이상 점수 |
| flight_result | str | null → success/failed (비행 후 업데이트) |
| created_at | str | ISO 8601 |

### fly2vec_swarm_missions (군집 분석)

| 필드 | 타입 | 용도 |
|------|------|------|
| vector | float[128] | R-GAT 군집 임베딩 |
| pattern | str | cooperative_search, collision_risk, coverage_overlap, relay_move |
| drone_count | int | 군집 드론 수 |
| anomaly_avg | float | 군집 평균 이상 점수 |
| proximity_edges | int | Proximity 엣지 수 |
| coverage_edges | int | Coverage 엣지 수 |

---

## 6. LangGraph 에이전트 노드/엣지 명세

### MissionPlannerState (TypedDict)

```python
class MissionPlannerState(Fly2VecState):
    pg_json: dict | None           # Planning Graph JSON
    mission_vector: list[float] | None  # 128d 임베딩
    anomaly_score: float | None    # 0~1
    similar_missions: list[dict] | None
    swarm_analysis: dict | None
    conflict_points: list[dict] | None
    retry_count: int               # 현재 루프 횟수
    is_verified: bool              # 검증 통과 여부
    error: str | None
```

### 노드 흐름

```
START → parse_input → generate_pg → embed_and_check
                                         │
                              ┌───────────┴───────────┐
                              │ _should_modify()      │
                              │  anomaly>0.5 OR       │
                              │  conflict>0 AND       │
                              │  retry<3              │
                              ▼                       ▼
                         modify_pg              finalize → END
                              │                       
                              └── embed_and_check (루프)
```

### 조건 분기 로직

```python
def _should_modify(state) -> "modify" | "finalize":
    if retry >= 3: return "finalize"  # 최대 시도 초과
    if anomaly > 0.5 or conflicts > 0: return "modify"
    return "finalize"  # 검증 통과
```

---

## 7. Docker Compose 서비스 구성

| 서비스 | 포트 | 이미지 | 네트워크 | healthcheck |
|--------|------|--------|---------|-------------|
| qdrant-fly2vec | 6340:6333 | qdrant/qdrant:latest | fly2vec-net | TCP 6333 |
| graph-embedding-mcp | 8050 | fly2vec (자체빌드) | fly2vec-net | - |
| swarm-analysis-mcp | 8051 | fly2vec (자체빌드) | fly2vec-net | - |
| fly2vec-agent | 8060 | fly2vec (자체빌드) | fly2vec-net | - |

**볼륨**: `fly2vec_qdrant_storage` (Qdrant 영구 저장)
**의존 순서**: qdrant → MCP 서버 2개 → agent

---

## 8. 기존 실험 코드 통합 전략

| 기존 파일 | 통합 위치 | 통합 방법 |
|-----------|----------|----------|
| `run_full_analysis.py` | `src/embedding/node2vec_encoder.py` | 그래프 구성 + Node2Vec 학습 로직 추출 |
| `run_gcn_analysis.py` | `src/embedding/gcn_encoder.py` | GCN 모델 + 정규화 로직 추출 |
| `run_smgat_analysis.py` | `src/embedding/rgat_encoder.py` | R-GAT 3종 엣지 + 앙상블 로직 추출 |
| `test_anomaly_ensemble.py` | `tests/test_embedding.py` | 테스트 케이스 변환 |
| `enrich_with_mapbox.py` | `src/data/loader.py` (Phase 7) | MapBox 보강 → Qdrant 벌크 로드 |
| `data/dataset_*.csv` | `data/` (심볼릭 링크) | 초기 데이터 로드용 |
| `results/*.csv` | `results/` | 검증 기준 데이터 |

---

## 9. 다음 단계 (Phase 7~8)

1. `src/data/loader.py` 작성 — CSV → PG JSON → 임베딩 → Qdrant 벌크 로드
2. `data/` 디렉토리에 기존 CSV 심볼릭 링크
3. `tests/` 작성 — MCP 도구별 단위 테스트
4. `docker compose up` E2E 검증
5. 학습된 모델 가중치를 `data/models/`에 저장하여 인코더가 로드
