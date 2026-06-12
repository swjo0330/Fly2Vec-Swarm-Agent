# Fly2Vec: 설계 제안서 vs 구현 코드 심층 리뷰

**리뷰일**: 2026-05-04  
**제안서**: `2026-05-02-fly2vec-combined-proposal.md`  
**코드 범위**: `fly2vec/src/` 전체 + `enrich_with_mapbox.py`

---

## 전체 요약

구현 코드는 제안서의 아키텍처 골격(LangGraph Agent + MCP 서버 2개 + Qdrant + A2A)을 충실히 따르고 있으며, AERION 프로젝트의 기존 패턴(base_graph_agent, MCP base, A2A interface)을 일관되게 재사용하고 있다. 그러나 **5단계 파이프라인 중 1단계(MapBox MCP 보강)와 경로 A(자연어 -> MapBox MCP 경유)의 핵심 통합이 누락**되어 있으며, 노드 피처 차원이 제안서 15d와 불일치한다. 아래 7개 리뷰 포인트별 상세 분석을 기술한다.

---

## 1. 파이프라인 정합성 (제안서 4.2절 vs 실제 코드)

### 설계 (제안서 4.2)
5단계 공통 파이프라인:
1. MapBox MCP 보강 (distance, bearing, reverse_geocode, point_in_polygon)
2. NetworkX 그래프 구성 (노드 15d, Sequential + Spatial KNN)
3. Node2Vec / GCN 임베딩 (128d)
4. Fly2Vec-Swarm 군집 분석 (R-GAT 3종 엣지)
5. Qdrant 저장 + RAG 검색

### 구현

| 단계 | 구현 상태 | 해당 코드 |
|------|-----------|-----------|
| 1. MapBox MCP 보강 | **미통합** | `enrich_with_mapbox.py`는 독립 스크립트로 존재. `fly2vec/src/` 파이프라인과 연결 없음. `GraphEmbeddingMCP.embed_mission()`은 MapBox 보강 없이 raw 좌표를 직접 그래프로 변환 |
| 2. NetworkX 그래프 구성 | **구현됨 (4d)** | `node2vec_encoder.py:build_graph()` -- Sequential + Spatial KNN 엣지 구성. 단, 노드 피처가 4d (lat, lon, alt, seq_ratio) |
| 3. Node2Vec/GCN 임베딩 | **구현됨** | `node2vec_encoder.py:encode_mission()`, `gcn_encoder.py:encode_mission()` -- 128d 출력. 가중 평균 풀링 포함 |
| 4. R-GAT 군집 분석 | **부분 구현** | `rgat_encoder.py:analyze()` -- 3종 엣지 구성 + 규칙 기반 패턴 분류. 실제 R-GAT 모델(PyG GATConv)은 미사용, 엣지 비율 기반 휴리스틱 |
| 5. Qdrant 저장 + RAG | **구현됨** | `fly2vec_qdrant.py`, `collections.py` -- 128d COSINE, 2개 컬렉션 |

### Gap
- **1단계 MapBox 보강이 파이프라인에 통합되지 않음**: `enrich_with_mapbox.py`가 `fly2vec/src/` 외부의 독립 실험 스크립트로만 존재. `GraphEmbeddingMCP.embed_mission()`이 MapBox 보강 단계를 건너뛰고 raw 좌표를 직접 처리
- **4단계 R-GAT**: PyTorch Geometric 기반 실제 R-GAT 모델이 아닌 규칙 기반 근사 (Proximity/Coverage 엣지 비율로 anomaly 계산)

### 심각도: Critical (1단계), Major (4단계)

### 수정 제안
1. `fly2vec/src/data/` 아래에 `mapbox_enricher.py` 모듈을 신설하여 `enrich_with_mapbox.py`의 `compute_route_features()`와 `reverse_geocode()` 로직을 통합
2. `GraphEmbeddingMCP.embed_mission()` 흐름에 MapBox 보강 단계를 삽입: `pg_json -> enrich(pg_json) -> build_graph(enriched) -> encode`
3. R-GAT는 현재 비학습 상태(제안서 15.3에서 한계로 명시)이므로, 최소한 PyG `RGATConv` 레이어를 통한 message passing은 수행하도록 `rgat_encoder.py`를 개선

---

## 2. 경로 A (자연어 입력) 흐름 (제안서 4.3절)

### 설계 (제안서 4.3)
```
자연어 -> LLM Agent 해석 -> MapBox MCP (geocode + directions) -> Douglas-Peucker -> PG JSON
```

### 구현
`mission_planner_agent.py:node_generate_pg()` (line 153-172):
```python
response = await self.model.ainvoke(
    [SystemMessage(content=SYSTEM_PROMPT + '\n\n' + generation_prompt)] + messages
)
pg_json = _parse_json_response(response.content)
```
LLM이 **직접 좌표를 생성**하고 있다. MapBox MCP 도구 호출(geocode, directions)이 전혀 없으며, Douglas-Peucker 간소화도 미구현.

### Gap
- MapBox MCP 호출 부재: LLM이 좌표를 환각(hallucinate)할 위험
- Douglas-Peucker 간소화 미구현
- 자연어에서 intent 추출(origin, destination, altitude) 로직 부재 -- LLM에 모든 것을 위임

### 심각도: Critical

### 수정 제안
1. `node_generate_pg()` 에서 intent 파싱 후 MapBox MCP 도구를 호출하는 흐름으로 변경:
   - `geocode` 도구로 출발지/목적지 좌표 획득
   - `directions` 도구로 경로 좌표 리스트 획득
   - Douglas-Peucker 로 간소화 후 PG JSON 생성
2. 또는 MCP 서버 목록에 MapBox MCP를 추가하고, LLM에게 도구 사용을 프롬프트로 유도 (ReAct/Tool-use 패턴)

---

## 3. 노드 피처 차원 (제안서 "15d" vs 실제)

### 설계 (제안서 3절, 8절)
> "각 WP를 노드(15차원 피처)로"

제안서에서 암시하는 15d 구성:
lat, lon, alt, speed, bearing, distance, hold_sec, seq_ratio, terrain_type, no_fly_zone, place_name_embedding, bearing_change, alt_change, speed_change, command_type

### 구현

| 인코더 | 실제 노드 피처 | 차원 |
|--------|---------------|------|
| `Node2VecEncoder.build_graph()` | lat, lon, alt, seq_ratio | **4d** (node attr로 저장되지만 Node2Vec은 이를 사용하지 않음 -- 구조만 학습) |
| `GCNEncoder.encode_mission()` | lat_norm, lon_norm, alt_norm, seq_ratio | **4d** |
| `RGATEncoder._build_swarm_graph()` | (10d 언급) hour_norm, season_norm, terrain_norm 추가 + 기존 WP 필드 | **명시적 피처 텐서 미구성** -- 벡터 수동 할당 (vector[0:10]) |

### Gap
- 제안서 15d 대비 실제 GCN은 4d -- **11d 부족**
- 누락된 피처: speed, bearing, distance, hold_sec, terrain_type, no_fly_zone, bearing_change, alt_change, command_type, place_name_embedding 등
- R-GAT 인코더는 노드 피처 텐서를 구성하지 않고 수동으로 128d 벡터에 통계값을 할당 (line 139-170)

### 심각도: Major

### 수정 제안
1. `node2vec_encoder.py:build_graph()`에서 노드에 저장하는 attribute를 15d로 확장:
   - speed, hold_sec, command (one-hot), bearing (이전 WP 대비), distance (이전 WP 대비)
   - MapBox 보강 필드: terrain_type, no_fly_zone
2. `gcn_encoder.py`의 `in_channels`를 4에서 15로 변경
3. R-GAT에서 실제 PyG 노드 피처 텐서를 구성

---

## 4. MapBox 보강 파생변수 (제안서 7절)

### 설계 (제안서 7절)
**노드 보강**: 장소명 (reverse_geocode), 비행금지 여부 (point_in_polygon), 지형 타입 (reverse_geocode 파싱)
**엣지 보강**: 정밀 거리 (distance_tool), 방위각 (bearing_tool), 방위 변화 (계산)

### 구현
`enrich_with_mapbox.py`에서 구현된 것:
- Haversine 거리 (로컬 계산) -- 구현됨
- Bearing 방위각 (로컬 계산) -- 구현됨
- Bearing 변화량 (로컬 계산) -- 구현됨
- reverse_geocode (MapBox API) -- 시작/종료/중심 3점만, 상위 50개 경로만

`enrich_with_mapbox.py`에서 **미구현**:
- point_in_polygon (비행금지구역) -- 미구현
- 지형 타입 분류 (도심/해안/산지/농경지) -- 미구현

**인코더에서의 활용 현황**:
- `node2vec_encoder.py`: MapBox 보강 필드를 전혀 사용하지 않음
- `gcn_encoder.py`: lat, lon, alt, seq_ratio만 사용 -- 보강 필드 미반영
- `rgat_encoder.py`: `terrain_norm`을 고도 기반으로 자체 계산 (MapBox 미사용)

### Gap
- `enrich_with_mapbox.py`의 결과(`enriched_routes.csv`)가 `fly2vec/src/` 파이프라인과 연결되지 않음
- point_in_polygon, 지형 타입 분류 미구현
- 인코더들이 보강 피처를 전혀 소비하지 않음

### 심각도: Major

### 수정 제안
1. `loader.py`에서 `enriched_routes.csv`를 읽어 PG JSON에 보강 필드를 병합하는 로직 추가
2. `build_graph()`에서 노드 attribute에 distance, bearing, bearing_change, place_name, terrain_type 추가
3. GCN/R-GAT 인코더에서 보강 피처를 입력 차원에 반영

---

## 5. 검증-수정 루프 (제안서 5절)

### 설계 (제안서 5절)
```
PG -> embed -> anomaly > 0.5 OR conflict > 0? -> YES: PG 수정 (max 3회) -> 재검증
                                                -> NO: 최종 PG
```

### 구현
`mission_planner_agent.py`의 LangGraph 구조:

```
parse_input -> generate_pg -> embed_and_check -> [_should_modify]
  -> 'modify': modify_pg -> embed_and_check (루프)
  -> 'finalize': finalize -> END
```

- `_should_modify()` (line 136-147): `anomaly_score > ANOMALY_THRESHOLD(0.5)` OR `len(conflicts) > 0` AND `retry < MAX_VERIFICATION_LOOPS(3)` -- **제안서와 정확히 일치**
- `node_embed_and_check()` (line 174-238): embed_mission -> anomaly_score -> find_similar -> detect_conflict 순차 호출 -- **4개 MCP 도구 호출 포함**
- `node_modify_pg()` (line 240-272): LLM에게 수정 요청 -> 파싱 -> retry_count 증가

### Gap
- **MapBox MCP 경유 누락**: 검증-수정 루프 전에 MapBox MCP -> Safety Validator 단계가 없음 (제안서 Figure: Strategic Planner -> MapBox MCP -> Safety Validator -> Graph Embedding MCP)
- **Safety Validator 미구현**: 제안서에서 언급한 "12 규칙 검증" 단계가 코드에 없음
- `analyze_swarm` 호출 누락: `embed_and_check`에서 `detect_conflict`만 호출하고 `analyze_swarm`은 호출하지 않음

### 심각도: Major (Safety Validator), Minor (analyze_swarm)

### 수정 제안
1. Safety Validator 노드를 `generate_pg`와 `embed_and_check` 사이에 추가
2. `embed_and_check`에서 `analyze_swarm` 도구도 호출하여 패턴 분류 결과를 `swarm_analysis` 상태에 저장
3. 수정 루프 시 MapBox MCP로 좌표 보정을 수행하도록 `modify_pg` 개선

---

## 6. MCP 도구 시그니처 (제안서 4.1절)

### 설계 (제안서 4.1, 17절)

**GraphEmbeddingMCP**:
- `embed_mission(pg_json) -> {vector: 128d}`
- `find_similar(vector, top_k) -> {similar_missions: [...]}`
- `anomaly_score(vector) -> {score: float, is_anomalous: bool}`

**SwarmAnalysisMCP**:
- `analyze_swarm(missions) -> {pattern, anomaly_avg, drone_scores}`
- `detect_conflict(missions) -> {conflicts: [...]}`
- `optimize_assignment(missions, drones) -> {assignments, coverage_score}`

### 구현

**GraphEmbeddingMCP** (`graph_embedding_mcp.py`):
- `embed_mission(pg_json, store)` -- store 매개변수 추가 (제안서에 없음, 유용한 확장)
- `find_similar(vector, top_k)` -- 제안서와 일치
- `anomaly_score(vector)` -- 3-Method 앙상블 근사 구현

**SwarmAnalysisMCP** (`swarm_analysis_mcp.py`):
- `analyze_swarm(missions)` -- 제안서와 일치
- `detect_conflict(missions, proximity_threshold_m)` -- threshold 매개변수 추가 (유용한 확장)
- `optimize_assignment(missions, drones)` -- 제안서와 일치 (단, 내부 구현은 순차 배정 간소화)

### Gap
- `anomaly_score`: 3-Method 앙상블 중 Isolation Forest가 **실제 모델 없이 거리 기반 근사** (line 117-118: `iso_score = min(centroid_score * 1.5, 1.0)`)
- `anomaly_score`: KNN Average Distance가 **Centroid Distance와 동일값 사용** (line 115: `knn_score = centroid_score`)
- `optimize_assignment`: 순차 배정 (`i % len(drones)`)으로 최적화 로직 부재

### 심각도: Major (anomaly_score 근사), Minor (optimize_assignment)

### 수정 제안
1. `anomaly_score`에 scikit-learn `IsolationForest`를 실제로 학습/적용 (오프라인 학습 후 모델 로드)
2. KNN Average Distance를 Qdrant top-K 평균 거리로 독립 계산
3. `optimize_assignment`에 최소한 거리 기반 그리디 배정 알고리즘 적용

---

## 7. 서버 내 검증-수정 루프 전체 흐름 (제안서 5절 Figure)

### 설계 (제안서 5절)
```
Strategic Planner -> MapBox MCP -> Safety Validator -> Graph Embedding MCP -> Fly2Vec-Swarm MCP
  -> anomaly > 0.5 OR conflict > 0?
    -> YES: PG 수정 -> 루프 재실행
    -> NO: 최종 PG -> A2A SSE -> 드론
```

### 구현
`MissionPlannerAgent` 전체 흐름:
```
parse_input -> generate_pg (LLM 직접) -> embed_and_check (Graph Embedding + Swarm conflict)
  -> anomaly > 0.5 OR conflict > 0?
    -> YES: modify_pg (LLM) -> embed_and_check (루프, max 3)
    -> NO: finalize -> END
```

A2A SSE 전달: `fly2vec_client.py` + `fly2vec_executor.py` + `fly2vec_a2a_interface.py`로 인프라 구현됨.

### Gap

| 제안서 단계 | 구현 상태 |
|-------------|-----------|
| MapBox MCP 경유 (geocode + directions) | **미구현** -- LLM 직접 좌표 생성 |
| Safety Validator (12 규칙) | **미구현** -- 검증 노드 없음 |
| Graph Embedding MCP 호출 | **구현됨** -- embed_mission + anomaly_score + find_similar |
| Fly2Vec-Swarm MCP 호출 | **부분** -- detect_conflict만, analyze_swarm 미호출 |
| 검증-수정 루프 (max 3) | **구현됨** -- _should_modify + modify_pg |
| A2A SSE -> 드론 | **인프라 구현됨** -- 실제 연동 미검증 |
| 비행 결과 축적 (flight_result -> Qdrant) | **미구현** |

### 심각도: Critical (MapBox + Safety Validator 누락으로 전체 흐름의 2단계가 빠짐)

### 수정 제안
1. LangGraph 노드 추가: `enrich_with_mapbox` (MapBox MCP 호출), `safety_validate` (12 규칙 검증)
2. 그래프 흐름 변경:
   ```
   parse_input -> generate_pg -> enrich_with_mapbox -> safety_validate -> embed_and_check -> ...
   ```
3. `embed_and_check`에 `analyze_swarm` 호출 추가
4. flight_result 수신 및 Qdrant 업데이트를 위한 별도 엔드포인트 또는 A2A 핸들러 구현

---

## 추가 발견사항

### 긍정적 측면

1. **AERION 프로젝트 패턴 일관성**: `Fly2VecBaseAgent`, `Fly2VecMCPBase`, `Fly2VecA2AAgent` 등 기존 AERION의 base class 패턴을 충실히 따름
2. **LangGraph 구조**: `RunnableConfig` 전달, `StateGraph` + conditional edges, `MessagesState` 확장 등 LangGraph 모범 패턴 준수
3. **Qdrant 통합**: `AsyncQdrantClient` 사용, 컬렉션 자동 생성, 표준 응답 포맷 등 프로덕션 수준의 설계
4. **데이터 증강기**: `augmentor.py`의 6단계 증강 파이프라인은 제안서 11.3절의 합성 시나리오 생성을 체계적으로 확장
5. **JSON 파싱 견고성**: `_parse_json_response()`의 4단계 fallback (순수 JSON -> ```json 블록 -> 첫 {...} -> fallback)
6. **에러 처리**: MCP 도구 호출 실패 시 graceful degradation (`embed_and_check`의 try/except)

### 주의 사항

1. **API 키 하드코딩**: `enrich_with_mapbox.py` line 42-44에 MapBox API 토큰이 소스코드에 직접 포함. 환경변수로 이동 필요
2. **GCN 모델 미학습**: `GCNModel`이 `eval()` 모드로 랜덤 초기 가중치로만 동작. 학습 루프 미구현
3. **R-GAT 5종 엣지 vs 3종**: `RGATEncoder.EDGE_TYPES`에 `temporal`과 `anomaly`가 추가되어 5종인데, 제안서는 3종(Sequential/Proximity/Coverage)만 정의. 코드의 확장은 합리적이나 문서와 불일치

---

## 우선순위별 정리

### Critical (반드시 수정)
1. **경로 A MapBox MCP 경유 누락** (리뷰 2): LLM 직접 좌표 생성은 환각 위험
2. **MapBox 보강 파이프라인 미통합** (리뷰 1, 4): 5단계 파이프라인의 1단계가 분리된 실험 스크립트로만 존재
3. **Safety Validator 미구현** (리뷰 7): 제안서 핵심 컴포넌트 부재

### Major (수정 권장)
4. **노드 피처 4d vs 15d** (리뷰 3): GCN/R-GAT 학습 품질에 직접 영향
5. **anomaly_score 3-Method 근사** (리뷰 6): Isolation Forest와 KNN이 실제 동작하지 않음
6. **R-GAT 실제 모델 미사용** (리뷰 1): 규칙 기반 휴리스틱만 적용
7. **analyze_swarm 미호출** (리뷰 5): 검증 루프에서 패턴 분류 결과 미활용

### Minor (개선 권장)
8. **API 키 하드코딩** (추가 발견): 환경변수로 이동
9. **optimize_assignment 간소화** (리뷰 6): 최소한 그리디 알고리즘 적용
10. **flight_result Qdrant 축적 미구현** (리뷰 7): RAG 강화 루프 미완성
11. **R-GAT 5종 vs 3종 엣지 문서 불일치** (추가 발견): 제안서 업데이트 또는 코드 정리 필요
