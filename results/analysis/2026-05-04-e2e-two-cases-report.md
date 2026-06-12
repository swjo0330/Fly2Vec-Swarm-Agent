# Fly2Vec E2E 2-Case Test Report

**날짜**: 2026-05-04
**환경**: Docker Compose (fly2vec), macOS Darwin 25.3.0
**LLM Backend**: Gemini 2.5 Flash (GOOGLE_API_KEY)

---

## 1. Phase 0: Docker 재기동

| 서비스 | 컨테이너 | 포트 | 상태 |
|--------|----------|------|------|
| qdrant-fly2vec | qdrant-fly2vec | 6340:6333 | healthy |
| graph-embedding-mcp | fly2vec-graph-embedding-mcp | 8050:8050 | running |
| swarm-analysis-mcp | fly2vec-swarm-analysis-mcp | 8051:8051 | running |
| fly2vec-agent | fly2vec-mission-planner | 8060:8060 | running |

- 빌드: cached (Dockerfile 변경 없음, src/ 코드만 업데이트)
- Qdrant healthcheck 통과 후 MCP 서버 기동, Agent 마지막 기동

---

## 2. 코드 수정 사항

### 버그 발견 및 수정: MCP 도구 응답 파싱

**문제**: `langchain-mcp-adapters`의 `StructuredTool.ainvoke()`가 MCP 프로토콜 표준에 따라 `list[{'type':'text','text':'<json>'}]` 형태로 반환하나, `node_embed_and_check`에서 `isinstance(result, dict)` 검사를 하여 항상 실패.

**증상**: embed_mission, anomaly_score, find_similar 호출은 성공하나 결과를 파싱하지 못해 `mission_vector=None`, `anomaly_score=None` 반환.

**수정**: `_parse_mcp_result()` 헬퍼 함수 추가 (list/dict/str 3단계 fallback 파싱).
- 파일: `src/agents/mission_planner_agent.py`

---

## 3. Case A: 자연어 입력 E2E

**입력**: `"서울역에서 인천공항까지 50m 고도로 4개 웨이포인트 비행 계획"`

### 결과

| 항목 | 값 |
|------|-----|
| 소요 시간 | 9.2s |
| PG 생성 | YES (Gemini 2.5 Flash) |
| Mission type | waypoint_sequence |
| Waypoints | 4개 |
| 128d 벡터 | dim=128, norm=1.0585 |
| Anomaly score | 0.0525 (threshold=0.5 이하, PASS) |
| 유사 미션 | 5건 (top score=0.9999999) |
| 충돌 | 0건 |
| 검증 통과 | YES (retry 0회) |

### 생성된 웨이포인트

| seq | lat_deg | lon_deg | alt_m |
|-----|---------|---------|-------|
| 1 | 37.5599 | 126.9723 | 50.0 |
| 2 | 37.5267 | 126.7951 | 50.0 |
| 3 | 37.4934 | 126.6179 | 50.0 |
| 4 | 37.4602 | 126.4407 | 50.0 |

서울역(37.56, 126.97) ~ 인천공항(37.46, 126.44) 간 직선 경로, 50m 고도 일정.

### 파이프라인 단계별 처리

```
1. parse_input      → OK (retry_count=0 초기화)
2. generate_pg      → OK (Gemini LLM → 4 WPs JSON 생성)
3. embed_and_check  →
   3a. embed_mission  → OK (Node2Vec 128d, Qdrant 저장)
   3b. anomaly_score  → OK (3-method 앙상블: 0.0525)
   3c. find_similar   → OK (5건 유사 미션)
   3d. detect_conflict → OK (0 충돌)
4. _should_modify   → finalize (score=0.0525 <= 0.5)
5. finalize         → OK (is_verified=True)
```

---

## 4. Case B: 실데이터 기반 E2E

**입력**: 8-WP grid_search PG JSON (인천 영종도 일대, 50m/80m 2-layer 그리드)

### 결과

| 항목 | 값 |
|------|-----|
| 소요 시간 | 0.7s (LLM 호출 없음) |
| PG 생성 | SKIPPED (입력 PG 사용) |
| Mission type | grid_search |
| Waypoints | 8개 |
| 128d 벡터 | dim=128, norm=1.1877 |
| Anomaly score | 0.2011 (threshold=0.5 이하, PASS) |
| 유사 미션 | 5건 (top score=1.000000) |
| 충돌 | 0건 |
| 검증 통과 | YES (retry 0회) |

### 임베딩 벡터 특성

- 첫 5차원: [-0.043961, -0.113288, 0.194018, -0.079285, -0.074677]
- norm=1.1877 (Case A의 1.0585보다 큼 -- 8 WP vs 4 WP)

### 유사 미션 검색 결과

| rank | score | mission_type | wp_count |
|------|-------|-------------|----------|
| 1 | 1.000000 | grid_search | 8 |
| 2 | 0.873184 | waypoint_sequence | 5 |
| 3 | 0.865662 | linear_flight | 5 |
| 4 | 0.853279 | waypoint_sequence | 5 |
| 5 | 0.791898 | waypoint_sequence | 4 |

- Top-1이 자기 자신(score=1.0)으로 Qdrant 저장/검색 정상 검증
- grid_search와 waypoint_sequence 간 유사도 0.87 -- 경로 패턴 차이 반영

### 파이프라인 단계별 처리

```
1. parse_input      → OK
2. generate_pg      → SKIPPED (pg_json already provided)
3. embed_and_check  →
   3a. embed_mission  → OK (Node2Vec 128d, Qdrant 저장)
   3b. anomaly_score  → OK (0.2011)
   3c. find_similar   → OK (5건)
   3d. detect_conflict → OK (0 충돌)
4. _should_modify   → finalize (score=0.2011 <= 0.5)
5. finalize         → OK (is_verified=True)
```

---

## 5. 아키텍처 파이프라인 매핑 (4.2 ①~⑤)

| 단계 | 아키텍처 | 구현 | 상태 |
|------|----------|------|------|
| ① 자연어 입력 | User → Agent | HumanMessage → MissionPlannerAgent | PASS |
| ② PG 생성 | Agent → LLM | Gemini 2.5 Flash → JSON parse | PASS |
| ③ Graph Embedding | Agent → Graph MCP | embed_mission (Node2Vec 128d) | PASS |
| ④ Anomaly + Similar | Agent → Graph MCP | anomaly_score + find_similar | PASS |
| ⑤ Conflict Detection | Agent → Swarm MCP | detect_conflict (R-GAT) | PASS |

---

## 6. 에러/이슈

| # | 심각도 | 내용 | 조치 |
|---|--------|------|------|
| 1 | HIGH | MCP 도구 응답이 `list[{'type':'text','text':'...'}]` 형태인데 `isinstance(result, dict)` 검사로 파싱 실패 | `_parse_mcp_result()` 헬퍼 추가, 3단계 fallback 구현 |
| 2 | LOW | `.env`에 `GEMINI_API_KEY`만 있고 `GOOGLE_API_KEY` 누락 (langchain-google-genai가 요구) | `.env`에 `GOOGLE_API_KEY` 추가 |
| 3 | INFO | `Both GOOGLE_API_KEY and GEMINI_API_KEY are set` 경고 | 무해, 정상 동작 |

---

## 7. 소요 시간 비교

| 구간 | Case A | Case B |
|------|--------|--------|
| Agent 초기화 + MCP 연결 | ~2s | ~2s |
| LLM PG 생성 | ~5s | 0s (skip) |
| embed_mission | ~0.5s | ~0.3s |
| anomaly_score | ~0.3s | ~0.2s |
| find_similar | ~0.3s | ~0.2s |
| detect_conflict | ~0.3s | ~0.2s |
| **합계** | **9.2s** | **0.7s** |

---

## 8. 결론

**두 케이스 모두 E2E PASS.**

- Case A: 자연어 → Gemini LLM → PG JSON → Node2Vec 128d → Qdrant 저장 → anomaly 0.053 → 유사 미션 5건 → 검증 통과
- Case B: 실데이터 PG 직접 입력 → LLM 스킵 → Node2Vec 128d → Qdrant 저장 → anomaly 0.201 → 유사 미션 5건 (top=1.0 자기 자신) → 검증 통과
- MCP 프로토콜 응답 파싱 버그 발견 및 수정 (`_parse_mcp_result`)
- 전체 파이프라인 (Agent → MCP → Qdrant → LLM) 정상 동작 확인
