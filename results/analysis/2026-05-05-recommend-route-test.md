# 경로 추천(recommend_route) E2E 테스트 결과

**작성일**: 2026-05-05 | **프로젝트**: Fly2Vec

---

## 1. 기능 설명

`recommend_route`는 현재 임무 경로와 **구조적으로 유사하면서 과거에 정상으로 확인된** 경로를 추천한다.
기존 `find_similar`가 유사도만 보는 반면, `recommend_route`는 **anomaly_score 필터**를 추가하여 안전한 경로만 반환한다.

```
find_similar:      유사 경로 top-K (정상/이상 무관)
recommend_route:   유사 + 정상 경로 top-K (anomaly ≤ 0.3만)
```

## 2. 테스트 결과

### 테스트 1: 정상 경로에서 추천
- 입력: grid_search 6WP (정상)
- recommend_routes: **1건** (grid_search, anomaly=0.05, 유사도 100%)
- 해석: 자기 자신이 정상이므로 정상 경로로 추천됨 ✅

### 테스트 2: 이상 벡터에서 정상 경로 추천
- 입력: collision_risk 패턴 벡터 (이상)
- recommend_routes: **1건** (grid_search, anomaly=0.05, 유사도 19.81%)
- 해석: 이상 경로에 대해 유사하면서 정상인 대안 경로를 제시 ✅

### 테스트 3: find_similar vs recommend_routes 비교

| 순위 | find_similar (필터 없음) | recommend_routes (anomaly ≤ 0.3) |
|------|------------------------|----------------------------------|
| #1 | **collision_risk anomaly=0.8** (100%) | grid_search **anomaly=0.05** (19.81%) |
| #2 | grid_search anomaly=N/A (27.74%) | — |
| #3 | linear_flight anomaly=N/A (27.22%) | — |

**핵심**: find_similar는 이상 경로(collision_risk)를 최상위로 반환하지만,
recommend_routes는 이를 필터링하고 **정상 경로만 추천**한다.

## 3. LLM Agent 활용 흐름

```
PG 생성 → embed → anomaly = 0.6 (> 0.5)
  → recommend_route(vector, max_anomaly=0.3) 호출
  → "유사하면서 정상인 과거 경로: grid_search 6WP (anomaly 0.05)"
  → LLM 프롬프트에 컨텍스트 주입:
      "추천 정상 경로: grid_search 6WP (유사도 20%, anomaly 0.05)"
  → LLM: "추천 경로의 격자 패턴을 참고하여 드론 간 간격 확대"
  → PG 수정 → 재검증 → anomaly = 0.2 → 통과
```

## 4. 현재 한계

- 기존 벌크 로드 2,261건에는 `anomaly_score` payload가 없음 → 필터 대상 제한
- 새로 embed_mission으로 저장되는 데이터부터 anomaly_score 포함
- 벌크 데이터 재로드 시 anomaly_score 추가 필요 (향후)

## 5. 구현 파일

| 파일 | 변경 |
|------|------|
| `fly2vec_qdrant.py` | `recommend_routes()` — Qdrant payload 필터 (anomaly ≤ max) |
| `graph_embedding_mcp.py` | `recommend_route` MCP 도구 (4번째) + embed 시 anomaly 저장 |
| `mission_planner_agent.py` | anomaly > 0.5일 때 추천 호출 + modify_pg에 컨텍스트 |
