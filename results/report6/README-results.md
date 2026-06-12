# report4 — SM-GAT 기반 군집 드론 임무 분석 결과 해석

> 2026-05-01 | R-GAT (Relational Graph Attention Network) | 500건 합성 군집 시나리오

---

## 1. 분석 개요

기존 Mission2Vec(report1~3)이 **단일 드론 경로**를 분석한 반면, SM-GAT(report4)는 **다수 드론의 군집 임무**를 분석한다.
2,068건 실 관제 데이터에서 3~5대 드론 조합으로 500건 합성 군집 시나리오를 생성하고,
R-GAT(3종 엣지: Sequential/Proximity/Coverage)로 군집 임베딩 128d를 생성한 후
3-Method 앙상블 이상 탐지를 수행하였다.

---

## 2. Mission2Vec vs SM-GAT 비교

| 항목 | Mission2Vec (report1~3) | SM-GAT (report4) |
|------|------------------------|-------------------|
| **분석 단위** | 단일 경로 | **군집 (3~5대)** |
| **모델** | Node2Vec / GCN | **R-GAT (3종 엣지)** |
| **데이터** | 2,068건 실 경로 | **500건 합성 군집** |
| **클러스터** | 63개 (GCN) | **3개** |
| **Noise 비율** | 25.1% (GCN) | **0.8%** |
| **Anomaly 평균** | 0.155 (GCN) | **0.156** |
| **95%ile 임계값** | 0.481 (GCN) | **0.335** |

---

## 3. 시나리오 타입별 분석

| 타입 | 건수 | anomaly avg | anomaly max | 해석 |
|------|------|------------|------------|------|
| **cooperative_search** | 125 | 0.083 | 0.361 | 영역 분리 정상 운용 |
| **collision_risk** | 125 | 0.208 | 0.707 | 경로 교차/근접 위험 |
| **coverage_overlap** | 125 | 0.220 | 0.974 | 수색 영역 중복 |
| **relay_move** | 125 | 0.115 | 0.421 | 일렬 릴레이 정상 이동 |

### 해석

- **collision_risk**: Proximity 엣지가 많아 근접 WP가 다수 → 앙상블 점수 높음 예상
- **coverage_overlap**: Coverage 엣지가 많아 같은 그리드에 다수 드론 → 중복 탐지
- **cooperative_search / relay_move**: 영역 분리 또는 일렬 배치로 정상 패턴 → 점수 낮음

---

## 4. 상위 anomaly 시나리오

```
  OVLP_035              score=0.974  type=coverage_overlap      drones=4  prox=904  cov=62
  OVLP_047              score=0.949  type=coverage_overlap      drones=4  prox=6806  cov=450
  OVLP_124              score=0.904  type=coverage_overlap      drones=3  prox=338  cov=22
  CONF_074              score=0.707  type=collision_risk        drones=3  prox=40  cov=12
  CONF_116              score=0.536  type=collision_risk        drones=4  prox=120  cov=24
  RLAY_100              score=0.421  type=relay_move            drones=4  prox=730  cov=16
  OVLP_110              score=0.402  type=coverage_overlap      drones=3  prox=2952  cov=66
  CONF_102              score=0.399  type=collision_risk        drones=3  prox=180  cov=0
  CONF_013              score=0.398  type=collision_risk        drones=3  prox=32  cov=4
  OVLP_010              score=0.397  type=coverage_overlap      drones=3  prox=352  cov=14
```

---

## 5. R-GAT 3종 엣지 분석

| 엣지 타입 | 의미 | 평균 엣지 수 |
|-----------|------|-------------|
| **Sequential** | 같은 드론 내 WP 순서 | 159 |
| **Proximity** | 다른 드론 간 500m 이내 | 1138 |
| **Coverage** | 같은 100m 그리드 셀 | 137 |

Proximity와 Coverage 엣지가 많을수록 충돌 위험/수색 중복 가능성 높음.

---

## 6. UMAP 시각화 해석

### umap_swarm_types.png
좌: 시나리오 타입별 색상 구분 — 4종 시나리오가 UMAP 공간에서 얼마나 분리되는지 확인
우: anomaly score 히트맵 — 붉은 점이 이상 시나리오

### umap_clusters.png
HDBSCAN 3개 클러스터 — 군집 임무 패턴의 자동 분류 결과

### analysis_summary.png
4패널: 타입별 점수 분포, 엣지 비율, 드론 수별 분포, 3-Method 비교

### rgat_attention_weights.png
R-GAT의 엣지 타입별 어텐션 가중치 — 어떤 관계에 모델이 주목하는지

### top_anomaly_scenarios.png
상위 15건 anomaly 시나리오 — 타입별 색상

---

## 7. 결과 파일 목록

| 파일 | 내용 |
|------|------|
| `swarm_analysis_full.csv` | 500건 전체 분석 결과 |
| `cluster_summary.csv` | HDBSCAN 클러스터 통계 |
| `top_anomaly_detail.csv` | 상위 20건 anomaly 상세 |
| `type_summary.csv` | 시나리오 타입별 통계 |
| `swarm_vectors.npy` | 128d R-GAT 군집 임베딩 벡터 |
| `umap_2d.npy / umap_3d.npy` | UMAP 좌표 |
| `umap_swarm_types.png` | UMAP 2D (타입 + anomaly) |
| `umap_clusters.png` | UMAP 2D (HDBSCAN) |
| `analysis_summary.png` | 4패널 분석 차트 |
| `rgat_attention_weights.png` | R-GAT 어텐션 가중치 |
| `top_anomaly_scenarios.png` | 상위 anomaly 바 차트 |
| `umap_3d.html` | 3D 인터랙티브 UMAP |

---

## 8. 토론

### SM-GAT의 의의

Mission2Vec이 "이 경로가 정상인가?"를 분석한다면, SM-GAT는 "이 군집 임무가 정상인가?"를 분석한다.
3종 엣지(Sequential/Proximity/Coverage)로 단일 경로로는 보이지 않는 **드론 간 상호작용 패턴**을 포착하며,
R-GAT의 어텐션 메커니즘으로 **어떤 드론 간 관계가 위험한지** 해석 가능한 결과를 제공한다.

### 한계

- 비학습(untrained) R-GAT: message passing만으로 피처 집계 — 학습 시 더 나은 분리 기대
- 합성 시나리오: 실 군집 운용 데이터 부재 → 현실 반영 한계
- Coverage 엣지 정의: 100m 그리드가 모든 임무 유형에 적합하지 않을 수 있음

### 향후 방향

- R-GAT 자기지도 학습 → 패턴 분류 정확도 향상
- 실 군집 시나리오 수집 (Gazebo SITL 3~5대)
- Temporal 축 추가 → 비행 중 동적 분석
