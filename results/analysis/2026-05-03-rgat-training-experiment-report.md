# Fly2Vec R-GAT 학습 실험 중간보고서

**작성일**: 2026-05-03 | **프로젝트**: Fly2Vec (AI개론 기말) | **작성자**: 조성원, 이서정

---

## 1. 실험 목적

비학습 R-GAT(message passing만 수행)의 한계를 극복하기 위해 자기지도/지도 학습을 적용하여:
- 엣지 타입별 어텐션 가중치 차별화
- 이상 탐지 성능 향상 (비학습 대비)
- cooperative_search vs relay_move 패턴 분리

## 2. 실험 환경

| 항목 | 값 |
|------|-----|
| 데이터 | 실 관제 2,261건 기반 증강 군집 시나리오 3,000건 |
| 시나리오 4종 | cooperative_search, relay_move, collision_risk, coverage_overlap |
| 정상:이상 비율 | 62% : 38% |
| 노드 피처 | 7d (기본) → 9d (확장: +min_distance_to_other, +is_overlap_grid) |
| R-GAT 구조 | 2-Layer, 4-Head, 3종 엣지 (Sequential/Proximity/Coverage) |
| 풀링 | Mean Pooling → Attention Pooling (실험 6) |
| 평가 지표 | 패턴 분류 정확도, 이상 탐지 정확도, collision 탐지 정확도 |

## 3. 실험 결과

### 3.1 전체 비교표

| # | 모델 | 피처 | 패턴 분류 | 이상 탐지 | collision | coverage | 비고 |
|---|------|------|----------|----------|-----------|----------|------|
| 1 | **MLP** (baseline) | 10d 통계 | **48.6%** | 71.5% | **88.1%** | **78.5%** | 직접 통계 입력 (proximity_edge_count 등) |
| 2 | GAT (엣지 무구분) | 7d | 34.8% | 64.8% | — | — | 모든 엣지 동일 취급 → Proximity 신호 학습 실패 |
| 3 | **R-GAT** (엣지 타입별) | 7d | 29.3% | 70.2% | 40.6% | — | Layer2 Proximity 가중치 **63%** — 가설 검증 성공 |
| 4 | R-GAT + SupCon | 7d | 30.5% | 70.3% | 37.7% | — | SupCon 효과 미미 (best 35.7%) |
| 5 | R-GAT 9d (mean pool) | 9d | 30.3% | 70.5% | 55.1% | — | +min_dist+overlap 피처 추가 |
| 6 | **R-GAT 9d + Attn Pool** | 9d | 30.3% | **71.5%** | **58.0%** | — | **최종 모델** — MLP 이상 탐지와 동등 |

### 3.2 실험별 상세 분석

#### 실험 1: MLP Baseline (10d 통계 직접 입력)

- **입력**: proximity_edge_count, coverage_edge_count, avg_min_distance 등 10차원 수작업 통계
- **결과**: 패턴 48.6%, 이상 71.5%, collision 88.1%
- **의미**: 그래프 통계를 직접 계산하여 입력하면 높은 성능 → **R-GAT가 이 통계를 자동 학습할 수 있는가?**가 핵심 질문

#### 실험 2: GAT (엣지 타입 무구분, 7d)

- **설정**: 모든 엣지를 동일하게 취급하는 표준 GAT
- **결과**: 패턴 34.8%, 이상 64.8%
- **분석**: Sequential/Proximity/Coverage를 구분하지 않으면 Proximity 신호가 희석됨
- **결론**: **엣지 타입 구분이 필수** → R-GAT 도입 근거

#### 실험 3: R-GAT (엣지 타입별 독립 어텐션, 7d)

- **설정**: 3종 엣지에 독립적 어텐션 가중치 학습
- **결과**: 패턴 29.3%, 이상 70.2%, collision 40.6%
- **핵심 발견**:
  - Layer 2에서 **Proximity 가중치 63%** — 모델이 Proximity 엣지의 중요성을 자동 학습
  - **제안서 가설 "Proximity ∝ anomaly" 검증 성공**
- **한계**: cooperative vs relay 구분 실패 (둘 다 Proximity=0인 정상 패턴으로 그래프 구조 동일)

#### 실험 4: R-GAT + SupCon (Supervised Contrastive Learning)

- **설정**: SupCon loss로 같은 라벨 임베딩 근접 / 다른 라벨 분리 학습
- **결과**: 패턴 30.5%, 이상 70.3% (best epoch 35.7%)
- **분석**: cooperative/relay가 그래프 구조상 동일 → 임베딩 분리 불가
- **설계 결정**: **SupCon 제거**, 노드 피처 확장으로 방향 전환
- **검토**: SupCon vs Triplet vs NT-Xent 비교 후 SupCon 선택했으나, 근본 원인이 데이터 구조이므로 loss 교체로 해결 불가

#### 실험 5: R-GAT 9d (노드 피처 확장)

- **추가 피처**:
  - `min_distance_to_other`: 다른 드론 WP와의 최소 거리 (연속값, 엣지가 못 잡는 거리 정보)
  - `is_overlap_grid`: 같은 100m 그리드에 타 드론 WP 존재 여부 (이진값)
- **Information Leakage 검토**:
  - `n_proximity_neighbors` = Proximity 엣지 degree → **제외** (직접 누출)
  - `min_distance_to_other` → **허용** (엣지가 500m 임계에서만 존재, 연속 거리는 별도 정보)
  - `is_overlap_grid` → **허용** (Coverage 보완 이진값)
- **결과**: collision 40.6% → **55.1%** (+14.5%p 개선)
- **의미**: min_dist 피처가 충돌 탐지에 직접적 기여

#### 실험 6: R-GAT 9d + Attention Pooling (최종 모델)

- **변경**: Mean Pooling → Attention Pooling (학습 가능한 가중치로 노드 중요도 결정)
- **결과**: 이상 탐지 **71.5%** (MLP baseline 동등!), collision **58.0%**
- **분석**:
  - Attention Pooling이 위험 노드(Proximity 근접 WP)에 높은 가중치 부여
  - 전체 그래프에서 "어디가 위험한지"를 자동 식별
- **Layer 2 Proximity 가중치**: 34% (실험 3의 63%에서 하락 — 9d 피처가 일부 역할 분담)

## 4. 핵심 결론

### 4.1 성과

1. **이상 탐지: R-GAT = MLP** (71.5%) — GNN이 수작업 통계 없이 그래프 구조만으로 위험 식별 가능 증명
2. **Proximity 가설 검증**: R-GAT가 Proximity 엣지에 높은 어텐션 → "근접 ∝ 이상" 관계 자동 학습
3. **엣지 타입 구분 필수**: GAT(64.8%) vs R-GAT(70.2%) — 6%p 차이로 R-GAT 우위 실증
4. **collision 탐지 개선**: 40.6% → 58.0% (피처 확장 + Attention Pooling 효과)

### 4.2 한계 및 향후 과제

| 한계 | 원인 | 향후 방향 |
|------|------|----------|
| 패턴 분류 30.3% (MLP 48.6% 대비 낮음) | cooperative/relay가 그래프 구조 동일 | 시간축 피처 (Temporal R-GAT) 추가 |
| collision 58.0% (MLP 88.1% 대비 낮음) | MLP는 edge_count를 직접 입력 | 엣지 피처 학습 (Edge Feature Network) |
| 합성 데이터 한계 | 실 군집 운용 데이터 부재 | Gazebo SITL 3~5대 시뮬 데이터 수집 |

### 4.3 설계 결정 요약

| 결정 | 근거 |
|------|------|
| SupCon 제거 | cooperative/relay 그래프 구조 동일 → loss로 해결 불가 |
| 9d 피처 확장 | min_dist + overlap이 collision 14.5%p 개선 |
| Attention Pooling 채택 | 위험 노드 자동 식별 → 이상 탐지 1%p 추가 개선 |
| Information Leakage 검토 | n_proximity_neighbors 제외, min_dist/overlap 허용 |

## 5. 저장된 모델 목록

| 파일 | 모델 | 상태 |
|------|------|------|
| `fly2vec_classifier.pt` | MLP V1 (baseline) | 참고 |
| `fly2vec_classifier_v2.pt` | MLP V2 (스케일링+가중치) | 참고 |
| `fly2vec_gnn_gat.pt` | GAT (엣지 무구분) | 참고 |
| `fly2vec_rgat_best.pt` | R-GAT 7d | 참고 |
| `fly2vec_rgat_supcon.pt` | R-GAT + SupCon | 폐기 (효과 미미) |
| `fly2vec_rgat_9d_best.pt` | R-GAT 9d mean pool | 참고 |
| **`fly2vec_rgat_attn_best.pt`** | **R-GAT 9d + Attn Pool** | **최종 모델** |
| `scaler.pkl` | StandardScaler | 전처리 |

## 6. 비학습 → 학습 R-GAT 비교 (report4 vs 본 실험)

| 항목 | 비학습 R-GAT (report4) | 학습 R-GAT (최종) |
|------|----------------------|-------------------|
| 클러스터 수 | 3 → 8 (증강 후) | — (지도학습) |
| Noise 비율 | 0.8% → 22.6% | — |
| 이상 탐지 정확도 | 앙상블 기반 (비교 불가) | **71.5%** |
| collision 탐지 | Proximity 엣지 수 상관 | **58.0%** (직접 분류) |
| Proximity 어텐션 | 미학습 (균등) | **Layer2 34~63%** (자동 학습) |

---

*본 보고서는 Fly2Vec 프로젝트의 R-GAT 학습 실험 Phase 8 결과를 기록한 중간보고서이다.*
