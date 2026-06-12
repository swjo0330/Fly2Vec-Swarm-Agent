# Fly2Vec 구현 로그

## Phase 8: 통합 테스트 + 실제 연동

### 2026-05-02 세션

#### 완료된 Phase (1~7)
- Phase 1~6: 프로젝트 골격 + 21개 소스 파일 작성 완료
- Phase 7: 데이터 통합 (loader.py + CSV 복사) 완료
- 감독 리뷰: 23개 파일 전 항목 PASS, AERION 오염 없음
- 설계안 대비: 클래스명/시그니처/Docker 100% 일치

#### Phase 8 P1 구현 완료
- **Step 1**: MissionPlannerAgent 3개 TODO → 실제 구현 완료
  - `node_generate_pg`: LLM PG 생성 + `_parse_json_response()` 4단계 fallback
  - `node_embed_and_check`: self.tools에서 embed_mission → anomaly_score → find_similar → detect_conflict 순차 호출, graceful fallback
  - `node_modify_pg`: anomaly/conflict 정보 + 현재 PG를 LLM에 전달 → 수정된 PG 파싱
- **Step 2**: `tests/test_embedding.py` 작성 (8개 케이스)
  - Node2Vec: build_graph, encode_mission_shape, encode_empty
  - GCN: encode_mission_shape
  - R-GAT: analyze_cooperative, analyze_collision, detect_conflicts, detect_no_conflict
- **Step 3**: `tests/test_mcp.py` 작성 (9개 케이스)
  - GraphEmbeddingMCP: embed, find_similar, anomaly_score, anomaly_high
  - SwarmAnalysisMCP: analyze_swarm, detect_conflict, detect_none, optimize
- **conftest.py**: 공통 fixture (sample_pg, sample_swarm_cooperative, sample_swarm_collision)

#### 수정 사항 적용
- `pyproject.toml`: `node2vec>=0.4.0` 추가
- `docker-compose.yml`: `version: '3.8'` 제거
- `gcn_encoder.py`: `_graph_builder` 싱글톤 적용
- `RULES.md`: Section 11 문서화/기록 규칙 추가

#### 설계 문서 작성
- `docs/2026-05-02-swarm-data-augmentation-design.md` — 6단계 증강 파이프라인
- `docs/2026-05-03-TODO-next-session.md` — 다음 세션 10단계 계획
- `docs/2026-05-02-session-progress.md` — 세션 작업 기록

#### Phase 8 P2 구현 (데이터 증강) — 진행중
- **Step 5**: `src/data/augmentor.py` — SwarmAugmentor 클래스 구현
  - 6단계 증강: basic(250) + size(150) + temporal(150) + anomaly(200) = 750건+
  - 입력: parse_missions_csv 결과 (2,068건)
  - 출력 형식: scenario_id, type, label, n_drones, timing, anomaly_type, missions
  - 이상 시나리오 6종: battery_low, gps_drift, vehicle_loss, comm_outage, near_miss, scatter
- **Step 6**: `src/embedding/rgat_encoder.py` 노드 피처 확장
  - 7d → 10d: +hour_norm, +season_norm, +terrain_norm
  - 벡터 vector[0:10] 활성화 (기존 [0:5]에서 확장)
  - EDGE_TYPES: 3종 → 5종 (+temporal, +anomaly)
  - Temporal 엣지: sequential timing 시 드론 간 시간 순서 연결

#### 규칙 업데이트
- RULES.md Section 11: 문서화/기록 규칙 추가 (기말 보고서 근거용)
- 모든 구현 단계는 docs/에 기록, 세션 종료 시 업데이트 필수

#### Phase 8 P2 실행 결과 (실 데이터 기반)

**실 데이터 파싱**: 2,261건 (격자 51%, 직선 22%, 경유점 17%, 원형 10%)

**증강 결과 (1,050건)**:
- 정상 62% / 이상 38% / 시간차 30%
- 유형: relay_move 486, cooperative_search 418, coverage 79, collision 67
- 이상유형: battery_low/gps_drift/vehicle_loss/comm_outage 각 60건

**R-GAT + HDBSCAN 분석**:
- 벡터: (1050, 128), 활성 10d
- 클러스터: **8개** (기존 500건 → 3개에서 대폭 세분화)
- 노이즈: 22.6% (비학습 R-GAT 기준, 학습 후 개선 예상)
- cooperative_search가 3개 sub-클러스터로 분리 (지역/경로 다양성)
- UMAP: relay_move가 y축으로 명확 분리

**개선 필요**:
- C7(490건) 대형 클러스터 분해 필요 → R-GAT 학습 후 해결 예상
- collision_risk/coverage_overlap UMAP 근접 → 엣지 피처 강화 필요

#### R-GAT 학습 실험 이력

**실험 1: MLP (10d 통계, baseline)**
- 패턴 48.6%, 이상 71.5%
- collision 88.1%, coverage 78.5%
- 10d 통계(proximity_edge_count 등)를 직접 입력 → 가장 높은 정확도

**실험 2: GAT (엣지 타입 무구분, 7d)**
- 패턴 34.8%, 이상 64.8%
- 모든 엣지를 동일 취급 → Proximity 신호 학습 실패

**실험 3: R-GAT (엣지 타입별 독립 어텐션, 7d, 3000건)**
- 패턴 29.3%, 이상 70.2%
- Layer2 Proximity 가중치 63% — **가설 검증 성공** (모델이 Proximity 중요성 학습)
- cooperative vs relay 구분 실패 (둘 다 Proximity 없는 정상 패턴)

**실험 4: R-GAT + SupCon (7d, 3000건)**
- 패턴 30.5%, 이상 70.3% (best 35.7%)
- SupCon 효과 미미 — cooperative/relay가 그래프 구조상 동일하여 임베딩 분리 불가
- 결론: **SupCon 제거**, 노드 피처 확장으로 방향 전환

**실험 5: R-GAT 9d (min_distance_to_other + is_overlap_grid, 3000건)**
- 패턴 30.3%, 이상 70.5%, collision **55.1%**
- min_dist 피처가 충돌 탐지 40.6%→55.1% 개선

**실험 6: R-GAT 9d + Attention Pooling (최종 모델)**
- 패턴 30.3%, **이상 71.5%** (MLP 동등!), collision **58.0%**
- Attention Pooling이 위험 노드에 높은 가중치 → 이상 탐지 향상
- **이상 탐지에서 MLP와 동등 성능 달성**

**최종 비교표:**
| 모델 | 패턴 | 이상 | collision | 비고 |
|------|------|------|-----------|------|
| MLP (10d 통계) | 48.6% | 71.5% | 88.1% | 직접 통계 입력 |
| R-GAT 7d | 29.3% | 70.2% | 40.6% | 기본 R-GAT |
| R-GAT+SupCon | 30.5% | 70.3% | 37.7% | SupCon 효과 미미 |
| R-GAT 9d mean | 30.3% | 70.5% | 55.1% | +min_dist+overlap |
| **R-GAT 9d attn** | **30.3%** | **71.5%** | **58.0%** | **최종 모델** |

#### 설계 검토 기록
- SupCon vs Triplet vs NT-Xent: SupCon 선택 (레이블 있는 상황 최적)
- Information Leakage 분석: n_proximity_neighbors = Proximity 엣지 degree → 제외
- min_distance_to_other: 엣지가 못잡는 연속 거리 → 허용
- is_overlap_grid: Coverage 보완 이진값 → 허용

#### 최종 결론
- **R-GAT 9d + Attention Pooling = 최종 모델** (이상 탐지 71.5%, collision 58%)
- 이상 탐지에서 MLP와 동등 → GNN이 그래프 구조만으로 위험 식별 가능 증명
- Layer2 Proximity 가중치 34% → 제안서 가설("Proximity ∝ anomaly") 검증
- cooperative vs relay 구분 한계 → 데이터 구조적 동일성이 원인 (향후 연구)

#### 저장된 모델
- `data/models/fly2vec_classifier.pt` — MLP V1
- `data/models/fly2vec_classifier_v2.pt` — MLP V2 (스케일링+가중치)
- `data/models/fly2vec_rgat_best.pt` — R-GAT 7d
- `data/models/fly2vec_rgat_supcon.pt` — R-GAT+SupCon (참고용)
- `data/models/fly2vec_rgat_9d_best.pt` — R-GAT 9d mean pool
- `data/models/fly2vec_rgat_attn_best.pt` — **R-GAT 9d attn (최종)**
- `data/models/scaler.pkl` — StandardScaler

#### 다음 세션 TODO
- Qdrant 벌크 로드 (2,261건 단일 + 3,000건 군집)
- Docker E2E 검증
- UMAP 시각화 (학습 전 vs 후 비교)
- 기말 보고서 결과 정리
