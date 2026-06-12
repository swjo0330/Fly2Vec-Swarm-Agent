#!/opt/anaconda3/bin/python
"""
3-Method 앙상블 이상 탐지 테스트/데모 스크립트

설계 문서: 2026-04-29-pg-graph-embedding-design.md §4, §5.2, §6

실행: python3 test_anomaly_ensemble.py

파이프라인:
  1. CSV 로드 + 필터링 (한국 범위, 고도 0-500m, 6+ WP)
  2. NetworkX DiGraph 구성 (Sequential + Spatial KNN 엣지)
  3. Node2Vec Random Walk → gensim Word2Vec → 128d 미션 벡터
  4. 3-Method 앙상블 이상 탐지
     - Method 1: Cluster Centroid Distance (KMeans k=4, cosine)
     - Method 2: Isolation Forest (n_estimators=100, contamination=0.05)
     - Method 3: KNN Average Distance (k=10, cosine)
  5. 합성 이상 주입 → 탐지 검증
  6. 결과 출력 + CSV 저장

의존성: numpy, pandas, networkx, gensim, scikit-learn
"""

import os
import sys
import math
import warnings
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# 설정 상수
# ──────────────────────────────────────────────

# 데이터 경로 (스크립트와 같은 디렉토리의 data/ 하위)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ → 프로젝트 루트
DATA_PATH = PROJECT_ROOT / "fly2vec" / "data" / "dataset_wp_spot.csv"
OUTPUT_PATH = SCRIPT_DIR / "anomaly_results.csv"

# 필터링 조건
LAT_MIN, LAT_MAX = 33.0, 38.0  # 한국 위도 범위
LNG_MIN, LNG_MAX = 124.0, 132.0  # 한국 경도 범위
ALT_MIN, ALT_MAX = 0.0, 500.0  # 고도 범위 (m)
MIN_WP_COUNT = 4  # 최소 웨이포인트 수

# Node2Vec 파라미터 (설계 문서 §4.3)
WALK_LENGTH = 30
NUM_WALKS = 20
P = 1.0  # Return parameter
Q = 0.25  # In-out parameter (DFS 편향, 경로 순서 선호)
EMBEDDING_DIM = 128
WORD2VEC_WINDOW = 10
WORD2VEC_MIN_COUNT = 1
WORD2VEC_EPOCHS = 5
WORD2VEC_WORKERS = 4

# Spatial KNN
SPATIAL_KNN_K = 3
SPATIAL_SEQ_GAP_MIN = 3  # 시퀀스 간격 최소 3 이상만 KNN 엣지

# 이상 탐지 파라미터
KMEANS_K = 4
ISO_FOREST_N_ESTIMATORS = 100
ISO_FOREST_CONTAMINATION = 0.05
KNN_K = 10

# 앙상블 가중치 (설계 문서 §5.2)
W_CENTROID = 0.3
W_ISOFOREST = 0.4
W_KNN = 0.3

# 합성 이상 수
N_PERMUTED = 10  # WP 순서 무작위 치환
N_NOISY = 10  # 좌표 노이즈 주입

# 재현성
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 haversine 거리 (m)"""
    R = 6371000.0  # 지구 반경 (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 방위각 (degrees, 0=North, 90=East)"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    theta = math.atan2(x, y)
    return (math.degrees(theta) + 360) % 360


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-Max 정규화 (0~1)"""
    smin, smax = s.min(), s.max()
    if smax - smin < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - smin) / (smax - smin)


# ──────────────────────────────────────────────
# 1단계: 데이터 로드 + 필터링
# ──────────────────────────────────────────────

def load_and_filter(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """CSV 로드 → 한국 범위 + 고도 + WP 수 필터링 → 경로별 그룹"""
    print("=" * 70)
    print("[1] 데이터 로드 + 필터링")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"  원본 WP 수: {len(df):,}")
    print(f"  원본 경로 수 (wp_spot_id): {df['wp_spot_id'].nunique():,}")

    # 한국 범위 필터
    mask_korea = (
        (df["wp_lat"] >= LAT_MIN) & (df["wp_lat"] <= LAT_MAX) &
        (df["wp_lng"] >= LNG_MIN) & (df["wp_lng"] <= LNG_MAX)
    )
    df = df[mask_korea].copy()
    print(f"  한국 범위 필터 후: {len(df):,} WP")

    # 고도 필터
    mask_alt = (df["wp_alt"] >= ALT_MIN) & (df["wp_alt"] <= ALT_MAX)
    df = df[mask_alt].copy()
    print(f"  고도 필터 (0-500m) 후: {len(df):,} WP")

    # 경로별 그룹 + WP 수 필터
    grouped = {name: group.sort_values("wp_seq").reset_index(drop=True)
               for name, group in df.groupby("wp_spot_id")}
    routes = {k: v for k, v in grouped.items() if len(v) >= MIN_WP_COUNT}

    total_wp = sum(len(v) for v in routes.values())
    print(f"  WP >= {MIN_WP_COUNT} 필터 후: {len(routes):,} 경로 / {total_wp:,} WP")

    # WP 수 분포 통계
    wp_counts = [len(v) for v in routes.values()]
    print(f"  WP 수 통계: min={min(wp_counts)}, median={int(np.median(wp_counts))}, "
          f"mean={np.mean(wp_counts):.1f}, max={max(wp_counts)}")
    print()
    return routes


# ──────────────────────────────────────────────
# 2단계: 그래프 구성 (NetworkX)
# ──────────────────────────────────────────────

def build_graph(route_df: pd.DataFrame) -> nx.DiGraph:
    """
    경로 DataFrame → NetworkX DiGraph 구성

    노드 피처: lat_norm, lon_norm, alt_norm, seq_ratio
    Sequential 엣지: WP_i → WP_{i+1}, 가중치 = 1/(1 + dist/1000)
    Spatial KNN 엣지: K=3, seq_gap >= 3
    """
    G = nx.DiGraph()
    n = len(route_df)

    # 정규화를 위한 값 계산
    lats = route_df["wp_lat"].values
    lngs = route_df["wp_lng"].values
    alts = route_df["wp_alt"].values

    lat_min, lat_max = lats.min(), lats.max()
    lng_min, lng_max = lngs.min(), lngs.max()
    alt_min, alt_max = alts.min(), alts.max()

    lat_range = lat_max - lat_min if lat_max - lat_min > 1e-9 else 1.0
    lng_range = lng_max - lng_min if lng_max - lng_min > 1e-9 else 1.0
    alt_range = alt_max - alt_min if alt_max - alt_min > 1e-9 else 1.0

    # 노드 생성
    for i in range(n):
        node_id = f"WP_{i}"
        G.add_node(node_id,
                   lat_norm=(lats[i] - lat_min) / lat_range,
                   lon_norm=(lngs[i] - lng_min) / lng_range,
                   alt_norm=(alts[i] - alt_min) / alt_range,
                   seq_ratio=i / max(n - 1, 1),
                   lat=lats[i],
                   lon=lngs[i],
                   alt=alts[i])

    # Sequential 엣지 (directed)
    for i in range(n - 1):
        dist = haversine(lats[i], lngs[i], lats[i + 1], lngs[i + 1])
        weight = 1.0 / (1.0 + dist / 1000.0)
        G.add_edge(f"WP_{i}", f"WP_{i + 1}",
                   edge_type="sequential",
                   weight=weight,
                   distance_m=dist)

    # Spatial KNN 엣지 (undirected로 양방향 추가)
    # seq_gap >= 3인 노드 쌍에서 공간적으로 가까운 K개 연결
    if n >= SPATIAL_SEQ_GAP_MIN + 1:
        for i in range(n):
            distances_to_others = []
            for j in range(n):
                seq_gap = abs(i - j)
                if seq_gap < SPATIAL_SEQ_GAP_MIN:
                    continue
                dist = haversine(lats[i], lngs[i], lats[j], lngs[j])
                distances_to_others.append((j, dist))

            # 가장 가까운 K개 선택
            distances_to_others.sort(key=lambda x: x[1])
            for j, dist in distances_to_others[:SPATIAL_KNN_K]:
                weight = 0.5 / (1.0 + dist / 1000.0)
                if not G.has_edge(f"WP_{i}", f"WP_{j}"):
                    G.add_edge(f"WP_{i}", f"WP_{j}",
                               edge_type="spatial_knn",
                               weight=weight,
                               distance_m=dist)
                if not G.has_edge(f"WP_{j}", f"WP_{i}"):
                    G.add_edge(f"WP_{j}", f"WP_{i}",
                               edge_type="spatial_knn",
                               weight=weight,
                               distance_m=dist)

    return G


# ──────────────────────────────────────────────
# 3단계: Node2Vec 임베딩
# ──────────────────────────────────────────────

def biased_random_walk(G: nx.DiGraph, start: str, walk_length: int,
                       p: float, q: float) -> List[str]:
    """
    Node2Vec 편향 랜덤 워크
    p: return parameter (큰 값 → 되돌아가기 억제)
    q: in-out parameter (작은 값 → DFS 편향, 멀리 탐색)
    """
    walk = [start]
    if len(G) <= 1:
        return walk

    for _ in range(walk_length - 1):
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if not neighbors:
            break

        if len(walk) == 1:
            # 첫 스텝: 균일 분포
            next_node = neighbors[np.random.randint(len(neighbors))]
        else:
            prev = walk[-2]
            # 편향 가중치 계산
            weights = []
            for nbr in neighbors:
                edge_weight = G[cur][nbr].get("weight", 1.0)
                if nbr == prev:
                    # 되돌아감: 1/p
                    alpha = 1.0 / p
                elif G.has_edge(prev, nbr) or G.has_edge(nbr, prev):
                    # prev의 이웃이기도 함 (BFS): 1
                    alpha = 1.0
                else:
                    # 새로운 방향 (DFS): 1/q
                    alpha = 1.0 / q
                weights.append(alpha * edge_weight)

            weights = np.array(weights)
            weights /= weights.sum()
            next_node = np.random.choice(neighbors, p=weights)

        walk.append(next_node)

    return walk


def generate_walks(G: nx.DiGraph, num_walks: int, walk_length: int,
                   p: float, q: float) -> List[List[str]]:
    """전체 노드에서 랜덤 워크 생성"""
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = biased_random_walk(G, node, walk_length, p, q)
            walks.append(walk)
    return walks


def embed_mission(G: nx.DiGraph) -> Optional[np.ndarray]:
    """
    그래프 → Node2Vec → 128d 미션 벡터

    가중 평균 풀링: 첫 WP x2.0, 마지막 WP x2.0, 나머지 x1.0
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return None

    # 랜덤 워크 생성
    walks = generate_walks(G, NUM_WALKS, WALK_LENGTH, P, Q)

    if not walks:
        return None

    # Word2Vec 학습
    model = Word2Vec(
        sentences=walks,
        vector_size=EMBEDDING_DIM,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        sg=1,  # Skip-gram
        workers=WORD2VEC_WORKERS,
        epochs=WORD2VEC_EPOCHS,
        seed=RANDOM_SEED
    )

    # 가중 평균 풀링
    embeddings = []
    weights = []
    n = len(nodes)

    for i, node in enumerate(sorted(nodes, key=lambda x: int(x.split("_")[1]))):
        if node not in model.wv:
            continue
        vec = model.wv[node]
        # 가중치: 첫/마지막 WP에 높은 가중치
        if i == 0 or i == n - 1:
            w = 2.0
        else:
            w = 1.0
        embeddings.append(vec)
        weights.append(w)

    if not embeddings:
        return None

    embeddings = np.array(embeddings)
    weights = np.array(weights)
    weights /= weights.sum()

    mission_vector = np.average(embeddings, axis=0, weights=weights)

    # L2 정규화
    norm = np.linalg.norm(mission_vector)
    if norm > 1e-9:
        mission_vector /= norm

    return mission_vector


def embed_all_routes(routes: Dict[str, pd.DataFrame],
                     max_routes: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
    """
    전체 경로 임베딩

    Returns:
        route_ids: 경로 ID 리스트
        vectors: (N, 128) 미션 벡터 행렬
    """
    print("=" * 70)
    print("[2-3] 그래프 구성 + Node2Vec 임베딩")
    print("=" * 70)

    route_ids = []
    vectors = []

    items = list(routes.items())
    if max_routes:
        items = items[:max_routes]

    total = len(items)
    t_start = time.time()

    for idx, (route_id, route_df) in enumerate(items):
        # 진행 상황 출력 (20% 단위)
        if (idx + 1) % max(1, total // 5) == 0 or idx == 0 or idx == total - 1:
            elapsed = time.time() - t_start
            print(f"  [{idx + 1}/{total}] {route_id} (WP={len(route_df)}) "
                  f"... {elapsed:.1f}s")

        # 그래프 구성
        G = build_graph(route_df)

        # 임베딩
        vec = embed_mission(G)
        if vec is not None:
            route_ids.append(route_id)
            vectors.append(vec)

    vectors = np.array(vectors)
    print(f"  임베딩 완료: {len(route_ids)} 경로 → {vectors.shape} 행렬")
    print(f"  총 소요 시간: {time.time() - t_start:.1f}s")
    print()
    return route_ids, vectors


# ──────────────────────────────────────────────
# 4단계: 3-Method 앙상블 이상 탐지
# ──────────────────────────────────────────────

class AnomalyEnsemble:
    """
    3-Method 앙상블 이상 탐지기

    Method 1: Cluster Centroid Distance (KMeans k=4, cosine)
    Method 2: Isolation Forest (n_estimators=100, contamination=0.05)
    Method 3: KNN Average Distance (k=10, cosine)
    """

    def __init__(self):
        self.kmeans = None
        self.centroids = None
        self.iso_forest = None
        self.train_vectors = None
        self.scaler_centroid = None
        self.scaler_knn = None
        # 각 방법의 학습 데이터 스코어 범위 (정규화용)
        self._train_centroid_max = 1.0
        self._train_knn_max = 1.0

    def fit(self, vectors: np.ndarray):
        """정상 미션 벡터로 학습"""
        print("  [앙상블 학습]")
        self.train_vectors = vectors.copy()

        # Method 1: KMeans 학습
        self.kmeans = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_SEED, n_init=10)
        self.kmeans.fit(vectors)
        self.centroids = self.kmeans.cluster_centers_
        # 학습 데이터의 centroid 거리 범위 기록
        train_centroid_scores = self._centroid_distance_raw(vectors)
        self._train_centroid_max = max(np.percentile(train_centroid_scores, 99), 1e-9)
        print(f"    Method 1 (KMeans k={KMEANS_K}): centroid 학습 완료")

        # Method 2: Isolation Forest 학습
        self.iso_forest = IsolationForest(
            n_estimators=ISO_FOREST_N_ESTIMATORS,
            contamination=ISO_FOREST_CONTAMINATION,
            random_state=RANDOM_SEED
        )
        self.iso_forest.fit(vectors)
        print(f"    Method 2 (IsolationForest n={ISO_FOREST_N_ESTIMATORS}): 학습 완료")

        # Method 2: Isolation Forest 학습 데이터의 스코어 범위 기록 (정규화용)
        iso_train_raw = self.iso_forest.score_samples(vectors)
        self._iso_train_min = iso_train_raw.min()
        self._iso_train_max = iso_train_raw.max()

        # Method 3: KNN 학습 데이터의 거리 범위 기록 (exclude_self=True로 자기 자신 제외)
        train_knn_scores = self._knn_distance_raw(vectors, exclude_self=True)
        self._train_knn_max = max(np.percentile(train_knn_scores, 99), 1e-9)
        print(f"    Method 3 (KNN k={KNN_K}): 학습 완료")
        print()

    def _centroid_distance_raw(self, vectors: np.ndarray) -> np.ndarray:
        """각 벡터의 가장 가까운 centroid까지 cosine 거리"""
        cos_dist = cosine_distances(vectors, self.centroids)  # (N, K)
        return cos_dist.min(axis=1)  # (N,)

    def _knn_distance_raw(self, vectors: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        """각 벡터의 학습 데이터 top-K 평균 cosine 거리

        Args:
            exclude_self: True일 때 자기 자신과의 거리(0)를 제외 (학습 데이터 평가 시)
        """
        cos_dist = cosine_distances(vectors, self.train_vectors)  # (N, M)
        if exclude_self:
            # 학습 데이터가 자기 자신을 쿼리할 때: 첫 번째 열(self-match, distance=0) 스킵
            k = min(KNN_K, cos_dist.shape[1] - 1)
            knn_dists = np.sort(cos_dist, axis=1)[:, 1:k+1]
        else:
            k = min(KNN_K, cos_dist.shape[1])
            # 각 행에서 가장 가까운 k개의 평균
            knn_dists = np.sort(cos_dist, axis=1)[:, :k]
        return knn_dists.mean(axis=1)  # (N,)

    def score(self, vectors: np.ndarray) -> Dict[str, np.ndarray]:
        """
        이상 점수 계산 (0~1, 높을수록 이상)

        Returns:
            dict with keys: centroid, isoforest, knn, ensemble
        """
        # Method 1: Centroid Distance (0~1 정규화)
        centroid_raw = self._centroid_distance_raw(vectors)
        centroid_score = np.clip(centroid_raw / self._train_centroid_max, 0, 1)

        # Method 2: Isolation Forest
        # score_samples(): 낮을수록 이상 → 반전 + 정규화 (학습 데이터 기준 고정 범위)
        iso_raw = self.iso_forest.score_samples(vectors)
        iso_score = 1.0 - (iso_raw - self._iso_train_min) / max(self._iso_train_max - self._iso_train_min, 1e-9)
        iso_score = np.clip(iso_score, 0, 1)

        # Method 3: KNN Average Distance (0~1 정규화)
        knn_raw = self._knn_distance_raw(vectors)
        knn_score = np.clip(knn_raw / self._train_knn_max, 0, 1)

        # 앙상블 가중 평균
        ensemble = W_CENTROID * centroid_score + W_ISOFOREST * iso_score + W_KNN * knn_score

        return {
            "centroid": centroid_score,
            "isoforest": iso_score,
            "knn": knn_score,
            "ensemble": ensemble
        }


# ──────────────────────────────────────────────
# 5단계: 합성 이상 주입
# ──────────────────────────────────────────────

def inject_permuted_anomalies(routes: Dict[str, pd.DataFrame],
                              route_ids: List[str],
                              n: int = N_PERMUTED) -> Tuple[List[str], np.ndarray]:
    """
    정상 경로의 WP 순서를 무작위 치환 → 합성 이상
    경로 구조가 파괴되므로 이상으로 탐지되어야 함
    """
    selected = np.random.choice(route_ids, size=min(n, len(route_ids)), replace=False)
    anomaly_ids = []
    anomaly_vectors = []

    for rid in selected:
        df = routes[rid].copy()
        # WP 순서 무작위 치환
        perm_idx = np.random.permutation(len(df))
        df_perm = df.iloc[perm_idx].reset_index(drop=True)
        # wp_seq을 다시 0-based로 재설정
        df_perm["wp_seq"] = range(1, len(df_perm) + 1)

        G = build_graph(df_perm)
        vec = embed_mission(G)
        if vec is not None:
            anomaly_ids.append(f"PERM_{rid}")
            anomaly_vectors.append(vec)

    return anomaly_ids, np.array(anomaly_vectors) if anomaly_vectors else np.empty((0, EMBEDDING_DIM))


def inject_noisy_anomalies(routes: Dict[str, pd.DataFrame],
                           route_ids: List[str],
                           n: int = N_NOISY,
                           noise_std: float = 0.05) -> Tuple[List[str], np.ndarray]:
    """
    정상 경로의 좌표에 큰 노이즈 추가 → 공간 이상
    (위도/경도에 ~0.05도 = ~5km 노이즈)
    """
    selected = np.random.choice(route_ids, size=min(n, len(route_ids)), replace=False)
    anomaly_ids = []
    anomaly_vectors = []

    for rid in selected:
        df = routes[rid].copy()
        # 좌표에 노이즈 주입
        df["wp_lat"] = df["wp_lat"] + np.random.normal(0, noise_std, len(df))
        df["wp_lng"] = df["wp_lng"] + np.random.normal(0, noise_std, len(df))
        # 고도에도 노이즈 (0-500m 범위 유지)
        df["wp_alt"] = np.clip(
            df["wp_alt"] + np.random.normal(0, 50, len(df)), ALT_MIN, ALT_MAX
        )

        G = build_graph(df)
        vec = embed_mission(G)
        if vec is not None:
            anomaly_ids.append(f"NOISE_{rid}")
            anomaly_vectors.append(vec)

    return anomaly_ids, np.array(anomaly_vectors) if anomaly_vectors else np.empty((0, EMBEDDING_DIM))


# ──────────────────────────────────────────────
# 6단계: 결과 출력 + CSV 저장
# ──────────────────────────────────────────────

def print_score_table(label: str, scores: Dict[str, np.ndarray]):
    """점수 통계 테이블 출력"""
    print(f"  {label}:")
    print(f"    {'방법':<20} {'평균':>8} {'표준편차':>8} {'최소':>8} {'최대':>8} {'중앙값':>8}")
    print(f"    {'-' * 60}")
    for method in ["centroid", "isoforest", "knn", "ensemble"]:
        vals = scores[method]
        if len(vals) == 0:
            continue
        print(f"    {method:<20} {vals.mean():>8.4f} {vals.std():>8.4f} "
              f"{vals.min():>8.4f} {vals.max():>8.4f} {np.median(vals):>8.4f}")
    print()


def interpret_score(score: float) -> str:
    """이상 점수 해석 (설계 문서 §5.2)"""
    if score < 0.3:
        return "정상"
    elif score < 0.5:
        return "주의"
    elif score < 0.7:
        return "이상"
    else:
        return "심각"


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────

def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  AERION — 3-Method 앙상블 이상 탐지 테스트/데모                     ║")
    print("║  설계: 2026-04-29-pg-graph-embedding-design.md §4, §5.2, §6       ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # ── 1단계: 데이터 로드 ──
    if not DATA_PATH.exists():
        print(f"[오류] 데이터 파일 없음: {DATA_PATH}")
        sys.exit(1)

    routes = load_and_filter(DATA_PATH)
    if len(routes) < 30:
        print(f"[오류] 필터 후 경로 수 부족 ({len(routes)}건). 최소 30건 필요.")
        sys.exit(1)

    # ── 2-3단계: 그래프 구성 + 임베딩 ──
    # 성능을 위해 최대 200개 경로만 사용 (전체 1,395건은 ~30분 소요)
    MAX_ROUTES = 200
    print(f"  [참고] 데모용으로 최대 {MAX_ROUTES}개 경로만 처리합니다.")
    print(f"  (전체 {len(routes)}건 처리 시 ~{len(routes) * 1.3 / 60:.0f}분 소요)")
    print()

    route_ids, vectors = embed_all_routes(routes, max_routes=MAX_ROUTES)

    if len(route_ids) < 30:
        print(f"[오류] 임베딩 성공 경로 부족 ({len(route_ids)}건)")
        sys.exit(1)

    # ── 4단계: 학습/테스트 분할 + 앙상블 학습 ──
    print("=" * 70)
    print("[4] 3-Method 앙상블 이상 탐지")
    print("=" * 70)

    n_total = len(route_ids)
    n_train = int(n_total * 0.8)

    # 셔플 후 분할
    perm = np.random.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_ids = [route_ids[i] for i in train_idx]
    test_ids = [route_ids[i] for i in test_idx]
    train_vectors = vectors[train_idx]
    test_vectors = vectors[test_idx]

    print(f"  학습: {len(train_ids)}건 / 테스트: {len(test_ids)}건")
    print()

    # 앙상블 학습
    ensemble = AnomalyEnsemble()
    ensemble.fit(train_vectors)

    # 테스트 데이터 점수 계산
    test_scores = ensemble.score(test_vectors)

    print("=" * 70)
    print("[5] 합성 이상 주입 + 탐지 검증")
    print("=" * 70)

    # ── 5단계: 합성 이상 주입 ──
    print(f"  순서 치환 이상 {N_PERMUTED}건 생성 중...")
    perm_ids, perm_vectors = inject_permuted_anomalies(routes, train_ids, N_PERMUTED)
    print(f"    → {len(perm_ids)}건 생성 완료")

    print(f"  좌표 노이즈 이상 {N_NOISY}건 생성 중...")
    noise_ids, noise_vectors = inject_noisy_anomalies(routes, train_ids, N_NOISY)
    print(f"    → {len(noise_ids)}건 생성 완료")
    print()

    # 이상 점수 계산
    perm_scores = ensemble.score(perm_vectors) if len(perm_vectors) > 0 else None
    noise_scores = ensemble.score(noise_vectors) if len(noise_vectors) > 0 else None

    # ── 6단계: 결과 출력 ──
    print("=" * 70)
    print("[6] 결과 분석")
    print("=" * 70)
    print()

    # 6-1. 점수 통계 비교
    print("─" * 60)
    print("A. 이상 점수 통계 비교")
    print("─" * 60)
    print()

    print_score_table("정상 테스트 데이터", test_scores)
    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        print_score_table("합성 이상 (WP 순서 치환)", perm_scores)
    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        print_score_table("합성 이상 (좌표 노이즈)", noise_scores)

    # 6-2. 앙상블 점수 비교 요약
    print("─" * 60)
    print("B. 앙상블 점수 비교 요약")
    print("─" * 60)
    print()

    normal_mean = test_scores["ensemble"].mean()
    perm_mean = perm_scores["ensemble"].mean() if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0 else 0
    noise_mean = noise_scores["ensemble"].mean() if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0 else 0

    print(f"  {'구분':<25} {'평균 앙상블 점수':>16} {'해석':>8}")
    print(f"  {'-' * 55}")
    print(f"  {'정상 테스트':<25} {normal_mean:>16.4f} {interpret_score(normal_mean):>8}")
    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        print(f"  {'WP 순서 치환 이상':<25} {perm_mean:>16.4f} {interpret_score(perm_mean):>8}")
    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        print(f"  {'좌표 노이즈 이상':<25} {noise_mean:>16.4f} {interpret_score(noise_mean):>8}")
    print()

    # 6-3. 탐지율 계산
    # 정상 테스트의 95 백분위수를 임계값으로 사용
    threshold = np.percentile(test_scores["ensemble"], 95)
    print("─" * 60)
    print("C. 탐지율 (정상 95백분위수 임계값)")
    print("─" * 60)
    print(f"  임계값: {threshold:.4f}")
    print()

    normal_flagged = (test_scores["ensemble"] > threshold).sum()
    print(f"  정상 오탐률 (FPR): {normal_flagged}/{len(test_ids)} "
          f"= {normal_flagged / max(len(test_ids), 1) * 100:.1f}%")

    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        perm_detected = (perm_scores["ensemble"] > threshold).sum()
        print(f"  순서 치환 탐지율: {perm_detected}/{len(perm_ids)} "
              f"= {perm_detected / max(len(perm_ids), 1) * 100:.1f}%")

    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        noise_detected = (noise_scores["ensemble"] > threshold).sum()
        print(f"  좌표 노이즈 탐지율: {noise_detected}/{len(noise_ids)} "
              f"= {noise_detected / max(len(noise_ids), 1) * 100:.1f}%")
    print()

    # 6-4. 개별 경로 점수 예시 (상위 5건)
    print("─" * 60)
    print("D. 테스트 경로 점수 예시 (앙상블 점수 상위 5건)")
    print("─" * 60)
    print()
    print(f"  {'경로 ID':<20} {'Centroid':>10} {'IsoForest':>10} {'KNN':>10} {'앙상블':>10} {'해석':>8}")
    print(f"  {'-' * 72}")

    top_idx = np.argsort(test_scores["ensemble"])[::-1][:5]
    for i in top_idx:
        rid = test_ids[i]
        c = test_scores["centroid"][i]
        iso = test_scores["isoforest"][i]
        k = test_scores["knn"][i]
        e = test_scores["ensemble"][i]
        print(f"  {rid:<20} {c:>10.4f} {iso:>10.4f} {k:>10.4f} {e:>10.4f} {interpret_score(e):>8}")
    print()

    # 6-5. 합성 이상 개별 점수
    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        print("─" * 60)
        print("E. 합성 이상 (WP 순서 치환) 개별 점수")
        print("─" * 60)
        print()
        print(f"  {'경로 ID':<25} {'Centroid':>10} {'IsoForest':>10} {'KNN':>10} {'앙상블':>10} {'해석':>8}")
        print(f"  {'-' * 77}")
        for i in range(len(perm_ids)):
            c = perm_scores["centroid"][i]
            iso = perm_scores["isoforest"][i]
            k = perm_scores["knn"][i]
            e = perm_scores["ensemble"][i]
            flag = " ◀ 탐지" if e > threshold else ""
            print(f"  {perm_ids[i]:<25} {c:>10.4f} {iso:>10.4f} {k:>10.4f} {e:>10.4f} {interpret_score(e):>8}{flag}")
        print()

    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        print("─" * 60)
        print("F. 합성 이상 (좌표 노이즈) 개별 점수")
        print("─" * 60)
        print()
        print(f"  {'경로 ID':<25} {'Centroid':>10} {'IsoForest':>10} {'KNN':>10} {'앙상블':>10} {'해석':>8}")
        print(f"  {'-' * 77}")
        for i in range(len(noise_ids)):
            c = noise_scores["centroid"][i]
            iso = noise_scores["isoforest"][i]
            k = noise_scores["knn"][i]
            e = noise_scores["ensemble"][i]
            flag = " ◀ 탐지" if e > threshold else ""
            print(f"  {noise_ids[i]:<25} {c:>10.4f} {iso:>10.4f} {k:>10.4f} {e:>10.4f} {interpret_score(e):>8}{flag}")
        print()

    # ── CSV 저장 ──
    print("─" * 60)
    print("G. 결과 CSV 저장")
    print("─" * 60)

    results = []

    # 정상 테스트
    for i, rid in enumerate(test_ids):
        results.append({
            "route_id": rid,
            "type": "normal_test",
            "centroid_score": test_scores["centroid"][i],
            "isoforest_score": test_scores["isoforest"][i],
            "knn_score": test_scores["knn"][i],
            "ensemble_score": test_scores["ensemble"][i],
            "interpretation": interpret_score(test_scores["ensemble"][i]),
            "detected": test_scores["ensemble"][i] > threshold
        })

    # 순서 치환 이상
    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        for i, rid in enumerate(perm_ids):
            results.append({
                "route_id": rid,
                "type": "anomaly_permuted",
                "centroid_score": perm_scores["centroid"][i],
                "isoforest_score": perm_scores["isoforest"][i],
                "knn_score": perm_scores["knn"][i],
                "ensemble_score": perm_scores["ensemble"][i],
                "interpretation": interpret_score(perm_scores["ensemble"][i]),
                "detected": perm_scores["ensemble"][i] > threshold
            })

    # 좌표 노이즈 이상
    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        for i, rid in enumerate(noise_ids):
            results.append({
                "route_id": rid,
                "type": "anomaly_noisy",
                "centroid_score": noise_scores["centroid"][i],
                "isoforest_score": noise_scores["isoforest"][i],
                "knn_score": noise_scores["knn"][i],
                "ensemble_score": noise_scores["ensemble"][i],
                "interpretation": interpret_score(noise_scores["ensemble"][i]),
                "detected": noise_scores["ensemble"][i] > threshold
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {OUTPUT_PATH}")
    print(f"  총 {len(result_df)}건 (정상 {len(test_ids)} + 이상 {len(perm_ids) + len(noise_ids)})")
    print()

    # ── 최종 요약 ──
    print("╔" + "═" * 68 + "╗")
    print("║  최종 요약                                                         ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  입력 경로: {len(routes):>6}건 (필터 후)                                   ║")
    print(f"║  처리 경로: {len(route_ids):>6}건 (임베딩 성공)                                ║")
    print(f"║  학습 데이터: {n_train:>4}건 / 테스트: {len(test_ids):>4}건                           ║")
    print(f"║  임베딩 차원: {EMBEDDING_DIM}d / Node2Vec (p={P}, q={Q})                     ║")
    print(f"║  앙상블: Centroid({W_CENTROID}) + IsoForest({W_ISOFOREST}) + KNN({W_KNN})          ║")
    print(f"║                                                                    ║")
    print(f"║  정상 평균 점수:       {normal_mean:.4f} ({interpret_score(normal_mean)})                          ║")
    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        print(f"║  순서치환 이상 평균:   {perm_mean:.4f} ({interpret_score(perm_mean)})                          ║")
    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        print(f"║  노이즈 이상 평균:    {noise_mean:.4f} ({interpret_score(noise_mean)})                          ║")
    print(f"║  탐지 임계값 (95%ile): {threshold:.4f}                                     ║")
    print("╚" + "═" * 68 + "╝")
    print()


if __name__ == "__main__":
    main()
