#!/opt/anaconda3/bin/python
"""
AERION - 전체 데이터셋 3-Method 앙상블 이상 탐지 + HDBSCAN + UMAP 시각화

기반: test_anomaly_ensemble.py
변경사항:
  1. 전체 경로 처리 (MAX_ROUTES=2000, 제한 없음)
  2. HDBSCAN 클러스터링 + 클러스터별 이상 임계값
  3. UMAP 2D 시각화 (PNG 저장)
  4. 종합 결과 CSV 저장 (anomaly_results_full.csv, cluster_summary.csv)
  5. 한국어 리포트 요약 출력

실행: python3 run_full_analysis.py
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

# UMAP / HDBSCAN / matplotlib
import umap
import hdbscan
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────
# 설정 상수
# ──────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ → 프로젝트 루트
DATA_PATH = PROJECT_ROOT / "fly2vec" / "data" / "dataset_wp_spot.csv"

# 자동 증가 디렉토리: results/report1, report2, report3, ...
def _next_report_dir() -> Path:
    base = PROJECT_ROOT / "results"
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("report")])
    if not existing:
        num = 1
    else:
        nums = [int(d.name.replace("report", "")) for d in existing if d.name.replace("report", "").isdigit()]
        num = max(nums) + 1 if nums else 1
    return base / f"report{num}"

RESULTS_DIR = _next_report_dir()
OUTPUT_CSV = RESULTS_DIR / "anomaly_results_full.csv"
CLUSTER_CSV = RESULTS_DIR / "cluster_summary.csv"
UMAP_PNG = RESULTS_DIR / "umap_clusters.png"

# 필터링 조건
LAT_MIN, LAT_MAX = 33.0, 38.0
LNG_MIN, LNG_MAX = 124.0, 132.0
ALT_MIN, ALT_MAX = 0.0, 500.0
MIN_WP_COUNT = 4

# Node2Vec 파라미터
WALK_LENGTH = 30
NUM_WALKS = 20
P = 1.0
Q = 0.25
EMBEDDING_DIM = 128
WORD2VEC_WINDOW = 10
WORD2VEC_MIN_COUNT = 1
WORD2VEC_EPOCHS = 5
WORD2VEC_WORKERS = 4

# Spatial KNN
SPATIAL_KNN_K = 3
SPATIAL_SEQ_GAP_MIN = 3

# 이상 탐지 파라미터
KMEANS_K = 4
ISO_FOREST_N_ESTIMATORS = 100
ISO_FOREST_CONTAMINATION = 0.05
KNN_K = 10

# 앙상블 가중치
W_CENTROID = 0.3
W_ISOFOREST = 0.4
W_KNN = 0.3

# HDBSCAN 파라미터
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5

# 합성 이상 수
N_PERMUTED = 15
N_NOISY = 15

# 전체 경로 처리
MAX_ROUTES = 2000  # 사실상 전체 처리

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def normalize_series(s: pd.Series) -> pd.Series:
    smin, smax = s.min(), s.max()
    if smax - smin < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - smin) / (smax - smin)


def interpret_score(score: float) -> str:
    if score < 0.3:
        return "정상"
    elif score < 0.5:
        return "주의"
    elif score < 0.7:
        return "이상"
    else:
        return "심각"


# ──────────────────────────────────────────────
# 1단계: 데이터 로드 + 필터링
# ──────────────────────────────────────────────

def load_and_filter(csv_path: Path) -> Dict[str, pd.DataFrame]:
    print("=" * 70)
    print("[1] 데이터 로드 + 필터링")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"  원본 WP 수: {len(df):,}")
    print(f"  원본 경로 수 (wp_spot_id): {df['wp_spot_id'].nunique():,}")

    mask_korea = (
        (df["wp_lat"] >= LAT_MIN) & (df["wp_lat"] <= LAT_MAX) &
        (df["wp_lng"] >= LNG_MIN) & (df["wp_lng"] <= LNG_MAX)
    )
    df = df[mask_korea].copy()
    print(f"  한국 범위 필터 후: {len(df):,} WP")

    mask_alt = (df["wp_alt"] >= ALT_MIN) & (df["wp_alt"] <= ALT_MAX)
    df = df[mask_alt].copy()
    print(f"  고도 필터 (0-500m) 후: {len(df):,} WP")

    grouped = {name: group.sort_values("wp_seq").reset_index(drop=True)
               for name, group in df.groupby("wp_spot_id")}
    routes = {k: v for k, v in grouped.items() if len(v) >= MIN_WP_COUNT}

    total_wp = sum(len(v) for v in routes.values())
    print(f"  WP >= {MIN_WP_COUNT} 필터 후: {len(routes):,} 경로 / {total_wp:,} WP")

    wp_counts = [len(v) for v in routes.values()]
    if wp_counts:
        print(f"  WP 수 통계: min={min(wp_counts)}, median={int(np.median(wp_counts))}, "
              f"mean={np.mean(wp_counts):.1f}, max={max(wp_counts)}")
    print()
    return routes


# ──────────────────────────────────────────────
# 2단계: 그래프 구성
# ──────────────────────────────────────────────

def build_graph(route_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    n = len(route_df)
    if n == 0:
        return G

    lats = route_df["wp_lat"].values
    lngs = route_df["wp_lng"].values
    alts = route_df["wp_alt"].values

    # 전역 정규화 (한국 중심) — 절대 위치 보존
    # 기존: 경로 내 min-max → 절대 위치 소실
    # 변경: 한국 전체 기준 → "인천 격자"와 "충남 격자"가 다른 좌표
    KOREA_LAT_CENTER, KOREA_LAT_RANGE = 35.5, 2.5   # 33~38
    KOREA_LNG_CENTER, KOREA_LNG_RANGE = 128.0, 4.0   # 124~132
    ALT_MAX_NORM = 500.0  # 0~500m

    for i in range(n):
        node_id = f"WP_{i}"
        G.add_node(node_id,
                   lat_norm=(lats[i] - KOREA_LAT_CENTER) / KOREA_LAT_RANGE,
                   lon_norm=(lngs[i] - KOREA_LNG_CENTER) / KOREA_LNG_RANGE,
                   alt_norm=alts[i] / ALT_MAX_NORM,
                   seq_ratio=i / max(n - 1, 1),
                   lat=lats[i], lon=lngs[i], alt=alts[i])

    # 엣지 가중치: 고도 변화 반영
    # 고도 1m = 수평 거리 0.5m로 환산하여 "유효 거리" 계산
    ALT_SENSITIVITY = 0.5
    for i in range(n - 1):
        dist = haversine(lats[i], lngs[i], lats[i + 1], lngs[i + 1])
        alt_delta = abs(alts[i + 1] - alts[i])
        effective_dist = (dist**2 + (alt_delta * ALT_SENSITIVITY)**2) ** 0.5
        weight = 1.0 / (1.0 + effective_dist / 1000.0)
        G.add_edge(f"WP_{i}", f"WP_{i + 1}",
                   edge_type="sequential", weight=weight,
                   distance_m=dist, alt_change_m=alt_delta)

    if n >= SPATIAL_SEQ_GAP_MIN + 1:
        for i in range(n):
            distances_to_others = []
            for j in range(n):
                if abs(i - j) < SPATIAL_SEQ_GAP_MIN:
                    continue
                dist = haversine(lats[i], lngs[i], lats[j], lngs[j])
                alt_delta = abs(alts[i] - alts[j])
                effective_dist = (dist**2 + (alt_delta * ALT_SENSITIVITY)**2) ** 0.5
                distances_to_others.append((j, effective_dist))
            distances_to_others.sort(key=lambda x: x[1])
            for j, eff_dist in distances_to_others[:SPATIAL_KNN_K]:
                weight = 0.5 / (1.0 + eff_dist / 1000.0)
                if not G.has_edge(f"WP_{i}", f"WP_{j}"):
                    G.add_edge(f"WP_{i}", f"WP_{j}",
                               edge_type="spatial_knn", weight=weight, distance_m=eff_dist)
                if not G.has_edge(f"WP_{j}", f"WP_{i}"):
                    G.add_edge(f"WP_{j}", f"WP_{i}",
                               edge_type="spatial_knn", weight=weight, distance_m=eff_dist)
    return G


# ──────────────────────────────────────────────
# 3단계: Node2Vec 임베딩
# ──────────────────────────────────────────────

def biased_random_walk(G: nx.DiGraph, start: str, walk_length: int,
                       p: float, q: float) -> List[str]:
    walk = [start]
    if len(G) <= 1:
        return walk
    for _ in range(walk_length - 1):
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if not neighbors:
            break
        if len(walk) == 1:
            next_node = neighbors[np.random.randint(len(neighbors))]
        else:
            prev = walk[-2]
            weights = []
            for nbr in neighbors:
                edge_weight = G[cur][nbr].get("weight", 1.0)
                if nbr == prev:
                    alpha = 1.0 / p
                elif G.has_edge(prev, nbr) or G.has_edge(nbr, prev):
                    alpha = 1.0
                else:
                    alpha = 1.0 / q
                weights.append(alpha * edge_weight)
            weights = np.array(weights)
            if weights.sum() < 1e-12:
                break
            weights /= weights.sum()
            next_node = np.random.choice(neighbors, p=weights)
        walk.append(next_node)
    return walk


def generate_walks(G: nx.DiGraph, num_walks: int, walk_length: int,
                   p: float, q: float) -> List[List[str]]:
    walks = []
    nodes = list(G.nodes())
    if not nodes:
        return walks
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = biased_random_walk(G, node, walk_length, p, q)
            walks.append(walk)
    return walks


def embed_mission(G: nx.DiGraph) -> Optional[np.ndarray]:
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return None

    walks = generate_walks(G, NUM_WALKS, WALK_LENGTH, P, Q)
    if not walks:
        return None

    model = Word2Vec(
        sentences=walks,
        vector_size=EMBEDDING_DIM,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        sg=1,
        workers=WORD2VEC_WORKERS,
        epochs=WORD2VEC_EPOCHS,
        seed=RANDOM_SEED
    )

    embeddings = []
    weights = []
    n = len(nodes)
    for i, node in enumerate(sorted(nodes, key=lambda x: int(x.split("_")[1]))):
        if node not in model.wv:
            continue
        vec = model.wv[node]
        w = 2.0 if (i == 0 or i == n - 1) else 1.0
        embeddings.append(vec)
        weights.append(w)

    if not embeddings:
        return None

    embeddings = np.array(embeddings)
    weights = np.array(weights)
    weights /= weights.sum()
    mission_vector = np.average(embeddings, axis=0, weights=weights)

    norm = np.linalg.norm(mission_vector)
    if norm > 1e-9:
        mission_vector /= norm
    return mission_vector


def embed_all_routes(routes: Dict[str, pd.DataFrame],
                     max_routes: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
    print("=" * 70)
    print("[2-3] 그래프 구성 + Node2Vec 임베딩 (전체 데이터셋)")
    print("=" * 70)

    route_ids = []
    vectors = []

    items = list(routes.items())
    if max_routes and max_routes < len(items):
        items = items[:max_routes]

    total = len(items)
    t_start = time.time()
    report_interval = max(1, total // 10)

    for idx, (route_id, route_df) in enumerate(items):
        if (idx + 1) % report_interval == 0 or idx == 0 or idx == total - 1:
            elapsed = time.time() - t_start
            eta = (elapsed / max(idx, 1)) * (total - idx) if idx > 0 else 0
            print(f"  [{idx + 1}/{total}] {route_id} (WP={len(route_df)}) "
                  f"... {elapsed:.1f}s (ETA: {eta:.0f}s)")

        G = build_graph(route_df)
        vec = embed_mission(G)
        if vec is not None:
            route_ids.append(route_id)
            vectors.append(vec)

    vectors = np.array(vectors) if vectors else np.empty((0, EMBEDDING_DIM))
    print(f"  임베딩 완료: {len(route_ids)} 경로 -> {vectors.shape} 행렬")
    print(f"  총 소요 시간: {time.time() - t_start:.1f}s")
    print()
    return route_ids, vectors


# ──────────────────────────────────────────────
# 4단계: 3-Method 앙상블 이상 탐지
# ──────────────────────────────────────────────

class AnomalyEnsemble:
    def __init__(self):
        self.kmeans = None
        self.centroids = None
        self.iso_forest = None
        self.train_vectors = None
        self._train_centroid_max = 1.0
        self._train_knn_max = 1.0

    def fit(self, vectors: np.ndarray):
        print("  [앙상블 학습]")
        self.train_vectors = vectors.copy()

        k = min(KMEANS_K, len(vectors))
        self.kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        self.kmeans.fit(vectors)
        self.centroids = self.kmeans.cluster_centers_
        train_centroid_scores = self._centroid_distance_raw(vectors)
        self._train_centroid_max = max(np.percentile(train_centroid_scores, 99), 1e-9)
        print(f"    Method 1 (KMeans k={k}): centroid 학습 완료")

        self.iso_forest = IsolationForest(
            n_estimators=ISO_FOREST_N_ESTIMATORS,
            contamination=ISO_FOREST_CONTAMINATION,
            random_state=RANDOM_SEED
        )
        self.iso_forest.fit(vectors)
        # Method 2: Isolation Forest 학습 데이터의 스코어 범위 기록 (정규화용)
        iso_train_raw = self.iso_forest.score_samples(vectors)
        self._iso_train_min = iso_train_raw.min()
        self._iso_train_max = iso_train_raw.max()
        print(f"    Method 2 (IsolationForest n={ISO_FOREST_N_ESTIMATORS}): 학습 완료")

        train_knn_scores = self._knn_distance_raw(vectors, exclude_self=True)
        self._train_knn_max = max(np.percentile(train_knn_scores, 99), 1e-9)
        print(f"    Method 3 (KNN k={KNN_K}): 학습 완료")
        print()

    def _centroid_distance_raw(self, vectors: np.ndarray) -> np.ndarray:
        cos_dist = cosine_distances(vectors, self.centroids)
        return cos_dist.min(axis=1)

    def _knn_distance_raw(self, vectors: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        cos_dist = cosine_distances(vectors, self.train_vectors)
        if exclude_self:
            # 학습 데이터가 자기 자신을 쿼리할 때: 첫 번째 열(self-match, distance=0) 스킵
            k = min(KNN_K, cos_dist.shape[1] - 1)
            knn_dists = np.sort(cos_dist, axis=1)[:, 1:k+1]
        else:
            k = min(KNN_K, cos_dist.shape[1])
            knn_dists = np.sort(cos_dist, axis=1)[:, :k]
        return knn_dists.mean(axis=1)

    def score(self, vectors: np.ndarray) -> Dict[str, np.ndarray]:
        centroid_raw = self._centroid_distance_raw(vectors)
        centroid_score = np.clip(centroid_raw / self._train_centroid_max, 0, 1)

        iso_raw = self.iso_forest.score_samples(vectors)
        iso_score = 1.0 - (iso_raw - self._iso_train_min) / max(self._iso_train_max - self._iso_train_min, 1e-9)
        iso_score = np.clip(iso_score, 0, 1)

        knn_raw = self._knn_distance_raw(vectors)
        knn_score = np.clip(knn_raw / self._train_knn_max, 0, 1)

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

def inject_permuted_anomalies(routes, route_ids, n=N_PERMUTED):
    selected = np.random.choice(route_ids, size=min(n, len(route_ids)), replace=False)
    anomaly_ids, anomaly_vectors = [], []
    for rid in selected:
        df = routes[rid].copy()
        perm_idx = np.random.permutation(len(df))
        df_perm = df.iloc[perm_idx].reset_index(drop=True)
        df_perm["wp_seq"] = range(1, len(df_perm) + 1)
        G = build_graph(df_perm)
        vec = embed_mission(G)
        if vec is not None:
            anomaly_ids.append(f"PERM_{rid}")
            anomaly_vectors.append(vec)
    return anomaly_ids, np.array(anomaly_vectors) if anomaly_vectors else np.empty((0, EMBEDDING_DIM))


def inject_noisy_anomalies(routes, route_ids, n=N_NOISY, noise_std=0.05):
    selected = np.random.choice(route_ids, size=min(n, len(route_ids)), replace=False)
    anomaly_ids, anomaly_vectors = [], []
    for rid in selected:
        df = routes[rid].copy()
        df["wp_lat"] = df["wp_lat"] + np.random.normal(0, noise_std, len(df))
        df["wp_lng"] = df["wp_lng"] + np.random.normal(0, noise_std, len(df))
        df["wp_alt"] = np.clip(df["wp_alt"] + np.random.normal(0, 50, len(df)), ALT_MIN, ALT_MAX)
        G = build_graph(df)
        vec = embed_mission(G)
        if vec is not None:
            anomaly_ids.append(f"NOISE_{rid}")
            anomaly_vectors.append(vec)
    return anomaly_ids, np.array(anomaly_vectors) if anomaly_vectors else np.empty((0, EMBEDDING_DIM))


# ──────────────────────────────────────────────
# HDBSCAN 클러스터링 + 클러스터별 임계값
# ──────────────────────────────────────────────

def run_hdbscan(vectors: np.ndarray):
    """HDBSCAN 클러스터링 수행"""
    print("=" * 70)
    print("[HDBSCAN] 클러스터링")
    print("=" * 70)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(vectors)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = (labels == -1).sum()
    print(f"  클러스터 수: {n_clusters}")
    print(f"  노이즈 포인트: {n_noise}")
    print()

    # 클러스터별 분포
    print(f"  {'클러스터':<12} {'경로 수':>8} {'비율':>8}")
    print(f"  {'-' * 32}")
    for label in sorted(unique_labels):
        count = (labels == label).sum()
        pct = count / len(labels) * 100
        name = "노이즈(-1)" if label == -1 else f"클러스터 {label}"
        print(f"  {name:<12} {count:>8} {pct:>7.1f}%")
    print()

    return labels, clusterer


def compute_per_cluster_thresholds(labels: np.ndarray, ensemble_scores: np.ndarray,
                                   percentile: float = 95.0) -> Dict[int, float]:
    """클러스터별 이상 임계값 계산 (95 백분위수)"""
    thresholds = {}
    unique_labels = set(labels)
    global_threshold = np.percentile(ensemble_scores, percentile)

    for label in unique_labels:
        mask = labels == label
        cluster_scores = ensemble_scores[mask]
        if len(cluster_scores) >= 5:
            thresholds[label] = np.percentile(cluster_scores, percentile)
        else:
            # 소수 클러스터는 전역 임계값 사용
            thresholds[label] = global_threshold

    return thresholds


# ──────────────────────────────────────────────
# UMAP 시각화
# ──────────────────────────────────────────────

def create_umap_plot(vectors: np.ndarray, labels: np.ndarray,
                     route_ids: List[str],
                     anomaly_vectors: np.ndarray = None,
                     anomaly_labels_type: np.ndarray = None,
                     detected_mask: np.ndarray = None,
                     output_path: Path = UMAP_PNG):
    """UMAP 2D scatter + HDBSCAN 클러스터 + 이상 마커"""
    print("=" * 70)
    print("[UMAP] 2D 시각화 생성")
    print("=" * 70)

    # 전체 벡터 합치기 (정상 + 합성이상)
    if anomaly_vectors is not None and len(anomaly_vectors) > 0:
        all_vectors = np.vstack([vectors, anomaly_vectors])
    else:
        all_vectors = vectors

    # UMAP 차원 축소
    n_neighbors = min(15, len(all_vectors) - 1)
    if n_neighbors < 2:
        print("  [경고] 데이터 부족으로 UMAP 생략")
        return

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED
    )
    embedding_2d = reducer.fit_transform(all_vectors)
    print(f"  UMAP 2D 완료: {all_vectors.shape} -> {embedding_2d.shape}")

    # 벡터 + 2D 좌표 저장 (3D UMAP 등 후속 분석용)
    np.save(RESULTS_DIR / "mission_vectors.npy", all_vectors)
    np.save(RESULTS_DIR / "umap_2d.npy", embedding_2d)

    # 3D UMAP
    reducer_3d = umap.UMAP(
        n_components=3, n_neighbors=n_neighbors,
        min_dist=0.1, metric="cosine", random_state=RANDOM_SEED
    )
    embedding_3d = reducer_3d.fit_transform(all_vectors)
    np.save(RESULTS_DIR / "umap_3d.npy", embedding_3d)
    print(f"  UMAP 3D 완료: {all_vectors.shape} -> {embedding_3d.shape}")

    # 3D 인터랙티브 HTML (plotly)
    try:
        import plotly.graph_objects as go
        fig3d = go.Figure()
        unique_labels = sorted(set(labels))
        colors = ["lightgray" if l == -1 else f"hsl({(l * 36) % 360}, 70%, 50%)" for l in unique_labels]
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            name = "Noise" if label == -1 else f"Cluster {label}"
            fig3d.add_trace(go.Scatter3d(
                x=embedding_3d[:n_normal][mask, 0],
                y=embedding_3d[:n_normal][mask, 1],
                z=embedding_3d[:n_normal][mask, 2],
                mode="markers", marker=dict(size=3, opacity=0.6),
                name=f"{name} (n={mask.sum()})"
            ))
        if anomaly_2d is not None and len(embedding_3d) > n_normal:
            anom_3d = embedding_3d[n_normal:]
            fig3d.add_trace(go.Scatter3d(
                x=anom_3d[:, 0], y=anom_3d[:, 1], z=anom_3d[:, 2],
                mode="markers", marker=dict(size=6, symbol="diamond", color="red", opacity=0.9),
                name=f"Synthetic Anomaly (n={len(anom_3d)})"
            ))
        fig3d.update_layout(
            title="Mission2Vec: 3D UMAP + HDBSCAN",
            scene=dict(
                xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"
            ),
            legend=dict(x=1.02, y=0.5, font=dict(size=9)),
            margin=dict(l=0, r=200, t=40, b=0)
        )
        fig3d.write_html(str(RESULTS_DIR / "umap_3d.html"))
        print(f"  3D HTML 저장: {RESULTS_DIR / 'umap_3d.html'}")
    except Exception as e:
        print(f"  3D HTML 생성 실패 (non-critical): {e}")

    n_normal = len(vectors)
    normal_2d = embedding_2d[:n_normal]
    anomaly_2d = embedding_2d[n_normal:] if len(all_vectors) > n_normal else None

    # 플롯 생성
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # 색상 맵 (HDBSCAN 클러스터)
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])

    # 노이즈(-1)는 회색, 나머지는 컬러맵
    cmap = cm.get_cmap("tab10", max(n_clusters, 1))

    for label in unique_labels:
        mask = labels == label
        if label == -1:
            color = "lightgray"
            marker = "x"
            alpha = 0.4
            lbl = f"Noise (n={mask.sum()})"
            zorder = 1
        else:
            color = cmap(label % 10)
            marker = "o"
            alpha = 0.6
            lbl = f"Cluster {label} (n={mask.sum()})"
            zorder = 2

        ax.scatter(normal_2d[mask, 0], normal_2d[mask, 1],
                   c=[color], marker=marker, alpha=alpha,
                   s=30, label=lbl, zorder=zorder, edgecolors="none")

    # 합성 이상 마커
    if anomaly_2d is not None and len(anomaly_2d) > 0:
        n_anom = len(anomaly_2d)
        if anomaly_labels_type is not None and len(anomaly_labels_type) == n_anom:
            # PERM vs NOISE 구분
            for i in range(n_anom):
                if anomaly_labels_type[i] == "PERM":
                    ax.scatter(anomaly_2d[i, 0], anomaly_2d[i, 1],
                               c="red", marker="^", s=120, zorder=10,
                               edgecolors="black", linewidths=0.8,
                               label="Synthetic Anomaly (Permuted)" if i == 0 else "")
                else:
                    ax.scatter(anomaly_2d[i, 0], anomaly_2d[i, 1],
                               c="orange", marker="s", s=120, zorder=10,
                               edgecolors="black", linewidths=0.8,
                               label="Synthetic Anomaly (Noise)" if anomaly_labels_type[i - 1:i] != ["NOISE"] else "")
        else:
            ax.scatter(anomaly_2d[:, 0], anomaly_2d[:, 1],
                       c="red", marker="^", s=120, zorder=10,
                       edgecolors="black", linewidths=0.8,
                       label=f"Synthetic Anomaly (n={n_anom})")

    # 탐지된 정상 경로 강조 (detected_mask)
    if detected_mask is not None:
        det_2d = normal_2d[detected_mask]
        if len(det_2d) > 0:
            ax.scatter(det_2d[:, 0], det_2d[:, 1],
                       facecolors="none", edgecolors="red", marker="o",
                       s=100, linewidths=1.5, zorder=9,
                       label=f"Detected Anomaly (n={len(det_2d)})")

    ax.set_title(f"Mission2Vec: UAV Mission Path Embedding (UMAP + HDBSCAN)\n"
                 f"{n_normal} routes, {n_clusters} clusters",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1 (structural similarity)", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2 (structural similarity)", fontsize=11)

    # 범례 정리 (중복 제거) — 그래프 밖 우측 하단
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc="lower left", bbox_to_anchor=(1.02, 0),
              fontsize=7, markerscale=1.2, borderaxespad=0,
              framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {output_path}")
    print()


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────

def main():
    print()
    print("=" * 70)
    print("  AERION - 전체 데이터셋 앙상블 이상 탐지 + HDBSCAN + UMAP")
    print("=" * 70)
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    if not DATA_PATH.exists():
        print(f"[오류] 데이터 파일 없음: {DATA_PATH}")
        sys.exit(1)

    routes = load_and_filter(DATA_PATH)
    if len(routes) < 30:
        print(f"[오류] 필터 후 경로 수 부족 ({len(routes)}건)")
        sys.exit(1)

    # 2-3. 전체 임베딩
    print(f"  [전체 {len(routes)}건 경로 처리 (제한 없음)]")
    print()
    route_ids, vectors = embed_all_routes(routes, max_routes=MAX_ROUTES)

    if len(route_ids) < 30:
        print(f"[오류] 임베딩 성공 경로 부족 ({len(route_ids)}건)")
        sys.exit(1)

    # 4. 학습/테스트 분할 + 앙상블
    print("=" * 70)
    print("[4] 3-Method 앙상블 이상 탐지")
    print("=" * 70)

    n_total = len(route_ids)
    n_train = int(n_total * 0.8)
    perm = np.random.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_ids = [route_ids[i] for i in train_idx]
    test_ids = [route_ids[i] for i in test_idx]
    train_vectors = vectors[train_idx]
    test_vectors = vectors[test_idx]
    print(f"  학습: {len(train_ids)}건 / 테스트: {len(test_ids)}건")
    print()

    ensemble = AnomalyEnsemble()
    ensemble.fit(train_vectors)

    # 전체 데이터 점수 (학습+테스트 모두)
    all_scores = ensemble.score(vectors)
    test_scores = ensemble.score(test_vectors)

    # 5. 합성 이상
    print("=" * 70)
    print("[5] 합성 이상 주입 + 탐지 검증")
    print("=" * 70)

    print(f"  순서 치환 이상 {N_PERMUTED}건 생성 중...")
    perm_ids, perm_vectors = inject_permuted_anomalies(routes, train_ids, N_PERMUTED)
    print(f"    -> {len(perm_ids)}건 생성 완료")

    print(f"  좌표 노이즈 이상 {N_NOISY}건 생성 중...")
    noise_ids, noise_vectors = inject_noisy_anomalies(routes, train_ids, N_NOISY)
    print(f"    -> {len(noise_ids)}건 생성 완료")
    print()

    perm_scores = ensemble.score(perm_vectors) if len(perm_vectors) > 0 else None
    noise_scores = ensemble.score(noise_vectors) if len(noise_vectors) > 0 else None

    # HDBSCAN 클러스터링 (전체 데이터)
    hdbscan_labels, clusterer = run_hdbscan(vectors)

    # 클러스터별 임계값
    per_cluster_thresholds = compute_per_cluster_thresholds(
        hdbscan_labels, all_scores["ensemble"])
    global_threshold = np.percentile(test_scores["ensemble"], 95)

    print(f"  전역 임계값 (테스트 95%ile): {global_threshold:.4f}")
    print(f"  클러스터별 임계값:")
    for label in sorted(per_cluster_thresholds.keys()):
        name = "노이즈(-1)" if label == -1 else f"클러스터 {label}"
        print(f"    {name}: {per_cluster_thresholds[label]:.4f}")
    print()

    # 클러스터별 이상 판정
    per_cluster_detected = np.zeros(len(vectors), dtype=bool)
    for i in range(len(vectors)):
        cl = hdbscan_labels[i]
        threshold_cl = per_cluster_thresholds.get(cl, global_threshold)
        if all_scores["ensemble"][i] > threshold_cl:
            per_cluster_detected[i] = True

    # UMAP 시각화
    # 합성 이상 벡터 + 타입 라벨 합치기
    synth_vectors_list = []
    synth_type_list = []
    if len(perm_vectors) > 0:
        synth_vectors_list.append(perm_vectors)
        synth_type_list.extend(["PERM"] * len(perm_vectors))
    if len(noise_vectors) > 0:
        synth_vectors_list.append(noise_vectors)
        synth_type_list.extend(["NOISE"] * len(noise_vectors))

    synth_vectors = np.vstack(synth_vectors_list) if synth_vectors_list else None
    synth_types = np.array(synth_type_list) if synth_type_list else None

    create_umap_plot(
        vectors, hdbscan_labels, route_ids,
        anomaly_vectors=synth_vectors,
        anomaly_labels_type=synth_types,
        detected_mask=per_cluster_detected,
        output_path=UMAP_PNG
    )

    # 6. 결과 출력
    print("=" * 70)
    print("[6] 결과 분석")
    print("=" * 70)
    print()

    # 점수 통계
    normal_mean = test_scores["ensemble"].mean()
    perm_mean = perm_scores["ensemble"].mean() if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0 else 0
    noise_mean = noise_scores["ensemble"].mean() if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0 else 0

    print(f"  {'구분':<30} {'평균':>8} {'표준편차':>8} {'최소':>8} {'최대':>8}")
    print(f"  {'-' * 60}")
    for name, scores in [("정상 테스트", test_scores),
                          ("합성(순서치환)", perm_scores),
                          ("합성(노이즈)", noise_scores)]:
        if scores is None:
            continue
        e = scores["ensemble"]
        print(f"  {name:<30} {e.mean():>8.4f} {e.std():>8.4f} {e.min():>8.4f} {e.max():>8.4f}")
    print()

    # 탐지율
    print(f"  {'='*60}")
    print(f"  탐지율 (전역 임계값 {global_threshold:.4f})")
    print(f"  {'='*60}")

    normal_flagged = (test_scores["ensemble"] > global_threshold).sum()
    print(f"  정상 오탐률 (FPR): {normal_flagged}/{len(test_ids)} "
          f"= {normal_flagged / max(len(test_ids), 1) * 100:.1f}%")

    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        perm_detected = (perm_scores["ensemble"] > global_threshold).sum()
        print(f"  순서 치환 탐지율: {perm_detected}/{len(perm_ids)} "
              f"= {perm_detected / max(len(perm_ids), 1) * 100:.1f}%")

    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        noise_detected = (noise_scores["ensemble"] > global_threshold).sum()
        print(f"  좌표 노이즈 탐지율: {noise_detected}/{len(noise_ids)} "
              f"= {noise_detected / max(len(noise_ids), 1) * 100:.1f}%")
    print()

    # 클러스터별 탐지율
    print(f"  {'='*60}")
    print(f"  클러스터별 이상 탐지 현황")
    print(f"  {'='*60}")
    print(f"  {'클러스터':<15} {'경로수':>8} {'탐지수':>8} {'탐지율':>8} {'임계값':>8}")
    print(f"  {'-' * 52}")
    for label in sorted(set(hdbscan_labels)):
        mask = hdbscan_labels == label
        n_in = mask.sum()
        n_det = per_cluster_detected[mask].sum()
        name = "노이즈(-1)" if label == -1 else f"클러스터 {label}"
        print(f"  {name:<15} {n_in:>8} {n_det:>8} {n_det/max(n_in,1)*100:>7.1f}% "
              f"{per_cluster_thresholds.get(label, 0):>8.4f}")
    print()

    # CSV 저장 - anomaly_results_full.csv
    results = []
    for i, rid in enumerate(route_ids):
        results.append({
            "route_id": rid,
            "type": "normal",
            "hdbscan_cluster": int(hdbscan_labels[i]),
            "centroid_score": all_scores["centroid"][i],
            "isoforest_score": all_scores["isoforest"][i],
            "knn_score": all_scores["knn"][i],
            "ensemble_score": all_scores["ensemble"][i],
            "interpretation": interpret_score(all_scores["ensemble"][i]),
            "detected_global": all_scores["ensemble"][i] > global_threshold,
            "detected_per_cluster": per_cluster_detected[i],
            "cluster_threshold": per_cluster_thresholds.get(int(hdbscan_labels[i]), global_threshold),
        })

    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        for i, rid in enumerate(perm_ids):
            results.append({
                "route_id": rid,
                "type": "anomaly_permuted",
                "hdbscan_cluster": -99,
                "centroid_score": perm_scores["centroid"][i],
                "isoforest_score": perm_scores["isoforest"][i],
                "knn_score": perm_scores["knn"][i],
                "ensemble_score": perm_scores["ensemble"][i],
                "interpretation": interpret_score(perm_scores["ensemble"][i]),
                "detected_global": perm_scores["ensemble"][i] > global_threshold,
                "detected_per_cluster": True,
                "cluster_threshold": global_threshold,
            })

    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        for i, rid in enumerate(noise_ids):
            results.append({
                "route_id": rid,
                "type": "anomaly_noisy",
                "hdbscan_cluster": -99,
                "centroid_score": noise_scores["centroid"][i],
                "isoforest_score": noise_scores["isoforest"][i],
                "knn_score": noise_scores["knn"][i],
                "ensemble_score": noise_scores["ensemble"][i],
                "interpretation": interpret_score(noise_scores["ensemble"][i]),
                "detected_global": noise_scores["ensemble"][i] > global_threshold,
                "detected_per_cluster": True,
                "cluster_threshold": global_threshold,
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  결과 CSV 저장: {OUTPUT_CSV}")
    print(f"    총 {len(result_df)}건 (정상 {len(route_ids)} + 합성이상 {len(perm_ids)+len(noise_ids)})")

    # CSV 저장 - cluster_summary.csv
    cluster_rows = []
    for label in sorted(set(hdbscan_labels)):
        mask = hdbscan_labels == label
        cl_scores = all_scores["ensemble"][mask]
        cluster_rows.append({
            "cluster": int(label),
            "count": int(mask.sum()),
            "pct": mask.sum() / len(hdbscan_labels) * 100,
            "score_mean": cl_scores.mean(),
            "score_std": cl_scores.std(),
            "score_min": cl_scores.min(),
            "score_max": cl_scores.max(),
            "score_p95": np.percentile(cl_scores, 95) if len(cl_scores) >= 5 else cl_scores.max(),
            "threshold": per_cluster_thresholds.get(label, global_threshold),
            "n_detected": int(per_cluster_detected[mask].sum()),
        })

    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(CLUSTER_CSV, index=False, encoding="utf-8-sig")
    print(f"  클러스터 요약 CSV 저장: {CLUSTER_CSV}")
    print()

    # 최종 한국어 리포트
    print()
    print("=" * 70)
    print("  [최종 리포트]")
    print("=" * 70)
    print()
    print(f"  1. 입력 데이터")
    print(f"     - 필터 후 전체 경로: {len(routes):,}건")
    print(f"     - 임베딩 성공: {len(route_ids):,}건 ({EMBEDDING_DIM}차원)")
    print(f"     - 학습/테스트 분할: {n_train}/{n_total - n_train}")
    print()
    print(f"  2. HDBSCAN 클러스터링")
    n_clusters = len([l for l in set(hdbscan_labels) if l >= 0])
    n_noise_pts = (hdbscan_labels == -1).sum()
    print(f"     - 클러스터 수: {n_clusters}개")
    print(f"     - 노이즈 포인트: {n_noise_pts}건 ({n_noise_pts/len(hdbscan_labels)*100:.1f}%)")
    print()
    print(f"  3. 이상 탐지 성능")
    print(f"     - 전역 임계값 (95%ile): {global_threshold:.4f}")
    print(f"     - 정상 평균 앙상블 점수: {normal_mean:.4f} ({interpret_score(normal_mean)})")
    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        perm_det = (perm_scores["ensemble"] > global_threshold).sum()
        print(f"     - 순서 치환 이상 평균: {perm_mean:.4f} / 탐지율: {perm_det}/{len(perm_ids)} "
              f"({perm_det/max(len(perm_ids),1)*100:.0f}%)")
    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        noise_det = (noise_scores["ensemble"] > global_threshold).sum()
        print(f"     - 좌표 노이즈 이상 평균: {noise_mean:.4f} / 탐지율: {noise_det}/{len(noise_ids)} "
              f"({noise_det/max(len(noise_ids),1)*100:.0f}%)")
    print()
    print(f"  4. 출력 파일")
    print(f"     - {OUTPUT_CSV}")
    print(f"     - {CLUSTER_CSV}")
    print(f"     - {UMAP_PNG}")
    print()
    print("=" * 70)
    print("  분석 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
