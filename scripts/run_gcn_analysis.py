#!/opt/anaconda3/bin/python
"""
AERION - GCN 기반 미션 경로 임베딩 + 3-Method 앙상블 이상 탐지 + HDBSCAN + UMAP

기반: run_full_analysis.py (Node2Vec 버전)
변경사항:
  1. Node2Vec → GCN (Graph Convolutional Network) 기반 임베딩
  2. 노드 피처: [lat_norm, lon_norm, alt_norm, seq_ratio] = 4차원 (한국 전역 정규화)
  3. GCN Encoder: 2-layer (4d → 64d → 128d), message passing으로 이웃 정보 집계
  4. 비학습(untrained) GCN 우선 사용 → 학습 없이도 구조 정보 반영
  5. 미션 벡터: 노드 임베딩 가중 평균 (시작/끝 가중치 2x)
  6. 나머지 파이프라인 동일 (3-Method 앙상블, HDBSCAN, UMAP)

실행: python3 run_gcn_analysis.py

필요 패키지: torch, torch_geometric, numpy, pandas, scikit-learn, umap-learn, hdbscan, matplotlib
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import math
import warnings
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

# PyTorch + PyG
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

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

# GCN 파라미터
GCN_HIDDEN_DIM = 64
GCN_OUT_DIM = 128         # 최종 임베딩 차원 (= EMBEDDING_DIM)
EMBEDDING_DIM = GCN_OUT_DIM
GCN_NODE_FEATURES = 4     # [lat_norm, lon_norm, alt_norm, seq_ratio]

# GCN 학습 파라미터 (autoencoder 모드 시 사용)
GCN_TRAIN_EPOCHS = 0      # 0 = 비학습(untrained) 모드 / >0 = autoencoder 학습 모드
GCN_LEARNING_RATE = 0.01

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
MAX_ROUTES = 2000

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


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
# 2단계: 그래프 구성 (Node2Vec 버전과 동일)
# ──────────────────────────────────────────────

def build_graph(route_df: pd.DataFrame) -> nx.DiGraph:
    """경로 데이터 → NetworkX 방향 그래프 (전역 정규화 + 고도 인식 엣지)"""
    G = nx.DiGraph()
    n = len(route_df)
    if n == 0:
        return G

    lats = route_df["wp_lat"].values
    lngs = route_df["wp_lng"].values
    alts = route_df["wp_alt"].values

    # 전역 정규화 (한국 중심) — 절대 위치 보존
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
    ALT_SENSITIVITY = 0.5
    for i in range(n - 1):
        dist = haversine(lats[i], lngs[i], lats[i + 1], lngs[i + 1])
        alt_delta = abs(alts[i + 1] - alts[i])
        effective_dist = (dist**2 + (alt_delta * ALT_SENSITIVITY)**2) ** 0.5
        weight = 1.0 / (1.0 + effective_dist / 1000.0)
        G.add_edge(f"WP_{i}", f"WP_{i + 1}",
                   edge_type="sequential", weight=weight,
                   distance_m=dist, alt_change_m=alt_delta)

    # Spatial KNN 엣지
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
# 3단계: GCN 기반 임베딩
# ──────────────────────────────────────────────

class GCNEncoder(torch.nn.Module):
    """2-layer GCN Encoder: 노드 피처 4d → 64d → 128d"""
    def __init__(self, in_channels: int = GCN_NODE_FEATURES,
                 hidden: int = GCN_HIDDEN_DIM,
                 out_channels: int = GCN_OUT_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1층: GCN + ReLU
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # 2층: GCN (활성화 없음 — 임베딩 공간)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def nx_to_pyg(G: nx.DiGraph) -> Optional[Data]:
    """NetworkX 그래프 → PyG Data 변환 (노드 피처 + 엣지 가중치 포함)"""
    nodes = sorted(G.nodes(), key=lambda x: int(x.split("_")[1]))
    if len(nodes) < 2:
        return None

    # 노드 피처 행렬: [lat_norm, lon_norm, alt_norm, seq_ratio]
    node_features = []
    for node in nodes:
        attrs = G.nodes[node]
        node_features.append([
            attrs["lat_norm"],
            attrs["lon_norm"],
            attrs["alt_norm"],
            attrs["seq_ratio"],
        ])
    x = torch.tensor(node_features, dtype=torch.float32)

    # 노드 ID → 인덱스 매핑
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # 엣지 인덱스 + 가중치 (방향 그래프 → 무방향으로 처리: GCN은 무방향 선호)
    edge_src, edge_dst, edge_weights = [], [], []
    seen_edges = set()
    for u, v, data in G.edges(data=True):
        if u not in node_to_idx or v not in node_to_idx:
            continue
        i, j = node_to_idx[u], node_to_idx[v]
        w = data.get("weight", 1.0)
        # 양방향 추가 (GCN 대칭 정규화를 위해)
        if (i, j) not in seen_edges:
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            edge_weights.extend([w, w])
            seen_edges.add((i, j))
            seen_edges.add((j, i))

    if not edge_src:
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                num_nodes=len(nodes))


# 공유 GCN 인코더 (비학습 모드: 랜덤 초기화된 가중치로 message passing)
# GCN은 학습 없이도 이웃 노드의 피처를 집계(aggregate)하므로
# 구조 정보가 임베딩에 반영됨
_shared_encoder: Optional[GCNEncoder] = None


def get_shared_encoder() -> GCNEncoder:
    """공유 GCN 인코더 반환 (한 번만 초기화)"""
    global _shared_encoder
    if _shared_encoder is None:
        _shared_encoder = GCNEncoder(
            in_channels=GCN_NODE_FEATURES,
            hidden=GCN_HIDDEN_DIM,
            out_channels=GCN_OUT_DIM,
        )
        _shared_encoder.eval()  # 비학습 모드: dropout 비활성화 등
        print(f"  GCN Encoder 초기화: {GCN_NODE_FEATURES}d → {GCN_HIDDEN_DIM}d → {GCN_OUT_DIM}d")
        total_params = sum(p.numel() for p in _shared_encoder.parameters())
        print(f"  파라미터 수: {total_params:,}")
    return _shared_encoder


def train_gcn_autoencoder(encoder: GCNEncoder, data_list: List[Data],
                          epochs: int = GCN_TRAIN_EPOCHS,
                          lr: float = GCN_LEARNING_RATE) -> float:
    """
    Graph Autoencoder 학습: 인접 행렬 재구성 (inner product decoder)
    - 엣지가 있는 노드 쌍: positive sample
    - 랜덤 노드 쌍: negative sample
    - Binary Cross Entropy loss
    """
    if epochs <= 0:
        return 0.0

    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    print(f"  [GCN Autoencoder 학습] {epochs} epochs, lr={lr}")
    total_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_graphs = 0

        for data in data_list:
            if data is None or data.num_nodes < 2:
                continue

            optimizer.zero_grad()

            # Forward: 노드 임베딩 계산
            z = encoder(data.x, data.edge_index, data.edge_weight)

            # Positive edges (실제 엣지)
            pos_edge_index = data.edge_index
            n_pos = pos_edge_index.shape[1]

            # Negative edges (랜덤 샘플링)
            n_neg = min(n_pos, data.num_nodes * (data.num_nodes - 1) - n_pos)
            if n_neg <= 0:
                continue

            neg_src = torch.randint(0, data.num_nodes, (n_neg,))
            neg_dst = torch.randint(0, data.num_nodes, (n_neg,))

            # Inner product decoder
            # Positive: z[src] . z[dst] → sigmoid → 1에 가까워야 함
            pos_pred = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_pred, torch.ones_like(pos_pred))

            # Negative: z[src] . z[dst] → sigmoid → 0에 가까워야 함
            neg_pred = (z[neg_src] * z[neg_dst]).sum(dim=1)
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_pred, torch.zeros_like(neg_pred))

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_graphs += 1

        avg_loss = epoch_loss / max(n_graphs, 1)
        total_loss = avg_loss

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    encoder.eval()
    return total_loss


def embed_mission_gcn(G: nx.DiGraph, encoder: GCNEncoder) -> Optional[np.ndarray]:
    """단일 경로 그래프 → GCN 임베딩 → 가중 평균 → 미션 벡터 (128d)"""
    data = nx_to_pyg(G)
    if data is None:
        return None

    with torch.no_grad():
        # GCN forward: 노드 임베딩 계산
        node_embeddings = encoder(data.x, data.edge_index, data.edge_weight)
        # shape: [num_nodes, GCN_OUT_DIM]

    embeddings = node_embeddings.numpy()
    n = len(embeddings)

    # 가중 평균 풀링 (시작/끝 노드 가중치 2배)
    weights = np.ones(n)
    weights[0] = 2.0
    weights[-1] = 2.0
    weights /= weights.sum()

    mission_vector = np.average(embeddings, axis=0, weights=weights)

    # L2 정규화
    norm = np.linalg.norm(mission_vector)
    if norm > 1e-9:
        mission_vector /= norm

    return mission_vector


def embed_all_routes_gcn(routes: Dict[str, pd.DataFrame],
                         max_routes: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
    """전체 경로 → GCN 임베딩"""
    print("=" * 70)
    print("[2-3] 그래프 구성 + GCN 임베딩 (전체 데이터셋)")
    print("=" * 70)

    encoder = get_shared_encoder()

    # 그래프 구성
    items = list(routes.items())
    if max_routes and max_routes < len(items):
        items = items[:max_routes]

    total = len(items)
    print(f"  총 {total}건 경로 처리 중...")

    # 1단계: 모든 그래프 + PyG Data 생성
    graphs = []
    pyg_data_list = []
    route_id_list = []

    t_start = time.time()
    for idx, (route_id, route_df) in enumerate(items):
        G = build_graph(route_df)
        data = nx_to_pyg(G)
        if data is not None:
            graphs.append(G)
            pyg_data_list.append(data)
            route_id_list.append(route_id)

    print(f"  그래프 구성 완료: {len(pyg_data_list)}건 ({time.time() - t_start:.1f}s)")

    # 2단계: (선택적) GCN Autoencoder 학습
    if GCN_TRAIN_EPOCHS > 0:
        print()
        final_loss = train_gcn_autoencoder(encoder, pyg_data_list,
                                           epochs=GCN_TRAIN_EPOCHS)
        print(f"  학습 완료: 최종 loss={final_loss:.4f}")
        print()
    else:
        print(f"  비학습(untrained) 모드: 랜덤 초기화 GCN으로 message passing 수행")
        print(f"  (GCN은 학습 없이도 이웃 노드 피처를 집계하여 구조 정보 반영)")
        print()

    # 3단계: 미션 벡터 추출
    route_ids = []
    vectors = []

    t_embed = time.time()
    report_interval = max(1, len(graphs) // 10)

    for idx, (G, route_id) in enumerate(zip(graphs, route_id_list)):
        if (idx + 1) % report_interval == 0 or idx == 0 or idx == len(graphs) - 1:
            elapsed = time.time() - t_embed
            eta = (elapsed / max(idx, 1)) * (len(graphs) - idx) if idx > 0 else 0
            print(f"  [{idx + 1}/{len(graphs)}] {route_id} (WP={G.number_of_nodes()}) "
                  f"... {elapsed:.1f}s (ETA: {eta:.0f}s)")

        vec = embed_mission_gcn(G, encoder)
        if vec is not None:
            route_ids.append(route_id)
            vectors.append(vec)

    vectors = np.array(vectors) if vectors else np.empty((0, EMBEDDING_DIM))
    total_time = time.time() - t_start
    print(f"  임베딩 완료: {len(route_ids)} 경로 -> {vectors.shape} 행렬")
    print(f"  총 소요 시간: {total_time:.1f}s")
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

def inject_permuted_anomalies(routes, route_ids, encoder, n=N_PERMUTED):
    """순서 치환 이상 생성 (GCN 기반)"""
    selected = np.random.choice(route_ids, size=min(n, len(route_ids)), replace=False)
    anomaly_ids, anomaly_vectors = [], []
    for rid in selected:
        df = routes[rid].copy()
        perm_idx = np.random.permutation(len(df))
        df_perm = df.iloc[perm_idx].reset_index(drop=True)
        df_perm["wp_seq"] = range(1, len(df_perm) + 1)
        G = build_graph(df_perm)
        vec = embed_mission_gcn(G, encoder)
        if vec is not None:
            anomaly_ids.append(f"PERM_{rid}")
            anomaly_vectors.append(vec)
    return anomaly_ids, np.array(anomaly_vectors) if anomaly_vectors else np.empty((0, EMBEDDING_DIM))


def inject_noisy_anomalies(routes, route_ids, encoder, n=N_NOISY, noise_std=0.05):
    """좌표 노이즈 이상 생성 (GCN 기반)"""
    selected = np.random.choice(route_ids, size=min(n, len(route_ids)), replace=False)
    anomaly_ids, anomaly_vectors = [], []
    for rid in selected:
        df = routes[rid].copy()
        df["wp_lat"] = df["wp_lat"] + np.random.normal(0, noise_std, len(df))
        df["wp_lng"] = df["wp_lng"] + np.random.normal(0, noise_std, len(df))
        df["wp_alt"] = np.clip(df["wp_alt"] + np.random.normal(0, 50, len(df)), ALT_MIN, ALT_MAX)
        G = build_graph(df)
        vec = embed_mission_gcn(G, encoder)
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
    print("[UMAP] 2D/3D 시각화 생성")
    print("=" * 70)

    # 전체 벡터 합치기 (정상 + 합성이상)
    if anomaly_vectors is not None and len(anomaly_vectors) > 0:
        all_vectors = np.vstack([vectors, anomaly_vectors])
    else:
        all_vectors = vectors

    n_neighbors = min(15, len(all_vectors) - 1)
    if n_neighbors < 2:
        print("  [경고] 데이터 부족으로 UMAP 생략")
        return

    # UMAP 2D
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED
    )
    embedding_2d = reducer.fit_transform(all_vectors)
    print(f"  UMAP 2D 완료: {all_vectors.shape} -> {embedding_2d.shape}")

    # 벡터 저장
    np.save(RESULTS_DIR / "mission_vectors.npy", all_vectors)
    np.save(RESULTS_DIR / "umap_2d.npy", embedding_2d)

    # UMAP 3D
    reducer_3d = umap.UMAP(
        n_components=3, n_neighbors=n_neighbors,
        min_dist=0.1, metric="cosine", random_state=RANDOM_SEED
    )
    embedding_3d = reducer_3d.fit_transform(all_vectors)
    np.save(RESULTS_DIR / "umap_3d.npy", embedding_3d)
    print(f"  UMAP 3D 완료: {all_vectors.shape} -> {embedding_3d.shape}")

    n_normal = len(vectors)

    # 3D 인터랙티브 HTML (plotly)
    try:
        import plotly.graph_objects as go
        fig3d = go.Figure()
        unique_labels_sorted = sorted(set(labels))
        for label in unique_labels_sorted:
            mask = labels == label
            name = "Noise" if label == -1 else f"Cluster {label}"
            fig3d.add_trace(go.Scatter3d(
                x=embedding_3d[:n_normal][mask, 0],
                y=embedding_3d[:n_normal][mask, 1],
                z=embedding_3d[:n_normal][mask, 2],
                mode="markers", marker=dict(size=3, opacity=0.6),
                name=f"{name} (n={mask.sum()})"
            ))
        if len(all_vectors) > n_normal:
            anom_3d = embedding_3d[n_normal:]
            fig3d.add_trace(go.Scatter3d(
                x=anom_3d[:, 0], y=anom_3d[:, 1], z=anom_3d[:, 2],
                mode="markers", marker=dict(size=6, symbol="diamond", color="red", opacity=0.9),
                name=f"Synthetic Anomaly (n={len(anom_3d)})"
            ))
        fig3d.update_layout(
            title="Mission2Vec (GCN): 3D UMAP + HDBSCAN",
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

    normal_2d = embedding_2d[:n_normal]
    anomaly_2d = embedding_2d[n_normal:] if len(all_vectors) > n_normal else None

    # 플롯 생성
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])
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
        perm_plotted = False
        noise_plotted = False
        if anomaly_labels_type is not None and len(anomaly_labels_type) == n_anom:
            for i in range(n_anom):
                if anomaly_labels_type[i] == "PERM":
                    label_str = "Synthetic Anomaly (Permuted)" if not perm_plotted else ""
                    ax.scatter(anomaly_2d[i, 0], anomaly_2d[i, 1],
                               c="red", marker="^", s=120, zorder=10,
                               edgecolors="black", linewidths=0.8, label=label_str)
                    perm_plotted = True
                else:
                    label_str = "Synthetic Anomaly (Noise)" if not noise_plotted else ""
                    ax.scatter(anomaly_2d[i, 0], anomaly_2d[i, 1],
                               c="orange", marker="s", s=120, zorder=10,
                               edgecolors="black", linewidths=0.8, label=label_str)
                    noise_plotted = True
        else:
            ax.scatter(anomaly_2d[:, 0], anomaly_2d[:, 1],
                       c="red", marker="^", s=120, zorder=10,
                       edgecolors="black", linewidths=0.8,
                       label=f"Synthetic Anomaly (n={n_anom})")

    # 탐지된 정상 경로 강조
    if detected_mask is not None:
        det_2d = normal_2d[detected_mask]
        if len(det_2d) > 0:
            ax.scatter(det_2d[:, 0], det_2d[:, 1],
                       facecolors="none", edgecolors="red", marker="o",
                       s=100, linewidths=1.5, zorder=9,
                       label=f"Detected Anomaly (n={len(det_2d)})")

    ax.set_title(f"Mission2Vec (GCN): UAV Mission Path Embedding (UMAP + HDBSCAN)\n"
                 f"{n_normal} routes, {n_clusters} clusters",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP Dimension 1 (structural similarity)", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2 (structural similarity)", fontsize=11)

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
# Analysis Summary PNG 생성
# ──────────────────────────────────────────────

def _create_analysis_summary(
    n_routes, n_clusters, n_noise, hdbscan_labels,
    all_scores, test_scores, perm_scores, noise_scores,
    perm_ids, noise_ids, global_threshold,
    per_cluster_thresholds, per_cluster_detected,
    output_path,
):
    """4-panel analysis summary PNG: score distribution, cluster sizes, detection rates, box plot"""
    print("=" * 70)
    print("[Summary] analysis_summary.png 생성")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("GCN Mission2Vec - Analysis Summary", fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Ensemble score distribution
    ax1 = axes[0, 0]
    ax1.hist(all_scores["ensemble"], bins=50, color="steelblue", alpha=0.7, label="Normal routes")
    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        ax1.hist(perm_scores["ensemble"], bins=15, color="red", alpha=0.7, label="Permuted anomaly")
    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        ax1.hist(noise_scores["ensemble"], bins=15, color="orange", alpha=0.7, label="Noisy anomaly")
    ax1.axvline(global_threshold, color="darkred", linestyle="--", linewidth=1.5,
                label=f"Threshold ({global_threshold:.3f})")
    ax1.set_title("Ensemble Score Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Ensemble Score")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)

    # Panel 2: Cluster size bar chart
    ax2 = axes[0, 1]
    unique_labels = sorted(set(hdbscan_labels))
    cluster_names = ["Noise" if l == -1 else f"C{l}" for l in unique_labels]
    cluster_sizes = [(hdbscan_labels == l).sum() for l in unique_labels]
    colors = ["lightgray" if l == -1 else cm.get_cmap("tab10")(l % 10) for l in unique_labels]
    ax2.bar(cluster_names, cluster_sizes, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_title("HDBSCAN Cluster Sizes", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Route Count")
    for i, (name, size) in enumerate(zip(cluster_names, cluster_sizes)):
        ax2.text(i, size + max(cluster_sizes) * 0.01, str(size), ha="center", fontsize=8)

    # Panel 3: Detection rate summary
    ax3 = axes[1, 0]
    categories = []
    rates = []
    bar_colors = []

    # Normal FPR
    normal_flagged = (test_scores["ensemble"] > global_threshold).sum()
    n_test = len(test_scores["ensemble"])
    fpr = normal_flagged / max(n_test, 1) * 100
    categories.append(f"Normal FPR\n({normal_flagged}/{n_test})")
    rates.append(fpr)
    bar_colors.append("steelblue")

    # Permuted detection
    if perm_scores is not None and len(perm_scores.get("ensemble", [])) > 0:
        perm_det = (perm_scores["ensemble"] > global_threshold).sum()
        perm_rate = perm_det / max(len(perm_ids), 1) * 100
        categories.append(f"Permuted DR\n({perm_det}/{len(perm_ids)})")
        rates.append(perm_rate)
        bar_colors.append("red")

    # Noise detection
    if noise_scores is not None and len(noise_scores.get("ensemble", [])) > 0:
        noise_det = (noise_scores["ensemble"] > global_threshold).sum()
        noise_rate = noise_det / max(len(noise_ids), 1) * 100
        categories.append(f"Noisy DR\n({noise_det}/{len(noise_ids)})")
        rates.append(noise_rate)
        bar_colors.append("orange")

    ax3.bar(categories, rates, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax3.set_title("Detection Rates (Global Threshold)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Rate (%)")
    ax3.set_ylim(0, 105)
    for i, r in enumerate(rates):
        ax3.text(i, r + 1, f"{r:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Panel 4: Box plot per cluster
    ax4 = axes[1, 1]
    cluster_score_data = []
    cluster_tick_labels = []
    for label in unique_labels:
        mask = hdbscan_labels == label
        cluster_score_data.append(all_scores["ensemble"][mask])
        cluster_tick_labels.append("Noise" if label == -1 else f"C{label}")
    bp = ax4.boxplot(cluster_score_data, labels=cluster_tick_labels, patch_artist=True)
    for i, (patch, label) in enumerate(zip(bp["boxes"], unique_labels)):
        color = "lightgray" if label == -1 else cm.get_cmap("tab10")(label % 10)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_title("Ensemble Score by Cluster", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Cluster")
    ax4.set_ylabel("Ensemble Score")
    ax4.axhline(global_threshold, color="darkred", linestyle="--", linewidth=1, alpha=0.7)

    # Summary text
    fig.text(0.5, 0.01,
             f"Routes: {n_routes} | Clusters: {n_clusters} | Noise: {n_noise} ({n_noise/max(n_routes,1)*100:.1f}%) | "
             f"Threshold: {global_threshold:.4f} | Method: GCN (untrained, {GCN_NODE_FEATURES}d->{GCN_HIDDEN_DIM}d->{GCN_OUT_DIM}d)",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
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
    print("  AERION - GCN 기반 미션 경로 임베딩 + 앙상블 이상 탐지")
    print("  (Node2Vec → GCN Graph Convolutional Network)")
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

    # 2-3. GCN 임베딩
    print(f"  [전체 {len(routes)}건 경로 처리 (제한 없음)]")
    gcn_mode = "비학습(untrained)" if GCN_TRAIN_EPOCHS == 0 else f"학습({GCN_TRAIN_EPOCHS} epochs)"
    print(f"  [GCN 모드: {gcn_mode}]")
    print()
    route_ids, vectors = embed_all_routes_gcn(routes, max_routes=MAX_ROUTES)

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

    all_scores = ensemble.score(vectors)
    test_scores = ensemble.score(test_vectors)

    # 5. 합성 이상
    print("=" * 70)
    print("[5] 합성 이상 주입 + 탐지 검증")
    print("=" * 70)

    encoder = get_shared_encoder()

    print(f"  순서 치환 이상 {N_PERMUTED}건 생성 중...")
    perm_ids, perm_vectors = inject_permuted_anomalies(routes, train_ids, encoder, N_PERMUTED)
    print(f"    -> {len(perm_ids)}건 생성 완료")

    print(f"  좌표 노이즈 이상 {N_NOISY}건 생성 중...")
    noise_ids, noise_vectors = inject_noisy_anomalies(routes, train_ids, encoder, N_NOISY)
    print(f"    -> {len(noise_ids)}건 생성 완료")
    print()

    perm_scores = ensemble.score(perm_vectors) if len(perm_vectors) > 0 else None
    noise_scores = ensemble.score(noise_vectors) if len(noise_vectors) > 0 else None

    # HDBSCAN
    hdbscan_labels, clusterer = run_hdbscan(vectors)

    per_cluster_thresholds = compute_per_cluster_thresholds(
        hdbscan_labels, all_scores["ensemble"])
    global_threshold = np.percentile(test_scores["ensemble"], 95)

    print(f"  전역 임계값 (테스트 95%ile): {global_threshold:.4f}")
    print(f"  클러스터별 임계값:")
    for label in sorted(per_cluster_thresholds.keys()):
        name = "노이즈(-1)" if label == -1 else f"클러스터 {label}"
        print(f"    {name}: {per_cluster_thresholds[label]:.4f}")
    print()

    per_cluster_detected = np.zeros(len(vectors), dtype=bool)
    for i in range(len(vectors)):
        cl = hdbscan_labels[i]
        threshold_cl = per_cluster_thresholds.get(cl, global_threshold)
        if all_scores["ensemble"][i] > threshold_cl:
            per_cluster_detected[i] = True

    # UMAP 시각화
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

    # analysis_summary.png 생성
    _create_analysis_summary(
        n_routes=len(route_ids),
        n_clusters=len([l for l in set(hdbscan_labels) if l >= 0]),
        n_noise=(hdbscan_labels == -1).sum(),
        hdbscan_labels=hdbscan_labels,
        all_scores=all_scores,
        test_scores=test_scores,
        perm_scores=perm_scores,
        noise_scores=noise_scores,
        perm_ids=perm_ids,
        noise_ids=noise_ids,
        global_threshold=global_threshold,
        per_cluster_thresholds=per_cluster_thresholds,
        per_cluster_detected=per_cluster_detected,
        output_path=RESULTS_DIR / "analysis_summary.png",
    )

    # 6. 결과 출력
    print("=" * 70)
    print("[6] 결과 분석")
    print("=" * 70)
    print()

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

    # 최종 리포트
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
    print(f"  2. GCN 임베딩 방법")
    print(f"     - 인코더: 2-layer GCN ({GCN_NODE_FEATURES}d -> {GCN_HIDDEN_DIM}d -> {GCN_OUT_DIM}d)")
    print(f"     - 노드 피처: [lat_norm, lon_norm, alt_norm, seq_ratio]")
    print(f"     - 정규화: 한국 전역 기준 (lat: 35.5 +/- 2.5, lng: 128.0 +/- 4.0)")
    print(f"     - 모드: {gcn_mode}")
    if GCN_TRAIN_EPOCHS == 0:
        print(f"     - 참고: GCN은 학습 없이도 message passing으로 이웃 노드 피처를 집계")
        print(f"       -> 그래프 구조가 임베딩에 반영됨 (위치 + 연결 패턴)")
    print(f"     - 풀링: 가중 평균 (시작/끝 노드 2x) + L2 정규화")
    print()
    print(f"  3. HDBSCAN 클러스터링")
    n_clusters = len([l for l in set(hdbscan_labels) if l >= 0])
    n_noise_pts = (hdbscan_labels == -1).sum()
    print(f"     - 클러스터 수: {n_clusters}개")
    print(f"     - 노이즈 포인트: {n_noise_pts}건 ({n_noise_pts/len(hdbscan_labels)*100:.1f}%)")
    print()
    print(f"  4. 이상 탐지 성능")
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
    print(f"  5. Node2Vec 대비 GCN 비교 요약")
    print(f"     - Node2Vec: 랜덤 워크 + Word2Vec -> 구조 임베딩 (학습 필요)")
    print(f"     - GCN: 노드 피처 + message passing -> 구조+위치 임베딩")
    print(f"     - GCN 장점: 노드 피처(위치/고도) 직접 활용, 학습 없이도 작동")
    print(f"     - GCN 장점: 처리 속도 빠름 (Word2Vec 학습 불필요)")
    print(f"     - GCN 단점: 랜덤 초기화 → 실행마다 결과 다를 수 있음 (seed 고정으로 완화)")
    if GCN_TRAIN_EPOCHS == 0:
        print(f"     - 현재 모드: 비학습 GCN (구조 집계만 활용)")
        print(f"     - 개선 옵션: GCN_TRAIN_EPOCHS > 0 설정 시 autoencoder 학습 가능")
    print()
    print(f"  6. 출력 파일")
    print(f"     - {OUTPUT_CSV}")
    print(f"     - {CLUSTER_CSV}")
    print(f"     - {UMAP_PNG}")
    print(f"     - {RESULTS_DIR / 'mission_vectors.npy'}")
    print(f"     - {RESULTS_DIR / 'umap_2d.npy'}")
    print(f"     - {RESULTS_DIR / 'umap_3d.npy'}")
    print()
    print("=" * 70)
    print("  GCN 분석 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
