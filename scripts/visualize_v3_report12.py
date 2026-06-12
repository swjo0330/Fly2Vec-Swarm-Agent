#!/usr/bin/env python3
"""v3 report12 결과 시각화 스크립트.

report12의 JSON 결과를 읽어 시각화 9종 + 임베딩/UMAP 파일을 생성한다.

생성 파일 목록 (results/report12/figures/):
  1. umap_swarm_types.png     — UMAP 2D 4종 시나리오 색상 분리
  2. umap_clusters.png        — UMAP + HDBSCAN 자동 군집화
  3. swarm_vectors.npy        — 32d 임베딩 벡터 (N × 32)
  4. umap_3d.npy              — UMAP 3D 좌표 (N × 3)
  5. cluster_summary.csv      — 군집별 count / dominant_type
  6. e1_ablation_chart.png    — 7조건 × 3지표 grouped bar
  7. e1_type_weights.png      — 7조건 × 5엣지 × 2레이어 히트맵
  8. baseline_chart.png       — R-GAT vs GCN vs GAT 비교
  9. feature_ablation_chart.png — 5그룹 제거 영향 delta chart
 10. confusion_matrix.png     — E-ALL 20d best 모델 confusion matrix

실행:
  cd /Users/.../proposal
  fly2vec/.venv/bin/python scripts/visualize_v3_report12.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

# 한글 폰트 설정 (macOS)
for _font in ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic', 'Malgun Gothic']:
    if any(_font in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _font
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 경로 설정 ──────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPORT_DIR   = PROJECT_ROOT / "results" / "report12"
MODEL_DIR    = PROJECT_ROOT / "fly2vec" / "data" / "models"
FIG_DIR      = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 각 실험 서브디렉토리
E1_DIR      = REPORT_DIR / "e1_ablation"
E1_20D_DIR  = REPORT_DIR / "e1_ablation_20d"
BASE_DIR    = REPORT_DIR / "baseline_comparison"
FEAT_DIR    = REPORT_DIR / "feature_ablation"

# ── 공통 상수 (run_v3_comprehensive.py와 동일하게 맞춤) ───
SEED          = 42
HIDDEN_DIM    = 64
OUT_DIM       = 32
N_EDGE_TYPES  = 5
HEADS         = 4
PROXIMITY_RADIUS_M = 500.0
COVERAGE_GRID_M    = 100.0

EDGE_NAMES  = ['Sequential', 'Proximity', 'Coverage', 'Temporal', 'Anomaly']
TYPE_MAP    = {"cooperative_search": 0, "relay_move": 1,
               "collision_risk": 2, "coverage_overlap": 3}
ANOMALY_MAP = {"cooperative_search": 0, "relay_move": 0,
               "collision_risk": 1, "coverage_overlap": 1}
CLASS_NAMES = ["cooperative_search", "relay_move", "collision_risk", "coverage_overlap"]
CLASS_NAMES_KR = ["협력 수색", "릴레이 이동", "충돌 위험", "수색 중복"]

# E1 조건 순서 및 한글 레이블
CONDITION_ORDER = ["E-S", "E-SP", "E-SPC", "E-SPCT", "E-ALL", "E-P", "E-C"]
CONDITION_LABELS = {
    "E-S":    "순서만",
    "E-SP":   "순서+근접",
    "E-SPC":  "+커버리지",
    "E-SPCT": "+시간",
    "E-ALL":  "전체(5종)",
    "E-P":    "근접만",
    "E-C":    "커버리지만",
}

# 시각화용 색상 팔레트
TYPE_COLORS = ['#4CAF50', '#2196F3', '#E91E63', '#FF9800']
METRIC_COLORS = {'pattern': '#4CAF50', 'anomaly': '#2196F3', 'collision': '#E91E63'}


# ══════════════════════════════════════════════════════════════
#  모델 정의 — run_v3_comprehensive.py와 완전 동일
# ══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


class RGATBlock(nn.Module):
    """Relational GAT — 엣지 타입별 독립 GAT + 학습 가중치."""
    def __init__(self, in_dim, out_dim, n_edge_types=5, heads=4):
        super().__init__()
        self.n_edge_types = n_edge_types
        self.gats = nn.ModuleList([
            GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=0.1)
            for _ in range(n_edge_types)
        ])
        self.type_weights = nn.Parameter(torch.ones(n_edge_types) / n_edge_types)

    def forward(self, x, edge_index, edge_type):
        w = F.softmax(self.type_weights, dim=0)
        out = torch.zeros(x.size(0), self.gats[0].out_channels, device=x.device)
        for et in range(self.n_edge_types):
            mask = (edge_type == et)
            if mask.sum() == 0:
                continue
            out = out + w[et] * self.gats[et](x, edge_index[:, mask])
        return out


class AttnPool(nn.Module):
    """어텐션 기반 그래프-레벨 풀링."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=0)
        return (x * weights).sum(dim=0)


class Fly2VecRGAT(nn.Module):
    """R-GAT 모델 (in_dim 파라미터화 — 15d/20d 공용)."""
    def __init__(self, in_dim=15):
        super().__init__()
        self.r1 = RGATBlock(in_dim, HIDDEN_DIM, N_EDGE_TYPES, HEADS)
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.r2 = RGATBlock(HIDDEN_DIM, OUT_DIM, N_EDGE_TYPES, HEADS)
        self.pool = AttnPool(OUT_DIM)
        self.th = nn.Linear(OUT_DIM, 4)
        self.ah = nn.Linear(OUT_DIM, 1)

    def forward(self, data):
        h = F.relu(self.bn(self.r1(data.x, data.edge_index, data.edge_type)))
        h = self.r2(h, data.edge_index, data.edge_type)
        emb = self.pool(h)
        return self.th(emb), torch.sigmoid(self.ah(emb)).squeeze(), emb


# ══════════════════════════════════════════════════════════════
#  그래프 빌드 유틸 — run_v3_comprehensive.py에서 그대로 복사
# ══════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _norm(v, lo, hi):
    return max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-6)))


def _build_node_features_20d(drones):
    """20d 노드 피처 빌드 (run_v3_comprehensive.py와 동일)."""
    all_feats, node_meta = [], []
    alts_map, bears_map, speeds_map = {}, {}, {}

    for did, dr in enumerate(drones):
        lats, lngs, alts = dr["lats"], dr["lngs"], dr["alts"]
        n = len(lats)
        speeds = dr.get("speeds", [2.0] * n)
        dists = [0.0] + [haversine(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bears = [0.0] + [bearing(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bcs = [0.0, 0.0] + [((bears[j] - bears[j-1] + 180) % 360 - 180) for j in range(2, n)]

        for i in range(n):
            gidx = len(all_feats)
            dist_next = dists[i + 1] if i < n - 1 else 0.0
            alt_change = alts[i] - alts[i - 1] if i > 0 else 0.0
            feat = [
                _norm(lats[i], 33, 38), _norm(lngs[i], 124, 132), _norm(alts[i], 0, 500),
                i / max(n - 1, 1), _norm(speeds[i], 0, 15),
                _norm(dists[i], 0, 1000), _norm(dist_next, 0, 1000),
                _norm(bears[i], 0, 360), _norm(abs(bcs[i]), 0, 180),
                0.0, _norm(abs(alt_change), 0, 100),
                1.0 if i == 0 else 0.0, 1.0 if i == n - 1 else 0.0,
                0.0, 0.0,
            ]
            all_feats.append(feat)
            node_meta.append({"did": did, "idx": i, "lat": lats[i], "lon": lngs[i],
                               "gidx": gidx})
            alts_map[gidx] = alts[i]
            bears_map[gidx] = bears[i]
            speeds_map[gidx] = speeds[i]

    # 드론 간 관계 피처 5개 추가
    for i, mi in enumerate(node_meta):
        min_dist = 500.0
        nearest_j = -1
        nearby_count = 0
        for j, mj in enumerate(node_meta):
            if mi["did"] == mj["did"]:
                continue
            d = haversine(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
            if d < min_dist:
                min_dist = d
                nearest_j = j
            if d < 100.0:
                nearby_count += 1

        all_feats[i].append(_norm(min_dist, 0, 500))

        if nearest_j >= 0:
            gi = mi["gidx"]
            gj = node_meta[nearest_j]["gidx"]
            all_feats[i].append(_norm(abs(speeds_map.get(gi, 2.0) - speeds_map.get(gj, 2.0)), 0, 10))
            bear_i = bears_map.get(gi, 0)
            bear_j = bears_map.get(gj, 0)
            hdiff = abs(((bear_i - bear_j + 180) % 360) - 180)
            all_feats[i].append(_norm(hdiff, 0, 180))
            all_feats[i].append(_norm(abs(alts_map.get(gi, 50) - alts_map.get(gj, 50)), 0, 100))
        else:
            all_feats[i].extend([1.0, 0.0, 1.0])

        all_feats[i].append(_norm(nearby_count, 0, 10))

    return all_feats, node_meta


def _build_edges(node_meta, active_types):
    """active_types 기준 엣지 생성 (run_v3_comprehensive.py와 동일)."""
    edges = {t: [] for t in range(5)}

    if 0 in active_types:
        for m in node_meta:
            if m["idx"] > 0:
                prev = m["gidx"] - 1
                if node_meta[prev]["did"] == m["did"]:
                    edges[0].append([prev, m["gidx"]])
                    edges[0].append([m["gidx"], prev])

    if 1 in active_types or 2 in active_types:
        for i, mi in enumerate(node_meta):
            for j, mj in enumerate(node_meta):
                if mi["did"] == mj["did"] or j <= i:
                    continue
                d = haversine(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
                if 1 in active_types and d < PROXIMITY_RADIUS_M:
                    edges[1].append([i, j])
                    edges[1].append([j, i])
                if 2 in active_types:
                    gi = (int(mi["lat"] * 1e5) // int(COVERAGE_GRID_M),
                          int(mi["lon"] * 1e5) // int(COVERAGE_GRID_M))
                    gj = (int(mj["lat"] * 1e5) // int(COVERAGE_GRID_M),
                          int(mj["lon"] * 1e5) // int(COVERAGE_GRID_M))
                    if gi == gj:
                        edges[2].append([i, j])
                        edges[2].append([j, i])

    if 3 in active_types:
        drone_last, drone_first = {}, {}
        for m in node_meta:
            if m["did"] not in drone_first:
                drone_first[m["did"]] = m["gidx"]
            drone_last[m["did"]] = m["gidx"]
        drone_ids = sorted(drone_last.keys())
        for k in range(len(drone_ids) - 1):
            src = drone_last[drone_ids[k]]
            dst = drone_first[drone_ids[k + 1]]
            edges[3].append([src, dst])
            edges[3].append([dst, src])

    if 4 in active_types and 1 in active_types:
        for i, mi in enumerate(node_meta):
            for j, mj in enumerate(node_meta):
                if mi["did"] == mj["did"] or j <= i:
                    continue
                d = haversine(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
                if d < 50.0:
                    edges[4].append([i, j])
                    edges[4].append([j, i])

    return edges


def build_graph_20d(drones, active_types):
    """20d PyG 그래프 빌드."""
    all_feats, node_meta = _build_node_features_20d(drones)
    x = torch.tensor(all_feats, dtype=torch.float32)
    edges = _build_edges(node_meta, active_types)

    all_edges, all_types = [], []
    for t in range(5):
        for e in edges[t]:
            all_edges.append(e)
            all_types.append(t)

    if not all_edges:
        all_edges = [[0, 0]]
        all_types = [0]

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_type  = torch.tensor(all_types, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    return data


# ══════════════════════════════════════════════════════════════
#  Step 1: 임베딩 추출 & UMAP 계산
# ══════════════════════════════════════════════════════════════

def extract_embeddings(raw_scenarios, model_path, active_types, in_dim=20, max_n=1500):
    """best 모델로 임베딩 벡터 추출.

    max_n: UMAP 속도를 위해 최대 시나리오 수 제한.
    반환값: (embeddings: np.ndarray [N×32], labels: list[int], type_names: list[str])
    """
    print(f"  모델 로드: {model_path.name}")
    model = Fly2VecRGAT(in_dim=in_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 균형 샘플링 — 클래스별 동수 추출
    by_type = {t: [] for t in range(4)}
    for sc in raw_scenarios:
        tid = TYPE_MAP[sc["type"]]
        by_type[tid].append(sc)

    per_class = max_n // 4
    sampled = []
    random.seed(SEED)
    for tid in range(4):
        pool = by_type[tid]
        n = min(per_class, len(pool))
        sampled.extend(random.sample(pool, n))
    random.shuffle(sampled)

    print(f"  임베딩 추출 중 ({len(sampled)}건)...")
    embeddings, labels, type_names = [], [], []

    with torch.no_grad():
        for i, sc in enumerate(sampled):
            data = build_graph_20d(sc["drones"], active_types)
            data.y_pattern = torch.tensor(TYPE_MAP[sc["type"]], dtype=torch.long)
            data.y_anomaly = torch.tensor(float(ANOMALY_MAP[sc["type"]]), dtype=torch.float32)
            _, _, emb = model(data)
            embeddings.append(emb.numpy())
            labels.append(TYPE_MAP[sc["type"]])
            type_names.append(sc["type"])
            if (i + 1) % 300 == 0:
                print(f"    {i+1}/{len(sampled)}")

    emb_array = np.array(embeddings)  # [N × 32]
    print(f"  임베딩 shape: {emb_array.shape}")
    return emb_array, labels, type_names


def compute_umap(embeddings, n_components=2, n_neighbors=20, min_dist=0.1):
    """UMAP 차원 축소."""
    try:
        import umap
    except ImportError:
        print("  [경고] umap-learn 미설치 — pip install umap-learn")
        raise

    print(f"  UMAP {n_components}D 계산 중...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=SEED,
        verbose=False,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(umap_2d, min_cluster_size=15, min_samples=5):
    """HDBSCAN 군집화."""
    try:
        import hdbscan
    except ImportError:
        print("  [경고] hdbscan 미설치 — pip install hdbscan")
        raise

    print("  HDBSCAN 군집화 중...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(umap_2d)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_count = (cluster_labels == -1).sum()
    print(f"  군집 수: {n_clusters} | 노이즈: {noise_count}건")
    return cluster_labels, n_clusters


# ══════════════════════════════════════════════════════════════
#  시각화 함수들
# ══════════════════════════════════════════════════════════════

def plot_umap_swarm_types(umap_2d, labels, type_names):
    """시각화 1: UMAP 2D scatter — 4종 시나리오 색상 분리."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors_arr = np.array([TYPE_COLORS[l] for l in labels])
    for tid, (cls, cls_kr, color) in enumerate(zip(CLASS_NAMES, CLASS_NAMES_KR, TYPE_COLORS)):
        mask = np.array(labels) == tid
        ax.scatter(
            umap_2d[mask, 0], umap_2d[mask, 1],
            c=color, label=f"{cls_kr} ({mask.sum()})",
            alpha=0.6, s=15, edgecolors='none'
        )

    ax.set_xlabel('UMAP 차원 1', fontsize=12)
    ax.set_ylabel('UMAP 차원 2', fontsize=12)
    ax.set_title('Fly2Vec-Swarm v3 임베딩 UMAP 시각화\n(E-ALL 20d R-GAT, 32차원 → 2차원)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, markerscale=2, framealpha=0.9)
    ax.grid(alpha=0.2)

    out = FIG_DIR / "umap_swarm_types.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {out}")


def plot_umap_clusters(umap_2d, cluster_labels, n_clusters, type_labels):
    """시각화 2: UMAP + HDBSCAN 군집 결과."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 왼쪽: HDBSCAN 군집 색상
    ax = axes[0]
    unique_clusters = sorted(set(cluster_labels))
    cmap = plt.colormaps['tab20']
    cluster_colors = {c: ('lightgray' if c == -1 else cmap(i / max(n_clusters, 1)))
                      for i, c in enumerate(unique_clusters)}

    for c in unique_clusters:
        mask = cluster_labels == c
        label = f'노이즈 ({mask.sum()})' if c == -1 else f'군집 {c} ({mask.sum()})'
        color = cluster_colors[c]
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=[color], label=label, alpha=0.6, s=15, edgecolors='none')

    ax.set_xlabel('UMAP 차원 1', fontsize=11)
    ax.set_ylabel('UMAP 차원 2', fontsize=11)
    ax.set_title(f'HDBSCAN 자동 군집화 ({n_clusters}개 군집)', fontsize=12, fontweight='bold')
    if n_clusters <= 12:
        ax.legend(fontsize=8, markerscale=2, ncol=2, framealpha=0.85)
    ax.grid(alpha=0.2)

    # 오른쪽: 동일 UMAP에 시나리오 타입 오버레이
    ax2 = axes[1]
    for tid, (cls_kr, color) in enumerate(zip(CLASS_NAMES_KR, TYPE_COLORS)):
        mask = np.array(type_labels) == tid
        ax2.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                    c=color, label=cls_kr, alpha=0.5, s=12, edgecolors='none')

    # 군집 중심 번호 표시
    for c in unique_clusters:
        if c == -1:
            continue
        mask = cluster_labels == c
        cx = umap_2d[mask, 0].mean()
        cy = umap_2d[mask, 1].mean()
        ax2.annotate(str(c), (cx, cy), fontsize=9, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax2.set_xlabel('UMAP 차원 1', fontsize=11)
    ax2.set_ylabel('UMAP 차원 2', fontsize=11)
    ax2.set_title('군집별 시나리오 타입 분포', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, markerscale=2, framealpha=0.85)
    ax2.grid(alpha=0.2)

    fig.suptitle('Fly2Vec-Swarm v3 UMAP + HDBSCAN 군집화', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = FIG_DIR / "umap_clusters.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {out}")


def save_embeddings_and_summary(emb_array, umap_2d, umap_3d, labels, cluster_labels):
    """npy 파일 저장 + cluster_summary.csv 생성."""
    # swarm_vectors.npy (32d)
    npy_path = FIG_DIR / "swarm_vectors.npy"
    np.save(npy_path, emb_array)
    print(f"  저장: {npy_path}  shape={emb_array.shape}")

    # umap_3d.npy
    npy3_path = FIG_DIR / "umap_3d.npy"
    np.save(npy3_path, umap_3d)
    print(f"  저장: {npy3_path}  shape={umap_3d.shape}")

    # cluster_summary.csv
    rows = []
    unique_clusters = sorted(set(cluster_labels))
    for c in unique_clusters:
        mask = cluster_labels == c
        cluster_labels_sel = np.array(labels)[mask]
        if len(cluster_labels_sel) == 0:
            continue
        counts = np.bincount(cluster_labels_sel, minlength=4)
        dominant_idx = counts.argmax()
        dominant_type = CLASS_NAMES[dominant_idx]
        rows.append({
            "cluster": c,
            "count": int(mask.sum()),
            "dominant_type": dominant_type,
            "dominant_count": int(counts[dominant_idx]),
            "purity": round(float(counts[dominant_idx] / max(mask.sum(), 1)), 3),
            "cooperative_search": int(counts[0]),
            "relay_move": int(counts[1]),
            "collision_risk": int(counts[2]),
            "coverage_overlap": int(counts[3]),
        })

    df = pd.DataFrame(rows)
    csv_path = FIG_DIR / "cluster_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {csv_path}  ({len(df)}행)")
    return df


def plot_e1_ablation_chart(e1_results):
    """시각화 3: E1 엣지 Ablation — 7조건 × 3지표 grouped bar chart."""
    conditions = [c for c in CONDITION_ORDER if c in e1_results]
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]
    x = np.arange(len(conditions))
    width = 0.25

    pattern   = [e1_results[c]["pattern_accuracy"]  * 100 for c in conditions]
    anomaly   = [e1_results[c]["anomaly_accuracy"]   * 100 for c in conditions]
    collision = [e1_results[c]["collision_recall"]   * 100 for c in conditions]

    fig, ax = plt.subplots(figsize=(14, 6))

    b1 = ax.bar(x - width, pattern,   width, label='패턴 분류',  color=METRIC_COLORS['pattern'],   alpha=0.88)
    b2 = ax.bar(x,         anomaly,   width, label='이상 탐지',  color=METRIC_COLORS['anomaly'],   alpha=0.88)
    b3 = ax.bar(x + width, collision, width, label='충돌 탐지',  color=METRIC_COLORS['collision'], alpha=0.88)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    # 누적 추가 vs 단독 구분선
    ax.axvline(x=4.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.text(2.0, 103, '← 순차 누적 추가 →', ha='center', fontsize=9, color='gray')
    ax.text(5.5, 103, '← 단독 →', ha='center', fontsize=9, color='gray')

    ax.set_xlabel('엣지 조합 조건', fontsize=12)
    ax.set_ylabel('정확도 / 재현율 (%)', fontsize=12)
    ax.set_title('E1 엣지 Ablation — 조건별 성능 비교 (15d R-GAT, v3)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = FIG_DIR / "e1_ablation_chart.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {out}")


def plot_e1_type_weights(e1_results):
    """시각화 4: E1 type_weights 히트맵 — 7조건 × 5엣지 × 2레이어."""
    conditions = [c for c in CONDITION_ORDER if c in e1_results]
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))

    for ax, layer_key, title in [
        (ax1, "layer1_type_weights", "Layer 1 어텐션 가중치"),
        (ax2, "layer2_type_weights", "Layer 2 어텐션 가중치"),
    ]:
        matrix = []
        for c in conditions:
            w = e1_results[c].get(layer_key, {})
            row = [w.get(en, 0.0) for en in EDGE_NAMES]
            matrix.append(row)
        matrix = np.array(matrix)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0.0, vmax=0.5)
        ax.set_xticks(range(5))
        ax.set_xticklabels(EDGE_NAMES, fontsize=9, rotation=30, ha='right')
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')

        for i in range(len(conditions)):
            for j in range(5):
                val = matrix[i, j]
                color = 'white' if val > 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.75, label='Softmax 가중치')
    fig.suptitle('E1 엣지 Ablation — R-GAT 어텐션 가중치 분포 (15d, v3)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = FIG_DIR / "e1_type_weights.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {out}")


def plot_baseline_chart(baseline_results):
    """시각화 5: Baseline 비교 — R-GAT vs GCN vs GAT."""
    # baseline_results: list of dicts with key 'model'
    model_order = ["R-GAT-15d", "GCN-15d", "GAT-15d"]
    model_labels = ["R-GAT\n(본 연구)", "GCN\n(베이스라인)", "GAT\n(베이스라인)"]
    by_model = {r["model"]: r for r in baseline_results}

    x = np.arange(len(model_order))
    width = 0.22
    metrics = [
        ("pattern_accuracy",  "패턴 분류",  METRIC_COLORS['pattern']),
        ("anomaly_accuracy",  "이상 탐지",  METRIC_COLORS['anomaly']),
        ("collision_recall",  "충돌 탐지",  METRIC_COLORS['collision']),
        ("f1_macro",          "F1 (macro)", '#9C27B0'),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    offsets = [-1.5, -0.5, 0.5, 1.5]

    bars_list = []
    for (metric, mlabel, color), offset in zip(metrics, offsets):
        vals = [by_model.get(m, {}).get(metric, 0) * 100 for m in model_order]
        bars = ax.bar(x + offset * width, vals, width,
                      label=mlabel, color=color, alpha=0.88)
        bars_list.append(bars)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5)

    # R-GAT 강조 박스
    ax.axvspan(-0.5, 0.5, alpha=0.06, color='gold')
    ax.text(0, 2, '★ 본 연구', ha='center', fontsize=9, color='goldenrod', fontweight='bold')

    ax.set_xlabel('모델', fontsize=12)
    ax.set_ylabel('정확도 / F1 (%)', fontsize=12)
    ax.set_title('Baseline 비교 — R-GAT vs GCN vs GAT (15d, E-SPC, v3)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = FIG_DIR / "baseline_chart.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {out}")


def plot_feature_ablation_chart(feat_results):
    """시각화 6: 피처 그룹 제거 영향 — delta chart."""
    # baseline 제외한 ablation 그룹만
    groups = [r for r in feat_results if r["group_id"] != "baseline"]
    baseline = next((r for r in feat_results if r["group_id"] == "baseline"), {})

    group_ids   = [r["group_id"] for r in groups]
    group_names = [r["group_name"] for r in groups]
    # 한글 레이블 축약
    short_names = [
        "위치\n(lat,lon)",
        "고도\n(alt,Δalt)",
        "속도+거리",
        "방향\n(bearing)",
        "드론간\n관계(5개)",
    ]

    delta_pattern   = [r.get("delta_pattern",   0) * 100 for r in groups]
    delta_anomaly   = [r.get("delta_anomaly",   0) * 100 for r in groups]
    delta_collision = [r.get("delta_collision", 0) * 100 for r in groups]
    delta_f1        = [r.get("delta_f1",        0) * 100 for r in groups]

    x = np.arange(len(groups))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))

    b1 = ax.bar(x - 1.5*width, delta_pattern,   width, label='Δ 패턴 분류',  color=METRIC_COLORS['pattern'],   alpha=0.88)
    b2 = ax.bar(x - 0.5*width, delta_anomaly,   width, label='Δ 이상 탐지',  color=METRIC_COLORS['anomaly'],   alpha=0.88)
    b3 = ax.bar(x + 0.5*width, delta_collision, width, label='Δ 충돌 탐지',  color=METRIC_COLORS['collision'], alpha=0.88)
    b4 = ax.bar(x + 1.5*width, delta_f1,        width, label='Δ F1 (macro)', color='#9C27B0', alpha=0.88)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            yoff = 3 if h >= 0 else -3
            ax.annotate(f'{h:+.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, yoff), textcoords="offset points",
                        ha='center', va=va, fontsize=7)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('제거된 피처 그룹', fontsize=12)
    ax.set_ylabel('성능 변화량 (%p, 기준선 대비)', fontsize=12)
    ax.set_title(f'피처 그룹 Ablation — 기준선(20d E-SPC) 대비 성능 변화\n'
                 f'(기준선: 패턴={baseline.get("pattern_accuracy",0):.1%}, '
                 f'이상={baseline.get("anomaly_accuracy",0):.1%}, '
                 f'충돌={baseline.get("collision_recall",0):.1%})',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out = FIG_DIR / "feature_ablation_chart.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {out}")


def plot_confusion_matrix(raw_scenarios, model_path, active_types, in_dim=20):
    """시각화 7: E-ALL 20d best 모델 confusion matrix.

    실제 추론을 수행하여 정확한 혼동 행렬을 생성한다.
    """
    print(f"  confusion matrix 추론 중...")
    model = Fly2VecRGAT(in_dim=in_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # test split 재현 (SEED 동일하게)
    indices = list(range(len(raw_scenarios)))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
    test_scenarios = [raw_scenarios[i] for i in test_idx]

    pat_preds, pat_trues = [], []
    with torch.no_grad():
        for sc in test_scenarios:
            data = build_graph_20d(sc["drones"], active_types)
            data.y_pattern = torch.tensor(TYPE_MAP[sc["type"]], dtype=torch.long)
            data.y_anomaly = torch.tensor(float(ANOMALY_MAP[sc["type"]]), dtype=torch.float32)
            pat_logits, _, _ = model(data)
            pat_preds.append(pat_logits.argmax().item())
            pat_trues.append(TYPE_MAP[sc["type"]])

    cm = confusion_matrix(pat_trues, pat_preds, labels=list(range(4)))
    acc = accuracy_score(pat_trues, pat_preds)
    print(f"  패턴 정확도 (재현): {acc:.1%}")

    # 정규화 (행별)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data_cm, title, fmt_fn in [
        (axes[0], cm,      '원본 카운트', lambda v: f'{v:.0f}'),
        (axes[1], cm_norm, '정규화 (행별 비율)', lambda v: f'{v:.2f}'),
    ]:
        im = ax.imshow(data_cm, cmap='Blues', aspect='auto')
        ax.set_xticks(range(4))
        ax.set_xticklabels(CLASS_NAMES_KR, fontsize=10, rotation=20, ha='right')
        ax.set_yticks(range(4))
        ax.set_yticklabels(CLASS_NAMES_KR, fontsize=10)
        ax.set_xlabel('예측 클래스', fontsize=11)
        ax.set_ylabel('실제 클래스', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

        thresh = data_cm.max() / 2.0
        for i in range(4):
            for j in range(4):
                val = data_cm[i, j]
                color = 'white' if val > thresh else 'black'
                ax.text(j, i, fmt_fn(val), ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')

    fig.suptitle(f'Confusion Matrix — E-ALL 20d R-GAT (정확도: {acc:.1%}, v3)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = FIG_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  저장: {out}")


# ══════════════════════════════════════════════════════════════
#  JSON 로드 헬퍼
# ══════════════════════════════════════════════════════════════

def load_e1_results():
    """E1 15d ablation JSON 로드 → {condition_id: dict}."""
    summary_path = E1_DIR / "e1_ablation_results.json"
    if not summary_path.exists():
        # 개별 파일에서 합치기
        results = {}
        for cid in CONDITION_ORDER:
            p = E1_DIR / f"e1_{cid}_result.json"
            if p.exists():
                with open(p) as f:
                    results[cid] = json.load(f)
        return results
    with open(summary_path) as f:
        data = json.load(f)
    return {r["condition_id"]: r for r in data["conditions"]}


def load_baseline_results():
    """Baseline 비교 JSON 로드 → list of dicts."""
    summary_path = BASE_DIR / "baseline_comparison_results.json"
    if not summary_path.exists():
        return []
    with open(summary_path) as f:
        data = json.load(f)
    return data.get("models", [])


def load_feature_ablation_results():
    """피처 ablation JSON 로드 → list of dicts."""
    summary_path = FEAT_DIR / "feature_ablation_results.json"
    if not summary_path.exists():
        return []
    with open(summary_path) as f:
        data = json.load(f)
    return data.get("groups", [])


def load_raw_scenarios():
    """v3 시나리오 캐시 로드."""
    pkl_path = REPORT_DIR / "raw_scenarios_v3.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"시나리오 캐시 없음: {pkl_path}")
    print(f"  시나리오 캐시 로드: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════════

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 65)
    print("Fly2Vec-Swarm v3 report12 시각화 스크립트")
    print(f"출력 디렉토리: {FIG_DIR}")
    print("=" * 65)

    # ── [A] JSON 기반 차트 (재훈련 불필요) ──────────────────────

    print("\n[1] E1 엣지 Ablation 결과 로드")
    e1_results = load_e1_results()
    print(f"  조건 수: {len(e1_results)}")
    if not e1_results:
        print("  [경고] E1 결과 없음 — e1_ablation_chart 건너뜀")
    else:
        print("\n[2] E1 Grouped Bar Chart 생성")
        plot_e1_ablation_chart(e1_results)

        print("\n[3] E1 Type Weights 히트맵 생성")
        plot_e1_type_weights(e1_results)

    print("\n[4] Baseline 비교 결과 로드")
    baseline_results = load_baseline_results()
    print(f"  모델 수: {len(baseline_results)}")
    if not baseline_results:
        print("  [경고] Baseline 결과 없음 — baseline_chart 건너뜀")
    else:
        print("\n[5] Baseline 비교 차트 생성")
        plot_baseline_chart(baseline_results)

    print("\n[6] 피처 Ablation 결과 로드")
    feat_results = load_feature_ablation_results()
    print(f"  그룹 수: {len(feat_results)}")
    if not feat_results:
        print("  [경고] 피처 Ablation 결과 없음 — feature_ablation_chart 건너뜀")
    else:
        print("\n[7] 피처 Ablation Delta Chart 생성")
        plot_feature_ablation_chart(feat_results)

    # ── [B] 모델 추론 기반 시각화 (UMAP, Confusion Matrix) ─────

    print("\n[8] 시나리오 캐시 로드")
    raw_scenarios = load_raw_scenarios()
    print(f"  시나리오 수: {len(raw_scenarios)}")

    # 베스트 모델: E-ALL 20d
    best_model_path = MODEL_DIR / "v3_e1_20d_E-ALL_best.pt"
    active_types_all = {0, 1, 2, 3, 4}  # E-ALL

    if not best_model_path.exists():
        print(f"  [경고] 모델 없음: {best_model_path}")
        print("  UMAP / confusion matrix 건너뜀")
    else:
        print("\n[9] 임베딩 추출 (E-ALL 20d R-GAT)")
        emb_array, labels, type_names = extract_embeddings(
            raw_scenarios,
            model_path=best_model_path,
            active_types=active_types_all,
            in_dim=20,
            max_n=1500,  # 시나리오 수 1500개 상한 (속도)
        )

        print("\n[10] UMAP 2D 계산")
        umap_2d = compute_umap(emb_array, n_components=2, n_neighbors=20, min_dist=0.1)

        print("\n[11] UMAP 2D scatter 시각화")
        plot_umap_swarm_types(umap_2d, labels, type_names)

        print("\n[12] HDBSCAN 군집화")
        cluster_labels, n_clusters = run_hdbscan(umap_2d, min_cluster_size=15)

        print("\n[13] UMAP + HDBSCAN 군집 시각화")
        plot_umap_clusters(umap_2d, cluster_labels, n_clusters, labels)

        print("\n[14] UMAP 3D 계산 및 임베딩 저장")
        umap_3d = compute_umap(emb_array, n_components=3, n_neighbors=20, min_dist=0.1)
        save_embeddings_and_summary(emb_array, umap_2d, umap_3d, labels, cluster_labels)

        print("\n[15] Confusion Matrix 생성 (E-ALL 20d, test set 재현)")
        plot_confusion_matrix(
            raw_scenarios,
            model_path=best_model_path,
            active_types=active_types_all,
            in_dim=20,
        )

    # ── 완료 요약 ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("시각화 완료")
    print(f"출력 파일 목록:")
    for p in sorted(FIG_DIR.iterdir()):
        size = p.stat().st_size
        unit = "KB" if size < 1_000_000 else "MB"
        size_str = f"{size/1024:.0f}{unit}" if size < 1_000_000 else f"{size/1024/1024:.1f}{unit}"
        print(f"  {p.name:<35} {size_str:>8}")
    print("=" * 65)


if __name__ == "__main__":
    main()
