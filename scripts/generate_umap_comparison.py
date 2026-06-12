#!/usr/bin/env python3
"""
Fly2Vec: R-GAT 학습 전/후 UMAP 비교 시각화
  - 학습 전: results/report4/swarm_vectors.npy (500건 × 128d, 비학습 R-GAT)
  - 학습 후: fly2vec_rgat_attn_best.pt 로드 → 같은 500건 시나리오 재추론
  - 결과: results/analysis/umap_before_after_comparison.png
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

import umap
import hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ → 프로젝트 루트
REPORT4_DIR = PROJECT_ROOT / "results" / "report4"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
MODEL_PATH = PROJECT_ROOT / "fly2vec" / "data" / "models" / "fly2vec_rgat_attn_best.pt"
DATA_PATH = PROJECT_ROOT / "fly2vec" / "data" / "dataset_wp_spot.csv"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ── 한국 좌표 정규화 상수 (run_smgat_analysis.py 동일) ──
KOREA_LAT_CENTER, KOREA_LAT_RANGE = 35.5, 2.5
KOREA_LNG_CENTER, KOREA_LNG_RANGE = 128.0, 4.0
ALT_MAX_NORM = 500.0
PROXIMITY_RADIUS_M = 500.0

LAT_MIN, LAT_MAX = 33.0, 38.0
LNG_MIN, LNG_MAX = 124.0, 132.0
ALT_MIN, ALT_MAX = 0.0, 500.0
MIN_WP_COUNT = 4
MAX_ROUTES = 2000
N_SWARM_SCENARIOS = 500
DRONES_PER_SWARM = [3, 4, 5]


# ================================================================
# 학습된 R-GAT 모델 재구성 (state_dict 키에서 역추적)
# ================================================================
# state_dict 분석:
#   r1.gats.{0-4}: GATConv(9, 64, heads=4, concat=False) — 5 edge types
#   r1.type_weights: [5]
#   bn: BatchNorm(64)
#   r2.gats.{0-4}: GATConv(64, 32, heads=4, concat=False)
#   r2.type_weights: [5]
#   pool.attn: Linear(32,32) -> ReLU -> Linear(32,1) — attention pooling
#   th: Linear(32, 4) — pattern classifier
#   ah: Linear(32, 1) — anomaly head

class RGATBlock(nn.Module):
    """엣지 타입별 독립 GAT + 가중합"""
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
        for etype in range(self.n_edge_types):
            mask = (edge_type == etype)
            if mask.sum() == 0:
                continue
            etype_edges = edge_index[:, mask]
            h = self.gats[etype](x, etype_edges)
            out = out + w[etype] * h
        return out


class AttnPool(nn.Module):
    """Attention Pooling"""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        # x: [n_nodes, dim]
        scores = self.attn(x)  # [n_nodes, 1]
        weights = F.softmax(scores, dim=0)  # [n_nodes, 1]
        return (x * weights).sum(dim=0)  # [dim]


class Fly2VecRGATAttn(nn.Module):
    """R-GAT 9d + Attention Pooling (학습된 최종 모델)"""
    def __init__(self, in_dim=9, hidden_dim=64, out_dim=32, n_edge_types=5, heads=4):
        super().__init__()
        self.r1 = RGATBlock(in_dim, hidden_dim, n_edge_types, heads)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.r2 = RGATBlock(hidden_dim, out_dim, n_edge_types, heads)
        self.pool = AttnPool(out_dim)
        self.th = nn.Linear(out_dim, 4)   # pattern classifier
        self.ah = nn.Linear(out_dim, 1)   # anomaly head

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        h = self.r1(x, edge_index, edge_type)
        h = self.bn(h)
        h = F.relu(h)
        h = self.r2(h, edge_index, edge_type)

        # graph embedding via attention pooling
        graph_emb = self.pool(h)  # [32d]
        return h, graph_emb


# ================================================================
# 데이터 재생성 (run_smgat_analysis.py 로직 복제)
# ================================================================

import math
from collections import defaultdict

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam)*math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def offset_coords(lat, lon, offset_m, bearing_deg):
    R = 6371000.0
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(offset_m/R) +
                      math.cos(lat1)*math.sin(offset_m/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(offset_m/R)*math.cos(lat1),
                              math.cos(offset_m/R) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def classify_route(route_df):
    n = len(route_df)
    lats, lngs = route_df["wp_lat"].values, route_df["wp_lng"].values
    bearing_changes = []
    for i in range(1, n-1):
        b1 = bearing(lats[i-1], lngs[i-1], lats[i], lngs[i])
        b2 = bearing(lats[i], lngs[i], lats[i+1], lngs[i+1])
        diff = abs(b2 - b1)
        if diff > 180: diff = 360 - diff
        bearing_changes.append(diff)
    avg_change = np.mean(bearing_changes) if bearing_changes else 0
    start_end_dist = haversine(lats[0], lngs[0], lats[-1], lngs[-1])
    total_dist = sum(haversine(lats[i], lngs[i], lats[i+1], lngs[i+1]) for i in range(n-1))
    if total_dist > 0 and start_end_dist / total_dist < 0.2 and n >= 8:
        return "grid_search"
    elif avg_change > 60:
        return "grid_search"
    elif avg_change < 15:
        return "linear_flight"
    else:
        return "waypoint_sequence"


def load_and_filter():
    df = pd.read_csv(DATA_PATH)
    mask = ((df["wp_lat"] >= LAT_MIN) & (df["wp_lat"] <= LAT_MAX) &
            (df["wp_lng"] >= LNG_MIN) & (df["wp_lng"] <= LNG_MAX) &
            (df["wp_alt"] >= ALT_MIN) & (df["wp_alt"] <= ALT_MAX))
    df = df[mask].copy()
    grouped = {name: group.sort_values("wp_seq").reset_index(drop=True)
               for name, group in df.groupby("wp_spot_id")}
    routes = {k: v for k, v in grouped.items() if len(v) >= MIN_WP_COUNT}
    if len(routes) > MAX_ROUTES:
        keys = list(routes.keys())
        np.random.shuffle(keys)
        routes = {k: routes[k] for k in keys[:MAX_ROUTES]}
    return routes


def generate_swarm_scenarios(routes):
    route_keys = list(routes.keys())
    route_types = {k: classify_route(v) for k, v in routes.items()}
    by_type = defaultdict(list)
    for k, t in route_types.items():
        by_type[t].append(k)

    scenarios = []

    # Type 1: cooperative_search
    n_coop = N_SWARM_SCENARIOS // 4
    for i in range(n_coop):
        n_drones = np.random.choice(DRONES_PER_SWARM)
        route_type = np.random.choice(list(by_type.keys()))
        candidates = by_type[route_type]
        if len(candidates) < n_drones:
            candidates = route_keys
        selected = np.random.choice(candidates, size=n_drones, replace=False)
        drone_routes = []
        for j, key in enumerate(selected):
            route_df = routes[key].copy()
            angle = (360/n_drones)*j
            offset_dist = np.random.uniform(500, 2000)
            route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                offset_coords(lat, lon, offset_dist, angle)
                for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])])
            drone_routes.append({"drone_id": j, "route_key": key, "route_df": route_df})
        scenarios.append({"scenario_id": f"COOP_{i:03d}", "type": "cooperative_search",
                          "label": "normal", "n_drones": n_drones, "drones": drone_routes})

    # Type 2: collision_risk
    n_conflict = N_SWARM_SCENARIOS // 4
    for i in range(n_conflict):
        n_drones = np.random.choice([3, 4])
        selected = np.random.choice(route_keys, size=n_drones, replace=False)
        drone_routes = []
        base_route = routes[selected[0]]
        mid_lat = base_route["wp_lat"].mean()
        mid_lon = base_route["wp_lng"].mean()
        for j, key in enumerate(selected):
            route_df = routes[key].copy()
            if j < 2:
                offset_dist = np.random.uniform(30, 150)
                angle = np.random.uniform(0, 360)
                route_df["wp_lat"] = mid_lat + (route_df["wp_lat"] - route_df["wp_lat"].mean())
                route_df["wp_lng"] = mid_lon + (route_df["wp_lng"] - route_df["wp_lng"].mean())
                route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                    offset_coords(lat, lon, offset_dist, angle)
                    for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])])
            else:
                offset_dist = np.random.uniform(1000, 3000)
                angle = np.random.uniform(0, 360)
                route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                    offset_coords(lat, lon, offset_dist, angle)
                    for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])])
            drone_routes.append({"drone_id": j, "route_key": key, "route_df": route_df})
        scenarios.append({"scenario_id": f"CONF_{i:03d}", "type": "collision_risk",
                          "label": "collision_risk", "n_drones": n_drones, "drones": drone_routes})

    # Type 3: coverage_overlap
    n_overlap = N_SWARM_SCENARIOS // 4
    for i in range(n_overlap):
        n_drones = np.random.choice([3, 4, 5])
        grid_routes = by_type.get("grid_search", route_keys[:20])
        if len(grid_routes) < n_drones:
            grid_routes = route_keys
        selected = np.random.choice(grid_routes, size=n_drones, replace=False)
        base_route = routes[selected[0]]
        center_lat = base_route["wp_lat"].mean()
        center_lon = base_route["wp_lng"].mean()
        drone_routes = []
        for j, key in enumerate(selected):
            route_df = routes[key].copy()
            route_df["wp_lat"] = center_lat + (route_df["wp_lat"] - route_df["wp_lat"].mean())
            route_df["wp_lng"] = center_lon + (route_df["wp_lng"] - route_df["wp_lng"].mean())
            small_offset = np.random.uniform(10, 80)
            angle = (360/n_drones)*j
            route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                offset_coords(lat, lon, small_offset, angle)
                for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])])
            drone_routes.append({"drone_id": j, "route_key": key, "route_df": route_df})
        scenarios.append({"scenario_id": f"OVLP_{i:03d}", "type": "coverage_overlap",
                          "label": "coverage_overlap", "n_drones": n_drones, "drones": drone_routes})

    # Type 4: relay_move
    n_relay = N_SWARM_SCENARIOS - n_coop - n_conflict - n_overlap
    for i in range(n_relay):
        n_drones = np.random.choice([3, 4])
        linear_routes = by_type.get("linear_flight", route_keys[:20])
        if len(linear_routes) < n_drones:
            linear_routes = route_keys
        selected = np.random.choice(linear_routes, size=n_drones, replace=False)
        drone_routes = []
        for j, key in enumerate(selected):
            route_df = routes[key].copy()
            offset_dist = j * np.random.uniform(800, 1500)
            route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                offset_coords(lat, lon, offset_dist, 90)
                for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])])
            drone_routes.append({"drone_id": j, "route_key": key, "route_df": route_df})
        scenarios.append({"scenario_id": f"RLAY_{i:03d}", "type": "relay_move",
                          "label": "normal", "n_drones": n_drones, "drones": drone_routes})

    np.random.shuffle(scenarios)
    return scenarios


def build_swarm_graph_9d(scenario):
    """군집 시나리오 → PyG Data (5종 엣지, 9d 노드 피처)

    9d features: [lat_norm, lon_norm, alt_norm, seq_ratio, speed_norm,
                  heading_norm, drone_id_norm, min_distance_to_other, is_overlap_grid]
    5 edge types: Sequential(0), Proximity(1), Coverage(2), Temporal(3), Anomaly(4)
    """
    drones = scenario["drones"]
    all_nodes_7d = []
    node_meta = []

    # Step 1: collect 7d features + metadata
    for drone in drones:
        drone_id = drone["drone_id"]
        df = drone["route_df"]
        n = len(df)
        lats = df["wp_lat"].values
        lngs = df["wp_lng"].values
        alts = df["wp_alt"].values
        speeds = df["wp_speed"].values if "wp_speed" in df.columns else np.full(n, 2.0)
        headings = df["wp_heading"].values if "wp_heading" in df.columns else np.zeros(n)

        for i in range(n):
            feat_7d = [
                (lats[i] - KOREA_LAT_CENTER) / KOREA_LAT_RANGE,
                (lngs[i] - KOREA_LNG_CENTER) / KOREA_LNG_RANGE,
                alts[i] / ALT_MAX_NORM,
                i / max(n-1, 1),
                min(speeds[i], 20.0) / 20.0,
                headings[i] / 360.0,
                drone_id / max(len(drones)-1, 1),
            ]
            all_nodes_7d.append(feat_7d)
            node_meta.append({
                "drone_id": drone_id,
                "wp_idx": i,
                "lat": lats[i],
                "lon": lngs[i],
                "alt": alts[i],
            })

    n_nodes = len(all_nodes_7d)

    # Step 2: compute min_distance_to_other and is_overlap_grid for 9d
    min_dists = np.ones(n_nodes) * 99999.0
    grid_cells = defaultdict(list)
    for idx, meta in enumerate(node_meta):
        cell_lat = int(meta["lat"] * 1000)
        cell_lon = int(meta["lon"] * 1000)
        grid_cells[(cell_lat, cell_lon)].append(idx)

    # min distance to other drones
    drone_ranges = []
    offset = 0
    for drone in drones:
        n_wp = len(drone["route_df"])
        drone_ranges.append((offset, offset + n_wp, drone["drone_id"]))
        offset += n_wp

    for i_range in range(len(drone_ranges)):
        for j_range in range(i_range + 1, len(drone_ranges)):
            s1, e1, _ = drone_ranges[i_range]
            s2, e2, _ = drone_ranges[j_range]
            for ni in range(s1, e1):
                for nj in range(s2, e2):
                    mi, mj = node_meta[ni], node_meta[nj]
                    dist = haversine(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
                    if dist < min_dists[ni]:
                        min_dists[ni] = dist
                    if dist < min_dists[nj]:
                        min_dists[nj] = dist

    # is_overlap_grid
    is_overlap = np.zeros(n_nodes)
    for cell, indices in grid_cells.items():
        drone_ids_in_cell = set(node_meta[idx]["drone_id"] for idx in indices)
        if len(drone_ids_in_cell) >= 2:
            for idx in indices:
                is_overlap[idx] = 1.0

    # Build 9d features
    all_nodes_9d = []
    for i in range(n_nodes):
        feat = all_nodes_7d[i] + [
            min(min_dists[i], PROXIMITY_RADIUS_M) / PROXIMITY_RADIUS_M,  # normalized
            is_overlap[i],
        ]
        all_nodes_9d.append(feat)

    x = torch.tensor(all_nodes_9d, dtype=torch.float32)

    # Step 3: edges (5 types: sequential=0, proximity=1, coverage=2, temporal=3, anomaly=4)
    edges_by_type = {i: [] for i in range(5)}

    # Sequential
    offset = 0
    for drone in drones:
        n_wp = len(drone["route_df"])
        for i in range(n_wp - 1):
            src, dst = offset + i, offset + i + 1
            edges_by_type[0].append([src, dst])
            edges_by_type[0].append([dst, src])
        offset += n_wp

    # Proximity
    for i_range in range(len(drone_ranges)):
        for j_range in range(i_range + 1, len(drone_ranges)):
            s1, e1, _ = drone_ranges[i_range]
            s2, e2, _ = drone_ranges[j_range]
            for ni in range(s1, e1):
                for nj in range(s2, e2):
                    mi, mj = node_meta[ni], node_meta[nj]
                    dist = haversine(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
                    if dist < PROXIMITY_RADIUS_M:
                        edges_by_type[1].append([ni, nj])
                        edges_by_type[1].append([nj, ni])

    # Coverage
    for cell, indices in grid_cells.items():
        drone_groups = defaultdict(list)
        for idx in indices:
            drone_groups[node_meta[idx]["drone_id"]].append(idx)
        if len(drone_groups) >= 2:
            dids = list(drone_groups.keys())
            for di in range(len(dids)):
                for dj in range(di + 1, len(dids)):
                    for ni in drone_groups[dids[di]]:
                        for nj in drone_groups[dids[dj]]:
                            edges_by_type[2].append([ni, nj])
                            edges_by_type[2].append([nj, ni])

    # Temporal/Anomaly: 비어있어도 됨 (edge_type mask 처리)

    all_edges = []
    all_edge_types = []
    for etype in range(5):
        for e in edges_by_type[etype]:
            all_edges.append(e)
            all_edge_types.append(etype)

    if len(all_edges) == 0:
        all_edges = [[0, 0]]
        all_edge_types = [0]

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(all_edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.edge_type = edge_type

    return data


# ================================================================
# MAIN
# ================================================================

def main():
    print()
    print("=" * 70)
    print("  Fly2Vec: R-GAT Before/After Training UMAP Comparison")
    print("=" * 70)
    print()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load report4 (before training) ──
    print("[1] Loading report4 data (before training)...")
    before_vectors = np.load(REPORT4_DIR / "swarm_vectors.npy")  # (500, 128)
    before_umap = np.load(REPORT4_DIR / "umap_2d.npy")          # (500, 2)
    stats_df = pd.read_csv(REPORT4_DIR / "swarm_analysis_full.csv")
    print(f"    Before vectors: {before_vectors.shape}")
    print(f"    Before UMAP: {before_umap.shape}")
    print(f"    Scenarios: {len(stats_df)}")
    print()

    # ── 2. Load trained model ──
    print("[2] Loading trained R-GAT model...")
    model = Fly2VecRGATAttn(in_dim=9, hidden_dim=64, out_dim=32, n_edge_types=5, heads=4)
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"    Model loaded: {MODEL_PATH.name}")
    print(f"    Output dim: 32d")
    print()

    # ── 3. Regenerate same 500 scenarios ──
    print("[3] Regenerating 500 swarm scenarios (same seed)...")
    routes = load_and_filter()
    scenarios = generate_swarm_scenarios(routes)
    print(f"    Generated: {len(scenarios)} scenarios")

    # Match scenario types with report4
    type_counts = defaultdict(int)
    for s in scenarios:
        type_counts[s["type"]] += 1
    for t, c in sorted(type_counts.items()):
        print(f"      {t}: {c}")
    print()

    # ── 4. Generate trained embeddings ──
    print("[4] Generating trained R-GAT embeddings...")
    trained_embeddings = []
    scenario_types = []
    t0 = time.time()

    for idx, scenario in enumerate(scenarios):
        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{len(scenarios)}...")

        try:
            data = build_swarm_graph_9d(scenario)
            with torch.no_grad():
                _, graph_emb = model(data)
            trained_embeddings.append(graph_emb.numpy())
            scenario_types.append(scenario["type"])
        except Exception as e:
            print(f"    [WARN] {scenario['scenario_id']}: {e}")
            # fallback: zero vector
            trained_embeddings.append(np.zeros(32))
            scenario_types.append(scenario["type"])

    trained_vectors = np.array(trained_embeddings)  # (500, 32)
    elapsed = time.time() - t0
    print(f"    Trained vectors: {trained_vectors.shape}, {elapsed:.1f}s")
    print()

    # ── 5. UMAP for trained vectors ──
    print("[5] Computing UMAP for trained vectors...")
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    trained_umap = reducer.fit_transform(trained_vectors)
    print(f"    Trained UMAP: {trained_umap.shape}")
    print()

    # ── 6. HDBSCAN clustering comparison ──
    print("[6] HDBSCAN clustering...")

    # Before
    clusterer_before = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=3, metric="euclidean")
    labels_before = clusterer_before.fit_predict(before_vectors)
    n_clusters_before = len(set(labels_before) - {-1})
    noise_before = (labels_before == -1).sum()
    noise_pct_before = 100 * noise_before / len(labels_before)

    # After
    clusterer_after = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=3, metric="euclidean")
    labels_after = clusterer_after.fit_predict(trained_vectors)
    n_clusters_after = len(set(labels_after) - {-1})
    noise_after = (labels_after == -1).sum()
    noise_pct_after = 100 * noise_after / len(labels_after)

    print(f"    Before: {n_clusters_before} clusters, {noise_before} noise ({noise_pct_before:.1f}%)")
    print(f"    After:  {n_clusters_after} clusters, {noise_after} noise ({noise_pct_after:.1f}%)")
    print()

    # ── 7. Scenario type separation analysis ──
    print("[7] Scenario type separation analysis...")

    # Use the scenario_types from regenerated scenarios (same seed → same order)
    types_list = scenario_types

    type_colors = {
        "cooperative_search": "#3498db",   # blue
        "collision_risk": "#e74c3c",       # red
        "coverage_overlap": "#f39c12",     # orange
        "relay_move": "#2ecc71",           # green
    }
    type_labels_short = {
        "cooperative_search": "Cooperative",
        "collision_risk": "Collision",
        "coverage_overlap": "Coverage",
        "relay_move": "Relay",
    }

    # Compute silhouette-like metric: average intra-type distance / inter-type distance
    from sklearn.metrics import silhouette_score
    type_to_int = {"cooperative_search": 0, "collision_risk": 1, "coverage_overlap": 2, "relay_move": 3}
    int_labels = np.array([type_to_int[t] for t in types_list])

    try:
        sil_before = silhouette_score(before_vectors, int_labels)
    except Exception:
        sil_before = 0.0
    try:
        sil_after = silhouette_score(trained_vectors, int_labels)
    except Exception:
        sil_after = 0.0

    print(f"    Silhouette (before): {sil_before:.4f}")
    print(f"    Silhouette (after):  {sil_after:.4f}")
    print()

    # ── 8. Generate comparison figure ──
    print("[8] Generating comparison figure...")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Use types from report4 CSV for 'before' and regenerated types for 'after'
    report4_types = stats_df["type"].values

    # LEFT: Before training
    ax = axes[0]
    for stype, color in type_colors.items():
        mask = report4_types == stype
        if mask.sum() > 0:
            ax.scatter(before_umap[mask, 0], before_umap[mask, 1],
                       c=color, label=type_labels_short[stype], s=25, alpha=0.7, edgecolors='white', linewidths=0.3)
    ax.set_title("Before Training (Untrained R-GAT, 128d)\n"
                 f"Clusters: {n_clusters_before} | Noise: {noise_pct_before:.1f}% | "
                 f"Silhouette: {sil_before:.3f}",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    # RIGHT: After training
    ax = axes[1]
    for stype, color in type_colors.items():
        mask = np.array(types_list) == stype
        if mask.sum() > 0:
            ax.scatter(trained_umap[mask, 0], trained_umap[mask, 1],
                       c=color, label=type_labels_short[stype], s=25, alpha=0.7, edgecolors='white', linewidths=0.3)
    ax.set_title("After Training (R-GAT 9d + Attn Pool, 32d)\n"
                 f"Clusters: {n_clusters_after} | Noise: {noise_pct_after:.1f}% | "
                 f"Silhouette: {sil_after:.3f}",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Fly2Vec: R-GAT Embedding UMAP Comparison (Before vs After Training)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = ANALYSIS_DIR / "umap_before_after_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")

    # ── 9. Save artifacts ──
    np.save(ANALYSIS_DIR / "trained_swarm_vectors.npy", trained_vectors)
    np.save(ANALYSIS_DIR / "trained_umap_2d.npy", trained_umap)
    print(f"    Saved: trained_swarm_vectors.npy ({trained_vectors.shape})")
    print(f"    Saved: trained_umap_2d.npy ({trained_umap.shape})")

    # ── 10. Summary ──
    print()
    print("=" * 70)
    print("  SUMMARY: Before vs After Training")
    print("=" * 70)
    print(f"  {'Metric':<30s} {'Before':>12s} {'After':>12s} {'Delta':>12s}")
    print(f"  {'-'*66}")
    print(f"  {'Embedding dim':<30s} {'128d':>12s} {'32d':>12s} {'':>12s}")
    print(f"  {'HDBSCAN clusters':<30s} {n_clusters_before:>12d} {n_clusters_after:>12d} {n_clusters_after - n_clusters_before:>+12d}")
    print(f"  {'Noise count':<30s} {noise_before:>12d} {noise_after:>12d} {noise_after - noise_before:>+12d}")
    print(f"  {'Noise %':<30s} {noise_pct_before:>11.1f}% {noise_pct_after:>11.1f}% {noise_pct_after - noise_pct_before:>+11.1f}%")
    print(f"  {'Silhouette score':<30s} {sil_before:>12.4f} {sil_after:>12.4f} {sil_after - sil_before:>+12.4f}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
