#!/opt/anaconda3/bin/python
"""
AERION - SM-GAT (Swarm Mission Graph Attention Network) 분석
  - 기존 2,068건 실 관제 데이터에서 500건 합성 군집 시나리오 생성
  - R-GAT (Relational GAT) + 3종 엣지 (Sequential/Proximity/Coverage)
  - 멀티태스크: 패턴 분류 / 이상 탐지 / 군집 임베딩
  - 결과: report4/ (UMAP, 어텐션 히트맵, 분석 차트, CSV)

실행: python3 run_smgat_analysis.py
필요: torch, torch_geometric, numpy, pandas, scikit-learn, umap-learn, hdbscan, matplotlib, plotly
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import math
import json
import warnings
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

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

# 필터링
LAT_MIN, LAT_MAX = 33.0, 38.0
LNG_MIN, LNG_MAX = 124.0, 132.0
ALT_MIN, ALT_MAX = 0.0, 500.0
MIN_WP_COUNT = 4

# 한국 전역 정규화
KOREA_LAT_CENTER, KOREA_LAT_RANGE = 35.5, 2.5
KOREA_LNG_CENTER, KOREA_LNG_RANGE = 128.0, 4.0
ALT_MAX_NORM = 500.0

# 군집 시나리오
N_SWARM_SCENARIOS = 3000
DRONES_PER_SWARM = [3, 4, 5]  # 3~5대 조합
PROXIMITY_RADIUS_M = 500.0
COVERAGE_GRID_M = 100.0

# R-GAT 파라미터
RGAT_IN_DIM = 15         # 15d enrichment: [lat, lon, alt, seq_ratio, speed, dist_prev, dist_next, bearing, bearing_change, hold_sec, alt_change, is_start, is_end, is_land, terrain]
RGAT_HIDDEN_DIM = 64
RGAT_OUT_DIM = 128
RGAT_HEADS = 4
N_EDGE_TYPES = 3         # Sequential, Proximity, Coverage

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 8
HDBSCAN_MIN_SAMPLES = 3

# 이상 탐지
KMEANS_K = 4
ISO_FOREST_N_ESTIMATORS = 100
ISO_FOREST_CONTAMINATION = 0.05
KNN_K = 10
W_CENTROID, W_ISOFOREST, W_KNN = 0.3, 0.4, 0.3

# 시각화
MAX_ROUTES = 2000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def interpret_score(score):
    if score < 0.3: return "normal"
    elif score < 0.5: return "caution"
    elif score < 0.7: return "anomaly"
    else: return "critical"


def offset_coords(lat, lon, offset_m, bearing_deg):
    """좌표를 특정 방향으로 offset_m 이동"""
    R = 6371000.0
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(offset_m / R) +
                      math.cos(lat1) * math.sin(offset_m / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(offset_m / R) * math.cos(lat1),
                              math.cos(offset_m / R) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


# ──────────────────────────────────────────────
# 1단계: 데이터 로드 + 필터링
# ──────────────────────────────────────────────

def load_and_filter(csv_path: Path) -> Dict[str, pd.DataFrame]:
    print("=" * 70)
    print("[1] 데이터 로드 + 필터링")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"  원본 WP: {len(df):,} / 경로: {df['wp_spot_id'].nunique():,}")

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

    total_wp = sum(len(v) for v in routes.values())
    print(f"  필터 후: {len(routes):,} 경로 / {total_wp:,} WP\n")
    return routes


# ──────────────────────────────────────────────
# 2단계: 합성 군집 시나리오 생성
# ──────────────────────────────────────────────

def classify_route(route_df):
    """경로 유형 자동 분류"""
    n = len(route_df)
    lats, lngs = route_df["wp_lat"].values, route_df["wp_lng"].values

    # 방위 변화 패턴
    bearing_changes = []
    for i in range(1, n - 1):
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


def generate_swarm_scenarios(routes: Dict[str, pd.DataFrame]) -> List[Dict]:
    """실 데이터에서 합성 군집 시나리오 생성"""
    print("=" * 70)
    print("[2] 합성 군집 시나리오 생성")
    print("=" * 70)

    route_keys = list(routes.keys())
    route_types = {k: classify_route(v) for k, v in routes.items()}

    # 유형별 분류
    by_type = defaultdict(list)
    for k, t in route_types.items():
        by_type[t].append(k)

    scenarios = []
    labels = []

    # === 시나리오 타입별 생성 ===

    # Type 1: 협력 수색 (같은 유형, 영역 분리) — 정상
    n_coop = N_SWARM_SCENARIOS // 4
    for i in range(n_coop):
        n_drones = np.random.choice(DRONES_PER_SWARM)
        # 같은 유형의 경로 선택
        route_type = np.random.choice(list(by_type.keys()))
        candidates = by_type[route_type]
        if len(candidates) < n_drones:
            candidates = route_keys
        selected = np.random.choice(candidates, size=n_drones, replace=False)

        # 영역 분리: 각 드론을 다른 방향으로 offset
        drone_routes = []
        for j, key in enumerate(selected):
            route_df = routes[key].copy()
            angle = (360 / n_drones) * j
            offset_dist = np.random.uniform(500, 2000)  # 500m~2km 분리
            route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                offset_coords(lat, lon, offset_dist, angle)
                for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])
            ])
            drone_routes.append({"drone_id": j, "route_key": key, "route_df": route_df})

        scenarios.append({
            "scenario_id": f"COOP_{i:03d}",
            "type": "cooperative_search",
            "label": "normal",
            "n_drones": n_drones,
            "drones": drone_routes
        })

    # Type 2: 충돌 위험 (경로 교차 의도 생성)
    n_conflict = N_SWARM_SCENARIOS // 4
    for i in range(n_conflict):
        n_drones = np.random.choice([3, 4])
        selected = np.random.choice(route_keys, size=n_drones, replace=False)

        drone_routes = []
        # 첫 두 드론의 경로를 겹치게 배치
        base_route = routes[selected[0]]
        mid_lat = base_route["wp_lat"].mean()
        mid_lon = base_route["wp_lng"].mean()

        for j, key in enumerate(selected):
            route_df = routes[key].copy()
            if j < 2:
                # 첫 두 드론은 같은 영역으로 수렴 (50~200m 이내)
                offset_dist = np.random.uniform(30, 150)
                angle = np.random.uniform(0, 360)
                route_df["wp_lat"] = mid_lat + (route_df["wp_lat"] - route_df["wp_lat"].mean())
                route_df["wp_lng"] = mid_lon + (route_df["wp_lng"] - route_df["wp_lng"].mean())
                route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                    offset_coords(lat, lon, offset_dist, angle)
                    for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])
                ])
            else:
                # 나머지는 멀리 배치
                offset_dist = np.random.uniform(1000, 3000)
                angle = np.random.uniform(0, 360)
                route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                    offset_coords(lat, lon, offset_dist, angle)
                    for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])
                ])

            drone_routes.append({"drone_id": j, "route_key": key, "route_df": route_df})

        scenarios.append({
            "scenario_id": f"CONF_{i:03d}",
            "type": "collision_risk",
            "label": "collision_risk",
            "n_drones": n_drones,
            "drones": drone_routes
        })

    # Type 3: 커버리지 중복 (같은 격자에 다수 드론)
    n_overlap = N_SWARM_SCENARIOS // 4
    for i in range(n_overlap):
        n_drones = np.random.choice([3, 4, 5])
        # 같은 유형의 격자 수색 경로들을 동일 위치에 배치
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
            # 모든 드론을 같은 중심으로 배치 (약간의 오프셋)
            route_df["wp_lat"] = center_lat + (route_df["wp_lat"] - route_df["wp_lat"].mean())
            route_df["wp_lng"] = center_lon + (route_df["wp_lng"] - route_df["wp_lng"].mean())
            small_offset = np.random.uniform(10, 80)
            angle = (360 / n_drones) * j
            route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                offset_coords(lat, lon, small_offset, angle)
                for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])
            ])
            drone_routes.append({"drone_id": j, "route_key": key, "route_df": route_df})

        scenarios.append({
            "scenario_id": f"OVLP_{i:03d}",
            "type": "coverage_overlap",
            "label": "coverage_overlap",
            "n_drones": n_drones,
            "drones": drone_routes
        })

    # Type 4: 릴레이 이동 (정상 — 직선 비행 연결)
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
            # 릴레이: 일렬로 간격 배치
            offset_dist = j * np.random.uniform(800, 1500)
            route_df["wp_lat"], route_df["wp_lng"] = zip(*[
                offset_coords(lat, lon, offset_dist, 90)  # 동쪽으로 일렬
                for lat, lon in zip(route_df["wp_lat"], route_df["wp_lng"])
            ])
            drone_routes.append({"drone_id": j, "route_key": key, "route_df": route_df})

        scenarios.append({
            "scenario_id": f"RLAY_{i:03d}",
            "type": "relay_move",
            "label": "normal",
            "n_drones": n_drones,
            "drones": drone_routes
        })

    np.random.shuffle(scenarios)

    # 통계
    type_counts = defaultdict(int)
    label_counts = defaultdict(int)
    for s in scenarios:
        type_counts[s["type"]] += 1
        label_counts[s["label"]] += 1

    print(f"  총 시나리오: {len(scenarios)}")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")
    print(f"  라벨 분포:")
    for l, c in sorted(label_counts.items()):
        print(f"    {l}: {c}")
    print()
    return scenarios


# ──────────────────────────────────────────────
# 3단계: 군집 그래프 구성 (3종 엣지)
# ──────────────────────────────────────────────

def build_swarm_graph(scenario: Dict) -> Tuple[Data, Dict]:
    """군집 시나리오 → PyG Data (3종 엣지)"""
    drones = scenario["drones"]
    all_nodes = []
    node_meta = []

    # 노드 수집
    for drone in drones:
        drone_id = drone["drone_id"]
        df = drone["route_df"]
        n = len(df)
        lats = df["wp_lat"].values
        lngs = df["wp_lng"].values
        alts = df["wp_alt"].values
        speeds = df["wp_speed"].values if "wp_speed" in df.columns else np.full(n, 2.0)
        headings = df["wp_heading"].values if "wp_heading" in df.columns else np.zeros(n)

        # 엣지별 거리/방위 사전 계산 (15d 피처용)
        dists_wp = [0.0] + [haversine(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bearings_wp = [0.0] + [bearing(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bearing_changes_wp = [0.0, 0.0] + [
            ((bearings_wp[j] - bearings_wp[j-1] + 180) % 360 - 180) for j in range(2, n)
        ]

        for i in range(n):
            _n = lambda v, lo, hi: max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-6)))
            dist_prev = dists_wp[i]
            dist_next = dists_wp[i + 1] if i < n - 1 else 0.0
            alt_change = alts[i] - alts[i - 1] if i > 0 else 0.0
            hold_sec = df["wp_wait"].values[i] if "wp_wait" in df.columns else 0.0

            node_feat = [
                _n(lats[i], 33.0, 38.0),           # lat_norm
                _n(lngs[i], 124.0, 132.0),          # lon_norm
                _n(alts[i], 0, 500),                 # alt_norm
                i / max(n - 1, 1),                   # seq_ratio
                _n(min(speeds[i], 15.0), 0, 15),     # speed_norm
                _n(dist_prev, 0, 1000),              # dist_prev_norm
                _n(dist_next, 0, 1000),              # dist_next_norm
                _n(bearings_wp[i], 0, 360),          # bearing_norm
                _n(abs(bearing_changes_wp[i]), 0, 180),  # bearing_change_norm
                _n(hold_sec, 0, 60),                 # hold_sec_norm
                _n(abs(alt_change), 0, 100),         # alt_change_norm
                1.0 if i == 0 else 0.0,              # is_start
                1.0 if i == n - 1 else 0.0,          # is_end
                0.0,                                  # is_land (CSV에 없음)
                0.0,                                  # terrain_type (placeholder)
            ]
            all_nodes.append(node_feat)
            node_meta.append({
                "drone_id": drone_id,
                "wp_idx": i,
                "lat": lats[i],
                "lon": lngs[i],
                "alt": alts[i],
                "global_idx": len(all_nodes) - 1
            })

    x = torch.tensor(all_nodes, dtype=torch.float32)
    n_nodes = len(all_nodes)

    # 엣지 수집 (타입별)
    edges_by_type = {0: [], 1: [], 2: []}  # 0=Sequential, 1=Proximity, 2=Coverage
    edge_attrs_by_type = {0: [], 1: [], 2: []}

    # === Sequential 엣지 (같은 드론 내 연속 WP) ===
    offset = 0
    for drone in drones:
        n_wp = len(drone["route_df"])
        for i in range(n_wp - 1):
            src, dst = offset + i, offset + i + 1
            s_meta, d_meta = node_meta[src], node_meta[dst]
            dist = haversine(s_meta["lat"], s_meta["lon"], d_meta["lat"], d_meta["lon"])
            alt_change = abs(d_meta["alt"] - s_meta["alt"])
            brng = bearing(s_meta["lat"], s_meta["lon"], d_meta["lat"], d_meta["lon"]) / 360.0

            attr = [dist / 5000.0, brng, alt_change / 100.0, 0, 0, 1.0]
            edges_by_type[0].append([src, dst])
            edge_attrs_by_type[0].append(attr)
            # 양방향
            edges_by_type[0].append([dst, src])
            edge_attrs_by_type[0].append(attr)
        offset += n_wp

    # === Proximity 엣지 (다른 드론 WP 간 근접) ===
    drone_ranges = []
    offset = 0
    for drone in drones:
        n_wp = len(drone["route_df"])
        drone_ranges.append((offset, offset + n_wp, drone["drone_id"]))
        offset += n_wp

    for i_range in range(len(drone_ranges)):
        for j_range in range(i_range + 1, len(drone_ranges)):
            s1, e1, did1 = drone_ranges[i_range]
            s2, e2, did2 = drone_ranges[j_range]
            for ni in range(s1, e1):
                for nj in range(s2, e2):
                    mi, mj = node_meta[ni], node_meta[nj]
                    dist = haversine(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
                    if dist < PROXIMITY_RADIUS_M:
                        alt_diff = abs(mi["alt"] - mj["alt"])
                        brng_val = bearing(mi["lat"], mi["lon"], mj["lat"], mj["lon"]) / 360.0
                        # TTC 근사 (단순 거리 기반)
                        ttc = dist / max(5.0, 1.0)  # 5m/s 가정
                        attr = [dist / PROXIMITY_RADIUS_M, brng_val, alt_diff / 100.0,
                                ttc / 100.0, 0, 0.5]
                        edges_by_type[1].append([ni, nj])
                        edge_attrs_by_type[1].append(attr)
                        edges_by_type[1].append([nj, ni])
                        edge_attrs_by_type[1].append(attr)

    # === Coverage 엣지 (같은 그리드 셀) ===
    grid_cells = defaultdict(list)
    for idx, meta in enumerate(node_meta):
        # 100m 그리드 셀 계산
        cell_lat = int(meta["lat"] * 1000)  # ~111m 단위
        cell_lon = int(meta["lon"] * 1000)  # ~90m 단위 (한국 위도)
        grid_cells[(cell_lat, cell_lon)].append(idx)

    for cell, indices in grid_cells.items():
        # 다른 드론의 WP가 같은 셀에 있으면 Coverage 엣지
        drone_groups = defaultdict(list)
        for idx in indices:
            drone_groups[node_meta[idx]["drone_id"]].append(idx)

        if len(drone_groups) >= 2:
            drone_ids = list(drone_groups.keys())
            for di in range(len(drone_ids)):
                for dj in range(di + 1, len(drone_ids)):
                    for ni in drone_groups[drone_ids[di]]:
                        for nj in drone_groups[drone_ids[dj]]:
                            attr = [0, 0, 0, 0, 0, 0.3]
                            edges_by_type[2].append([ni, nj])
                            edge_attrs_by_type[2].append(attr)
                            edges_by_type[2].append([nj, ni])
                            edge_attrs_by_type[2].append(attr)

    # PyG Data 구성
    all_edges = []
    all_edge_attrs = []
    all_edge_types = []

    for etype in range(N_EDGE_TYPES):
        edges = edges_by_type[etype]
        attrs = edge_attrs_by_type[etype]
        all_edges.extend(edges)
        all_edge_attrs.extend(attrs)
        all_edge_types.extend([etype] * len(edges))

    if len(all_edges) == 0:
        # 최소 self-loop 추가
        all_edges = [[0, 0]]
        all_edge_attrs = [[0, 0, 0, 0, 0, 0]]
        all_edge_types = [0]

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(all_edge_attrs, dtype=torch.float32)
    edge_type = torch.tensor(all_edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_type = edge_type
    data.n_nodes = n_nodes

    # 엣지 통계
    stats = {
        "n_nodes": n_nodes,
        "n_edges_sequential": len(edges_by_type[0]),
        "n_edges_proximity": len(edges_by_type[1]),
        "n_edges_coverage": len(edges_by_type[2]),
        "n_edges_total": len(all_edges),
        "node_meta": node_meta,
        "drone_ranges": drone_ranges
    }
    return data, stats


# ──────────────────────────────────────────────
# 4단계: R-GAT 모델 (Relational Graph Attention Network)
# ──────────────────────────────────────────────

class RelationalGAT(nn.Module):
    """
    R-GAT: 엣지 타입별 독립 GATConv + 타입별 가중합
    3종 엣지(Sequential/Proximity/Coverage)에 대해 독립적 어텐션 학습
    """
    def __init__(self, in_dim=RGAT_IN_DIM, hidden_dim=RGAT_HIDDEN_DIM,
                 out_dim=RGAT_OUT_DIM, heads=RGAT_HEADS, n_edge_types=N_EDGE_TYPES):
        super().__init__()
        self.n_edge_types = n_edge_types

        # 엣지 타입별 독립 GAT Layer 1
        self.gat1_layers = nn.ModuleList([
            GATConv(in_dim, hidden_dim, heads=heads, concat=False, dropout=0.1)
            for _ in range(n_edge_types)
        ])
        # 타입별 가중치 (학습 가능)
        self.type_weights1 = nn.Parameter(torch.ones(n_edge_types) / n_edge_types)

        # 엣지 타입별 독립 GAT Layer 2
        self.gat2_layers = nn.ModuleList([
            GATConv(hidden_dim, out_dim, heads=heads, concat=False, dropout=0.1)
            for _ in range(n_edge_types)
        ])
        self.type_weights2 = nn.Parameter(torch.ones(n_edge_types) / n_edge_types)

        # 멀티태스크 헤드
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 32), nn.ReLU(),
            nn.Linear(32, 4)  # 4 패턴: cooperative_search, collision_risk, coverage_overlap, relay_move
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(out_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        attention_weights = {}

        # Layer 1: 타입별 GAT + 가중합
        w1 = F.softmax(self.type_weights1, dim=0)
        h1 = torch.zeros(x.size(0), RGAT_HIDDEN_DIM)
        for etype in range(self.n_edge_types):
            mask = (edge_type == etype)
            if mask.sum() == 0:
                continue
            etype_edges = edge_index[:, mask]
            out, (edge_idx, attn) = self.gat1_layers[etype](x, etype_edges, return_attention_weights=True)
            h1 = h1 + w1[etype] * out
            attention_weights[f"layer1_type{etype}"] = attn.detach()

        h1 = F.relu(h1)

        # Layer 2: 타입별 GAT + 가중합
        w2 = F.softmax(self.type_weights2, dim=0)
        h2 = torch.zeros(h1.size(0), RGAT_OUT_DIM)
        for etype in range(self.n_edge_types):
            mask = (edge_type == etype)
            if mask.sum() == 0:
                continue
            etype_edges = edge_index[:, mask]
            out, (edge_idx, attn) = self.gat2_layers[etype](h1, etype_edges, return_attention_weights=True)
            h2 = h2 + w2[etype] * out
            attention_weights[f"layer2_type{etype}"] = attn.detach()

        # 그래프 풀링 (가중 평균)
        graph_emb = h2.mean(dim=0)  # [128d]

        # 멀티태스크 출력
        pattern_logits = self.classifier(graph_emb.unsqueeze(0))  # [1, 4]
        anomaly_scores = self.anomaly_head(h2)  # [n_nodes, 1]

        return h2, graph_emb, pattern_logits, anomaly_scores, attention_weights


# ──────────────────────────────────────────────
# 5단계: 전체 분석 파이프라인
# ──────────────────────────────────────────────

def run_smgat_pipeline(scenarios: List[Dict]) -> Tuple[np.ndarray, pd.DataFrame, List[Dict]]:
    """전체 시나리오에 대해 R-GAT 임베딩 + 분석"""
    print("=" * 70)
    print("[3] R-GAT 임베딩 생성")
    print("=" * 70)

    model = RelationalGAT()
    model.eval()  # 비학습 모드 (random init → message passing)

    embeddings = []
    all_stats = []
    all_attention = []
    pattern_labels = []
    edge_summaries = []

    t0 = time.time()
    for idx, scenario in enumerate(scenarios):
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {idx + 1}/{len(scenarios)} 처리 중... ({elapsed:.1f}s)")

        try:
            data, stats = build_swarm_graph(scenario)

            with torch.no_grad():
                node_emb, graph_emb, pattern_logits, anomaly_scores, attn = model(data)

            embeddings.append(graph_emb.numpy())
            pattern_labels.append(scenario["type"])

            # 어텐션 가중치 저장 (상위 anomaly 시나리오용)
            type_weights = F.softmax(model.type_weights2, dim=0).detach().numpy()

            all_stats.append({
                "scenario_id": scenario["scenario_id"],
                "type": scenario["type"],
                "label": scenario["label"],
                "n_drones": scenario["n_drones"],
                "n_nodes": stats["n_nodes"],
                "n_edges_seq": stats["n_edges_sequential"],
                "n_edges_prox": stats["n_edges_proximity"],
                "n_edges_cov": stats["n_edges_coverage"],
                "n_edges_total": stats["n_edges_total"],
                "mean_node_anomaly": anomaly_scores.mean().item(),
                "max_node_anomaly": anomaly_scores.max().item(),
                "pattern_pred": ["cooperative_search", "collision_risk",
                                 "coverage_overlap", "relay_move"][pattern_logits.argmax().item()],
                "type_weight_seq": type_weights[0],
                "type_weight_prox": type_weights[1],
                "type_weight_cov": type_weights[2],
            })

            edge_summaries.append({
                "scenario_id": scenario["scenario_id"],
                "proximity_ratio": stats["n_edges_proximity"] / max(stats["n_edges_total"], 1),
                "coverage_ratio": stats["n_edges_coverage"] / max(stats["n_edges_total"], 1),
            })

        except Exception as e:
            print(f"  [WARN] {scenario['scenario_id']} 실패: {e}")
            continue

    elapsed = time.time() - t0
    print(f"  완료: {len(embeddings)} 시나리오, {elapsed:.1f}s")

    emb_matrix = np.array(embeddings)
    stats_df = pd.DataFrame(all_stats)

    print(f"  임베딩 shape: {emb_matrix.shape}")
    print()

    return emb_matrix, stats_df, edge_summaries


# ──────────────────────────────────────────────
# 6단계: 3-Method 앙상블 이상 탐지 (군집 레벨)
# ──────────────────────────────────────────────

def run_anomaly_detection(emb_matrix: np.ndarray, stats_df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("[4] 3-Method 앙상블 이상 탐지 (군집 레벨)")
    print("=" * 70)

    n = len(emb_matrix)

    # Method 1: Centroid Distance
    kmeans = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_SEED, n_init=10)
    clusters = kmeans.fit_predict(emb_matrix)
    centroid_dists = np.zeros(n)
    for i in range(n):
        centroid_dists[i] = np.linalg.norm(emb_matrix[i] - kmeans.cluster_centers_[clusters[i]])
    cd_min, cd_max = centroid_dists.min(), centroid_dists.max()
    if cd_max - cd_min > 1e-9:
        centroid_scores = (centroid_dists - cd_min) / (cd_max - cd_min)
    else:
        centroid_scores = np.zeros(n)

    # Method 2: Isolation Forest
    iso = IsolationForest(n_estimators=ISO_FOREST_N_ESTIMATORS,
                          contamination=ISO_FOREST_CONTAMINATION,
                          random_state=RANDOM_SEED)
    iso.fit(emb_matrix)
    raw_iso = iso.decision_function(emb_matrix)
    iso_min, iso_max = raw_iso.min(), raw_iso.max()
    if iso_max - iso_min > 1e-9:
        iso_scores = 1.0 - (raw_iso - iso_min) / (iso_max - iso_min)
    else:
        iso_scores = np.zeros(n)

    # Method 3: KNN Average Distance
    cos_dist = cosine_distances(emb_matrix)
    np.fill_diagonal(cos_dist, np.inf)
    knn_scores = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(cos_dist[i])
        k = min(KNN_K, n - 1)
        knn_scores[i] = np.mean(sorted_dists[:k])
    knn_min, knn_max = knn_scores.min(), knn_scores.max()
    if knn_max - knn_min > 1e-9:
        knn_scores = (knn_scores - knn_min) / (knn_max - knn_min)
    else:
        knn_scores = np.zeros(n)

    # 앙상블
    ensemble = W_CENTROID * centroid_scores + W_ISOFOREST * iso_scores + W_KNN * knn_scores

    stats_df["centroid_score"] = centroid_scores
    stats_df["isoforest_score"] = iso_scores
    stats_df["knn_score"] = knn_scores
    stats_df["ensemble_score"] = ensemble
    stats_df["interpretation"] = [interpret_score(s) for s in ensemble]

    print(f"  앙상블 점수 통계:")
    print(f"    mean={ensemble.mean():.3f}, std={ensemble.std():.3f}")
    print(f"    p50={np.percentile(ensemble, 50):.3f}, p95={np.percentile(ensemble, 95):.3f}")
    print(f"    max={ensemble.max():.3f}")

    # 라벨별 통계
    for label in stats_df["label"].unique():
        mask = stats_df["label"] == label
        scores = ensemble[mask]
        print(f"    {label}: mean={scores.mean():.3f}, max={scores.max():.3f}")
    print()

    return stats_df


# ──────────────────────────────────────────────
# 7단계: HDBSCAN 클러스터링
# ──────────────────────────────────────────────

def run_clustering(emb_matrix: np.ndarray, stats_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print("[5] HDBSCAN 클러스터링")
    print("=" * 70)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(emb_matrix)
    stats_df["hdbscan_cluster"] = labels

    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = (labels == -1).sum()

    print(f"  클러스터 수: {n_clusters}")
    print(f"  Noise: {n_noise} ({100*n_noise/len(labels):.1f}%)")

    # 클러스터 요약
    cluster_rows = []
    for cl in unique_labels:
        mask = labels == cl
        cl_scores = stats_df.loc[mask, "ensemble_score"]
        cl_types = stats_df.loc[mask, "type"].value_counts()
        dominant_type = cl_types.index[0] if len(cl_types) > 0 else "unknown"

        cluster_rows.append({
            "cluster": f"C{cl}" if cl >= 0 else "Noise",
            "count": mask.sum(),
            "pct": f"{100 * mask.sum() / len(labels):.1f}",
            "score_mean": f"{cl_scores.mean():.3f}",
            "score_max": f"{cl_scores.max():.3f}",
            "dominant_type": dominant_type,
            "type_distribution": dict(cl_types),
            "n_prox_edges_avg": f"{stats_df.loc[mask, 'n_edges_prox'].mean():.0f}",
            "n_cov_edges_avg": f"{stats_df.loc[mask, 'n_edges_cov'].mean():.0f}",
        })

    cluster_df = pd.DataFrame(cluster_rows)

    # 상위 클러스터 출력
    print(f"\n  상위 클러스터:")
    for _, row in cluster_df.head(10).iterrows():
        print(f"    {row['cluster']}: {row['count']}건, "
              f"score={row['score_mean']}, "
              f"type={row['dominant_type']}, "
              f"prox={row['n_prox_edges_avg']}, cov={row['n_cov_edges_avg']}")
    print()

    return stats_df, cluster_df


# ──────────────────────────────────────────────
# 8단계: UMAP 시각화
# ──────────────────────────────────────────────

def run_visualization(emb_matrix: np.ndarray, stats_df: pd.DataFrame,
                      cluster_df: pd.DataFrame, edge_summaries: List[Dict]):
    print("=" * 70)
    print("[6] 시각화 생성")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- UMAP 2D ---
    print("  UMAP 2D...")
    reducer_2d = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    umap_2d = reducer_2d.fit_transform(emb_matrix)
    np.save(RESULTS_DIR / "umap_2d.npy", umap_2d)

    # --- UMAP 3D ---
    print("  UMAP 3D...")
    reducer_3d = umap.UMAP(n_components=3, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    umap_3d = reducer_3d.fit_transform(emb_matrix)
    np.save(RESULTS_DIR / "umap_3d.npy", umap_3d)

    # --- Plot 1: UMAP 2D by scenario type ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    type_colors = {
        "cooperative_search": "#2ecc71",
        "collision_risk": "#e74c3c",
        "coverage_overlap": "#f39c12",
        "relay_move": "#3498db"
    }

    ax = axes[0]
    for stype, color in type_colors.items():
        mask = stats_df["type"] == stype
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=color, label=stype, s=20, alpha=0.7)
    ax.set_title("UMAP 2D — Swarm Scenario Types", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # --- Plot 2: UMAP 2D by anomaly score ---
    ax = axes[1]
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1],
                    c=stats_df["ensemble_score"], cmap="RdYlGn_r", s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Ensemble Anomaly Score")
    ax.set_title("UMAP 2D — Anomaly Scores", fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "umap_swarm_types.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: umap_swarm_types.png")

    # --- Plot 3: HDBSCAN 클러스터 ---
    fig, ax = plt.subplots(figsize=(12, 8))
    clusters = stats_df["hdbscan_cluster"].values
    unique_clusters = sorted(set(clusters))
    colors = cm.tab20(np.linspace(0, 1, max(len(unique_clusters), 1)))

    for idx, cl in enumerate(unique_clusters):
        mask = clusters == cl
        label = "Noise" if cl == -1 else f"C{cl}"
        color = "gray" if cl == -1 else colors[idx % len(colors)]
        alpha = 0.3 if cl == -1 else 0.7
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=[color], label=label, s=15, alpha=alpha)

    ax.set_title(f"UMAP 2D — HDBSCAN Clusters ({len([c for c in unique_clusters if c >= 0])} clusters)", fontsize=14)
    ax.legend(fontsize=7, ncol=3, loc="best")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "umap_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: umap_clusters.png")

    # --- Plot 4: 분석 서머리 (4패널) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 4a: 타입별 앙상블 점수 분포
    ax = axes[0, 0]
    types = ["cooperative_search", "collision_risk", "coverage_overlap", "relay_move"]
    type_scores = [stats_df.loc[stats_df["type"] == t, "ensemble_score"].values for t in types]
    bp = ax.boxplot(type_scores, labels=["Coop\nSearch", "Collision\nRisk", "Coverage\nOverlap", "Relay\nMove"],
                    patch_artist=True)
    colors_box = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Ensemble Anomaly Score")
    ax.set_title("Anomaly Score by Scenario Type")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Anomaly threshold")
    ax.legend()

    # 4b: 엣지 타입 분포
    ax = axes[0, 1]
    edge_data = pd.DataFrame(edge_summaries)
    if len(edge_data) > 0:
        for stype, color in type_colors.items():
            mask = stats_df["type"] == stype
            if mask.sum() == 0:
                continue
            indices = stats_df.index[mask]
            valid_indices = [i for i in indices if i < len(edge_data)]
            if valid_indices:
                ax.scatter(edge_data.loc[valid_indices, "proximity_ratio"],
                           edge_data.loc[valid_indices, "coverage_ratio"],
                           c=color, label=stype, s=20, alpha=0.6)
    ax.set_xlabel("Proximity Edge Ratio")
    ax.set_ylabel("Coverage Edge Ratio")
    ax.set_title("Edge Type Distribution by Scenario")
    ax.legend(fontsize=8)

    # 4c: 드론 수별 anomaly 분포
    ax = axes[1, 0]
    for nd in sorted(stats_df["n_drones"].unique()):
        mask = stats_df["n_drones"] == nd
        scores = stats_df.loc[mask, "ensemble_score"]
        ax.hist(scores, bins=20, alpha=0.5, label=f"{nd} drones")
    ax.set_xlabel("Ensemble Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Anomaly Distribution by Drone Count")
    ax.legend()

    # 4d: 3-Method 비교
    ax = axes[1, 1]
    methods = ["centroid_score", "isoforest_score", "knn_score"]
    method_names = ["Centroid Dist", "Isolation Forest", "KNN Avg Dist"]
    for method, name in zip(methods, method_names):
        ax.hist(stats_df[method], bins=30, alpha=0.4, label=name)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("3-Method Score Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "analysis_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: analysis_summary.png")

    # --- Plot 5: R-GAT 어텐션 가중치 분석 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    type_names = ["Sequential", "Proximity", "Coverage"]
    for stype in types:
        mask = stats_df["type"] == stype
        weights = stats_df.loc[mask, ["type_weight_seq", "type_weight_prox", "type_weight_cov"]].mean()
        ax.bar([f"{stype}\n{tn}" for tn in type_names], weights.values,
               color=type_colors[stype], alpha=0.7, width=0.6)
    ax.set_ylabel("R-GAT Type Weight (softmax)")
    ax.set_title("R-GAT Edge Type Attention Weights by Scenario")
    ax.tick_params(axis='x', labelsize=7)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rgat_attention_weights.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: rgat_attention_weights.png")

    # --- 3D UMAP HTML (plotly) ---
    try:
        import plotly.express as px
        plotly_df = pd.DataFrame({
            "UMAP-1": umap_3d[:, 0],
            "UMAP-2": umap_3d[:, 1],
            "UMAP-3": umap_3d[:, 2],
            "type": stats_df["type"],
            "score": stats_df["ensemble_score"],
            "scenario_id": stats_df["scenario_id"],
            "n_drones": stats_df["n_drones"],
        })
        fig3d = px.scatter_3d(plotly_df, x="UMAP-1", y="UMAP-2", z="UMAP-3",
                              color="type", hover_data=["scenario_id", "score", "n_drones"],
                              title="SM-GAT Swarm Mission Embeddings (3D UMAP)",
                              color_discrete_map=type_colors, opacity=0.7)
        fig3d.write_html(str(RESULTS_DIR / "umap_3d.html"))
        print(f"  저장: umap_3d.html")
    except ImportError:
        print("  [SKIP] plotly 미설치 — 3D HTML 생략")

    # --- Plot 6: 상위 anomaly 시나리오 상세 ---
    fig, ax = plt.subplots(figsize=(14, 6))
    top_anomaly = stats_df.nlargest(15, "ensemble_score")
    bars = ax.barh(range(len(top_anomaly)),
                   top_anomaly["ensemble_score"],
                   color=[type_colors.get(t, "gray") for t in top_anomaly["type"]])
    ax.set_yticks(range(len(top_anomaly)))
    ax.set_yticklabels([f"{row['scenario_id']} ({row['type'][:8]}, {row['n_drones']}d)"
                        for _, row in top_anomaly.iterrows()], fontsize=8)
    ax.set_xlabel("Ensemble Anomaly Score")
    ax.set_title("Top 15 Anomalous Swarm Scenarios")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_anomaly_scenarios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: top_anomaly_scenarios.png")

    print()


# ──────────────────────────────────────────────
# 9단계: CSV 저장
# ──────────────────────────────────────────────

def save_results(emb_matrix, stats_df, cluster_df):
    print("=" * 70)
    print("[7] 결과 저장")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 벡터
    np.save(RESULTS_DIR / "swarm_vectors.npy", emb_matrix)
    print(f"  저장: swarm_vectors.npy ({emb_matrix.shape})")

    # 전체 결과 CSV
    out_cols = ["scenario_id", "type", "label", "n_drones", "n_nodes",
                "n_edges_seq", "n_edges_prox", "n_edges_cov", "n_edges_total",
                "mean_node_anomaly", "max_node_anomaly", "pattern_pred",
                "centroid_score", "isoforest_score", "knn_score",
                "ensemble_score", "interpretation", "hdbscan_cluster",
                "type_weight_seq", "type_weight_prox", "type_weight_cov"]
    valid_cols = [c for c in out_cols if c in stats_df.columns]
    stats_df[valid_cols].to_csv(RESULTS_DIR / "swarm_analysis_full.csv", index=False)
    print(f"  저장: swarm_analysis_full.csv ({len(stats_df)} rows)")

    # 클러스터 요약
    cluster_df.to_csv(RESULTS_DIR / "cluster_summary.csv", index=False)
    print(f"  저장: cluster_summary.csv ({len(cluster_df)} rows)")

    # 상위 anomaly 상세
    top = stats_df.nlargest(20, "ensemble_score")
    top[valid_cols].to_csv(RESULTS_DIR / "top_anomaly_detail.csv", index=False)
    print(f"  저장: top_anomaly_detail.csv (top 20)")

    # 타입별 통계
    type_summary = stats_df.groupby("type").agg({
        "ensemble_score": ["mean", "std", "max"],
        "n_edges_prox": "mean",
        "n_edges_cov": "mean",
        "n_drones": "mean"
    }).round(3)
    type_summary.to_csv(RESULTS_DIR / "type_summary.csv")
    print(f"  저장: type_summary.csv")

    print()


# ──────────────────────────────────────────────
# 10단계: README 생성
# ──────────────────────────────────────────────

def generate_readme(stats_df, cluster_df):
    """결과 해석 README 자동 생성"""

    n_total = len(stats_df)
    n_clusters = len(cluster_df[cluster_df["cluster"] != "Noise"])
    noise_row = cluster_df[cluster_df["cluster"] == "Noise"]
    noise_pct = noise_row["pct"].values[0] if len(noise_row) > 0 else "0"

    ens = stats_df["ensemble_score"]
    p95 = np.percentile(ens, 95)

    # 타입별 통계
    type_stats = {}
    for t in stats_df["type"].unique():
        mask = stats_df["type"] == t
        scores = ens[mask]
        type_stats[t] = {"mean": scores.mean(), "max": scores.max(), "count": mask.sum()}

    # 상위 anomaly
    top10 = stats_df.nlargest(10, "ensemble_score")

    readme = f"""# report4 — SM-GAT 기반 군집 드론 임무 분석 결과 해석

> 2026-05-01 | R-GAT (Relational Graph Attention Network) | {n_total}건 합성 군집 시나리오

---

## 1. 분석 개요

기존 Mission2Vec(report1~3)이 **단일 드론 경로**를 분석한 반면, SM-GAT(report4)는 **다수 드론의 군집 임무**를 분석한다.
2,068건 실 관제 데이터에서 3~5대 드론 조합으로 {n_total}건 합성 군집 시나리오를 생성하고,
R-GAT(3종 엣지: Sequential/Proximity/Coverage)로 군집 임베딩 128d를 생성한 후
3-Method 앙상블 이상 탐지를 수행하였다.

---

## 2. Mission2Vec vs SM-GAT 비교

| 항목 | Mission2Vec (report1~3) | SM-GAT (report4) |
|------|------------------------|-------------------|
| **분석 단위** | 단일 경로 | **군집 (3~5대)** |
| **모델** | Node2Vec / GCN | **R-GAT (3종 엣지)** |
| **데이터** | 2,068건 실 경로 | **{n_total}건 합성 군집** |
| **클러스터** | 63개 (GCN) | **{n_clusters}개** |
| **Noise 비율** | 25.1% (GCN) | **{noise_pct}%** |
| **Anomaly 평균** | 0.155 (GCN) | **{ens.mean():.3f}** |
| **95%ile 임계값** | 0.481 (GCN) | **{p95:.3f}** |

---

## 3. 시나리오 타입별 분석

| 타입 | 건수 | anomaly avg | anomaly max | 해석 |
|------|------|------------|------------|------|
"""
    for t in ["cooperative_search", "collision_risk", "coverage_overlap", "relay_move"]:
        if t in type_stats:
            s = type_stats[t]
            desc_map = {
                "cooperative_search": "영역 분리 정상 운용",
                "collision_risk": "경로 교차/근접 위험",
                "coverage_overlap": "수색 영역 중복",
                "relay_move": "일렬 릴레이 정상 이동"
            }
            readme += f"| **{t}** | {s['count']} | {s['mean']:.3f} | {s['max']:.3f} | {desc_map.get(t, '')} |\n"

    readme += f"""
### 해석

- **collision_risk**: Proximity 엣지가 많아 근접 WP가 다수 → 앙상블 점수 높음 예상
- **coverage_overlap**: Coverage 엣지가 많아 같은 그리드에 다수 드론 → 중복 탐지
- **cooperative_search / relay_move**: 영역 분리 또는 일렬 배치로 정상 패턴 → 점수 낮음

---

## 4. 상위 anomaly 시나리오

```
"""
    for _, row in top10.iterrows():
        readme += f"  {row['scenario_id']:20s}  score={row['ensemble_score']:.3f}  type={row['type']:20s}  drones={row['n_drones']}  prox={row['n_edges_prox']}  cov={row['n_edges_cov']}\n"

    readme += f"""```

---

## 5. R-GAT 3종 엣지 분석

| 엣지 타입 | 의미 | 평균 엣지 수 |
|-----------|------|-------------|
| **Sequential** | 같은 드론 내 WP 순서 | {stats_df['n_edges_seq'].mean():.0f} |
| **Proximity** | 다른 드론 간 500m 이내 | {stats_df['n_edges_prox'].mean():.0f} |
| **Coverage** | 같은 100m 그리드 셀 | {stats_df['n_edges_cov'].mean():.0f} |

Proximity와 Coverage 엣지가 많을수록 충돌 위험/수색 중복 가능성 높음.

---

## 6. UMAP 시각화 해석

### umap_swarm_types.png
좌: 시나리오 타입별 색상 구분 — 4종 시나리오가 UMAP 공간에서 얼마나 분리되는지 확인
우: anomaly score 히트맵 — 붉은 점이 이상 시나리오

### umap_clusters.png
HDBSCAN {n_clusters}개 클러스터 — 군집 임무 패턴의 자동 분류 결과

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
| `swarm_analysis_full.csv` | {n_total}건 전체 분석 결과 |
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
"""

    with open(RESULTS_DIR / "README-results.md", "w") as f:
        f.write(readme)
    print(f"  저장: README-results.md")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  SM-GAT: Swarm Mission Graph Attention Network Analysis            ║")
    print("║  R-GAT (3종 엣지) + 합성 군집 시나리오 + 3-Method 앙상블          ║")
    print("╚" + "═" * 68 + "╝")
    print()

    t_start = time.time()

    # 1. 데이터 로드
    routes = load_and_filter(DATA_PATH)

    # 2. 군집 시나리오 생성
    scenarios = generate_swarm_scenarios(routes)

    # 3. R-GAT 임베딩
    emb_matrix, stats_df, edge_summaries = run_smgat_pipeline(scenarios)

    # 4. 이상 탐지
    stats_df = run_anomaly_detection(emb_matrix, stats_df)

    # 5. 클러스터링
    stats_df, cluster_df = run_clustering(emb_matrix, stats_df)

    # 6. 시각화
    run_visualization(emb_matrix, stats_df, cluster_df, edge_summaries)

    # 7. 저장
    save_results(emb_matrix, stats_df, cluster_df)

    # 8. README
    generate_readme(stats_df, cluster_df)

    elapsed = time.time() - t_start
    print("=" * 70)
    print(f"  완료! 총 {elapsed:.1f}s")
    print(f"  결과: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
