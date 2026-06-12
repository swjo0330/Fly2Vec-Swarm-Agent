#!/usr/bin/env python3
"""v3 종합 실험 스크립트 — 6개 실험을 단일 스크립트로 실행.

v3 합성 데이터(gen_scenario_v3)를 사용하여 교수 피드백 누락 항목을 모두 보완:

  Exp 1: E1 엣지 Ablation (7 조건, 15d R-GAT)
  Exp 2: GCN/GAT Baseline 비교 (15d, E-SPC)
  Exp 3: E1-20d 엣지 Ablation (3 조건, 20d R-GAT)
  Exp 4: 노드 피처 그룹 Ablation (5 그룹, 20d, E-SPC)
  Exp 5: E7 드론 수 변화 (2, 3, 5대)
  Exp 6: Case Study — FN 그래프 구조 분석

실행:
  cd /Users/.../proposal
  fly2vec/.venv/bin/python scripts/run_v3_comprehensive.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import math, json, random, time, pickle, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ── v3 시나리오 생성 ──
from swarm_scenario_gen import gen_scenario_v3 as gen_scenario

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "fly2vec" / "data" / "dataset_wp_spot.csv"
MODEL_DIR = PROJECT_ROOT / "fly2vec" / "data" / "models"
REPORT_DIR = PROJECT_ROOT / "results" / "report12"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 공통 파라미터 ──
HIDDEN_DIM = 64
OUT_DIM = 32
N_EDGE_TYPES = 5
HEADS = 4
N_SCENARIOS = 3000
EPOCHS = 50
LR = 0.001
PROXIMITY_RADIUS_M = 500.0
COVERAGE_GRID_M = 100.0

EDGE_NAMES = ['Sequential', 'Proximity', 'Coverage', 'Temporal', 'Anomaly']
TYPE_MAP = {"cooperative_search": 0, "relay_move": 1, "collision_risk": 2, "coverage_overlap": 3}
ANOMALY_MAP = {"cooperative_search": 0, "relay_move": 0, "collision_risk": 1, "coverage_overlap": 1}
CLASS_NAMES = ["cooperative_search", "relay_move", "collision_risk", "coverage_overlap"]

# ── 7개 E1 실험 조건 ──
CONDITIONS_7 = [
    {"id": "E-S",    "name": "경로 순서 단독",         "active": {0}},
    {"id": "E-SP",   "name": "순서+근접",              "active": {0, 1}},
    {"id": "E-SPC",  "name": "순서+근접+커버리지",      "active": {0, 1, 2}},
    {"id": "E-SPCT", "name": "순서+근접+커버리지+시간",  "active": {0, 1, 2, 3}},
    {"id": "E-ALL",  "name": "전체(5종)",              "active": {0, 1, 2, 3, 4}},
    {"id": "E-P",    "name": "근접 단독",              "active": {1}},
    {"id": "E-C",    "name": "커버리지 단독",           "active": {2}},
]

# ── 3개 20d 핵심 조건 ──
CONDITIONS_3 = [
    {"id": "E-SPC",  "name": "순서+근접+커버리지",      "active": {0, 1, 2}},
    {"id": "E-SPCT", "name": "순서+근접+커버리지+시간",  "active": {0, 1, 2, 3}},
    {"id": "E-ALL",  "name": "전체(5종)",              "active": {0, 1, 2, 3, 4}},
]

# ── 피처 그룹 ablation (20d 기준) ──
FEATURE_GROUPS = [
    {"id": "no-position",   "name": "위치 제거 (lat, lon)",         "indices": [0, 1]},
    {"id": "no-altitude",   "name": "고도 제거 (alt, alt_change)",   "indices": [2, 10]},
    {"id": "no-speed-dist", "name": "속도+거리 제거",                "indices": [4, 5, 6]},
    {"id": "no-direction",  "name": "방향 제거 (bearing, bc)",       "indices": [7, 8]},
    {"id": "no-inter-drone","name": "드론간 관계 제거 (5개)",         "indices": [15, 16, 17, 18, 19]},
]


# ═══════════════════════════════════════════════════════════
#  유틸리티 함수
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
#  모델 정의
# ═══════════════════════════════════════════════════════════

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


class GCNBaseline(nn.Module):
    """GCN 베이스라인 — 엣지 타입 무시, 2층 GCNConv."""
    def __init__(self, in_dim=15):
        super().__init__()
        self.conv1 = GCNConv(in_dim, HIDDEN_DIM)
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, OUT_DIM)
        self.pool = AttnPool(OUT_DIM)
        self.th = nn.Linear(OUT_DIM, 4)
        self.ah = nn.Linear(OUT_DIM, 1)

    def forward(self, data):
        h = F.relu(self.bn(self.conv1(data.x, data.edge_index)))
        h = self.conv2(h, data.edge_index)
        emb = self.pool(h)
        return self.th(emb), torch.sigmoid(self.ah(emb)).squeeze(), emb


class GATBaseline(nn.Module):
    """GAT 베이스라인 — 엣지 타입 무시, 단일 GATConv(heads=4)."""
    def __init__(self, in_dim=15):
        super().__init__()
        self.conv1 = GATConv(in_dim, HIDDEN_DIM, heads=4, concat=False, dropout=0.1)
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.conv2 = GATConv(HIDDEN_DIM, OUT_DIM, heads=4, concat=False, dropout=0.1)
        self.pool = AttnPool(OUT_DIM)
        self.th = nn.Linear(OUT_DIM, 4)
        self.ah = nn.Linear(OUT_DIM, 1)

    def forward(self, data):
        h = F.relu(self.bn(self.conv1(data.x, data.edge_index)))
        h = self.conv2(h, data.edge_index)
        emb = self.pool(h)
        return self.th(emb), torch.sigmoid(self.ah(emb)).squeeze(), emb


# ═══════════════════════════════════════════════════════════
#  데이터 로드 & 시나리오 생성
# ═══════════════════════════════════════════════════════════

def load_routes():
    """실 관제 CSV에서 유효 경로 로드."""
    df = pd.read_csv(DATA_PATH)
    mask = ((df["wp_lat"] >= 33) & (df["wp_lat"] <= 38) &
            (df["wp_lng"] >= 124) & (df["wp_lng"] <= 132) &
            (df["wp_alt"] >= 0) & (df["wp_alt"] <= 500))
    df = df[mask]
    grouped = {n: g.sort_values("wp_seq").reset_index(drop=True)
               for n, g in df.groupby("wp_spot_id")}
    routes = {k: v for k, v in grouped.items() if len(v) >= 4}
    return routes


def generate_raw_scenarios(routes, rk, n_scenarios=N_SCENARIOS, cache_name="raw_scenarios_v3.pkl",
                           fixed_n_drones=None):
    """v3 시나리오 생성 (캐시 지원). fixed_n_drones: E7용 고정 드론 수."""
    cache_path = REPORT_DIR / cache_name
    if cache_path.exists():
        print(f"  캐시 로드: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    scenarios = []
    for i in range(n_scenarios):
        drones, typ = gen_scenario(i, routes, rk)
        # 드론 수 고정 시: 생성 후 자르거나 재생성
        if fixed_n_drones is not None:
            # 간단히 재생성 — gen_scenario_v3 내부의 random.choice를 오버라이드
            while len(drones) != fixed_n_drones:
                drones, typ = gen_scenario(i * 10000 + len(scenarios), routes, rk)
        scenarios.append({"drones": drones, "type": typ})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_scenarios} 시나리오 생성")

    with open(cache_path, "wb") as f:
        pickle.dump(scenarios, f)
    print(f"  캐시 저장: {cache_path} ({len(scenarios)}건)")
    return scenarios


def generate_fixed_drone_scenarios(routes, rk, n_drones, n_scenarios=1000):
    """E7용: 정확히 n_drones 드론 수를 가진 시나리오만 생성."""
    cache_name = f"raw_scenarios_v3_{n_drones}d.pkl"
    cache_path = REPORT_DIR / cache_name
    if cache_path.exists():
        print(f"  캐시 로드: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    scenarios = []
    attempts = 0
    max_attempts = n_scenarios * 20  # 충분한 시도
    while len(scenarios) < n_scenarios and attempts < max_attempts:
        attempts += 1
        drones, typ = gen_scenario(attempts, routes, rk)
        if len(drones) == n_drones:
            scenarios.append({"drones": drones, "type": typ})
            if len(scenarios) % 200 == 0:
                print(f"  {len(scenarios)}/{n_scenarios} ({n_drones}대) 시나리오 수집")

    # 부족하면 드론 수 강제 조정
    if len(scenarios) < n_scenarios:
        print(f"  경고: {n_drones}대 시나리오 {len(scenarios)}/{n_scenarios}건만 수집")

    with open(cache_path, "wb") as f:
        pickle.dump(scenarios, f)
    print(f"  캐시 저장: {cache_path} ({len(scenarios)}건)")
    return scenarios


# ═══════════════════════════════════════════════════════════
#  그래프 빌드 함수 (15d / 20d)
# ═══════════════════════════════════════════════════════════

def _build_node_features_15d(drones):
    """15d 노드 피처 + 메타 정보 반환."""
    all_feats, node_meta = [], []
    for did, dr in enumerate(drones):
        lats, lngs, alts = dr["lats"], dr["lngs"], dr["alts"]
        n = len(lats)
        speeds = dr.get("speeds", [2.0] * n)
        dists = [0.0] + [haversine(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bears = [0.0] + [bearing(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bcs = [0.0, 0.0] + [((bears[j] - bears[j-1] + 180) % 360 - 180) for j in range(2, n)]

        for i in range(n):
            dist_next = dists[i + 1] if i < n - 1 else 0.0
            alt_change = alts[i] - alts[i - 1] if i > 0 else 0.0
            feat = [
                _norm(lats[i], 33, 38), _norm(lngs[i], 124, 132), _norm(alts[i], 0, 500),
                i / max(n - 1, 1), _norm(speeds[i], 0, 15),
                _norm(dists[i], 0, 1000), _norm(dist_next, 0, 1000),
                _norm(bears[i], 0, 360), _norm(abs(bcs[i]), 0, 180),
                0.0, _norm(abs(alt_change), 0, 100),
                1.0 if i == 0 else 0.0, 1.0 if i == n - 1 else 0.0,
                0.0, 0.0
            ]
            all_feats.append(feat)
            node_meta.append({"did": did, "idx": i, "lat": lats[i], "lon": lngs[i],
                              "gidx": len(all_feats) - 1})
    return all_feats, node_meta


def _build_node_features_20d(drones):
    """20d 노드 피처 — 15d + 5개 드론 간 관계 피처."""
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

        # 16: min_inter_drone_dist
        all_feats[i].append(_norm(min_dist, 0, 500))

        if nearest_j >= 0:
            gi = mi["gidx"]
            gj = node_meta[nearest_j]["gidx"]
            # 17: relative_speed
            all_feats[i].append(_norm(abs(speeds_map.get(gi, 2.0) - speeds_map.get(gj, 2.0)), 0, 10))
            # 18: heading_divergence
            bear_i = bears_map.get(gi, 0)
            bear_j = bears_map.get(gj, 0)
            hdiff = abs(((bear_i - bear_j + 180) % 360) - 180)
            all_feats[i].append(_norm(hdiff, 0, 180))
            # 19: altitude_separation
            all_feats[i].append(_norm(abs(alts_map.get(gi, 50) - alts_map.get(gj, 50)), 0, 100))
        else:
            all_feats[i].extend([1.0, 0.0, 1.0])

        # 20: temporal_overlap
        all_feats[i].append(_norm(nearby_count, 0, 10))

    return all_feats, node_meta


def _build_edges(node_meta, active_types):
    """active_types에 해당하는 엣지만 생성."""
    edges = {t: [] for t in range(5)}

    # Sequential (type 0)
    if 0 in active_types:
        for m in node_meta:
            if m["idx"] > 0:
                prev = m["gidx"] - 1
                if node_meta[prev]["did"] == m["did"]:
                    edges[0].append([prev, m["gidx"]])
                    edges[0].append([m["gidx"], prev])

    # Proximity (type 1) + Coverage (type 2)
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

    # Temporal (type 3)
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

    # Anomaly (type 4) — 극근접 50m 미만
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


def _edges_to_pyg(x_tensor, edges, node_meta):
    """엣지 딕셔너리 → PyG Data 객체."""
    all_edges, all_types = [], []
    for t in range(5):
        for e in edges[t]:
            all_edges.append(e)
            all_types.append(t)

    if not all_edges:
        all_edges = [[0, 0]]
        all_types = [0]

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(all_types, dtype=torch.long)
    edge_counts = {EDGE_NAMES[t]: len(edges[t]) for t in range(5)}

    data = Data(x=x_tensor, edge_index=edge_index, edge_type=edge_type)
    data.edge_counts = edge_counts
    return data


def build_graph_15d(drones, active_types):
    """15d 그래프 빌드."""
    all_feats, node_meta = _build_node_features_15d(drones)
    x = torch.tensor(all_feats, dtype=torch.float32)
    edges = _build_edges(node_meta, active_types)
    return _edges_to_pyg(x, edges, node_meta), node_meta, edges


def build_graph_20d(drones, active_types):
    """20d 그래프 빌드 (15d + 5개 드론 간 관계 피처)."""
    all_feats, node_meta = _build_node_features_20d(drones)
    x = torch.tensor(all_feats, dtype=torch.float32)
    edges = _build_edges(node_meta, active_types)
    return _edges_to_pyg(x, edges, node_meta), node_meta, edges


# ═══════════════════════════════════════════════════════════
#  학습 & 평가 공통 함수
# ═══════════════════════════════════════════════════════════

def train_and_evaluate(model, train_data, test_data, model_path, label=""):
    """모델 학습 → best 로드 → 평가 → 결과 dict 반환."""
    t_start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    best_anomaly_acc = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        random.shuffle(train_data)
        for data in train_data:
            optimizer.zero_grad()
            pat_logits, anom_pred, _ = model(data)
            l1 = ce_loss(pat_logits.unsqueeze(0), data.y_pattern.unsqueeze(0))
            l2 = bce_loss(anom_pred, data.y_anomaly)
            loss = l1 + l2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            anom_preds, anom_trues = [], []
            with torch.no_grad():
                for data in test_data:
                    _, anom_pred, _ = model(data)
                    anom_preds.append(1 if anom_pred.item() > 0.5 else 0)
                    anom_trues.append(int(data.y_anomaly.item()))
            anom_acc = accuracy_score(anom_trues, anom_preds)
            avg_loss = total_loss / len(train_data)
            print(f"    [{label}] Epoch {epoch+1:3d} | loss={avg_loss:.4f} | anomaly={anom_acc:.1%}")

            if anom_acc > best_anomaly_acc:
                best_anomaly_acc = anom_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_path)

    training_time = time.time() - t_start

    # 최종 평가 — best 모델 로드
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    pat_preds, pat_trues, anom_preds, anom_trues = [], [], [], []
    with torch.no_grad():
        for data in test_data:
            pat_logits, anom_pred, _ = model(data)
            pat_preds.append(pat_logits.argmax().item())
            pat_trues.append(data.y_pattern.item())
            anom_preds.append(1 if anom_pred.item() > 0.5 else 0)
            anom_trues.append(int(data.y_anomaly.item()))

    pat_acc = accuracy_score(pat_trues, pat_preds)
    anom_acc = accuracy_score(anom_trues, anom_preds)
    f1_macro = f1_score(pat_trues, pat_preds, average='macro', zero_division=0)

    col_correct = sum(1 for p, t in zip(pat_preds, pat_trues) if t == 2 and p == 2)
    col_total = sum(1 for t in pat_trues if t == 2)
    col_recall = col_correct / max(col_total, 1)

    report = classification_report(
        pat_trues, pat_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    per_class_recall = {cls: round(report[cls]["recall"], 4) for cls in CLASS_NAMES}
    per_class_f1 = {cls: round(report[cls]["f1-score"], 4) for cls in CLASS_NAMES}

    # type_weights 추출 (R-GAT만 해당)
    layer1_weights, layer2_weights = {}, {}
    if hasattr(model, 'r1') and hasattr(model.r1, 'type_weights'):
        w1 = F.softmax(model.r1.type_weights, dim=0).detach().numpy()
        w2 = F.softmax(model.r2.type_weights, dim=0).detach().numpy()
        layer1_weights = {EDGE_NAMES[i]: round(float(w1[i]), 4) for i in range(5)}
        layer2_weights = {EDGE_NAMES[i]: round(float(w2[i]), 4) for i in range(5)}

    sample_edge_counts = test_data[0].edge_counts if hasattr(test_data[0], 'edge_counts') else {}

    result = {
        "best_epoch": best_epoch,
        "pattern_accuracy": round(pat_acc, 4),
        "anomaly_accuracy": round(anom_acc, 4),
        "collision_recall": round(col_recall, 4),
        "f1_macro": round(f1_macro, 4),
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "layer1_type_weights": layer1_weights,
        "layer2_type_weights": layer2_weights,
        "sample_edge_counts": sample_edge_counts,
        "training_time_sec": round(training_time, 1),
        "n_train": len(train_data),
        "n_test": len(test_data),
    }

    print(f"    [{label}] 결과: pattern={pat_acc:.1%} | anomaly={anom_acc:.1%} | "
          f"collision={col_recall:.1%} | F1={f1_macro:.3f} | {training_time:.0f}s")

    # FN/FP 인덱스도 반환 (Case Study용)
    fn_indices = [i for i, (p, t) in enumerate(zip(anom_preds, anom_trues)) if t == 1 and p == 0]
    tp_indices = [i for i, (p, t) in enumerate(zip(anom_preds, anom_trues)) if t == 1 and p == 1]
    result["_fn_indices"] = fn_indices
    result["_tp_indices"] = tp_indices

    return result


def _build_datasets_15d(raw_scenarios, active_types):
    """시나리오 → 15d PyG Data 리스트."""
    datasets = []
    for sc in raw_scenarios:
        data, _, _ = build_graph_15d(sc["drones"], active_types)
        data.y_pattern = torch.tensor(TYPE_MAP[sc["type"]], dtype=torch.long)
        data.y_anomaly = torch.tensor(float(ANOMALY_MAP[sc["type"]]), dtype=torch.float32)
        datasets.append(data)
    return datasets


def _build_datasets_20d(raw_scenarios, active_types):
    """시나리오 → 20d PyG Data 리스트."""
    datasets = []
    for sc in raw_scenarios:
        data, _, _ = build_graph_20d(sc["drones"], active_types)
        data.y_pattern = torch.tensor(TYPE_MAP[sc["type"]], dtype=torch.long)
        data.y_anomaly = torch.tensor(float(ANOMALY_MAP[sc["type"]]), dtype=torch.float32)
        datasets.append(data)
    return datasets


def _save_json(data, path):
    """JSON 저장 (내부 키 제거)."""
    clean = copy.deepcopy(data)
    # _로 시작하는 내부 키 제거
    if isinstance(clean, dict):
        for k in list(clean.keys()):
            if k.startswith("_"):
                del clean[k]
            elif isinstance(clean[k], list):
                for item in clean[k]:
                    if isinstance(item, dict):
                        for kk in list(item.keys()):
                            if kk.startswith("_"):
                                del item[kk]
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
    print(f"  저장: {path}")


def print_table(results, title=""):
    """결과 비교표 출력."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"{'조건':<20} {'패턴':>6} {'이상':>6} {'충돌':>6} {'F1':>6} {'시간':>6}")
    print(f"{'-' * 56}")
    for r in results:
        label = r.get("condition_id", r.get("model", r.get("group_id", "?")))
        print(f"{label:<20} "
              f"{r['pattern_accuracy']:>5.1%} {r['anomaly_accuracy']:>5.1%} "
              f"{r['collision_recall']:>5.1%} {r['f1_macro']:>5.3f} "
              f"{r['training_time_sec']:>5.0f}s")
    print(f"{'=' * 80}")


# ═══════════════════════════════════════════════════════════
#  Exp 1: E1 엣지 Ablation (7 조건, 15d)
# ═══════════════════════════════════════════════════════════

def run_exp1(raw_scenarios):
    """E1 엣지 Ablation — v3 데이터, 15d, 7개 조건."""
    exp_dir = REPORT_DIR / "e1_ablation"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"# Exp 1: E1 엣지 Ablation (7 조건, 15d, v3)")
    print(f"{'#' * 60}")

    all_results = []
    for cond in CONDITIONS_7:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        print(f"\n  그래프 생성: {cond['id']} ({cond['name']})...")
        t0 = time.time()
        datasets = _build_datasets_15d(raw_scenarios, cond["active"])
        print(f"  그래프 완료: {time.time() - t0:.1f}s")

        train_data, test_data = train_test_split(datasets, test_size=0.2, random_state=SEED)
        model = Fly2VecRGAT(in_dim=15)
        model_path = MODEL_DIR / f"v3_e1_{cond['id']}_best.pt"

        result = train_and_evaluate(model, train_data, test_data, model_path, label=cond['id'])
        result["condition_id"] = cond["id"]
        result["condition_name"] = cond["name"]
        result["active_edge_types"] = sorted(list(cond["active"]))
        result["active_edge_names"] = [EDGE_NAMES[t] for t in sorted(cond["active"])]
        all_results.append(result)

        cond_path = exp_dir / f"e1_{cond['id']}_result.json"
        _save_json(result, cond_path)

    summary = {
        "experiment": "E1_Edge_Ablation_v3",
        "date": time.strftime("%Y-%m-%d"),
        "data_version": "v3",
        "config": {"n_scenarios": N_SCENARIOS, "epochs": EPOCHS, "seed": SEED,
                    "feature_dim": 15},
        "conditions": all_results,
    }
    _save_json(summary, exp_dir / "e1_ablation_results.json")
    print_table(all_results, "Exp 1 결과: E1 엣지 Ablation (15d, v3)")
    return all_results


# ═══════════════════════════════════════════════════════════
#  Exp 2: GCN/GAT Baseline 비교 (15d, E-SPC)
# ═══════════════════════════════════════════════════════════

def run_exp2(raw_scenarios):
    """GCN/GAT Baseline 비교 — 15d, E-SPC 조건."""
    exp_dir = REPORT_DIR / "baseline_comparison"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"# Exp 2: GCN/GAT Baseline 비교 (15d, E-SPC, v3)")
    print(f"{'#' * 60}")

    # E-SPC 그래프 데이터 생성
    active = {0, 1, 2}
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"\n  E-SPC 그래프 생성...")
    t0 = time.time()
    datasets = _build_datasets_15d(raw_scenarios, active)
    print(f"  그래프 완료: {time.time() - t0:.1f}s")

    train_data, test_data = train_test_split(datasets, test_size=0.2, random_state=SEED)

    models = [
        ("R-GAT-15d", Fly2VecRGAT(in_dim=15)),
        ("GCN-15d",   GCNBaseline(in_dim=15)),
        ("GAT-15d",   GATBaseline(in_dim=15)),
    ]

    all_results = []
    for model_name, model in models:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        model_path = MODEL_DIR / f"v3_baseline_{model_name}_best.pt"
        result = train_and_evaluate(model, list(train_data), list(test_data),
                                    model_path, label=model_name)
        result["model"] = model_name
        result["condition"] = "E-SPC"
        all_results.append(result)

        res_path = exp_dir / f"{model_name}_result.json"
        _save_json(result, res_path)

    summary = {
        "experiment": "Baseline_Comparison_v3",
        "date": time.strftime("%Y-%m-%d"),
        "data_version": "v3",
        "config": {"n_scenarios": N_SCENARIOS, "epochs": EPOCHS, "seed": SEED,
                    "feature_dim": 15, "condition": "E-SPC"},
        "models": all_results,
    }
    _save_json(summary, exp_dir / "baseline_comparison_results.json")
    print_table(all_results, "Exp 2 결과: GCN/GAT/R-GAT Baseline (15d, E-SPC, v3)")
    return all_results


# ═══════════════════════════════════════════════════════════
#  Exp 3: E1-20d 엣지 Ablation (3 조건)
# ═══════════════════════════════════════════════════════════

def run_exp3(raw_scenarios):
    """E1-20d 엣지 Ablation — 3 핵심 조건."""
    exp_dir = REPORT_DIR / "e1_ablation_20d"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"# Exp 3: E1-20d 엣지 Ablation (3 조건, v3)")
    print(f"{'#' * 60}")

    all_results = []
    for cond in CONDITIONS_3:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        print(f"\n  20d 그래프 생성: {cond['id']}...")
        t0 = time.time()
        datasets = _build_datasets_20d(raw_scenarios, cond["active"])
        print(f"  그래프 완료: {time.time() - t0:.1f}s")

        train_data, test_data = train_test_split(datasets, test_size=0.2, random_state=SEED)
        model = Fly2VecRGAT(in_dim=20)
        model_path = MODEL_DIR / f"v3_e1_20d_{cond['id']}_best.pt"

        result = train_and_evaluate(model, train_data, test_data, model_path, label=f"20d-{cond['id']}")
        result["condition_id"] = cond["id"]
        result["condition_name"] = cond["name"]
        result["active_edge_types"] = sorted(list(cond["active"]))
        result["active_edge_names"] = [EDGE_NAMES[t] for t in sorted(cond["active"])]
        all_results.append(result)

        cond_path = exp_dir / f"e1_20d_{cond['id']}_result.json"
        _save_json(result, cond_path)

    summary = {
        "experiment": "E1_Edge_Ablation_20d_v3",
        "date": time.strftime("%Y-%m-%d"),
        "data_version": "v3",
        "config": {"n_scenarios": N_SCENARIOS, "epochs": EPOCHS, "seed": SEED,
                    "feature_dim": 20},
        "conditions": all_results,
    }
    _save_json(summary, exp_dir / "e1_ablation_20d_results.json")
    print_table(all_results, "Exp 3 결과: E1-20d 엣지 Ablation (v3)")
    return all_results


# ═══════════════════════════════════════════════════════════
#  Exp 4: 노드 피처 그룹 Ablation (20d, E-SPC)
# ═══════════════════════════════════════════════════════════

def run_exp4(raw_scenarios):
    """노드 피처 그룹 ablation — 20d에서 그룹별 0-마스킹."""
    exp_dir = REPORT_DIR / "feature_ablation"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"# Exp 4: 노드 피처 그룹 Ablation (20d, E-SPC, v3)")
    print(f"{'#' * 60}")

    # 기준선: 마스킹 없는 20d E-SPC
    active = {0, 1, 2}
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"\n  20d E-SPC 기준선 그래프 생성...")
    t0 = time.time()
    base_datasets = _build_datasets_20d(raw_scenarios, active)
    print(f"  그래프 완료: {time.time() - t0:.1f}s")

    # 기준선 학습
    train_base, test_base = train_test_split(base_datasets, test_size=0.2, random_state=SEED)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = Fly2VecRGAT(in_dim=20)
    model_path = MODEL_DIR / "v3_feat_ablation_baseline_best.pt"
    baseline_result = train_and_evaluate(model, list(train_base), list(test_base),
                                          model_path, label="baseline-20d")
    baseline_result["group_id"] = "baseline"
    baseline_result["group_name"] = "마스킹 없음 (기준선)"
    baseline_result["masked_indices"] = []

    all_results = [baseline_result]

    # 그룹별 마스킹 실험
    for group in FEATURE_GROUPS:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        print(f"\n  피처 마스킹: {group['id']} ({group['name']}) — indices {group['indices']}")

        # 데이터 복사 후 마스킹
        masked_datasets = []
        for data in base_datasets:
            md = Data(
                x=data.x.clone(),
                edge_index=data.edge_index.clone(),
                edge_type=data.edge_type.clone(),
            )
            md.y_pattern = data.y_pattern.clone()
            md.y_anomaly = data.y_anomaly.clone()
            md.edge_counts = data.edge_counts
            # 지정 인덱스 0으로 마스킹
            for idx in group["indices"]:
                if idx < md.x.size(1):
                    md.x[:, idx] = 0.0
            masked_datasets.append(md)

        train_masked, test_masked = train_test_split(masked_datasets, test_size=0.2, random_state=SEED)
        model = Fly2VecRGAT(in_dim=20)
        model_path = MODEL_DIR / f"v3_feat_ablation_{group['id']}_best.pt"

        result = train_and_evaluate(model, train_masked, test_masked,
                                     model_path, label=group['id'])
        result["group_id"] = group["id"]
        result["group_name"] = group["name"]
        result["masked_indices"] = group["indices"]

        # 기준선 대비 변화량
        result["delta_pattern"] = round(result["pattern_accuracy"] - baseline_result["pattern_accuracy"], 4)
        result["delta_anomaly"] = round(result["anomaly_accuracy"] - baseline_result["anomaly_accuracy"], 4)
        result["delta_collision"] = round(result["collision_recall"] - baseline_result["collision_recall"], 4)
        result["delta_f1"] = round(result["f1_macro"] - baseline_result["f1_macro"], 4)

        all_results.append(result)

        res_path = exp_dir / f"feat_{group['id']}_result.json"
        _save_json(result, res_path)

    summary = {
        "experiment": "Feature_Group_Ablation_v3",
        "date": time.strftime("%Y-%m-%d"),
        "data_version": "v3",
        "config": {"n_scenarios": N_SCENARIOS, "epochs": EPOCHS, "seed": SEED,
                    "feature_dim": 20, "condition": "E-SPC"},
        "groups": all_results,
    }
    _save_json(summary, exp_dir / "feature_ablation_results.json")

    # 결과표 (delta 포함)
    print(f"\n{'=' * 90}")
    print(f"Exp 4: 노드 피처 그룹 Ablation (20d, E-SPC, v3)")
    print(f"{'=' * 90}")
    print(f"{'그룹':<20} {'패턴':>6} {'이상':>6} {'충돌':>6} {'F1':>6} | {'d패턴':>7} {'d이상':>7} {'d충돌':>7}")
    print(f"{'-' * 90}")
    for r in all_results:
        dp = r.get("delta_pattern", 0)
        da = r.get("delta_anomaly", 0)
        dc = r.get("delta_collision", 0)
        print(f"{r['group_id']:<20} "
              f"{r['pattern_accuracy']:>5.1%} {r['anomaly_accuracy']:>5.1%} "
              f"{r['collision_recall']:>5.1%} {r['f1_macro']:>5.3f} | "
              f"{dp:>+6.1%}p {da:>+6.1%}p {dc:>+6.1%}p")
    print(f"{'=' * 90}")

    return all_results


# ═══════════════════════════════════════════════════════════
#  Exp 5: E7 드론 수 변화 (2, 3, 5대)
# ═══════════════════════════════════════════════════════════

def run_exp5(routes, rk):
    """E7 드론 수 변화 — 그래프 통계 + R-GAT 15d 성능 비교."""
    exp_dir = REPORT_DIR / "e7_drone_count"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"# Exp 5: E7 드론 수 변화 (3, 4, 5대, v3)")
    print(f"{'#' * 60}")

    drone_counts = [3, 4, 5]
    active = {0, 1, 2}  # E-SPC
    all_results = []

    for nd in drone_counts:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        print(f"\n  === {nd}대 드론 시나리오 ===")

        scenarios = generate_fixed_drone_scenarios(routes, rk, nd, n_scenarios=1000)
        print(f"  시나리오 수: {len(scenarios)}건")

        # 분포 확인
        type_dist = defaultdict(int)
        for s in scenarios:
            type_dist[s["type"]] += 1
        print(f"  분포: {dict(type_dist)}")

        # 그래프 통계 수집
        print(f"  그래프 생성 + 통계 수집...")
        t0 = time.time()
        datasets = []
        graph_stats = {"n_nodes": [], "n_edges_total": [], "n_edges_proximity": [],
                       "n_edges_coverage": [], "n_edges_sequential": [],
                       "edge_density": [], "proximity_ratio": []}

        for sc in scenarios:
            data, node_meta, edges = build_graph_15d(sc["drones"], active)
            data.y_pattern = torch.tensor(TYPE_MAP[sc["type"]], dtype=torch.long)
            data.y_anomaly = torch.tensor(float(ANOMALY_MAP[sc["type"]]), dtype=torch.float32)
            datasets.append(data)

            n_nodes = data.x.size(0)
            total_edges = sum(len(edges[t]) for t in range(5))
            prox_edges = len(edges[1])
            cov_edges = len(edges[2])
            seq_edges = len(edges[0])
            density = total_edges / max(n_nodes * (n_nodes - 1), 1)
            prox_ratio = prox_edges / max(total_edges, 1)

            graph_stats["n_nodes"].append(n_nodes)
            graph_stats["n_edges_total"].append(total_edges)
            graph_stats["n_edges_proximity"].append(prox_edges)
            graph_stats["n_edges_coverage"].append(cov_edges)
            graph_stats["n_edges_sequential"].append(seq_edges)
            graph_stats["edge_density"].append(density)
            graph_stats["proximity_ratio"].append(prox_ratio)

        print(f"  그래프 완료: {time.time() - t0:.1f}s")

        # 통계 요약
        stats_summary = {}
        for k, vals in graph_stats.items():
            if len(vals) == 0:
                stats_summary[k] = {"mean": 0, "std": 0, "min": 0, "max": 0}
            else:
                stats_summary[k] = {
                    "mean": round(float(np.mean(vals)), 2),
                    "std": round(float(np.std(vals)), 2),
                    "min": round(float(np.min(vals)), 2),
                    "max": round(float(np.max(vals)), 2),
                }

        print(f"  통계: nodes={stats_summary['n_nodes']['mean']:.0f} | "
              f"edges={stats_summary['n_edges_total']['mean']:.0f} | "
              f"density={stats_summary['edge_density']['mean']:.4f} | "
              f"prox_ratio={stats_summary['proximity_ratio']['mean']:.2%}")

        # R-GAT 학습
        if len(datasets) >= 50:
            train_data, test_data = train_test_split(datasets, test_size=0.2, random_state=SEED)
            random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
            model = Fly2VecRGAT(in_dim=15)
            model_path = MODEL_DIR / f"v3_e7_{nd}d_best.pt"
            result = train_and_evaluate(model, train_data, test_data,
                                         model_path, label=f"{nd}d")
        else:
            result = {"pattern_accuracy": 0, "anomaly_accuracy": 0,
                      "collision_recall": 0, "f1_macro": 0, "training_time_sec": 0}

        result["n_drones"] = nd
        result["n_scenarios"] = len(scenarios)
        result["type_distribution"] = dict(type_dist)
        result["graph_statistics"] = stats_summary
        all_results.append(result)

        res_path = exp_dir / f"e7_{nd}d_result.json"
        _save_json(result, res_path)

    summary = {
        "experiment": "E7_Drone_Count_v3",
        "date": time.strftime("%Y-%m-%d"),
        "data_version": "v3",
        "config": {"epochs": EPOCHS, "seed": SEED, "feature_dim": 15, "condition": "E-SPC"},
        "drone_counts": all_results,
    }
    _save_json(summary, exp_dir / "e7_drone_count_results.json")

    # 결과표
    print(f"\n{'=' * 80}")
    print(f"Exp 5: E7 드론 수 변화 (v3)")
    print(f"{'=' * 80}")
    print(f"{'드론':>4} {'시나리오':>6} {'노드':>6} {'엣지':>8} {'밀도':>8} {'Prox%':>6} | "
          f"{'패턴':>6} {'이상':>6} {'충돌':>6}")
    print(f"{'-' * 80}")
    for r in all_results:
        gs = r["graph_statistics"]
        print(f"{r['n_drones']:>4}대 {r['n_scenarios']:>6} "
              f"{gs['n_nodes']['mean']:>5.0f} {gs['n_edges_total']['mean']:>7.0f} "
              f"{gs['edge_density']['mean']:>7.4f} {gs['proximity_ratio']['mean']:>5.1%} | "
              f"{r['pattern_accuracy']:>5.1%} {r['anomaly_accuracy']:>5.1%} "
              f"{r['collision_recall']:>5.1%}")
    print(f"{'=' * 80}")

    return all_results


# ═══════════════════════════════════════════════════════════
#  Exp 6: Case Study — FN 그래프 구조 분석
# ═══════════════════════════════════════════════════════════

def run_exp6(raw_scenarios):
    """FN 케이스 그래프 구조 분석 — E-SPC 20d best 모델 기준."""
    exp_dir = REPORT_DIR / "case_study"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#' * 60}")
    print(f"# Exp 6: Case Study — FN 그래프 구조 분석 (v3)")
    print(f"{'#' * 60}")

    # E-SPC 20d 그래프 생성
    active = {0, 1, 2}
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"\n  20d E-SPC 그래프 생성...")
    t0 = time.time()

    datasets_with_meta = []
    for sc_idx, sc in enumerate(raw_scenarios):
        data, node_meta, edges = build_graph_20d(sc["drones"], active)
        data.y_pattern = torch.tensor(TYPE_MAP[sc["type"]], dtype=torch.long)
        data.y_anomaly = torch.tensor(float(ANOMALY_MAP[sc["type"]]), dtype=torch.float32)
        # 그래프 통계 저장
        n_nodes = data.x.size(0)
        total_edges = sum(len(edges[t]) for t in range(5))
        prox_edges = len(edges[1])

        # Proximity 엣지 거리 통계
        prox_dists = []
        for i, mi in enumerate(node_meta):
            for j, mj in enumerate(node_meta):
                if mi["did"] == mj["did"] or j <= i:
                    continue
                d = haversine(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
                if d < PROXIMITY_RADIUS_M:
                    prox_dists.append(d)

        data._graph_stats = {
            "scenario_idx": sc_idx,
            "type": sc["type"],
            "n_nodes": n_nodes,
            "n_proximity_edges": prox_edges,
            "n_coverage_edges": len(edges[2]),
            "n_sequential_edges": len(edges[0]),
            "total_edges": total_edges,
            "avg_proximity_dist": round(float(np.mean(prox_dists)), 1) if prox_dists else 0.0,
            "n_drones": len(sc["drones"]),
        }
        datasets_with_meta.append(data)

    print(f"  그래프 완료: {time.time() - t0:.1f}s")

    # train/test 분할
    indices = list(range(len(datasets_with_meta)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
    train_data = [datasets_with_meta[i] for i in train_idx]
    test_data = [datasets_with_meta[i] for i in test_idx]

    # 학습
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = Fly2VecRGAT(in_dim=20)
    model_path = MODEL_DIR / "v3_case_study_best.pt"
    result = train_and_evaluate(model, train_data, test_data, model_path, label="case-study")

    fn_indices = result["_fn_indices"]  # test_data 내 인덱스
    tp_indices = result["_tp_indices"]

    print(f"\n  FN: {len(fn_indices)}건 | TP: {len(tp_indices)}건")

    # FN vs TP 그래프 통계 비교
    def collect_stats(indices):
        stats = {"n_proximity_edges": [], "avg_proximity_dist": [],
                 "n_coverage_edges": [], "total_edges": [], "n_nodes": [],
                 "types": defaultdict(int)}
        for i in indices:
            gs = test_data[i]._graph_stats
            stats["n_proximity_edges"].append(gs["n_proximity_edges"])
            stats["avg_proximity_dist"].append(gs["avg_proximity_dist"])
            stats["n_coverage_edges"].append(gs["n_coverage_edges"])
            stats["total_edges"].append(gs["total_edges"])
            stats["n_nodes"].append(gs["n_nodes"])
            stats["types"][gs["type"]] += 1
        return stats

    fn_stats = collect_stats(fn_indices) if fn_indices else None
    tp_stats = collect_stats(tp_indices) if tp_indices else None

    def summarize_stats(stats, label):
        if not stats or not stats["n_proximity_edges"]:
            return {"label": label, "count": 0}
        return {
            "label": label,
            "count": len(stats["n_proximity_edges"]),
            "avg_proximity_edges": round(float(np.mean(stats["n_proximity_edges"])), 1),
            "avg_proximity_dist": round(float(np.mean(stats["avg_proximity_dist"])), 1),
            "avg_coverage_edges": round(float(np.mean(stats["n_coverage_edges"])), 1),
            "avg_total_edges": round(float(np.mean(stats["total_edges"])), 1),
            "avg_n_nodes": round(float(np.mean(stats["n_nodes"])), 1),
            "type_distribution": dict(stats["types"]),
        }

    fn_summary = summarize_stats(fn_stats, "FN (이상→정상 오분류)")
    tp_summary = summarize_stats(tp_stats, "TP (이상 정탐지)")

    # 대표 FN 3건 상세
    fn_examples = []
    for i in fn_indices[:3]:
        gs = test_data[i]._graph_stats
        fn_examples.append(gs)

    case_study = {
        "experiment": "Case_Study_FN_Analysis_v3",
        "date": time.strftime("%Y-%m-%d"),
        "data_version": "v3",
        "model_performance": {
            "pattern_accuracy": result["pattern_accuracy"],
            "anomaly_accuracy": result["anomaly_accuracy"],
            "collision_recall": result["collision_recall"],
        },
        "fn_count": len(fn_indices),
        "tp_count": len(tp_indices),
        "fn_summary": fn_summary,
        "tp_summary": tp_summary,
        "fn_examples": fn_examples,
    }
    _save_json(case_study, exp_dir / "case_study_results.json")

    # 비교 출력
    print(f"\n{'=' * 70}")
    print(f"Exp 6: FN vs TP 그래프 구조 비교")
    print(f"{'=' * 70}")
    if fn_summary.get("count", 0) > 0 and tp_summary.get("count", 0) > 0:
        print(f"{'지표':<25} {'FN':>12} {'TP':>12} {'차이':>12}")
        print(f"{'-' * 61}")
        for key in ["avg_proximity_edges", "avg_proximity_dist", "avg_coverage_edges",
                     "avg_total_edges", "avg_n_nodes"]:
            fn_v = fn_summary.get(key, 0)
            tp_v = tp_summary.get(key, 0)
            diff = fn_v - tp_v
            print(f"{key:<25} {fn_v:>11.1f} {tp_v:>11.1f} {diff:>+11.1f}")
        print(f"\n  FN 타입 분포: {fn_summary.get('type_distribution', {})}")
        print(f"  TP 타입 분포: {tp_summary.get('type_distribution', {})}")
    else:
        print(f"  FN 또는 TP가 0건 — 비교 불가")
    print(f"{'=' * 70}")

    if fn_examples:
        print(f"\n  대표 FN 3건:")
        for i, ex in enumerate(fn_examples):
            print(f"    [{i+1}] type={ex['type']} | prox={ex['n_proximity_edges']} "
                  f"| cov={ex['n_coverage_edges']} | avg_dist={ex['avg_proximity_dist']:.0f}m "
                  f"| total={ex['total_edges']}")

    return case_study


# ═══════════════════════════════════════════════════════════
#  메인 실행
# ═══════════════════════════════════════════════════════════

def main():
    total_start = time.time()
    print("=" * 60)
    print("v3 종합 실험 스크립트 — 6개 실험 일괄 실행")
    print(f"시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 데이터 로드 ──
    print("\n[0] 데이터 로드")
    routes = load_routes()
    rk = list(routes.keys())
    random.seed(SEED)
    random.shuffle(rk)
    print(f"  경로: {len(routes)}건")

    # ── 공통 v3 시나리오 생성 (Exp 1, 2, 3, 4, 6 공유) ──
    print(f"\n[0-1] v3 시나리오 {N_SCENARIOS}건 생성")
    random.seed(SEED); np.random.seed(SEED)
    raw_scenarios = generate_raw_scenarios(routes, rk)
    type_dist = defaultdict(int)
    for s in raw_scenarios:
        type_dist[s["type"]] += 1
    print(f"  시나리오: {len(raw_scenarios)}건")
    print(f"  분포: {dict(type_dist)}")

    # ── Exp 1: E1 엣지 Ablation (15d, 7 조건) ──
    exp1_results = run_exp1(raw_scenarios)

    # ── Exp 2: GCN/GAT Baseline 비교 (15d, E-SPC) ──
    exp2_results = run_exp2(raw_scenarios)

    # ── Exp 3: E1-20d 엣지 Ablation (3 조건) ──
    exp3_results = run_exp3(raw_scenarios)

    # ── Exp 4: 노드 피처 그룹 Ablation (20d, E-SPC) ──
    exp4_results = run_exp4(raw_scenarios)

    # ── Exp 5: E7 드론 수 변화 (2, 3, 5대) ──
    exp5_results = run_exp5(routes, rk)

    # ── Exp 6: Case Study FN 분석 ──
    exp6_results = run_exp6(raw_scenarios)

    # ── 전체 요약 ──
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"v3 종합 실험 완료")
    print(f"총 소요: {total_time:.0f}초 ({total_time/60:.1f}분)")
    print(f"결과 디렉토리: {REPORT_DIR}")
    print(f"{'=' * 60}")

    # 전체 통합 결과 저장
    final_summary = {
        "experiment": "v3_Comprehensive_All",
        "date": time.strftime("%Y-%m-%d"),
        "data_version": "v3",
        "total_time_sec": round(total_time, 1),
        "n_scenarios": N_SCENARIOS,
        "experiments": {
            "exp1_e1_ablation_15d": f"{len(exp1_results)} conditions",
            "exp2_baseline_comparison": f"{len(exp2_results)} models",
            "exp3_e1_ablation_20d": f"{len(exp3_results)} conditions",
            "exp4_feature_ablation": f"{len(exp4_results)} groups",
            "exp5_e7_drone_count": f"{len(exp5_results)} counts",
            "exp6_case_study": "FN analysis",
        },
    }
    _save_json(final_summary, REPORT_DIR / "v3_comprehensive_summary.json")
    print(f"전체 요약: {REPORT_DIR / 'v3_comprehensive_summary.json'}")


if __name__ == "__main__":
    main()
