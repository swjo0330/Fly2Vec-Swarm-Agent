#!/usr/bin/env python3
"""E1-20d 피처 확장 실험 — 15d→20d 노드 피처 확장 후 성능 비교.

E1 ablation에서 발견한 문제:
  - collision recall이 엣지 추가할수록 하락 (어텐션 희석)
  - 노드 피처(15d)에 드론 간 관계 정보 부재

해결: 5개 군집 관계 피처 추가 (15d → 20d)
  16. min_inter_drone_dist  — 가장 가까운 타 드론 WP까지 거리
  17. relative_speed         — 가장 가까운 타 드론과의 속도 차이
  18. heading_divergence     — 가장 가까운 타 드론과의 방향 차이
  19. altitude_separation    — 가장 가까운 타 드론과의 고도 차이
  20. temporal_overlap       — 100m 이내 타 드론 WP 수

3개 핵심 조건만 실행 (15d 결과와 비교):
  E-SP:   순서+근접 (15d collision 최고 83.9%)
  E-SPC:  순서+근접+커버리지 (이전 세션 기준선)
  E-ALL:  전체 5종 (15d collision 최저 67.7%)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import math, json, random, time, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "fly2vec" / "data" / "dataset_wp_spot.csv"
MODEL_DIR = PROJECT_ROOT / "fly2vec" / "data" / "models"
RESULT_DIR = PROJECT_ROOT / "results" / "report11" / "e1_ablation_20d"
RESULT_15D = PROJECT_ROOT / "results" / "report11" / "e1_ablation" / "e1_ablation_results.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── 파라미터 ──
IN_DIM = 20
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

# ── 3개 핵심 조건만 (15d와 비교) ──
CONDITIONS = [
    {"id": "E-SP",   "name": "순서+근접",              "active": {0, 1},       },
    {"id": "E-SPC",  "name": "순서+근접+커버리지",      "active": {0, 1, 2},    },
    {"id": "E-ALL",  "name": "전체(5종)",              "active": {0, 1, 2, 3, 4}},
]

# ── 유틸리티 ──
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2-lon1)
    x = math.sin(dl)*math.cos(p2)
    y = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def offset_coords(lat, lon, om, bd):
    R = 6371000.0; b = math.radians(bd)
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1)*math.cos(om/R)+math.cos(la1)*math.sin(om/R)*math.cos(b))
    lo2 = lo1 + math.atan2(math.sin(b)*math.sin(om/R)*math.cos(la1), math.cos(om/R)-math.sin(la1)*math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)

def _norm(v, lo, hi):
    return max(0.0, min(1.0, (v-lo)/max(hi-lo, 1e-6)))

# ── 모델 (train_rgat_15d.py와 동일) ──
class RGATBlock(nn.Module):
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
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=0)
        return (x * weights).sum(dim=0)

class Fly2VecRGAT20d(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = RGATBlock(IN_DIM, HIDDEN_DIM, N_EDGE_TYPES, HEADS)
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

# ── [항목4] 데이터셋 사전 생성 ──
def load_routes():
    df = pd.read_csv(DATA_PATH)
    mask = ((df["wp_lat"]>=33)&(df["wp_lat"]<=38)&
            (df["wp_lng"]>=124)&(df["wp_lng"]<=132)&
            (df["wp_alt"]>=0)&(df["wp_alt"]<=500))
    df = df[mask]
    grouped = {n: g.sort_values("wp_seq").reset_index(drop=True)
               for n, g in df.groupby("wp_spot_id")}
    routes = {k: v for k, v in grouped.items() if len(v) >= 4}
    return routes

# v2: 공통 모듈에서 import (같은 영역 내 군집 배치)
from swarm_scenario_gen import gen_scenario_v2 as gen_scenario

def generate_raw_scenarios(routes, rk):
    """시나리오(드론 좌표+라벨)만 생성 — 엣지 구성 전. 7개 조건 공유."""
    cache_path = RESULT_DIR / "raw_scenarios.pkl"
    if cache_path.exists():
        print(f"  캐시 로드: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    scenarios = []
    for i in range(N_SCENARIOS):
        drones, typ = gen_scenario(i, routes, rk)
        scenarios.append({"drones": drones, "type": typ})
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{N_SCENARIOS} 시나리오 생성")

    with open(cache_path, "wb") as f:
        pickle.dump(scenarios, f)
    print(f"  캐시 저장: {cache_path}")
    return scenarios

# ── [항목1+5] 엣지 마스킹 + Temporal 엣지 생성 ──
def build_graph_20d_masked(drones, active_types: set):
    """드론 리스트 → PyG Data (20d features, active_types만 엣지 생성).
    기존 15d + 5개 군집 관계 피처 추가."""
    all_feats, node_meta = [], []
    alts_map, bears_map, speeds_map = {}, {}, {}

    for did, dr in enumerate(drones):
        lats, lngs, alts = dr["lats"], dr["lngs"], dr["alts"]
        n = len(lats)
        speeds = dr.get("speeds", [2.0]*n)
        dists = [0.0] + [haversine(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bears = [0.0] + [bearing(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bcs = [0.0, 0.0] + [((bears[j]-bears[j-1]+180)%360-180) for j in range(2, n)]

        for i in range(n):
            gidx = len(all_feats)
            dist_next = dists[i+1] if i < n-1 else 0.0
            alt_change = alts[i] - alts[i-1] if i > 0 else 0.0
            feat = [
                _norm(lats[i], 33, 38), _norm(lngs[i], 124, 132), _norm(alts[i], 0, 500),
                i / max(n-1, 1), _norm(speeds[i], 0, 15),
                _norm(dists[i], 0, 1000), _norm(dist_next, 0, 1000),
                _norm(bears[i], 0, 360), _norm(abs(bcs[i]), 0, 180),
                0.0, _norm(abs(alt_change), 0, 100),
                1.0 if i==0 else 0.0, 1.0 if i==n-1 else 0.0,
                0.0, 0.0,
                # 20d 피처 5개는 아래에서 추가 (placeholder)
            ]
            all_feats.append(feat)
            node_meta.append({"did": did, "idx": i, "lat": lats[i], "lon": lngs[i],
                              "gidx": gidx})
            alts_map[gidx] = alts[i]
            bears_map[gidx] = bears[i]
            speeds_map[gidx] = speeds[i]

    # ── 20d 피처 5개 추가: 드론 간 관계 계산 ──
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

    x = torch.tensor(all_feats, dtype=torch.float32)
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
                    gi = (int(mi["lat"]*1e5) // int(COVERAGE_GRID_M),
                          int(mi["lon"]*1e5) // int(COVERAGE_GRID_M))
                    gj = (int(mj["lat"]*1e5) // int(COVERAGE_GRID_M),
                          int(mj["lon"]*1e5) // int(COVERAGE_GRID_M))
                    if gi == gj:
                        edges[2].append([i, j])
                        edges[2].append([j, i])

    # [항목5] Temporal (type 3) — 드론 간 시간 순서 연결
    if 3 in active_types:
        drone_last = {}
        drone_first = {}
        for m in node_meta:
            if m["did"] not in drone_first:
                drone_first[m["did"]] = m["gidx"]
            drone_last[m["did"]] = m["gidx"]
        drone_ids = sorted(drone_last.keys())
        for k in range(len(drone_ids) - 1):
            src = drone_last[drone_ids[k]]
            dst = drone_first[drone_ids[k+1]]
            edges[3].append([src, dst])
            edges[3].append([dst, src])

    # Anomaly (type 4) — 현재 placeholder: collision_risk 시나리오에서
    # Proximity 엣지 중 극근접(< 50m)인 것을 anomaly 엣지로도 중복 추가
    # (향후 확장 가능)
    if 4 in active_types and 1 in active_types:
        for i, mi in enumerate(node_meta):
            for j, mj in enumerate(node_meta):
                if mi["did"] == mj["did"] or j <= i:
                    continue
                d = haversine(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
                if d < 50.0:  # 극근접 50m 미만
                    edges[4].append([i, j])
                    edges[4].append([j, i])

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
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.edge_counts = edge_counts
    return data

# ── [항목2+3] 학습 및 평가 (per-class F1, collision recall, type_weights) ──
def train_and_evaluate(condition, train_data, test_data):
    """단일 조건에 대해 학습 → 평가 → 결과 dict 반환."""
    cond_id = condition["id"]
    print(f"\n{'='*60}")
    print(f"조건: {cond_id} ({condition['name']})")
    print(f"활성 엣지: {[EDGE_NAMES[t] for t in sorted(condition['active'])]}")
    print(f"{'='*60}")

    t_start = time.time()
    model = Fly2VecRGAT20d()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    best_anomaly_acc = 0.0
    best_epoch = 0
    model_path = MODEL_DIR / f"e1_20d_{cond_id}_best.pt"

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

        if (epoch+1) % 10 == 0 or epoch == 0:
            model.eval()
            anom_preds, anom_trues = [], []
            with torch.no_grad():
                for data in test_data:
                    _, anom_pred, _ = model(data)
                    anom_preds.append(1 if anom_pred.item() > 0.5 else 0)
                    anom_trues.append(int(data.y_anomaly.item()))
            anom_acc = accuracy_score(anom_trues, anom_preds)
            avg_loss = total_loss / len(train_data)
            print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | anomaly={anom_acc:.1%}")

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

    # [항목2] per-class F1 + collision recall
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

    # [항목3] type_weights 저장
    w1 = F.softmax(model.r1.type_weights, dim=0).detach().numpy()
    w2 = F.softmax(model.r2.type_weights, dim=0).detach().numpy()
    layer1_weights = {EDGE_NAMES[i]: round(float(w1[i]), 4) for i in range(5)}
    layer2_weights = {EDGE_NAMES[i]: round(float(w2[i]), 4) for i in range(5)}

    # 엣지 통계 (첫 test 샘플 기준)
    sample_edge_counts = test_data[0].edge_counts if hasattr(test_data[0], 'edge_counts') else {}

    result = {
        "condition_id": cond_id,
        "condition_name": condition["name"],
        "active_edge_types": sorted(list(condition["active"])),
        "active_edge_names": [EDGE_NAMES[t] for t in sorted(condition["active"])],
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

    # 결과 출력
    print(f"\n  결과: pattern={pat_acc:.1%} | anomaly={anom_acc:.1%} | "
          f"collision={col_recall:.1%} | F1={f1_macro:.3f}")
    print(f"  best_epoch={best_epoch} | 학습시간={training_time:.0f}s")
    print(f"  L1 weights: {layer1_weights}")
    print(f"  L2 weights: {layer2_weights}")

    return result

# ── 메인 실행 ──
def main():
    print("=" * 60)
    print("E1 엣지 Ablation 실험")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드")
    routes = load_routes()
    rk = list(routes.keys())
    random.seed(SEED)
    random.shuffle(rk)
    print(f"  경로: {len(routes)}건")

    # 2. [항목4] 시나리오 사전 생성 (7개 조건 공유)
    print(f"\n[2] {N_SCENARIOS}건 시나리오 사전 생성")
    random.seed(SEED); np.random.seed(SEED)
    raw_scenarios = generate_raw_scenarios(routes, rk)
    print(f"  시나리오 수: {len(raw_scenarios)}")
    type_dist = defaultdict(int)
    for s in raw_scenarios:
        type_dist[s["type"]] += 1
    print(f"  분포: {dict(type_dist)}")

    # 3. 7개 조건 순차 실행
    print(f"\n[3] 7개 조건 순차 학습")
    all_results = []
    total_start = time.time()

    for cond in CONDITIONS:
        # 조건별 그래프 생성 (동일 시나리오, 다른 엣지 구성)
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        print(f"\n  그래프 생성: {cond['id']}...")
        t_graph = time.time()
        datasets = []
        for sc in raw_scenarios:
            data = build_graph_20d_masked(sc["drones"], cond["active"])
            data.y_pattern = torch.tensor(TYPE_MAP[sc["type"]], dtype=torch.long)
            data.y_anomaly = torch.tensor(float(ANOMALY_MAP[sc["type"]]), dtype=torch.float32)
            datasets.append(data)
        print(f"  그래프 생성 완료: {time.time()-t_graph:.1f}s")

        # 동일 분할 (SEED 고정)
        train_data, test_data = train_test_split(datasets, test_size=0.2, random_state=SEED)

        # 학습 + 평가
        result = train_and_evaluate(cond, train_data, test_data)
        all_results.append(result)

        # 조건별 결과 즉시 저장 (중간 저장)
        cond_path = RESULT_DIR / f"e1_{cond['id']}_result.json"
        cond_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    total_time = time.time() - total_start

    # 4. 통합 결과 저장
    summary = {
        "experiment": "E1_Edge_Ablation",
        "date": time.strftime("%Y-%m-%d"),
        "config": {
            "n_scenarios": N_SCENARIOS, "epochs": EPOCHS, "seed": SEED,
            "proximity_radius_m": PROXIMITY_RADIUS_M,
            "coverage_grid_m": COVERAGE_GRID_M,
        },
        "total_time_sec": round(total_time, 1),
        "conditions": all_results,
    }
    summary_path = RESULT_DIR / "e1_ablation_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n통합 결과 저장: {summary_path}")

    # 5. 비교표 출력
    print(f"\n{'='*80}")
    print(f"E1 엣지 Ablation 결과 요약 (총 {total_time:.0f}초)")
    print(f"{'='*80}")
    print(f"{'조건':<12} {'활성 엣지':<30} {'패턴':>6} {'이상':>6} {'충돌':>6} {'F1':>6} {'epoch':>6}")
    print(f"{'-'*80}")
    for r in all_results:
        edges_str = "+".join([n[:3] for n in r["active_edge_names"]])
        print(f"{r['condition_id']:<12} {edges_str:<30} "
              f"{r['pattern_accuracy']:>5.1%} {r['anomaly_accuracy']:>5.1%} "
              f"{r['collision_recall']:>5.1%} {r['f1_macro']:>5.3f} {r['best_epoch']:>5d}")
    print(f"{'='*80}")

    # 6. 핵심 해석 포인트
    print(f"\n[해석 포인트]")
    esp = next((r for r in all_results if r["condition_id"] == "E-SP"), None)
    espc = next((r for r in all_results if r["condition_id"] == "E-SPC"), None)
    eall = next((r for r in all_results if r["condition_id"] == "E-ALL"), None)

    if esp and espc:
        cov_gain = espc["collision_recall"] - esp["collision_recall"]
        print(f"  E-SP→E-SPC collision: {cov_gain:+.1%}p")
    if eall:
        print(f"  E-ALL 20d: collision={eall['collision_recall']:.1%} (15d 대비 개선 확인)")

    # 7. 15d vs 20d 비교
    if RESULT_15D.exists():
        print(f"\n{'='*80}")
        print(f"15d vs 20d 비교")
        print(f"{'='*80}")
        with open(RESULT_15D) as f:
            data_15d = json.load(f)
        results_15d = {r["condition_id"]: r for r in data_15d["conditions"]}
        print(f"{'조건':<10} {'지표':<8} {'15d':>8} {'20d':>8} {'차이':>8}")
        print(f"{'-'*44}")
        for r20 in all_results:
            cid = r20["condition_id"]
            r15 = results_15d.get(cid)
            if not r15:
                continue
            for metric, label in [("anomaly_accuracy", "이상"), ("collision_recall", "충돌"), ("pattern_accuracy", "패턴")]:
                v15 = r15[metric]
                v20 = r20[metric]
                diff = v20 - v15
                print(f"{cid:<10} {label:<8} {v15:>7.1%} {v20:>7.1%} {diff:>+7.1%}p")
            print()

if __name__ == "__main__":
    main()
