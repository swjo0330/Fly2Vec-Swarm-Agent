#!/usr/bin/env python3
"""R-GAT 15d 학습 실험 — 기존 9d(실험6) 대비 enrichment 15d 효과 검증.

기존: R-GAT 9d + Attn Pool → 이상 71.5%, collision 58.0%
신규: R-GAT 15d + Attn Pool → ?
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import math, json, random, time
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
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ → 프로젝트 루트
DATA_PATH = PROJECT_ROOT / "fly2vec" / "data" / "dataset_wp_spot.csv"
MODEL_DIR = PROJECT_ROOT / "fly2vec" / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 파라미터
IN_DIM = 15
HIDDEN_DIM = 64
OUT_DIM = 32
N_EDGE_TYPES = 5
HEADS = 4
N_SCENARIOS = 3000
EPOCHS = 50
LR = 0.001
PROXIMITY_RADIUS_M = 500.0
COVERAGE_GRID_M = 100.0

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

# ── 모델 ──
class RGATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_edge_types=5, heads=4):
        super().__init__()
        self.n_edge_types = n_edge_types
        self.gats = nn.ModuleList([GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=0.1) for _ in range(n_edge_types)])
        self.type_weights = nn.Parameter(torch.ones(n_edge_types) / n_edge_types)
    def forward(self, x, edge_index, edge_type):
        w = F.softmax(self.type_weights, dim=0)
        out = torch.zeros(x.size(0), self.gats[0].out_channels, device=x.device)
        for et in range(self.n_edge_types):
            mask = (edge_type == et)
            if mask.sum() == 0: continue
            out = out + w[et] * self.gats[et](x, edge_index[:, mask])
        return out

class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=0)
        return (x * weights).sum(dim=0)

class Fly2VecRGAT15d(nn.Module):
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

# ── 데이터 로드 ──
print("=" * 60)
print("[1] 데이터 로드")
df = pd.read_csv(DATA_PATH)
mask = ((df["wp_lat"]>=33)&(df["wp_lat"]<=38)&(df["wp_lng"]>=124)&(df["wp_lng"]<=132)&(df["wp_alt"]>=0)&(df["wp_alt"]<=500))
df = df[mask]
grouped = {n: g.sort_values("wp_seq").reset_index(drop=True) for n,g in df.groupby("wp_spot_id")}
routes = {k:v for k,v in grouped.items() if len(v)>=4}
rk = list(routes.keys())
random.shuffle(rk)
print(f"  경로: {len(routes)}건")

# ── 시나리오 생성 + 그래프 변환 ──
print(f"\n[2] {N_SCENARIOS}건 시나리오 생성 + 15d 그래프 변환")
TYPE_MAP = {"cooperative_search": 0, "relay_move": 1, "collision_risk": 2, "coverage_overlap": 3}
ANOMALY_MAP = {"cooperative_search": 0, "relay_move": 0, "collision_risk": 1, "coverage_overlap": 1}

def _norm(v, lo, hi): return max(0.0, min(1.0, (v-lo)/max(hi-lo, 1e-6)))

def build_graph_15d(drones):
    """드론 리스트 → PyG Data (15d features, 5-type edges)."""
    all_feats, node_meta = [], []
    for did, dr in enumerate(drones):
        lats, lngs, alts = dr["lats"], dr["lngs"], dr["alts"]
        n = len(lats)
        speeds = dr.get("speeds", [2.0]*n)
        # 엣지별 거리/방위 사전 계산
        dists = [0.0] + [haversine(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bears = [0.0] + [bearing(lats[j-1], lngs[j-1], lats[j], lngs[j]) for j in range(1, n)]
        bcs = [0.0, 0.0] + [((bears[j]-bears[j-1]+180)%360-180) for j in range(2, n)]

        for i in range(n):
            dist_next = dists[i+1] if i < n-1 else 0.0
            alt_change = alts[i] - alts[i-1] if i > 0 else 0.0
            feat = [
                _norm(lats[i], 33, 38), _norm(lngs[i], 124, 132), _norm(alts[i], 0, 500),
                i / max(n-1, 1), _norm(speeds[i], 0, 15),
                _norm(dists[i], 0, 1000), _norm(dist_next, 0, 1000),
                _norm(bears[i], 0, 360), _norm(abs(bcs[i]), 0, 180),
                0.0,  # hold_sec
                _norm(abs(alt_change), 0, 100),
                1.0 if i==0 else 0.0, 1.0 if i==n-1 else 0.0,
                0.0, 0.0  # is_land, terrain
            ]
            all_feats.append(feat)
            node_meta.append({"did": did, "idx": i, "lat": lats[i], "lon": lngs[i], "gidx": len(all_feats)-1})

    x = torch.tensor(all_feats, dtype=torch.float32)
    edges = {t: [] for t in range(5)}

    # Sequential (type 0)
    for m in node_meta:
        if m["idx"] > 0:
            prev = m["gidx"] - 1
            if node_meta[prev]["did"] == m["did"]:
                edges[0].append([prev, m["gidx"]])
                edges[0].append([m["gidx"], prev])

    # Proximity (type 1) + Coverage (type 2) — 엣지 피처(거리) 포함
    edge_dists = {t: [] for t in range(5)}
    # Sequential 거리
    for e in edges[0]:
        i, j = e
        d = haversine(node_meta[i]["lat"], node_meta[i]["lon"], node_meta[j]["lat"], node_meta[j]["lon"])
        edge_dists[0].append(min(d / 1000.0, 1.0))  # 0~1km 정규화

    for i, mi in enumerate(node_meta):
        for j, mj in enumerate(node_meta):
            if mi["did"] == mj["did"] or j <= i: continue
            d = haversine(mi["lat"], mi["lon"], mj["lat"], mj["lon"])
            if d < PROXIMITY_RADIUS_M:
                edges[1].append([i, j]); edges[1].append([j, i])
                dist_norm = d / PROXIMITY_RADIUS_M  # 0~1 (0=매우 근접, 1=경계)
                edge_dists[1].append(dist_norm); edge_dists[1].append(dist_norm)
            gi = (int(mi["lat"]*1e5) // int(COVERAGE_GRID_M), int(mi["lon"]*1e5) // int(COVERAGE_GRID_M))
            gj = (int(mj["lat"]*1e5) // int(COVERAGE_GRID_M), int(mj["lon"]*1e5) // int(COVERAGE_GRID_M))
            if gi == gj:
                edges[2].append([i, j]); edges[2].append([j, i])
                edge_dists[2].append(d / COVERAGE_GRID_M); edge_dists[2].append(d / COVERAGE_GRID_M)

    all_edges, all_types, all_edge_feats = [], [], []
    for t in range(5):
        for idx, e in enumerate(edges[t]):
            all_edges.append(e); all_types.append(t)
            if idx < len(edge_dists[t]):
                all_edge_feats.append([edge_dists[t][idx]])
            else:
                all_edge_feats.append([0.0])

    if not all_edges:
        all_edges = [[0, 0]]; all_types = [0]; all_edge_feats = [[0.0]]

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(all_types, dtype=torch.long)
    edge_attr = torch.tensor(all_edge_feats, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr)

def gen_scenario(idx):
    n_drones = random.choice([3, 4, 5])
    typ = random.choices(["cooperative_search", "relay_move", "collision_risk", "coverage_overlap"],
                         weights=[40, 35, 10, 15])[0]
    base_keys = random.sample(rk[:500], min(n_drones, 500))
    drones = []
    for di, bk in enumerate(base_keys):
        r = routes[bk]
        lats, lngs, alts = r["wp_lat"].tolist(), r["wp_lng"].tolist(), r["wp_alt"].tolist()
        if typ == "cooperative_search":
            ol, olo = offset_coords(np.mean(lats), np.mean(lngs), 800*(di+1), 120*di)
            lats = [l + (ol - np.mean(lats)) for l in lats]
            lngs = [l + (olo - np.mean(lngs)) for l in lngs]
        elif typ == "collision_risk":
            # 실데이터 위치 유지, 첫 드론 중심으로 다른 드론을 근접 배치 (30~150m)
            if di == 0:
                _col_center = (np.mean(lats), np.mean(lngs))
            else:
                proximity_m = random.uniform(30, 150)
                proximity_bearing = random.uniform(0, 360)
                ol, olo = offset_coords(_col_center[0], _col_center[1], proximity_m, proximity_bearing)
                lats = [l - np.mean(lats) + ol for l in lats]
                lngs = [l - np.mean(lngs) + olo for l in lngs]
        elif typ == "coverage_overlap":
            center_lat, center_lon = np.mean(lats), np.mean(lngs)
            off = random.uniform(-0.0005, 0.0005)
            lats = [l + off for l in lats]
            lngs = [l + off*0.8 for l in lngs]
        elif typ == "relay_move":
            ol, olo = offset_coords(np.mean(lats), np.mean(lngs), 1200*di, random.uniform(0, 360))
            lats = [l - np.mean(lats) + ol for l in lats]
            lngs = [l - np.mean(lngs) + olo for l in lngs]
        drones.append({"lats": lats, "lngs": lngs, "alts": alts})
    return drones, typ

datasets = []
t0 = time.time()
for i in range(N_SCENARIOS):
    drones, typ = gen_scenario(i)
    data = build_graph_15d(drones)
    data.y_pattern = torch.tensor(TYPE_MAP[typ], dtype=torch.long)
    data.y_anomaly = torch.tensor(float(ANOMALY_MAP[typ]), dtype=torch.float32)
    datasets.append(data)
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{N_SCENARIOS} ({time.time()-t0:.1f}s)")
print(f"  완료: {len(datasets)}건, {time.time()-t0:.1f}s")

# ── Train/Test 분리 ──
train_data, test_data = train_test_split(datasets, test_size=0.2, random_state=SEED)
print(f"\n[3] Train: {len(train_data)}, Test: {len(test_data)}")

# ── 학습 ──
print(f"\n[4] R-GAT 15d 학습 (epochs={EPOCHS})")
model = Fly2VecRGAT15d()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()

best_anomaly_acc = 0.0
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

    # 평가
    if (epoch+1) % 5 == 0 or epoch == 0:
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

        # collision 탐지율
        col_correct = sum(1 for p,t in zip(pat_preds, pat_trues) if t==2 and p==2)
        col_total = sum(1 for t in pat_trues if t==2)
        col_acc = col_correct / max(col_total, 1)

        print(f"  Epoch {epoch+1:3d} | loss={total_loss/len(train_data):.4f} | "
              f"pattern={pat_acc:.1%} | anomaly={anom_acc:.1%} | collision={col_acc:.1%}")

        if anom_acc > best_anomaly_acc:
            best_anomaly_acc = anom_acc
            torch.save(model.state_dict(), MODEL_DIR / "fly2vec_rgat_15d_best.pt")

# ── 최종 평가 ──
print(f"\n[5] 최종 평가")
model.load_state_dict(torch.load(MODEL_DIR / "fly2vec_rgat_15d_best.pt", weights_only=True))
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
col_correct = sum(1 for p,t in zip(pat_preds, pat_trues) if t==2 and p==2)
col_total = sum(1 for t in pat_trues if t==2)
col_acc = col_correct / max(col_total, 1)

# 어텐션 가중치
w1 = F.softmax(model.r1.type_weights, dim=0).detach().numpy()
w2 = F.softmax(model.r2.type_weights, dim=0).detach().numpy()
edge_names = ['Sequential', 'Proximity', 'Coverage', 'Temporal', 'Anomaly']

print(f"\n{'='*60}")
print(f"R-GAT 15d 최종 결과")
print(f"{'='*60}")
print(f"  패턴 분류:  {pat_acc:.1%}")
print(f"  이상 탐지:  {anom_acc:.1%}")
print(f"  collision:  {col_acc:.1%}")
print(f"\n  Layer 1 어텐션:")
for n, w in zip(edge_names, w1): print(f"    {n}: {w:.1%}")
print(f"\n  Layer 2 어텐션:")
for n, w in zip(edge_names, w2): print(f"    {n}: {w:.1%}")

# 비교표
print(f"\n{'='*60}")
print(f"9d vs 15d 비교")
print(f"{'='*60}")
print(f"  {'모델':<25} {'패턴':>8} {'이상':>8} {'collision':>10}")
print(f"  {'R-GAT 9d+Attn (기존)':<25} {'30.3%':>8} {'71.5%':>8} {'58.0%':>10}")
print(f"  {'R-GAT 15d+Attn (신규)':<25} {pat_acc:>7.1%} {anom_acc:>7.1%} {col_acc:>9.1%}")

# 결과 저장
result = {
    "model": "R-GAT 15d + Attn Pool",
    "in_dim": IN_DIM, "hidden_dim": HIDDEN_DIM, "out_dim": OUT_DIM,
    "n_scenarios": N_SCENARIOS, "epochs": EPOCHS,
    "pattern_accuracy": round(pat_acc, 4),
    "anomaly_accuracy": round(anom_acc, 4),
    "collision_accuracy": round(col_acc, 4),
    "layer1_attention": {n: round(float(w), 4) for n, w in zip(edge_names, w1)},
    "layer2_attention": {n: round(float(w), 4) for n, w in zip(edge_names, w2)},
}
result_path = PROJECT_ROOT / "results" / "analysis" / "2026-05-04-rgat-15d-training-result.json"
result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\n  결과 저장: {result_path}")
print(f"  모델 저장: {MODEL_DIR / 'fly2vec_rgat_15d_best.pt'}")
