#!/usr/bin/env python3
"""Cesium 3D 시각화 레이어 생성 — report12 v3 데이터 기반.

5개 레이어 JSON 생성:
  1. risk-heatmap.json    — WP별 위험도 색상
  2. e8-attention.json    — 어텐션 엣지 관계선 시각화
  3. e1-ablation.json     — E1 엣지 조건 비교 시각화 (7조건)
  4. error-cases.json     — FP/FN 오탐지 하이라이트
  5. route-compare.json   — 위험 → 안전 추천 경로 비교

출력 디렉토리: results/report12/cesium_layers/
JSON 포맷: swarm_all_cesium.html loadLayerEntities 함수 호환
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math, json, random, pickle, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# ── 재현성 고정 ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 경로 설정 ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# v3 시나리오 캐시: report12에 저장된 raw_scenarios_v3.pkl 사용
SCENARIO_PATH = PROJECT_ROOT / "results" / "report12" / "raw_scenarios_v3.pkl"

# 최적 모델: v3 20d E-ALL (report12 실험 기준 best)
MODEL_PATH = PROJECT_ROOT / "fly2vec" / "data" / "models" / "e1_20d_E-ALL_best.pt"

# E1 결과 JSON 디렉토리
E1_RESULT_DIR = PROJECT_ROOT / "results" / "report12" / "e1_ablation"

# 출력 디렉토리
OUTPUT_DIR = PROJECT_ROOT / "results" / "report12" / "cesium_layers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 모델 파라미터 (20d 기준) ──
IN_DIM = 20
HIDDEN_DIM = 64
OUT_DIM = 32
N_EDGE_TYPES = 5
HEADS = 4

# ── 데이터 파라미터 ──
PROXIMITY_RADIUS_M = 500.0
COVERAGE_GRID_M = 100.0

# ── 분류 매핑 ──
TYPE_MAP = {
    "cooperative_search": 0,
    "relay_move": 1,
    "collision_risk": 2,
    "coverage_overlap": 3
}
ANOMALY_MAP = {
    "cooperative_search": 0,
    "relay_move": 0,
    "collision_risk": 1,
    "coverage_overlap": 1
}
CLASS_NAMES = ["cooperative_search", "relay_move", "collision_risk", "coverage_overlap"]
KR = {
    "cooperative_search": "협력수색",
    "relay_move": "릴레이",
    "collision_risk": "충돌위험",
    "coverage_overlap": "수색중복"
}

# ── 엣지 명칭 ──
EDGE_NAMES = ['Sequential', 'Proximity', 'Coverage', 'Temporal', 'Anomaly']
EDGE_KR = ['순서', '근접', '커버리지', '시간', '이상']

# ── 드론별 색상 팔레트 ──
DRONE_COLORS = ["#1976D2", "#F57C00", "#388E3C", "#C62828", "#7B1FA2"]

# ── E1 7개 조건 정의 ──
CONDITIONS_7 = [
    {"id": "E-S",    "name": "경로 순서 단독",          "active": {0}},
    {"id": "E-SP",   "name": "순서+근접",               "active": {0, 1}},
    {"id": "E-SPC",  "name": "순서+근접+커버리지",       "active": {0, 1, 2}},
    {"id": "E-SPCT", "name": "순서+근접+커버리지+시간",   "active": {0, 1, 2, 3}},
    {"id": "E-ALL",  "name": "전체(5종)",               "active": {0, 1, 2, 3, 4}},
    {"id": "E-P",    "name": "근접 단독",               "active": {1}},
    {"id": "E-C",    "name": "커버리지 단독",            "active": {2}},
]


# ═══════════════════════════════════════════════════════════════
#  유틸리티 함수
# ═══════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    """두 좌표 간 거리(m) 계산."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    """두 좌표 간 방위각(0~360) 계산."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _norm(v, lo, hi):
    """0~1 정규화."""
    return max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-6)))


def risk_color(risk):
    """위험도 0.0~1.0 → 그린→옐로→레드 16진 색상."""
    if risk < 0.3:
        r, g = int(255 * risk * 3), 255
    elif risk < 0.7:
        r, g = 255, int(255 * (1 - (risk - 0.3) / 0.4))
    else:
        r, g = 255, 0
    return f"#{r:02x}{g:02x}00"


# ═══════════════════════════════════════════════════════════════
#  모델 정의 (run_v3_comprehensive.py 와 동일 — 가중치 로드용)
# ═══════════════════════════════════════════════════════════════

class RGATBlock(nn.Module):
    """Relational GAT — 엣지 타입별 독립 GATConv + 학습 가중치."""
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

    def forward_with_attention(self, x, edge_index, edge_type):
        """어텐션 가중치를 함께 반환 (e8-attention 레이어용)."""
        w = F.softmax(self.type_weights, dim=0)
        out = torch.zeros(x.size(0), self.gats[0].out_channels, device=x.device)
        attn_weights_per_type = {}
        for et in range(self.n_edge_types):
            mask = (edge_type == et)
            if mask.sum() == 0:
                continue
            # GATConv return_attention_weights=True
            h, (ei, aw) = self.gats[et](
                x, edge_index[:, mask], return_attention_weights=True
            )
            out = out + w[et] * h
            attn_weights_per_type[et] = {
                "edge_index": ei.detach(),
                "attn": aw.detach().squeeze(-1),  # [E]
                "type_weight": w[et].item(),
            }
        return out, attn_weights_per_type


class AttnPool(nn.Module):
    """어텐션 기반 그래프-레벨 풀링."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=0)
        return (x * weights).sum(dim=0)


class Fly2VecRGAT(nn.Module):
    """R-GAT 모델 (in_dim 파라미터화 — 20d 기준)."""
    def __init__(self, in_dim=20):
        super().__init__()
        self.r1 = RGATBlock(in_dim, HIDDEN_DIM, N_EDGE_TYPES, HEADS)
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.r2 = RGATBlock(HIDDEN_DIM, OUT_DIM, N_EDGE_TYPES, HEADS)
        self.pool = AttnPool(OUT_DIM)
        self.th = nn.Linear(OUT_DIM, 4)   # 패턴 분류 헤드
        self.ah = nn.Linear(OUT_DIM, 1)   # 이상 탐지 헤드

    def forward(self, data):
        h = F.relu(self.bn(self.r1(data.x, data.edge_index, data.edge_type)))
        h = self.r2(h, data.edge_index, data.edge_type)
        emb = self.pool(h)
        return self.th(emb), torch.sigmoid(self.ah(emb)).squeeze(), emb

    def get_node_risk(self, data):
        """노드별 위험도 점수 반환 (풀링 전)."""
        h = F.relu(self.bn(self.r1(data.x, data.edge_index, data.edge_type)))
        h = self.r2(h, data.edge_index, data.edge_type)
        return torch.sigmoid(self.ah(h)).squeeze(-1)  # [N]

    def get_attention_edges(self, data):
        """Layer1 Proximity 어텐션 가중치 + 엣지 반환."""
        h, attn_map = self.r1.forward_with_attention(
            data.x, data.edge_index, data.edge_type
        )
        return attn_map  # dict: edge_type → {edge_index, attn, type_weight}


# ═══════════════════════════════════════════════════════════════
#  그래프 빌드 함수 (20d — run_v3_comprehensive.py 기준)
# ═══════════════════════════════════════════════════════════════

def _build_node_features_20d(drones):
    """20d 노드 피처 + 노드 메타 반환."""
    all_feats, node_meta = [], []
    alts_map, bears_map, speeds_map = {}, {}, {}

    for did, dr in enumerate(drones):
        lats, lngs, alts = dr["lats"], dr["lngs"], dr["alts"]
        n = len(lats)
        speeds = dr.get("speeds", [2.0] * n)
        dists = [0.0] + [haversine(lats[j-1], lngs[j-1], lats[j], lngs[j])
                         for j in range(1, n)]
        bears = [0.0] + [bearing(lats[j-1], lngs[j-1], lats[j], lngs[j])
                         for j in range(1, n)]
        bcs = [0.0, 0.0] + [((bears[j] - bears[j-1] + 180) % 360 - 180)
                             for j in range(2, n)]

        for i in range(n):
            gidx = len(all_feats)
            dist_next = dists[i + 1] if i < n - 1 else 0.0
            alt_change = alts[i] - alts[i - 1] if i > 0 else 0.0
            feat = [
                _norm(lats[i], 33, 38),       # 0: lat 정규화
                _norm(lngs[i], 124, 132),      # 1: lon 정규화
                _norm(alts[i], 0, 500),        # 2: alt 정규화
                i / max(n - 1, 1),             # 3: 순서 비율
                _norm(speeds[i], 0, 15),       # 4: 속도
                _norm(dists[i], 0, 1000),      # 5: 이전 거리
                _norm(dist_next, 0, 1000),     # 6: 다음 거리
                _norm(bears[i], 0, 360),       # 7: 방위각
                _norm(abs(bcs[i]), 0, 180),    # 8: 방위각 변화
                0.0,                           # 9: 예비
                _norm(abs(alt_change), 0, 100),# 10: 고도 변화
                1.0 if i == 0 else 0.0,        # 11: 출발점 플래그
                1.0 if i == n - 1 else 0.0,    # 12: 도착점 플래그
                0.0, 0.0,                      # 13~14: 예비
            ]
            all_feats.append(feat)
            node_meta.append({
                "did": did, "idx": i,
                "lat": lats[i], "lon": lngs[i], "alt": alts[i],
                "gidx": gidx
            })
            alts_map[gidx] = alts[i]
            bears_map[gidx] = bears[i]
            speeds_map[gidx] = speeds[i]

    # 드론 간 관계 피처 5개 추가 (15~19번)
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

        all_feats[i].append(_norm(min_dist, 0, 500))  # 15: min 드론간 거리

        if nearest_j >= 0:
            gi = mi["gidx"]
            gj = node_meta[nearest_j]["gidx"]
            all_feats[i].append(
                _norm(abs(speeds_map.get(gi, 2.0) - speeds_map.get(gj, 2.0)), 0, 10)
            )  # 16: 상대 속도
            bear_i = bears_map.get(gi, 0)
            bear_j = bears_map.get(gj, 0)
            hdiff = abs(((bear_i - bear_j + 180) % 360) - 180)
            all_feats[i].append(_norm(hdiff, 0, 180))  # 17: 헤딩 차이
            all_feats[i].append(
                _norm(abs(alts_map.get(gi, 50) - alts_map.get(gj, 50)), 0, 100)
            )  # 18: 고도 분리
        else:
            all_feats[i].extend([1.0, 0.0, 1.0])

        all_feats[i].append(_norm(nearby_count, 0, 10))  # 19: 근접 드론 수

    return all_feats, node_meta


def _build_edges(node_meta, active_types):
    """active_types에 해당하는 엣지만 생성, 딕셔너리 반환."""
    edges = {t: [] for t in range(5)}

    # Sequential (type 0): 같은 드론 연속 WP
    if 0 in active_types:
        for m in node_meta:
            if m["idx"] > 0:
                prev = m["gidx"] - 1
                if node_meta[prev]["did"] == m["did"]:
                    edges[0].append([prev, m["gidx"]])
                    edges[0].append([m["gidx"], prev])

    # Proximity (type 1) + Coverage (type 2): 드론 간 공간 관계
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

    # Temporal (type 3): 드론 간 시퀀스 연결
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

    # Anomaly (type 4): 극근접 50m 미만
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


def build_graph_20d(drones, active_types=None):
    """20d 그래프 빌드 → PyG Data + node_meta 반환."""
    if active_types is None:
        active_types = {0, 1, 2, 3, 4}  # E-ALL 기본값

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
    edge_type = torch.tensor(all_types, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    return data, node_meta, edges


def load_model():
    """E-ALL 20d 최적 모델 로드."""
    model = Fly2VecRGAT(in_dim=IN_DIM)
    state = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_scenarios():
    """report12 v3 시나리오 캐시 로드 + train/test 분리."""
    with open(SCENARIO_PATH, "rb") as f:
        raw = pickle.load(f)
    train_sc, test_sc = train_test_split(raw, test_size=0.2, random_state=SEED)
    return raw, train_sc, test_sc


# ═══════════════════════════════════════════════════════════════
#  레이어 1: risk-heatmap.json — WP별 위험도 색상
# ═══════════════════════════════════════════════════════════════

def generate_risk_heatmap(model, test_sc):
    """테스트 시나리오별 노드 위험도 점수를 Cesium 포인트/폴리라인으로 표현."""
    print("[risk-heatmap] 생성 시작 ...")

    cases = []
    with torch.no_grad():
        for idx, sc in enumerate(test_sc):
            typ = sc["type"]
            data, node_meta, _ = build_graph_20d(sc["drones"])
            node_risk = model.get_node_risk(data).numpy()

            avg_risk = float(np.mean(node_risk))
            max_risk = float(np.max(node_risk))
            ents = []

            # ── 입력: 드론 경로 폴리라인 ──
            for di, dr in enumerate(sc["drones"]):
                dc = DRONE_COLORS[di % len(DRONE_COLORS)]
                agl_list = [max(float(a), 0.0) for a in dr["alts"]]
                positions = [
                    [float(lat), float(lon), agl]
                    for lat, lon, agl in zip(dr["lats"], dr["lngs"], agl_list)
                ]
                ents.append({
                    "type": "polyline",
                    "id": f"c{idx}_path_d{di}",
                    "positions": positions,
                    "color": dc, "width": 3,
                    "label": f"입력: D{di} 경로",
                })
                # 출발/도착 포인트
                ents.append({
                    "type": "point", "id": f"c{idx}_dep_d{di}",
                    "position": [float(dr["lats"][0]), float(dr["lngs"][0]), agl_list[0]],
                    "color": "#00C853", "size": 14,
                    "label": f"▶ D{di} 출발", "font_size": 13,
                    "heightReference": "RELATIVE_TO_GROUND",
                })
                ents.append({
                    "type": "point", "id": f"c{idx}_arr_d{di}",
                    "position": [float(dr["lats"][-1]), float(dr["lngs"][-1]), agl_list[-1]],
                    "color": "#FF6E40", "size": 14,
                    "label": f"■ D{di} 도착", "font_size": 13,
                    "heightReference": "RELATIVE_TO_GROUND",
                })

            # ── 출력: 노드별 위험도 포인트 (경로와 동일 AGL) ──
            for m in node_meta:
                risk = float(node_risk[m["gidx"]]) if m["gidx"] < len(node_risk) else 0.0
                agl = max(float(m["alt"]), 0.0)
                color = risk_color(risk)
                size = max(6, int(risk * 18))

                if risk > 0.7:
                    label = f"위험 {risk:.0%} D{m['did']}"
                elif risk > 0.4:
                    label = f"주의 {risk:.0%}"
                else:
                    label = ""

                ents.append({
                    "type": "point",
                    "id": f"c{idx}_wp_{m['gidx']}",
                    "position": [float(m["lat"]), float(m["lon"]), agl],
                    "color": color, "size": size,
                    "label": label, "font_size": 13,
                    "heightReference": "RELATIVE_TO_GROUND",
                    "risk_score": round(risk, 3),
                })

            # ── 케이스 제목 라벨 (AGL + 50m) ──
            cx = float(np.mean([l for d in sc["drones"] for l in d["lats"]]))
            cy = float(np.mean([l for d in sc["drones"] for l in d["lngs"]]))
            avg_agl = float(np.mean([max(float(a), 0.0) for d in sc["drones"] for a in d["alts"]]))
            ents.append({
                "type": "point", "id": f"c{idx}_title",
                "position": [cx, cy, avg_agl + 50],
                "color": "#FFFFFF", "size": 1,
                "label": (
                    f"케이스 #{idx + 1} ({KR[typ]})\n"
                    f"━━ 입력 ━━\n"
                    f"드론 {len(sc['drones'])}대 군집 경로 (색상 선)\n"
                    f"━━ R-GAT 판정 ━━\n"
                    f"평균위험 {avg_risk:.0%} | 최대위험 {max_risk:.0%}\n"
                    f"안전(초록) → 주의(노랑) → 위험(빨강)"
                ),
                "font_size": 15,
                "heightReference": "RELATIVE_TO_GROUND",
                "label_bg": "rgba(20,20,40,0.92)",
            })

            cases.append({
                "index": idx, "type": typ, "type_kr": KR[typ],
                "n_drones": len(sc["drones"]),
                "avg_risk": round(avg_risk, 3),
                "max_risk": round(max_risk, 3),
                "entities": ents,
            })

            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(test_sc)} 처리 완료")

    layer = {
        "layer_id": "risk-heatmap",
        "name": "WP 위험 색상",
        "description": "R-GAT 노드별 위험도 점수 시각화 (report12 v3 20d E-ALL)",
        "total_cases": len(cases),
        "cases": cases,
    }
    out = OUTPUT_DIR / "risk-heatmap.json"
    out.write_text(json.dumps(layer, ensure_ascii=False))
    mb = out.stat().st_size / 1024 / 1024
    print(f"  저장: {out.name} ({len(cases)}건, {mb:.1f}MB)")


# ═══════════════════════════════════════════════════════════════
#  레이어 2: e8-attention.json — 어텐션 엣지 관계선
# ═══════════════════════════════════════════════════════════════

def generate_e8_attention(model, test_sc):
    """충돌 시나리오에서 Proximity 어텐션 가중치가 높은 엣지를 관계선으로 표현."""
    print("[e8-attention] 생성 시작 ...")

    # 충돌위험 시나리오만 사용 (최대 50개)
    collisions = [s for s in test_sc if s["type"] == "collision_risk"][:50]
    print(f"  충돌 시나리오: {len(collisions)}개 사용")

    entities = []
    with torch.no_grad():
        for si, sc in enumerate(collisions):
            data, node_meta, _ = build_graph_20d(sc["drones"])

            # ── 드론 경로 배경선 (회색) ──
            for di, dr in enumerate(sc["drones"]):
                agl_list = [max(float(a), 0.0) for a in dr["alts"]]
                positions = [
                    [float(lat), float(lon), agl]
                    for lat, lon, agl in zip(dr["lats"], dr["lngs"], agl_list)
                ]
                entities.append({
                    "type": "polyline",
                    "id": f"attn_{si}_d{di}",
                    "positions": positions,
                    "color": "#78909C", "width": 2,
                    "label": f"충돌#{si+1} D{di} 경로",
                })

            # ── Layer1 어텐션 추출 ──
            try:
                attn_map = model.get_attention_edges(data)
            except Exception as e:
                # GATConv return_attention_weights 미지원 fallback
                attn_map = {}

            # Proximity(type=1) 어텐션 엣지 렌더링
            if 1 in attn_map:
                attn_info = attn_map[1]
                ei = attn_info["edge_index"]  # [2, E]
                aw = attn_info["attn"]         # [E] — 헤드 평균값

                # multi-head → 평균
                aw_np = aw.numpy()
                if aw_np.ndim > 1:
                    aw_np = aw_np.mean(axis=-1)  # [E, heads] → [E]
                threshold = max(float(np.percentile(aw_np, 80)), 0.3) if len(aw_np) > 0 else 0.3

                for e_idx in range(ei.shape[1]):
                    src, dst = int(ei[0, e_idx]), int(ei[1, e_idx])
                    aw_val = float(aw_np[e_idx]) if e_idx < len(aw_np) else 0.0
                    if aw_val < threshold:
                        continue
                    if src >= len(node_meta) or dst >= len(node_meta):
                        continue
                    msrc, mdst = node_meta[src], node_meta[dst]
                    if msrc["did"] == mdst["did"]:
                        continue  # 같은 드론 내부 엣지 제외

                    agl_src = max(float(msrc["alt"]), 0.0)
                    agl_dst = max(float(mdst["alt"]), 0.0)
                    intensity = min(1.0, aw_val)
                    r = int(255 * intensity)
                    g = int(255 * (1 - intensity))
                    color = f"#{r:02x}{g:02x}40"
                    width = max(2, int(aw_val * 10))

                    entities.append({
                        "type": "polyline",
                        "id": f"attn_{si}_prox_{src}_{dst}",
                        "positions": [
                            [float(msrc["lat"]), float(msrc["lon"]), agl_src],
                            [float(mdst["lat"]), float(mdst["lon"]), agl_dst],
                        ],
                        "color": color, "width": width,
                        "attention_weight": round(aw_val, 4),
                        "edge_type": "Proximity",
                        "label": f"어텐션 {aw_val:.2f} (D{msrc['did']}↔D{mdst['did']})",
                    })
            else:
                # 어텐션 추출 실패 시 최근접 쌍 폴백
                for di in range(len(sc["drones"])):
                    for dj in range(di + 1, len(sc["drones"])):
                        d1, d2 = sc["drones"][di], sc["drones"][dj]
                        min_dist, bp = 1e9, None
                        for i in range(len(d1["lats"])):
                            for j in range(len(d2["lats"])):
                                d = haversine(d1["lats"][i], d1["lngs"][i],
                                              d2["lats"][j], d2["lngs"][j])
                                if d < min_dist:
                                    min_dist = d
                                    bp = (i, j)
                        if min_dist < PROXIMITY_RADIUS_M and bp:
                            i, j = bp
                            agl1 = max(float(d1["alts"][i]), 0.0)
                            agl2 = max(float(d2["alts"][j]), 0.0)
                            attn_proxy = max(0.0, 1.0 - min_dist / PROXIMITY_RADIUS_M)
                            w = max(3, int(8 * attn_proxy))
                            entities.append({
                                "type": "polyline",
                                "id": f"attn_{si}_prox_{di}_{dj}",
                                "positions": [
                                    [float(d1["lats"][i]), float(d1["lngs"][i]), agl1],
                                    [float(d2["lats"][j]), float(d2["lngs"][j]), agl2],
                                ],
                                "color": "#FF1744", "width": w,
                                "attention_weight": round(attn_proxy, 4),
                                "edge_type": "Proximity",
                                "label": f"근접 {min_dist:.0f}m (D{di}↔D{dj})",
                            })

    layer = {
        "layer_id": "e8-attention",
        "name": "E8 어텐션 관계선",
        "description": "충돌 시나리오 Proximity 어텐션 가중치 시각화",
        "n_scenarios": len(collisions),
        "entities": entities,
    }
    out = OUTPUT_DIR / "e8-attention.json"
    out.write_text(json.dumps(layer, ensure_ascii=False))
    print(f"  저장: {out.name} ({len(entities)} entities)")


# ═══════════════════════════════════════════════════════════════
#  레이어 3: e1-ablation.json — 엣지 조건 비교 (7조건)
# ═══════════════════════════════════════════════════════════════

def generate_e1_ablation(raw_sc):
    """7개 E1 조건별 정확도를 3층 고도 분리로 비교 시각화."""
    print("[e1-ablation] 생성 시작 ...")

    # 충돌위험 시나리오 5개만 사용 (비교 대상)
    collisions = [s for s in raw_sc if s["type"] == "collision_risk"][:5]

    # E1 결과 JSON 로드 (정확도 수치 표시용)
    e1_results = {}
    for cond in CONDITIONS_7:
        result_file = E1_RESULT_DIR / f"e1_{cond['id']}_result.json"
        if result_file.exists():
            with open(result_file) as f:
                e1_results[cond["id"]] = json.load(f)

    # 7조건 → 3개 대표 층으로 그룹화하여 시각화
    # 하단(AGL+0): E-S, E-P, E-C — 단순 단독 조건
    # 중단(AGL+40): E-SP, E-SPC — 복합 조건
    # 상단(AGL+80): E-SPCT, E-ALL — 최종 조건
    cond_layout = [
        {"id": "E-S",    "color": "#9E9E9E", "alt_add": 0,  "width": 2, "group": "단독"},
        {"id": "E-P",    "color": "#607D8B", "alt_add": 10, "width": 2, "group": "단독"},
        {"id": "E-C",    "color": "#455A64", "alt_add": 20, "width": 2, "group": "단독"},
        {"id": "E-SP",   "color": "#1565C0", "alt_add": 40, "width": 3, "group": "복합"},
        {"id": "E-SPC",  "color": "#2196F3", "alt_add": 55, "width": 4, "group": "복합"},
        {"id": "E-SPCT", "color": "#00BCD4", "alt_add": 75, "width": 4, "group": "최종"},
        {"id": "E-ALL",  "color": "#4CAF50", "alt_add": 90, "width": 5, "group": "최종"},
    ]

    entities = []
    for si, sc in enumerate(collisions):
        avg_agl = float(np.mean([max(float(a), 0.0)
                                 for d in sc["drones"] for a in d["alts"]]))
        cx = float(np.mean([l for d in sc["drones"] for l in d["lats"]]))
        cy = float(np.mean([l for d in sc["drones"] for l in d["lngs"]]))

        for ci, cond_vis in enumerate(cond_layout):
            cid = cond_vis["id"]
            # E1 결과에서 정확도 가져오기
            res = e1_results.get(cid, {})
            anom_acc = res.get("anomaly_accuracy", 0.0)
            col_rec = res.get("collision_recall", 0.0)
            pat_acc = res.get("pattern_accuracy", 0.0)

            for di, dr in enumerate(sc["drones"]):
                agl_list = [max(float(a), 0.0) + cond_vis["alt_add"] for a in dr["alts"]]
                positions = [
                    [float(lat), float(lon), agl]
                    for lat, lon, agl in zip(dr["lats"], dr["lngs"], agl_list)
                ]
                entities.append({
                    "type": "polyline",
                    "id": f"abl_{si}_{ci}_d{di}",
                    "positions": positions,
                    "color": cond_vis["color"],
                    "width": cond_vis["width"],
                    "label": f"#{si+1} {cid} D{di}",
                })

            # 조건별 성능 라벨
            label_lines = [f"{cid} — {CONDITIONS_7[ci]['name']}"]
            if res:
                label_lines.append(f"이상탐지 {anom_acc:.0%} | 충돌recall {col_rec:.0%}")
                label_lines.append(f"패턴정확도 {pat_acc:.0%}")
            else:
                label_lines.append("(결과 없음)")
            label_lines.append(f"그룹: {cond_vis['group']}")

            entities.append({
                "type": "point",
                "id": f"abl_{si}_{ci}_label",
                "position": [cx + si * 0.002, cy, avg_agl + cond_vis["alt_add"] + 15],
                "color": cond_vis["color"], "size": 14,
                "label": "\n".join(label_lines),
                "font_size": 13,
                "heightReference": "RELATIVE_TO_GROUND",
            })

        # 시나리오 제목 라벨
        entities.append({
            "type": "point",
            "id": f"abl_{si}_title",
            "position": [cx, cy, avg_agl + 110],
            "color": "#FFFFFF", "size": 1,
            "label": (
                f"E1 엣지 비교 — 충돌 시나리오 #{si+1}\n"
                f"아래→위: 단독조건(회색) → 복합(파랑) → 최종(초록)\n"
                f"E-ALL이 최고 성능"
            ),
            "font_size": 14,
            "heightReference": "RELATIVE_TO_GROUND",
            "label_bg": "rgba(10,20,40,0.90)",
        })

    layer = {
        "layer_id": "e1-ablation",
        "name": "E1 엣지 Ablation 비교",
        "description": "7개 엣지 조건 성능 비교 — 고도 분리로 시각화",
        "n_conditions": len(CONDITIONS_7),
        "n_scenarios": len(collisions),
        "entities": entities,
    }
    out = OUTPUT_DIR / "e1-ablation.json"
    out.write_text(json.dumps(layer, ensure_ascii=False))
    print(f"  저장: {out.name} ({len(entities)} entities)")


# ═══════════════════════════════════════════════════════════════
#  레이어 4: error-cases.json — FP/FN 오탐지 하이라이트
# ═══════════════════════════════════════════════════════════════

def generate_error_cases(model, test_sc):
    """추론 오류 케이스(FP: 오탐, FN: 미탐)를 색상으로 구분하여 표시."""
    print("[error-cases] 생성 시작 ...")

    entities = []
    counts = defaultdict(int)

    with torch.no_grad():
        for idx, sc in enumerate(test_sc):
            typ = sc["type"]
            true_l = TYPE_MAP[typ]

            data, node_meta, _ = build_graph_20d(sc["drones"])
            logits, anomaly_score, emb = model(data)
            pred_l = int(logits.argmax().item())

            if pred_l == true_l:
                continue  # 정답은 스킵

            true_anom = ANOMALY_MAP[typ]
            pred_anom = ANOMALY_MAP[CLASS_NAMES[pred_l]]

            # 오류 유형 분류
            if {true_l, pred_l} <= {0, 1}:
                # 안전 클래스 내 혼동 (cooperative ↔ relay)
                err, color, width = "혼동", "#FFD700", 3
            elif true_anom == 0 and pred_anom == 1:
                # 실제 안전인데 위험으로 예측 → FP (거짓 양성)
                err, color, width = "FP", "#FF9800", 4
            elif true_anom == 1 and pred_anom == 0:
                # 실제 위험인데 안전으로 예측 → FN (거짓 음성, 더 위험)
                err, color, width = "FN", "#FF0000", 5
            else:
                err, color, width = "기타", "#E91E63", 3

            counts[err] += 1

            # 드론 경로 표시
            for di, dr in enumerate(sc["drones"]):
                agl_list = [max(float(a), 0.0) for a in dr["alts"]]
                positions = [
                    [float(lat), float(lon), agl]
                    for lat, lon, agl in zip(dr["lats"], dr["lngs"], agl_list)
                ]
                entities.append({
                    "type": "polyline",
                    "id": f"err_{idx}_d{di}",
                    "positions": positions,
                    "color": color, "width": width,
                    "label": f"{err}: {KR[typ]}→{KR[CLASS_NAMES[pred_l]]} D{di}",
                })
                entities.append({
                    "type": "point",
                    "id": f"err_{idx}_dep_d{di}",
                    "position": [float(dr["lats"][0]), float(dr["lngs"][0]), agl_list[0]],
                    "color": "#00E676", "size": 12,
                    "label": f"▶ D{di}",
                    "font_size": 11,
                    "heightReference": "RELATIVE_TO_GROUND",
                })

            # 오류 라벨
            cx = float(np.mean([l for d in sc["drones"] for l in d["lats"]]))
            cy = float(np.mean([l for d in sc["drones"] for l in d["lngs"]]))
            avg_agl = float(np.mean([max(float(a), 0.0) for d in sc["drones"] for a in d["alts"]]))

            if err == "FN":
                icon = "FN(미탐)"
                bg = "rgba(180,0,0,0.88)"
            elif err == "FP":
                icon = "FP(오탐)"
                bg = "rgba(200,100,0,0.88)"
            elif err == "혼동":
                icon = "혼동"
                bg = "rgba(100,100,0,0.88)"
            else:
                icon = "기타"
                bg = "rgba(100,0,60,0.88)"

            entities.append({
                "type": "point",
                "id": f"err_{idx}_label",
                "position": [cx, cy, avg_agl + 45],
                "color": "#FFFFFF", "size": 16,
                "label": (
                    f"{icon}: {KR[typ]}→{KR[CLASS_NAMES[pred_l]]}\n"
                    f"실제: {KR[typ]} | 예측: {KR[CLASS_NAMES[pred_l]]}\n"
                    f"이상탐지 점수: {float(anomaly_score):.3f}"
                ),
                "font_size": 14,
                "heightReference": "RELATIVE_TO_GROUND",
                "label_bg": bg,
            })

    print(f"  오류 분포: {dict(counts)}")
    layer = {
        "layer_id": "error-cases",
        "name": "오탐지(FP/FN) 하이라이트",
        "description": "R-GAT 오분류 케이스 시각화 — FN(빨강): 미탐, FP(주황): 오탐",
        "error_counts": dict(counts),
        "entities": entities,
    }
    out = OUTPUT_DIR / "error-cases.json"
    out.write_text(json.dumps(layer, ensure_ascii=False))
    print(f"  저장: {out.name} ({len(entities)} entities, {sum(counts.values())}건 오류)")


# ═══════════════════════════════════════════════════════════════
#  레이어 5: route-compare.json — 위험→안전 추천 경로 비교
# ═══════════════════════════════════════════════════════════════

def generate_route_compare(raw_sc):
    """충돌위험 시나리오(빨강) vs 유사 협력수색 시나리오(초록) 경로 비교."""
    print("[route-compare] 생성 시작 ...")

    from scripts.swarm_scenario_gen import offset_coords as oc
    collisions = [s for s in raw_sc if s["type"] == "collision_risk"][:30]
    PUSH_DIST = 350.0  # 고정 밀어내기 거리
    TOP_N = 3  # 시나리오당 가장 위험한 N개 WP만 수정

    entities = []
    for si, col in enumerate(collisions):
        drones = col["drones"]
        n_drones = len(drones)

        # ── 모든 드론 간 WP 쌍의 거리 계산 → 가장 가까운 TOP_N 쌍만 수정 ──
        all_pairs = []
        for di in range(n_drones):
            for dj in range(di + 1, n_drones):
                # 각 드론 쌍에서 가장 가까운 WP 쌍 1개만
                best_wi, best_wj, best_d = 0, 0, 1e9
                for wi in range(len(drones[di]["lats"])):
                    for wj in range(len(drones[dj]["lats"])):
                        d = haversine(
                            drones[di]["lats"][wi], drones[di]["lngs"][wi],
                            drones[dj]["lats"][wj], drones[dj]["lngs"][wj]
                        )
                        if d < best_d:
                            best_d = d
                            best_wi, best_wj = wi, wj
                all_pairs.append({"di": di, "wi": best_wi, "dj": dj, "wj": best_wj, "dist": best_d})

        # 가장 위험한 TOP_N 쌍
        all_pairs.sort(key=lambda x: x["dist"])
        conflict_pairs = all_pairs[:TOP_N]
        conflict_wps = set()
        for cp in conflict_pairs:
            conflict_wps.add((cp["dj"], cp["wj"]))

        if not conflict_pairs:
            continue

        n_conflicts = len(conflict_pairs)
        min_dist = conflict_pairs[0]["dist"]

        # ── ① 원래 경로 (빨강, 충돌 WP 노란색 강조) ──
        for di, dr in enumerate(drones):
            agl_list = [max(float(a), 0.0) for a in dr["alts"]]
            positions = [
                [float(lat), float(lon), agl]
                for lat, lon, agl in zip(dr["lats"], dr["lngs"], agl_list)
            ]
            entities.append({
                "type": "polyline",
                "id": f"rc_{si}_orig_d{di}",
                "positions": positions,
                "color": "#FF1744", "width": 3,
            })
            # 충돌 WP 강조 (노란색 큰 점)
            for wi in range(len(dr["lats"])):
                if (di, wi) in conflict_wps:
                    entities.append({
                        "type": "point",
                        "id": f"rc_{si}_conflict_d{di}_w{wi}",
                        "position": positions[wi],
                        "color": "#FFEB3B", "size": 10,
                        "outline_color": "#333333",
                    })

        # ── ② 수정된 전체 경로 생성 (충돌 WP만 밀어냄) ──
        # 먼저 모든 충돌 WP의 밀어진 좌표 계산
        pushed = {}  # (di, wi) → (new_lat, new_lon)
        for cp in conflict_pairs:
            di, wi = cp["dj"], cp["wj"]
            orig_lat = float(drones[di]["lats"][wi])
            orig_lon = float(drones[di]["lngs"][wi])
            other_lat = float(drones[cp["di"]]["lats"][cp["wi"]])
            other_lon = float(drones[cp["di"]]["lngs"][cp["wi"]])
            away_bearing = bearing(other_lat, other_lon, orig_lat, orig_lon)
            new_lat, new_lon = oc(orig_lat, orig_lon, PUSH_DIST, away_bearing)
            pushed[(di, wi)] = (new_lat, new_lon)

        # 수정된 경로: 충돌 WP는 밀어진 좌표, 나머지는 원래 좌표
        for di, dr in enumerate(drones):
            mod_positions = []
            for wi in range(len(dr["lats"])):
                agl = max(float(dr["alts"][wi]), 0.0)
                if (di, wi) in pushed:
                    lat, lon = pushed[(di, wi)]
                else:
                    lat = float(dr["lats"][wi])
                    lon = float(dr["lngs"][wi])
                mod_positions.append([lat, lon, agl])

            entities.append({
                "type": "polyline",
                "id": f"rc_{si}_mod_d{di}",
                "positions": mod_positions,
                "color": "#00E676", "width": 3,
            })

        # ── ③ 흰 점선: 원래 WP → 밀어진 WP ──
        for (di, wi), (new_lat, new_lon) in pushed.items():
            orig_lat = float(drones[di]["lats"][wi])
            orig_lon = float(drones[di]["lngs"][wi])
            orig_agl = max(float(drones[di]["alts"][wi]), 0.0)
            move_dist = haversine(orig_lat, orig_lon, new_lat, new_lon)

            entities.append({
                "type": "polyline",
                "id": f"rc_{si}_push_d{di}_w{wi}",
                "positions": [
                    [orig_lat, orig_lon, orig_agl + 5],
                    [new_lat, new_lon, orig_agl + 5],
                ],
                "color": "rgba(255,255,255,0.5)", "width": 1,
                "dash": True,
            })

        # ── ④ 설명 라벨 ──
        cx = float(np.mean([l for d in drones for l in d["lats"]]))
        cy = float(np.mean([l for d in drones for l in d["lngs"]]))
        avg_agl = float(np.mean([max(float(a), 0.0) for d in drones for a in d["alts"]]))
        entities.append({
            "type": "point",
            "id": f"rc_{si}_info",
            "position": [cx, cy, avg_agl + 65],
            "color": "#FFFFFF", "size": 14,
            "label": f"추천#{si+1} | 충돌 {n_conflicts}곳 (최소{min_dist:.0f}m) | 노랑→초록 {PUSH_DIST:.0f}m 이동",
            "font_size": 12,
            "label_bg": "rgba(20,20,40,0.92)",
        })

    layer = {
        "layer_id": "route-compare",
        "name": "추천 경로 비교",
        "description": "충돌위험→협력수색 안전 경로 추천 비교 (10쌍)",
        "n_pairs": len(collisions),
        "entities": entities,
    }
    out = OUTPUT_DIR / "route-compare.json"
    out.write_text(json.dumps(layer, ensure_ascii=False))
    print(f"  저장: {out.name} ({len(entities)} entities)")


# ═══════════════════════════════════════════════════════════════
#  레이어 6: temporal-handoff.json — Temporal(시간동기) 핸드오프 관계선
# ═══════════════════════════════════════════════════════════════

def generate_temporal_handoff(raw_sc):
    """relay 시나리오에서 드론i 끝→드론i+1 시작 핸드오프 관계선 시각화."""
    print("[temporal-handoff] 생성 시작 ...")

    relays = [s for s in raw_sc if s["type"] == "relay_move"][:30]
    entities = []

    for si, sc in enumerate(relays):
        drones = sc["drones"]
        # 각 드론 경로 표시 (색상 구분)
        drone_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
        for di, dr in enumerate(drones):
            positions = [
                [float(lat), float(lon), max(float(alt), 0.0)]
                for lat, lon, alt in zip(dr["lats"], dr["lngs"], dr["alts"])
            ]
            entities.append({
                "type": "polyline",
                "id": f"relay_{si}_d{di}",
                "positions": positions,
                "color": drone_colors[di % len(drone_colors)],
                "width": 3,
                "label": f"릴레이#{si+1} D{di} ({len(dr['lats'])}WP)",
            })
            # 시작점/끝점 마커
            entities.append({
                "type": "point",
                "id": f"relay_{si}_d{di}_start",
                "position": positions[0],
                "color": "#00E676", "size": 12,
                "label": f"D{di} 출발",
            })
            entities.append({
                "type": "point",
                "id": f"relay_{si}_d{di}_end",
                "position": positions[-1],
                "color": "#FF6D00", "size": 12,
                "label": f"D{di} 도착",
            })

        # 핸드오프 연결선 (드론i 끝→드론i+1 시작)
        for di in range(len(drones) - 1):
            end_lat = float(drones[di]["lats"][-1])
            end_lon = float(drones[di]["lngs"][-1])
            end_alt = max(float(drones[di]["alts"][-1]), 0.0)
            start_lat = float(drones[di + 1]["lats"][0])
            start_lon = float(drones[di + 1]["lngs"][0])
            start_alt = max(float(drones[di + 1]["alts"][0]), 0.0)

            # 핸드오프 거리 계산
            R = 6371000.0
            dlat = math.radians(start_lat - end_lat)
            dlon = math.radians(start_lon - end_lon)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(end_lat)) * math.cos(math.radians(start_lat)) * math.sin(dlon / 2) ** 2
            dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            entities.append({
                "type": "polyline",
                "id": f"handoff_{si}_d{di}to{di+1}",
                "positions": [
                    [end_lat, end_lon, end_alt + 5],
                    [start_lat, start_lon, start_alt + 5],
                ],
                "color": "#00BCD4",
                "width": 4,
                "label": f"핸드오프 D{di}→D{di+1} ({dist:.0f}m)",
                "label_bg": "rgba(0,150,136,0.7)",
            })

    layer = {
        "layer_id": "temporal-handoff",
        "name": "Temporal(시간동기) 핸드오프",
        "description": "relay 시나리오에서 드론 간 임무 인계 연결 (끝→시작)",
        "n_scenarios": len(relays),
        "entities": entities,
    }
    out = OUTPUT_DIR / "temporal-handoff.json"
    out.write_text(json.dumps(layer, ensure_ascii=False))
    print(f"  저장: {out.name} ({len(entities)} entities, {len(relays)} 시나리오)")


# ═══════════════════════════════════════════════════════════════
#  레이어 7: model-comparison.json — GCN/GAT/R-GAT 비교
# ═══════════════════════════════════════════════════════════════

def generate_model_comparison(test_sc):
    """같은 시나리오를 GCN/GAT/R-GAT 3개 모델로 판정, 고도 분리 표시."""
    print("[model-comparison] 생성 시작 ...")

    from torch_geometric.nn import GCNConv

    # 3개 모델 로드
    model_configs = [
        {"name": "GCN", "path": "v3_baseline_GCN-15d_best.pt", "color": "#9E9E9E", "alt_offset": 0},
        {"name": "GAT", "path": "v3_baseline_GAT-15d_best.pt", "color": "#2196F3", "alt_offset": 30},
        {"name": "R-GAT", "path": "v3_baseline_R-GAT-15d_best.pt", "color": "#4CAF50", "alt_offset": 60},
    ]

    # GCN/GAT baseline 모델 클래스
    class GCNBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(15, HIDDEN_DIM)
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
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(15, HIDDEN_DIM, heads=HEADS, concat=False, dropout=0.1)
            self.bn = nn.BatchNorm1d(HIDDEN_DIM)
            self.conv2 = GATConv(HIDDEN_DIM, OUT_DIM, heads=HEADS, concat=False, dropout=0.1)
            self.pool = AttnPool(OUT_DIM)
            self.th = nn.Linear(OUT_DIM, 4)
            self.ah = nn.Linear(OUT_DIM, 1)
        def forward(self, data):
            h = F.relu(self.bn(self.conv1(data.x, data.edge_index)))
            h = self.conv2(h, data.edge_index)
            emb = self.pool(h)
            return self.th(emb), torch.sigmoid(self.ah(emb)).squeeze(), emb

    # 충돌+수색중복 시나리오만 (이상 유형)
    anomaly_sc = [s for s in test_sc if s["type"] in ("collision_risk", "coverage_overlap")][:20]
    entities = []
    MODEL_BASE = PROJECT_ROOT / "fly2vec" / "data" / "models"

    for mc in model_configs:
        mp = MODEL_BASE / mc["path"]
        if not mp.exists():
            print(f"  경고: {mc['path']} 없음 — 건너뜀")
            continue

        if mc["name"] == "GCN":
            m = GCNBaseline()
        elif mc["name"] == "GAT":
            m = GATBaseline()
        else:
            m = Fly2VecRGAT(in_dim=15)
        m.load_state_dict(torch.load(mp, map_location="cpu", weights_only=True))
        m.eval()

        with torch.no_grad():
            for si, sc in enumerate(anomaly_sc):
                # 15d 그래프 (20d 빌드 후 앞 15개만 사용)
                all_feats, node_meta = _build_node_features_20d(sc["drones"])
                all_feats_15d = [f[:15] for f in all_feats]
                edges = _build_edges(node_meta, {0, 1, 2})  # E-SPC
                x = torch.tensor(all_feats_15d, dtype=torch.float32)
                ei_list, et_list = [], []
                for t in range(5):
                    for e in edges[t]:
                        ei_list.append(e)
                        et_list.append(t)
                if ei_list:
                    edge_index = torch.tensor(ei_list, dtype=torch.long).T
                    edge_type = torch.tensor(et_list, dtype=torch.long)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_type = torch.zeros(0, dtype=torch.long)
                data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

                logits, anomaly_prob, emb = m(data)
                pred_type = int(logits.argmax())
                pred_anomaly = float(anomaly_prob)
                type_names = ["cooperative", "relay", "collision", "coverage"]

                # 경로 표시 (고도 offset)
                for di, dr in enumerate(sc["drones"]):
                    positions = [
                        [float(lat), float(lon), max(float(alt), 0.0) + mc["alt_offset"]]
                        for lat, lon, alt in zip(dr["lats"], dr["lngs"], dr["alts"])
                    ]
                    actual = sc["type"].replace("_search", "").replace("_risk", "").replace("_overlap", "").replace("_move", "")
                    entities.append({
                        "type": "polyline",
                        "id": f"mc_{mc['name']}_{si}_d{di}",
                        "positions": positions,
                        "color": mc["color"],
                        "width": 3,
                        "label": f"{mc['name']} #{si+1} | 판정:{type_names[pred_type]} | 이상:{pred_anomaly:.0%} | 실제:{actual}",
                        "label_bg": "rgba(50,50,50,0.7)",
                    })

    layer = {
        "layer_id": "model-comparison",
        "name": "모델 비교 (GCN/GAT/R-GAT)",
        "description": "같은 시나리오를 3개 모델로 판정 — 고도 분리 (GCN 0m / GAT +30m / R-GAT +60m)",
        "n_scenarios": len(anomaly_sc),
        "entities": entities,
    }
    out = OUTPUT_DIR / "model-comparison.json"
    out.write_text(json.dumps(layer, ensure_ascii=False))
    print(f"  저장: {out.name} ({len(entities)} entities)")


# ═══════════════════════════════════════════════════════════════
#  레이어 8: feature-upgrade.json — 15d vs 20d 위험도 비교
# ═══════════════════════════════════════════════════════════════

def generate_feature_upgrade(model_20d, test_sc):
    """같은 WP의 15d vs 20d 위험도 차이 시각화."""
    print("[feature-upgrade] 생성 시작 ...")

    # 15d 모델 로드
    model_15d_path = PROJECT_ROOT / "fly2vec" / "data" / "models" / "v3_e1_E-SPC_best.pt"
    if not model_15d_path.exists():
        print(f"  경고: {model_15d_path.name} 없음 — 건너뜀")
        return

    model_15d = Fly2VecRGAT(in_dim=15)
    model_15d.load_state_dict(torch.load(model_15d_path, map_location="cpu", weights_only=True))
    model_15d.eval()

    # 이상 시나리오만
    anomaly_sc = [s for s in test_sc if s["type"] in ("collision_risk", "coverage_overlap")][:15]
    entities = []

    with torch.no_grad():
        for si, sc in enumerate(anomaly_sc):
            drones = sc["drones"]

            # 15d 추론 (20d 빌드 후 앞 15개)
            feats_all, nm_15d = _build_node_features_20d(drones)
            feats_15d = [f[:15] for f in feats_all]
            edges_15d = _build_edges(nm_15d, {0, 1, 2})
            x_15d = torch.tensor(feats_15d, dtype=torch.float32)
            ei_list, et_list = [], []
            for t in range(5):
                for e in edges_15d[t]:
                    ei_list.append(e); et_list.append(t)
            if ei_list:
                edge_index = torch.tensor(ei_list, dtype=torch.long).T
                edge_type = torch.tensor(et_list, dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_type = torch.zeros(0, dtype=torch.long)
            data_15d = Data(x=x_15d, edge_index=edge_index, edge_type=edge_type)
            _, anom_15d, _ = model_15d(data_15d)

            # 20d 추론
            data_20d, nm_20d, _ = build_graph_20d(drones)
            _, anom_20d, _ = model_20d(data_20d)

            diff = float(anom_20d) - float(anom_15d)
            # 15d 경로 (빨강 계열, 아래층)
            for di, dr in enumerate(drones):
                positions = [
                    [float(lat), float(lon), max(float(alt), 0.0)]
                    for lat, lon, alt in zip(dr["lats"], dr["lngs"], dr["alts"])
                ]
                entities.append({
                    "type": "polyline",
                    "id": f"fu15_{si}_d{di}",
                    "positions": positions,
                    "color": "#E91E63", "width": 3,
                    "label": f"15d #{si+1} D{di} | 이상:{float(anom_15d):.0%}",
                })

            # 20d 경로 (초록 계열, +30m 위층)
            for di, dr in enumerate(drones):
                positions = [
                    [float(lat), float(lon), max(float(alt), 0.0) + 30]
                    for lat, lon, alt in zip(dr["lats"], dr["lngs"], dr["alts"])
                ]
                entities.append({
                    "type": "polyline",
                    "id": f"fu20_{si}_d{di}",
                    "positions": positions,
                    "color": "#4CAF50", "width": 3,
                    "label": f"20d #{si+1} D{di} | 이상:{float(anom_20d):.0%} (차이:{diff:+.0%})",
                    "label_bg": "rgba(50,50,50,0.7)",
                })

    layer = {
        "layer_id": "feature-upgrade",
        "name": "15d vs 20d Feature(피처) 비교",
        "description": "같은 시나리오의 15d(빨강, 아래)와 20d(초록, 위) 이상탐지 점수 비교",
        "n_scenarios": len(anomaly_sc),
        "entities": entities,
    }
    out = OUTPUT_DIR / "feature-upgrade.json"
    out.write_text(json.dumps(layer, ensure_ascii=False))
    print(f"  저장: {out.name} ({len(entities)} entities)")


# ═══════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Cesium 레이어 생성 — report12 v3 20d E-ALL")
    print("=" * 65)
    t0 = time.time()

    # 시나리오 로드
    print("\n[데이터] 시나리오 로드 중 ...")
    raw_sc, train_sc, test_sc = load_scenarios()
    print(f"  전체: {len(raw_sc)}건 | 학습: {len(train_sc)} | 테스트: {len(test_sc)}")
    for typ, cnt in sorted(
        ((t, sum(1 for s in raw_sc if s["type"] == t)) for t in TYPE_MAP),
        key=lambda x: x[1], reverse=True
    ):
        print(f"    {KR[typ]:6s}: {cnt}건")

    # 모델 로드
    print(f"\n[모델] {MODEL_PATH.name} 로드 중 ...")
    model = load_model()
    print("  로드 완료")

    print()

    # 레이어 1: 위험도 히트맵
    generate_risk_heatmap(model, test_sc)

    # 레이어 2: 어텐션 엣지
    generate_e8_attention(model, test_sc)

    # 레이어 3: E1 엣지 비교 (모델 추론 불필요 — 결과 JSON 기반)
    generate_e1_ablation(raw_sc)

    # 레이어 4: 오탐지 하이라이트
    generate_error_cases(model, test_sc)

    # 레이어 5: 추천 경로 비교 (모델 추론 불필요 — 시나리오 비교)
    generate_route_compare(raw_sc)

    # 레이어 6: Temporal(시간동기) 핸드오프 — relay 시나리오
    generate_temporal_handoff(raw_sc)

    # 레이어 7: GCN/GAT/R-GAT 모델 비교
    generate_model_comparison(test_sc)

    # 레이어 8: 15d vs 20d 위험도 비교
    generate_feature_upgrade(model, test_sc)

    # Cesium HTML 복사 (report10 기반 → report12에 독립 사본)
    src_html = PROJECT_ROOT / "results" / "report10" / "swarm_all_cesium.html"
    dst_html = OUTPUT_DIR.parent / "swarm_all_cesium.html"
    if src_html.exists():
        import shutil
        shutil.copy2(src_html, dst_html)
        print(f"\n[HTML] {dst_html.name} 복사 완료")
    else:
        print(f"\n[경고] {src_html} 없음 — HTML 수동 복사 필요")

    # 최종 요약
    elapsed = time.time() - t0
    print(f"\n완료 ({elapsed:.0f}초)")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print(f"Cesium 열기: cd {OUTPUT_DIR.parent} && python3 -m http.server 8888")
    print()
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        kb = f.stat().st_size / 1024
        print(f"  {f.name:30s} {kb:8.0f} KB")


if __name__ == "__main__":
    main()
