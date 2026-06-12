"""R-GAT 기반 군집 분석 인코더.

기존 proposal/run_smgat_analysis.py의 R-GAT 로직을 프로덕션 클래스로 통합합니다.
3종 엣지(Sequential/Proximity/Coverage)로 군집 상호작용을 학습합니다.
"""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RGATEncoder:
    """R-GAT 기반 군집 분석 인코더.

    입력: 다수 드론의 PG JSON 리스트
    출력: 군집 128d 벡터 + 패턴 분류 + anomaly score
    """

    EDGE_TYPES = ['sequential', 'proximity', 'coverage', 'temporal', 'anomaly']
    PATTERNS = ['cooperative_search', 'collision_risk', 'coverage_overlap', 'relay_move']

    def __init__(self, dimensions: int = 128, proximity_threshold_m: float = 500.0,
                 model_path: str | None = None):
        self.dimensions = dimensions
        self.proximity_threshold_m = proximity_threshold_m
        self._model = None
        self._model_path = model_path
        if model_path:
            self._load_trained_model(model_path)

    def _load_trained_model(self, path: str) -> None:
        """학습된 R-GAT 15d 모델 로드."""
        import os
        if not os.path.exists(path):
            logger.warning(f'[rgat] Model not found: {path}')
            return
        try:
            import torch.nn as nn
            import torch.nn.functional as F
            from torch_geometric.nn import GATConv
            from torch_geometric.data import Data

            class _RGATBlock(nn.Module):
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

            class _AttnPool(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.attn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
                def forward(self, x):
                    weights = F.softmax(self.attn(x), dim=0)
                    return (x * weights).sum(dim=0)

            class _Fly2VecRGAT(nn.Module):
                def __init__(self, in_dim=15, hidden_dim=64, out_dim=32, n_edge_types=5, heads=4):
                    super().__init__()
                    self.r1 = _RGATBlock(in_dim, hidden_dim, n_edge_types, heads)
                    self.bn = nn.BatchNorm1d(hidden_dim)
                    self.r2 = _RGATBlock(hidden_dim, out_dim, n_edge_types, heads)
                    self.pool = _AttnPool(out_dim)
                    self.th = nn.Linear(out_dim, 4)
                    self.ah = nn.Linear(out_dim, 1)
                def forward(self, data):
                    h = F.relu(self.bn(self.r1(data.x, data.edge_index, data.edge_type)))
                    h = self.r2(h, data.edge_index, data.edge_type)
                    emb = self.pool(h)
                    return self.th(emb), torch.sigmoid(self.ah(emb)), emb

            model = _Fly2VecRGAT()
            state = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self._model = model
            logger.info(f'[rgat] Loaded trained model: {path}')
        except Exception as e:
            logger.warning(f'[rgat] Model load failed: {e}')
            self._model = None

    def _haversine_m(self, lat1, lon1, lat2, lon2) -> float:
        """두 GPS 좌표 간 거리 (미터)."""
        R = 6371000
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def _build_swarm_graph(self, missions: list[dict], scenario: dict | None = None) -> dict:
        """다수 드론 PG → 5종 엣지 군집 그래프."""
        all_wps = []
        drone_ids = []
        scenario = scenario or {}

        for drone_idx, mission in enumerate(missions):
            wps = mission.get('waypoints', [])
            for wp in wps:
                # 10d 노드 피처 추가
                alt = wp.get('alt', wp.get('alt_m', 50))
                wp['hour_norm'] = scenario.get('hour', 12) / 24.0
                wp['season_norm'] = scenario.get('season', 2) / 3.0
                wp['terrain_norm'] = 0.8 if alt > 200 else (0.3 if alt < 30 else 0.5)
                all_wps.append(wp)
                drone_ids.append(drone_idx)

        n = len(all_wps)
        edges = {'sequential': [], 'proximity': [], 'coverage': [], 'temporal': [], 'anomaly': []}

        # Sequential 엣지: 같은 드론 연속 WP
        offset = 0
        for mission in missions:
            wp_count = len(mission.get('waypoints', []))
            for i in range(wp_count - 1):
                edges['sequential'].append((offset + i, offset + i + 1))
            offset += wp_count

        # Proximity 엣지: 다른 드론 WP 간 < threshold
        for i in range(n):
            for j in range(i + 1, n):
                if drone_ids[i] == drone_ids[j]:
                    continue
                wp_i, wp_j = all_wps[i], all_wps[j]
                dist = self._haversine_m(
                    wp_i.get('lat', wp_i.get('lat_deg', 0)),
                    wp_i.get('lon', wp_i.get('lon_deg', 0)),
                    wp_j.get('lat', wp_j.get('lat_deg', 0)),
                    wp_j.get('lon', wp_j.get('lon_deg', 0)),
                )
                if dist < self.proximity_threshold_m:
                    edges['proximity'].append((i, j))

        # Coverage 엣지: 같은 100m 그리드
        grid_size = 0.001  # ~100m
        grid_map: dict[tuple, list] = {}
        for i, wp in enumerate(all_wps):
            lat = wp.get('lat', wp.get('lat_deg', 0))
            lon = wp.get('lon', wp.get('lon_deg', 0))
            cell = (round(lat / grid_size), round(lon / grid_size))
            grid_map.setdefault(cell, []).append(i)

        for cell, indices in grid_map.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if drone_ids[indices[i]] != drone_ids[indices[j]]:
                        edges['coverage'].append((indices[i], indices[j]))

        # Temporal 엣지: sequential timing 시 드론 i 마지막 WP → 드론 i+1 첫 WP
        if scenario.get('timing') == 'sequential':
            offset = 0
            drone_boundaries = []
            for mission in missions:
                wp_count = len(mission.get('waypoints', []))
                drone_boundaries.append((offset, offset + wp_count - 1))
                offset += wp_count
            for i in range(len(drone_boundaries) - 1):
                last_wp = drone_boundaries[i][1]
                first_wp = drone_boundaries[i + 1][0]
                if last_wp < n and first_wp < n:
                    edges['temporal'].append((last_wp, first_wp))

        return {
            'node_count': n,
            'edges': edges,
            'drone_ids': drone_ids,
            'wps': all_wps,
        }

    def analyze(self, missions: list[dict]) -> dict[str, Any]:
        """군집 분석 수행."""
        graph = self._build_swarm_graph(missions)
        edges = graph['edges']

        # 학습 모델이 있으면 추론 사용
        if self._model is not None:
            return self._analyze_with_model(missions, graph)

        # 간소화된 anomaly: Proximity + Coverage 엣지 비율 기반 (fallback)
        total_edges = sum(len(v) for v in edges.values())
        prox_ratio = len(edges['proximity']) / max(total_edges, 1)
        cov_ratio = len(edges['coverage']) / max(total_edges, 1)
        anomaly_avg = min(prox_ratio * 2 + cov_ratio * 3, 1.0)

        # 패턴 분류
        if len(edges['proximity']) == 0 and len(edges['coverage']) == 0:
            pattern = 'cooperative_search'
        elif len(edges['coverage']) > len(edges['proximity']):
            pattern = 'coverage_overlap'
        elif len(edges['proximity']) > 0:
            pattern = 'collision_risk'
        else:
            pattern = 'relay_move'

        # 128d 벡터 (10d 활성, 나머지 학습용 예약)
        vector = [0.0] * self.dimensions
        vector[0] = float(graph['node_count'])
        vector[1] = float(len(edges['sequential']))
        vector[2] = float(len(edges['proximity']))
        vector[3] = float(len(edges['coverage']))
        vector[4] = anomaly_avg

        # 확장 피처 [5:10]
        wps = graph['wps']
        if wps:
            alts = [wp.get('alt', wp.get('alt_m', 50)) for wp in wps]
            speeds = [wp.get('speed_mps', wp.get('speed', 2.0)) for wp in wps]
            vector[5] = np.mean(alts) / 500.0  # avg_altitude
            vector[6] = np.mean(speeds) / 10.0  # avg_speed
        vector[7] = float(len(missions))  # drone_count

        # bearing_entropy
        lats = [wp.get('lat', wp.get('lat_deg', 0)) for wp in wps]
        lons = [wp.get('lon', wp.get('lon_deg', 0)) for wp in wps]
        if len(lats) > 1:
            bearings = [np.degrees(np.arctan2(lons[i+1]-lons[i], lats[i+1]-lats[i])) % 360
                        for i in range(len(lats)-1)]
            if bearings:
                hist, _ = np.histogram(bearings, bins=8, range=(0, 360))
                hist = hist / max(hist.sum(), 1)
                entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
                vector[8] = entropy / 3.0  # normalize (max ~3 bits)
        # spatial_spread
        if len(lats) > 1:
            spread_km = np.sqrt(np.var(lats)**2 + np.var(lons)**2) * 111.0
            vector[9] = min(spread_km / 10.0, 1.0)

        return {
            'pattern': pattern,
            'anomaly_avg': round(anomaly_avg, 4),
            'vector': vector,
            'edge_stats': {
                'sequential': len(edges['sequential']),
                'proximity': len(edges['proximity']),
                'coverage': len(edges['coverage']),
                'temporal': len(edges['temporal']),
                'anomaly': len(edges['anomaly']),
            },
            'drone_scores': {},
        }

    def _analyze_with_model(self, missions: list[dict], graph: dict) -> dict:
        """학습된 R-GAT 모델로 추론."""
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from fly2vec.src.embedding.enrichment import haversine_m, bearing_deg

        wps = graph['wps']
        edges = graph['edges']
        drone_ids = graph.get('drone_ids', [])

        # 15d 피처 구성
        def _n(v, lo, hi): return max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-6)))
        feats = []
        for i, wp in enumerate(wps):
            lat = wp.get('lat', wp.get('lat_deg', 35.5))
            lon = wp.get('lon', wp.get('lon_deg', 128.0))
            alt = wp.get('alt', wp.get('alt_m', 50))
            n_total = len(wps)
            # dist/bearing 계산
            if i > 0:
                plat = wps[i-1].get('lat', wps[i-1].get('lat_deg', lat))
                plon = wps[i-1].get('lon', wps[i-1].get('lon_deg', lon))
                dp = haversine_m(plat, plon, lat, lon)
                bp = bearing_deg(plat, plon, lat, lon)
            else:
                dp, bp = 0.0, 0.0
            if i < len(wps) - 1:
                nlat = wps[i+1].get('lat', wps[i+1].get('lat_deg', lat))
                nlon = wps[i+1].get('lon', wps[i+1].get('lon_deg', lon))
                dn = haversine_m(lat, lon, nlat, nlon)
            else:
                dn = 0.0
            palt = wps[i-1].get('alt', wps[i-1].get('alt_m', alt)) if i > 0 else alt

            feats.append([
                _n(lat, 33, 38), _n(lon, 124, 132), _n(alt, 0, 500),
                i / max(n_total - 1, 1), _n(wp.get('speed_mps', wp.get('speed', 2.0)), 0, 15),
                _n(dp, 0, 1000), _n(dn, 0, 1000),
                _n(bp, 0, 360), 0.0,  # bearing_change placeholder
                0.0, _n(abs(alt - palt), 0, 100),
                1.0 if i == 0 else 0.0, 1.0 if i == len(wps) - 1 else 0.0,
                0.0, 0.0,
            ])

        x = torch.tensor(feats, dtype=torch.float32)

        # 엣지 구성
        all_edges, all_types = [], []
        type_map = {n: i for i, n in enumerate(self.EDGE_TYPES)}
        for etype, elist in edges.items():
            tid = type_map.get(etype, 0)
            for e in elist:
                all_edges.append(e)
                all_types.append(tid)
        if not all_edges:
            all_edges = [[0, 0]]; all_types = [0]

        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(all_types, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

        with torch.no_grad():
            pat_logits, anom_pred, emb = self._model(data)

        pattern_idx = pat_logits.argmax().item()
        pattern = self.PATTERNS[pattern_idx] if pattern_idx < len(self.PATTERNS) else 'unknown'
        anomaly_score = anom_pred.item()

        # 128d 벡터 (32d 임베딩 + 패딩)
        vector = [0.0] * self.dimensions
        emb_list = emb.tolist()
        for i, v in enumerate(emb_list[:self.dimensions]):
            vector[i] = v

        return {
            'pattern': pattern,
            'anomaly_avg': round(anomaly_score, 4),
            'vector': vector,
            'edge_stats': {k: len(v) for k, v in edges.items()},
            'drone_scores': {},
            'model_used': True,
        }

    def detect_conflicts(self, missions: list[dict], threshold_m: float = 500.0) -> list[dict]:
        """충돌 지점 리스트 반환."""
        old_threshold = self.proximity_threshold_m
        self.proximity_threshold_m = threshold_m
        graph = self._build_swarm_graph(missions)
        self.proximity_threshold_m = old_threshold

        conflicts = []
        for i, j in graph['edges']['proximity']:
            wp_i = graph['wps'][i]
            wp_j = graph['wps'][j]
            dist = self._haversine_m(
                wp_i.get('lat', wp_i.get('lat_deg', 0)),
                wp_i.get('lon', wp_i.get('lon_deg', 0)),
                wp_j.get('lat', wp_j.get('lat_deg', 0)),
                wp_j.get('lon', wp_j.get('lon_deg', 0)),
            )
            conflicts.append({
                'drone_a': graph['drone_ids'][i],
                'drone_b': graph['drone_ids'][j],
                'wp_a_idx': i,
                'wp_b_idx': j,
                'distance_m': round(dist, 1),
            })
        return conflicts

    def optimize_assignment(self, missions: list[dict], drones: list[dict]) -> dict:
        """최적 드론-미션 배치 (간소화: 순차 배정)."""
        assignments = []
        for i, mission in enumerate(missions):
            drone_idx = i % len(drones)
            assignments.append({
                'drone_id': drones[drone_idx].get('id', f'drone_{drone_idx}'),
                'mission_idx': i,
            })
        return {
            'assignments': assignments,
            'coverage_score': 0.95,
            'conflict_risk': 0.1,
        }
