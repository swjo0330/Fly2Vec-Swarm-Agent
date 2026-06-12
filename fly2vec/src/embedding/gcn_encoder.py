"""GCN 기반 미션 경로 인코더.

기존 proposal/run_gcn_analysis.py의 GCN 학습 로직을 프로덕션 클래스로 통합합니다.
Node2Vec 한계(위치 미반영)를 극복: 4d → 64d → 128d Message Passing.
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fly2vec.src.embedding.enrichment import build_node_features_15d, enrich_pg_features
from fly2vec.src.embedding.node2vec_encoder import Node2VecEncoder

logger = logging.getLogger(__name__)


class GCNModel(torch.nn.Module):
    """2-Layer GCN: 15d → 64d → 128d."""

    def __init__(self, in_channels: int = 15, hidden: int = 64, out_channels: int = 128):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GCNEncoder:
    """GCN 기반 PG → 128d 벡터 인코더.

    노드 피처 [lat_norm, lon_norm, alt_norm, seq_ratio] (4d) →
    Message Passing → 128d WP 벡터 → 가중 평균 풀링 → 미션 벡터.
    """

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions
        self.model = GCNModel(in_channels=15, out_channels=dimensions)
        self.model.eval()
        self._graph_builder = Node2VecEncoder(dimensions=dimensions)

    def encode_mission(self, pg_json: dict) -> list[float]:
        """PG JSON → 128d 미션 벡터 (GCN, 15d 노드 피처)."""
        # 보강 (enrichment가 안 된 PG도 처리)
        enriched = enrich_pg_features(pg_json)
        G = self._graph_builder.build_graph(enriched)

        if len(G.nodes) < 2:
            return [0.0] * self.dimensions

        # 노드 피처: 15d (enrichment.py에서 보강된 WP 데이터 사용)
        wps = enriched.get('waypoints', [])
        features = []
        for node in sorted(G.nodes):
            if node < len(wps):
                features.append(build_node_features_15d(wps[node]))
            else:
                features.append([0.0] * 15)

        x = torch.tensor(features, dtype=torch.float32)

        # 엣지 인덱스
        edges = list(G.edges)
        if not edges:
            return [0.0] * self.dimensions
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # GCN forward
        with torch.no_grad():
            node_embeddings = self.model(x, edge_index)

        # 가중 평균 풀링
        weights = torch.ones(len(G.nodes))
        weights[0] = 2.0
        weights[-1] = 2.0
        weights = weights / weights.sum()

        mission_vector = (node_embeddings * weights.unsqueeze(1)).sum(dim=0)
        return mission_vector.tolist()
