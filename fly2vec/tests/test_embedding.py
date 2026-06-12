"""Fly2Vec Embedding 인코더 단위 테스트."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest


class TestNode2VecEncoder:
    """Node2VecEncoder 단위 테스트."""

    def _get_encoder(self):
        from fly2vec.src.embedding.node2vec_encoder import Node2VecEncoder
        return Node2VecEncoder(dimensions=128)

    def test_build_graph_nodes(self, sample_pg):
        """6 WP → 6 nodes."""
        encoder = self._get_encoder()
        G = encoder.build_graph(sample_pg)
        assert len(G.nodes) == 6

    def test_build_graph_sequential_edges(self, sample_pg):
        """6 WP → sequential edges 5개."""
        encoder = self._get_encoder()
        G = encoder.build_graph(sample_pg)
        seq_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'sequential']
        assert len(seq_edges) == 5

    def test_build_graph_spatial_knn_edges(self, sample_pg):
        """Spatial KNN 엣지가 존재."""
        encoder = self._get_encoder()
        G = encoder.build_graph(sample_pg)
        knn_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'spatial_knn']
        assert len(knn_edges) > 0

    def test_encode_mission_shape(self, sample_pg):
        """반환 벡터가 list[float], 길이 128."""
        encoder = self._get_encoder()
        vector = encoder.encode_mission(sample_pg)
        assert isinstance(vector, list)
        assert len(vector) == 128
        assert all(isinstance(v, float) for v in vector)

    def test_encode_empty_pg(self):
        """WP 0~1개 → [0.0]*128 fallback."""
        encoder = self._get_encoder()
        # 빈 PG
        vector = encoder.encode_mission({'waypoints': []})
        assert vector == [0.0] * 128
        # 1개 WP
        vector = encoder.encode_mission({'waypoints': [{'lat_deg': 37.0, 'lon_deg': 127.0, 'alt_m': 50, 'seq': 0}]})
        assert vector == [0.0] * 128


class TestGCNEncoder:
    """GCNEncoder 단위 테스트."""

    def test_encode_mission_shape(self, sample_pg):
        """128d 벡터, float 값."""
        from fly2vec.src.embedding.gcn_encoder import GCNEncoder
        encoder = GCNEncoder(dimensions=128)
        vector = encoder.encode_mission(sample_pg)
        assert isinstance(vector, list)
        assert len(vector) == 128
        assert all(isinstance(v, float) for v in vector)


class TestRGATEncoder:
    """RGATEncoder 단위 테스트."""

    def _get_encoder(self):
        from fly2vec.src.embedding.rgat_encoder import RGATEncoder
        return RGATEncoder(dimensions=128)

    def test_analyze_cooperative(self, sample_swarm_cooperative):
        """영역 분리 3대 → pattern='cooperative_search', anomaly_avg < 0.3."""
        encoder = self._get_encoder()
        result = encoder.analyze(sample_swarm_cooperative)
        assert result['pattern'] == 'cooperative_search'
        assert result['anomaly_avg'] < 0.3

    def test_analyze_collision(self, sample_swarm_collision):
        """근접 배치 → pattern='collision_risk', proximity edges > 0."""
        encoder = self._get_encoder()
        result = encoder.analyze(sample_swarm_collision)
        assert result['pattern'] == 'collision_risk'
        assert result['edge_stats']['proximity'] > 0

    def test_detect_conflicts_found(self, sample_swarm_collision):
        """30m 이내 WP → conflicts 비어있지 않음."""
        encoder = self._get_encoder()
        conflicts = encoder.detect_conflicts(sample_swarm_collision, threshold_m=500.0)
        assert len(conflicts) > 0
        assert all('distance_m' in c for c in conflicts)

    def test_detect_no_conflict(self, sample_swarm_cooperative):
        """2km 분리 → conflicts 빈 리스트."""
        encoder = self._get_encoder()
        conflicts = encoder.detect_conflicts(sample_swarm_cooperative, threshold_m=100.0)
        # 2km+ 분리된 경로이므로 100m 임계값에서는 충돌 없음
        assert len(conflicts) == 0
