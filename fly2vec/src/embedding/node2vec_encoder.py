"""Node2Vec 기반 미션 경로 인코더.

기존 실험 코드(run_full_analysis.py, run_gcn_analysis.py)의 핵심 로직을
프로덕션 클래스로 통합합니다.
"""

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class Node2VecEncoder:
    """Node2Vec 기반 PG → 128d 벡터 인코더.

    기존 proposal/run_full_analysis.py의 그래프 구성 + Node2Vec 학습 로직을
    MCP 서비스용 클래스로 래핑합니다.
    """

    def __init__(
        self,
        dimensions: int = 128,
        walk_length: int = 30,
        num_walks: int = 20,
        p: float = 1.0,
        q: float = 0.25,
    ):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self._model = None

    def build_graph(self, pg_json: dict) -> nx.DiGraph:
        """PG JSON → NetworkX 그래프 (Sequential + Spatial KNN 엣지)."""
        G = nx.DiGraph()
        waypoints = pg_json.get('waypoints', [])

        for i, wp in enumerate(waypoints):
            G.add_node(i, **{
                'lat': wp.get('lat', wp.get('lat_deg', 0)),
                'lon': wp.get('lon', wp.get('lon_deg', 0)),
                'alt': wp.get('alt', wp.get('alt_m', 0)),
                'seq_ratio': i / max(len(waypoints) - 1, 1),
            })

        # Sequential 엣지
        for i in range(len(waypoints) - 1):
            G.add_edge(i, i + 1, edge_type='sequential')

        # Spatial KNN 엣지 (K=3, seq_gap >= 3)
        positions = np.array([[G.nodes[i]['lat'], G.nodes[i]['lon']] for i in G.nodes])
        for i in range(len(waypoints)):
            if len(positions) < 4:
                break
            dists = np.linalg.norm(positions - positions[i], axis=1)
            sorted_idx = np.argsort(dists)
            k_count = 0
            for j in sorted_idx:
                if j == i or abs(j - i) < 3:
                    continue
                G.add_edge(i, int(j), edge_type='spatial_knn')
                k_count += 1
                if k_count >= 3:
                    break

        return G

    def encode_mission(self, pg_json: dict) -> list[float]:
        """PG JSON → 128d 미션 벡터.

        Node2Vec으로 WP별 벡터 생성 후 가중 평균 풀링.
        """
        G = self.build_graph(pg_json)

        if len(G.nodes) < 2:
            return [0.0] * self.dimensions

        try:
            from gensim.models import Word2Vec
            from node2vec import Node2Vec as N2V

            n2v = N2V(
                G,
                dimensions=self.dimensions,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                p=self.p,
                q=self.q,
                workers=1,
                quiet=True,
            )
            model = n2v.fit(window=5, min_count=1, batch_words=4)

            # 가중 평균 풀링 (첫/끝 WP ×2.0)
            vectors = []
            weights = []
            for node in G.nodes:
                if str(node) in model.wv:
                    vectors.append(model.wv[str(node)])
                    w = 1.0
                    if node == 0 or node == len(G.nodes) - 1:
                        w = 2.0
                    weights.append(w)

            if not vectors:
                return [0.0] * self.dimensions

            vectors = np.array(vectors)
            weights = np.array(weights) / sum(weights)
            mission_vector = np.average(vectors, axis=0, weights=weights)
            return mission_vector.tolist()

        except ImportError:
            logger.warning('node2vec not installed, using random embedding')
            return np.random.randn(self.dimensions).tolist()
