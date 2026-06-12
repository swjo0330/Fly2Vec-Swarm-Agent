"""Graph Embedding MCP 서버.

도구: embed_mission, find_similar, anomaly_score
"""

import logging
import os
from typing import Any

import numpy as np
from pydantic import Field

from fly2vec.src.mcp_servers.fly2vec_mcp_base import Fly2VecMCPBase
from fly2vec.src.qdrant.fly2vec_qdrant import Fly2VecQdrantManager
from fly2vec.src.qdrant.collections import (
    ANOMALY_THRESHOLD,
    COLLECTION_MISSION,
    DEFAULT_TOP_K,
    VECTOR_DIM,
)

logger = logging.getLogger(__name__)


class GraphEmbeddingMCP(Fly2VecMCPBase):
    """Graph Embedding MCP 서버.

    PG JSON → 그래프 → 128d 벡터 임베딩 → Qdrant 저장/검색/이상탐지.
    """

    def __init__(self, port: int = 8050, **kwargs):
        super().__init__(
            server_name='fly2vec-graph-embedding',
            port=port,
            server_instructions='드론 임무 경로를 그래프 임베딩하여 유사도 검색 및 이상 탐지를 수행합니다.',
            **kwargs,
        )

    def _initialize_clients(self) -> None:
        self.qdrant = Fly2VecQdrantManager()
        self._encoder = None  # lazy init
        self._centroid = None  # anomaly centroid 캐시
        self._iso_forest = None  # IsolationForest 캐시

    def _get_encoder(self):
        """임베딩 인코더 lazy 로드."""
        if self._encoder is None:
            from fly2vec.src.embedding.node2vec_encoder import Node2VecEncoder
            self._encoder = Node2VecEncoder()
        return self._encoder

    def _register_tools(self) -> None:
        self._reg_embed_mission()
        self._reg_find_similar()
        self._reg_anomaly_score()
        self._reg_recommend_route()

    def _reg_embed_mission(self) -> None:
        @self.mcp.tool()
        async def embed_mission(
            pg_json: dict = Field(..., description='Planning Graph JSON (waypoints 포함)'),
            store: bool = Field(True, description='Qdrant에 저장 여부'),
        ) -> dict[str, Any]:
            """PG JSON을 128d 벡터로 임베딩합니다."""
            try:
                from fly2vec.src.embedding.enrichment import enrich_pg_features

                enriched_pg = enrich_pg_features(pg_json)
                encoder = self._get_encoder()
                vector = encoder.encode_mission(enriched_pg)

                result = {
                    'vector': vector,
                    'dimension': len(vector),
                    'route_stats': enriched_pg.get('route_stats', {}),
                }

                if store:
                    # anomaly_score 사전 계산 (recommend_route 필터용)
                    _anomaly = 0.0
                    try:
                        if self._centroid is not None:
                            import numpy as np
                            _v = np.array(vector)
                            _c = np.array(self._centroid)
                            _anomaly = float(max(0, 1 - np.dot(_v, _c) / (np.linalg.norm(_v) * np.linalg.norm(_c) + 1e-10)))
                    except Exception:
                        pass
                    # R1: payload 3→7 확장 (RouteRecommenderMCP 4축 스코어링용)
                    _wps = pg_json.get('waypoints', [])
                    _rs = enriched_pg.get('route_stats', {}) if 'enriched_pg' in dir() else {}
                    _speeds = [w.get('speed_mps', 0) for w in _wps if w.get('speed_mps')]
                    _avg_speed = sum(_speeds) / len(_speeds) if _speeds else 2.0
                    point_id = await self.qdrant.store_mission_vector(
                        vector=vector,
                        payload={
                            'mission_type': pg_json.get('mission_type', 'unknown'),
                            'wp_count': len(_wps),
                            'anomaly_score': round(_anomaly, 4),
                            'total_distance_m': round(_rs.get('total_distance_m', 0.0), 1),
                            'avg_bearing_change': round(_rs.get('avg_bearing_change', 0.0), 1),
                            'avg_speed_mps': round(_avg_speed, 2),
                            'terrain_type': 0,
                        },
                    )
                    result['point_id'] = point_id

                return self.create_response(True, 'embed_mission', result)
            except Exception as e:
                return self.create_error(str(e), 'embed_mission', 'embed_mission')

    def _reg_find_similar(self) -> None:
        @self.mcp.tool()
        async def find_similar(
            vector: list[float] = Field(..., description='128d 쿼리 벡터'),
            top_k: int = Field(DEFAULT_TOP_K, description='반환할 유사 미션 수'),
        ) -> dict[str, Any]:
            """유사 미션을 Qdrant에서 검색합니다."""
            try:
                results = await self.qdrant.find_similar(vector=vector, top_k=top_k)
                return self.create_response(True, 'find_similar', {
                    'similar_missions': results,
                    'count': len(results),
                })
            except Exception as e:
                return self.create_error(str(e), 'find_similar', 'find_similar')

    def _reg_anomaly_score(self) -> None:
        @self.mcp.tool()
        async def anomaly_score(
            vector: list[float] = Field(..., description='128d 쿼리 벡터'),
        ) -> dict[str, Any]:
            """3-Method 앙상블 이상 점수를 계산합니다 (0~1)."""
            try:
                v = np.array(vector)

                # 1. Centroid Distance (30%)
                if self._centroid is None:
                    samples = await self.qdrant.scroll_vectors(limit=500)
                    if samples:
                        self._centroid = np.mean(samples, axis=0)
                centroid = self._centroid if self._centroid is not None else np.zeros(VECTOR_DIM)
                cos_sim = float(np.dot(v, centroid) / (np.linalg.norm(v) * np.linalg.norm(centroid) + 1e-10))
                centroid_score = max(0.0, 1.0 - cos_sim)

                # 2. Isolation Forest (40%)
                if self._iso_forest is None:
                    samples = await self.qdrant.scroll_vectors(limit=300)
                    if samples and len(samples) >= 10:
                        from sklearn.ensemble import IsolationForest
                        self._iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        self._iso_forest.fit(np.array(samples))
                if self._iso_forest is not None:
                    raw_iso = self._iso_forest.score_samples(v.reshape(1, -1))[0]
                    iso_score = float(max(0.0, min(1.0, -raw_iso)))
                else:
                    iso_score = min(centroid_score * 1.5, 1.0)

                # 3. KNN Average Distance (30%)
                similar = await self.qdrant.find_similar(vector=vector, top_k=10)
                if similar:
                    avg_sim = sum(s['score'] for s in similar) / len(similar)
                    knn_score = max(0.0, min(1.0, 1.0 - avg_sim))
                else:
                    knn_score = centroid_score

                # 앙상블
                final_score = 0.3 * centroid_score + 0.4 * iso_score + 0.3 * knn_score
                final_score = round(min(max(final_score, 0.0), 1.0), 4)

                return self.create_response(True, 'anomaly_score', {
                    'score': final_score,
                    'is_anomalous': final_score > ANOMALY_THRESHOLD,
                    'breakdown': {
                        'centroid_distance': round(centroid_score, 4),
                        'isolation_forest': round(iso_score, 4),
                        'knn_average': round(knn_score, 4),
                    },
                })
            except Exception as e:
                return self.create_error(str(e), 'anomaly_score', 'anomaly_score')

    def _reg_recommend_route(self) -> None:
        @self.mcp.tool()
        async def recommend_route(
            vector: list[float] = Field(..., description='128d 쿼리 벡터'),
            top_k: int = Field(5, description='추천 경로 수'),
            max_anomaly: float = Field(0.3, description='최대 허용 anomaly score'),
        ) -> dict[str, Any]:
            """유사하면서 정상인 경로를 추천합니다.

            현재 임무와 구조적으로 유사하되 과거에 정상으로 확인된 경로를 반환.
            anomaly_score가 낮은 경로만 필터링하여 추천.
            """
            try:
                results = await self.qdrant.recommend_routes(
                    vector=vector, top_k=top_k, max_anomaly=max_anomaly,
                )
                return self.create_response(True, 'recommend_route', {
                    'recommended_routes': results,
                    'count': len(results),
                    'filter': f'anomaly_score <= {max_anomaly}',
                })
            except Exception as e:
                return self.create_error(str(e), 'recommend_route', 'recommend_route')


if __name__ == '__main__':
    server = GraphEmbeddingMCP(port=int(os.getenv('GRAPH_EMBEDDING_MCP_PORT', '8050')))
    server.run()
