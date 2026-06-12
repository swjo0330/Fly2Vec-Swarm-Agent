"""Fly2Vec-Swarm Analysis MCP 서버.

도구: analyze_swarm, detect_conflict, optimize_assignment
"""

import logging
import os
from typing import Any

from pydantic import Field

from fly2vec.src.mcp_servers.fly2vec_mcp_base import Fly2VecMCPBase
from fly2vec.src.qdrant.fly2vec_qdrant import Fly2VecQdrantManager
from fly2vec.src.qdrant.collections import COLLECTION_SWARM

logger = logging.getLogger(__name__)


class SwarmAnalysisMCP(Fly2VecMCPBase):
    """Fly2Vec-Swarm 군집 분석 MCP 서버.

    R-GAT 3종 엣지(Sequential/Proximity/Coverage)로 군집 임무를 분석합니다.
    """

    def __init__(self, port: int = 8051, **kwargs):
        super().__init__(
            server_name='fly2vec-swarm-analysis',
            port=port,
            server_instructions='군집 드론 임무의 충돌 탐지, 패턴 분류, 경로 추천을 수행합니다.',
            **kwargs,
        )

    def _initialize_clients(self) -> None:
        self.qdrant = Fly2VecQdrantManager()
        self._rgat_model = None  # lazy init

    def _get_rgat(self):
        """R-GAT 모델 lazy 로드 (학습 모델 우선)."""
        if self._rgat_model is None:
            import os
            from fly2vec.src.embedding.rgat_encoder import RGATEncoder
            model_path = os.getenv('RGAT_MODEL_PATH', '/app/fly2vec/data/models/fly2vec_rgat_15d_best.pt')
            # 로컬 경로 fallback
            local_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'fly2vec_rgat_15d_best.pt')
            if os.path.exists(model_path):
                self._rgat_model = RGATEncoder(model_path=model_path)
            elif os.path.exists(local_path):
                self._rgat_model = RGATEncoder(model_path=local_path)
            else:
                logger.warning('[swarm] R-GAT model not found, using rule-based fallback')
                self._rgat_model = RGATEncoder()
        return self._rgat_model

    def _register_tools(self) -> None:
        self._reg_analyze_swarm()
        self._reg_detect_conflict()
        self._reg_optimize_assignment()

    def _reg_analyze_swarm(self) -> None:
        @self.mcp.tool()
        async def analyze_swarm(
            missions: list[dict] = Field(..., description='드론별 PG JSON 리스트'),
        ) -> dict[str, Any]:
            """군집 임무를 R-GAT로 분석합니다.

            반환: 패턴 분류, anomaly score, 경로 추천.
            """
            try:
                rgat = self._get_rgat()
                analysis = rgat.analyze(missions)

                # Qdrant에 군집 벡터 저장
                if analysis.get('vector'):
                    await self.qdrant.store_mission_vector(
                        vector=analysis['vector'],
                        payload={
                            'pattern': analysis.get('pattern', 'unknown'),
                            'drone_count': len(missions),
                            'anomaly_avg': analysis.get('anomaly_avg', 0.0),
                        },
                        collection=COLLECTION_SWARM,
                    )

                # RAG: 유사 군집 임무 검색
                similar_swarms = []
                if analysis.get('vector'):
                    similar_swarms = await self.qdrant.find_similar(
                        vector=analysis['vector'],
                        top_k=5,
                        collection=COLLECTION_SWARM,
                    )

                return self.create_response(True, 'analyze_swarm', {
                    'pattern': analysis.get('pattern'),
                    'anomaly_avg': analysis.get('anomaly_avg'),
                    'drone_scores': analysis.get('drone_scores', {}),
                    'edge_stats': analysis.get('edge_stats', {}),
                    'similar_swarms': similar_swarms,
                    'model_used': analysis.get('model_used', False),
                })
            except Exception as e:
                return self.create_error(str(e), 'analyze_swarm', 'analyze_swarm')

    def _reg_detect_conflict(self) -> None:
        @self.mcp.tool()
        async def detect_conflict(
            missions: list[dict] = Field(..., description='드론별 PG JSON 리스트'),
            proximity_threshold_m: float = Field(500.0, description='근접 임계값 (m)'),
        ) -> dict[str, Any]:
            """드론 간 충돌 지점을 탐지합니다.

            Proximity 엣지 기반: 다른 드론 WP 간 거리 < threshold.
            """
            try:
                rgat = self._get_rgat()
                conflicts = rgat.detect_conflicts(missions, proximity_threshold_m)

                return self.create_response(True, 'detect_conflict', {
                    'conflict_count': len(conflicts),
                    'conflicts': conflicts,
                    'has_conflict': len(conflicts) > 0,
                })
            except Exception as e:
                return self.create_error(str(e), 'detect_conflict', 'detect_conflict')

    def _reg_optimize_assignment(self) -> None:
        @self.mcp.tool()
        async def optimize_assignment(
            missions: list[dict] = Field(..., description='후보 미션 PG JSON 리스트'),
            drones: list[dict] = Field(..., description='드론 상태 리스트 [{id, position, battery}]'),
        ) -> dict[str, Any]:
            """최적 드론-미션 배치를 추천합니다.

            Coverage 엣지 최소화 + Proximity 엣지 최소화 기준.
            """
            try:
                rgat = self._get_rgat()
                assignment = rgat.optimize_assignment(missions, drones)

                return self.create_response(True, 'optimize_assignment', {
                    'assignments': assignment.get('assignments', []),
                    'coverage_score': assignment.get('coverage_score', 0.0),
                    'conflict_risk': assignment.get('conflict_risk', 0.0),
                })
            except Exception as e:
                return self.create_error(str(e), 'optimize_assignment', 'optimize_assignment')


if __name__ == '__main__':
    server = SwarmAnalysisMCP(port=int(os.getenv('SWARM_ANALYSIS_MCP_PORT', '8051')))
    server.run()
