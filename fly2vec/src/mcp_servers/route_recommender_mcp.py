"""RouteRecommenderMCP — 4축 다기준 경로 추천 서버.

Port: 8052
Score = 0.35×Safety + 0.25×Efficiency + 0.25×Similarity + 0.15×SwarmCompat

교수 피드백 Task 4 "경로 추천 (optimal?)" 대응.
"""
import logging
import math
import os

from pydantic import Field

from fly2vec.src.mcp_servers.fly2vec_mcp_base import Fly2VecMCPBase
from fly2vec.src.qdrant.fly2vec_qdrant import Fly2VecQdrantManager

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"safety": 0.35, "efficiency": 0.25, "similarity": 0.25, "swarm_compat": 0.15}


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / max(na * nb, 1e-9)


def _compute_safety(anomaly_score: float) -> float:
    return 1.0 - (anomaly_score ** 0.7)


def _compute_efficiency(payload: dict, query_distance: float) -> float:
    dist = payload.get('total_distance_m', 0)
    if dist <= 0 or query_distance <= 0:
        return 0.5  # 데이터 없으면 중립
    dist_ratio = min(dist / query_distance, 2.0)
    dist_eff = 1.0 - abs(1.0 - dist_ratio) * 0.5
    speed = payload.get('avg_speed_mps', 3.0)
    speed_eff = min(speed / 5.0, 1.0)
    wp_count = payload.get('wp_count', 5)
    wp_eff = 1.0 - min(wp_count / 30, 1.0) * 0.3
    return dist_eff * 0.5 + speed_eff * 0.3 + wp_eff * 0.2


def _compute_swarm_compat(candidate_vector: list[float], swarm_vectors: list[list[float]]) -> float:
    if not swarm_vectors:
        return 1.0
    max_sim = max(_cosine_sim(candidate_vector, sv) for sv in swarm_vectors)
    return max(0.0, min(1.0, 1.0 - max_sim * 0.8))  # 클리핑 (음수 cosine 방어)


class RouteRecommenderMCP(Fly2VecMCPBase):
    """4축 다기준 경로 추천 MCP 서버."""

    def __init__(self, port: int = 8052, **kwargs):
        super().__init__(
            server_name='route-recommender-mcp',
            port=port,
            server_instructions='4축 다기준 경로 추천: Safety, Efficiency, Similarity, SwarmCompat',
            **kwargs,
        )

    def _initialize_clients(self) -> None:
        self.qdrant = Fly2VecQdrantManager()
        logger.info('[route-recommender] Qdrant 초기화')

    def _register_tools(self) -> None:
        self._reg_recommend_safe_route()
        self._reg_explain_recommendation()
        self._reg_compare_routes()
        self._reg_suggest_modification()

    def _reg_recommend_safe_route(self) -> None:
        @self.mcp.tool()
        async def recommend_safe_route(
            vector: list[float] = Field(..., description='128d 쿼리 벡터'),
            mission_type: str = Field('unknown', description='임무 유형 필터'),
            top_k: int = Field(5, description='반환 경로 수'),
            max_anomaly: float = Field(0.5, description='최대 anomaly score'),
            swarm_vectors: list[list[float]] = Field(default_factory=list, description='군집 내 타 드론 벡터'),
            weights: dict = Field(default_factory=lambda: DEFAULT_WEIGHTS.copy(), description='4축 가중치'),
        ) -> dict:
            """4축 다기준 경로 추천."""
            try:
                # 1. Qdrant 후보 검색
                mt = mission_type if mission_type != 'unknown' else None
                candidates = await self.qdrant.recommend_multi_criteria(
                    vector=vector, mission_type=mt, max_anomaly=max_anomaly, top_k=top_k * 4,
                )
                if not candidates:
                    return self.create_response(True, 'recommend_safe_route', {
                        'recommended_routes': [], 'count': 0, 'message': '후보 없음',
                    })

                # 2. 쿼리 거리 추정 (벡터 norm 기반 간접 추정)
                query_distance = sum(v ** 2 for v in vector[:5]) * 1000  # 근사치

                # 3. 4축 스코어링
                scored = []
                w = {**DEFAULT_WEIGHTS, **weights}
                for cand in candidates:
                    payload = cand.get('payload', {})
                    sim = cand.get('score', 0.5)

                    safety = _compute_safety(float(payload.get('anomaly_score', 0.5) or 0.5))
                    efficiency = _compute_efficiency(payload, query_distance)
                    similarity = sim
                    swarm = _compute_swarm_compat(vector, swarm_vectors)

                    total = (w['safety'] * safety + w['efficiency'] * efficiency +
                             w['similarity'] * similarity + w['swarm_compat'] * swarm)

                    scored.append({
                        'point_id': cand.get('id', ''),
                        'total_score': round(total, 4),
                        'score_breakdown': {
                            'safety': round(safety, 4),
                            'efficiency': round(efficiency, 4),
                            'similarity': round(similarity, 4),
                            'swarm_compat': round(swarm, 4),
                        },
                        'payload': payload,
                    })

                # 4. 정렬 + top_k
                scored.sort(key=lambda x: x['total_score'], reverse=True)
                result = scored[:top_k]

                for i, r in enumerate(result):
                    r['rank'] = i + 1

                return self.create_response(True, 'recommend_safe_route', {
                    'recommended_routes': result,
                    'count': len(result),
                    'weights_used': w,
                })
            except Exception as e:
                return self.create_error(str(e), 'recommend_safe_route', 'recommend')

    def _reg_explain_recommendation(self) -> None:
        @self.mcp.tool()
        async def explain_recommendation(
            point_id: str = Field(..., description='추천 경로 point_id'),
            query_mission_type: str = Field('unknown', description='쿼리 임무 유형'),
        ) -> dict:
            """추천 이유 설명 (템플릿 기반)."""
            try:
                payload = await self.qdrant.get_by_id(point_id)
                if not payload:
                    return self.create_response(False, 'explain_recommendation', '포인트 없음')

                anomaly = payload.get('anomaly_score', 0.5)
                mt = payload.get('mission_type', 'unknown')
                wp = payload.get('wp_count', 0)
                dist = payload.get('total_distance_m', 0)

                safety = _compute_safety(anomaly)
                reasons = []
                if safety > 0.8:
                    reasons.append(f'안전성이 높습니다 (anomaly {anomaly:.2f}, 안전 점수 {safety:.2f})')
                if mt == query_mission_type:
                    reasons.append(f'동일한 임무 유형({mt})으로 구조적 유사성이 높습니다')
                if wp > 0:
                    reasons.append(f'{wp}개 웨이포인트, 총 {dist:.0f}m 경로입니다')
                if not reasons:
                    reasons.append('유사도 기반 추천입니다')

                explanation = '. '.join(reasons) + '.'

                return self.create_response(True, 'explain_recommendation', {
                    'point_id': point_id,
                    'payload': payload,
                    'explanation': explanation,
                    'safety_score': round(safety, 4),
                })
            except Exception as e:
                return self.create_error(str(e), 'explain_recommendation', 'explain')

    def _reg_compare_routes(self) -> None:
        @self.mcp.tool()
        async def compare_routes(
            route_ids: list[str] = Field(..., description='비교할 point_id 목록 (최대 5)'),
            query_vector: list[float] = Field(..., description='128d 쿼리 벡터'),
            swarm_vectors: list[list[float]] = Field(default_factory=list, description='군집 벡터'),
        ) -> dict:
            """복수 경로 4축 비교 + 레이더 차트 데이터."""
            try:
                query_distance = sum(v ** 2 for v in query_vector[:5]) * 1000
                comparison = []

                for pid in route_ids[:5]:
                    payload = await self.qdrant.get_by_id(pid)
                    if not payload:
                        continue

                    safety = _compute_safety(float(payload.get('anomaly_score', 0.5) or 0.5))
                    efficiency = _compute_efficiency(payload, query_distance)
                    swarm = _compute_swarm_compat(query_vector, swarm_vectors)

                    total = 0.35 * safety + 0.25 * efficiency + 0.25 * 0.5 + 0.15 * swarm

                    strengths, weaknesses = [], []
                    if safety > 0.8: strengths.append(f'안전 ({safety:.2f})')
                    else: weaknesses.append(f'안전 낮음 ({safety:.2f})')
                    if efficiency > 0.7: strengths.append(f'효율적')
                    else: weaknesses.append(f'비효율 ({efficiency:.2f})')

                    comparison.append({
                        'point_id': pid,
                        'scores': {
                            'safety': round(safety, 4),
                            'efficiency': round(efficiency, 4),
                            'similarity': 0.5,  # 개별 ID 기준이라 cosine 미계산
                            'swarm_compat': round(swarm, 4),
                        },
                        'total_score': round(total, 4),
                        'strengths': strengths,
                        'weaknesses': weaknesses,
                        'payload': payload,
                    })

                comparison.sort(key=lambda x: x['total_score'], reverse=True)
                for i, c in enumerate(comparison):
                    c['rank'] = i + 1

                winner = comparison[0] if comparison else None
                radar_data = {
                    'axes': ['Safety', 'Efficiency', 'Similarity', 'SwarmCompat'],
                    'series': [
                        {'label': f"Route {c['point_id'][:8]}", 'values': list(c['scores'].values())}
                        for c in comparison
                    ],
                }

                return self.create_response(True, 'compare_routes', {
                    'comparison': comparison,
                    'winner': {'point_id': winner['point_id'], 'total_score': winner['total_score']} if winner else None,
                    'radar_data': radar_data,
                })
            except Exception as e:
                return self.create_error(str(e), 'compare_routes', 'compare')

    def _reg_suggest_modification(self) -> None:
        @self.mcp.tool()
        async def suggest_modification(
            route_pg: dict = Field(..., description='현재 PG JSON'),
            risk_points: list[int] = Field(default_factory=list, description='위험 WP 인덱스'),
            target_anomaly: float = Field(0.3, description='목표 anomaly score'),
        ) -> dict:
            """위험 구간 수정 제안 (템플릿 기반)."""
            try:
                waypoints = route_pg.get('waypoints', [])
                if not waypoints:
                    return self.create_response(False, 'suggest_modification', '웨이포인트 없음')

                # 위험 WP 자동 식별 (bearing_change > 120 또는 명시적 risk_points)
                identified_risk = list(risk_points)
                for i, wp in enumerate(waypoints):
                    bc = abs(wp.get('bearing_change_deg', 0))
                    if bc > 120 and i not in identified_risk:
                        identified_risk.append(i)

                changes = []
                for idx in identified_risk:
                    if idx >= len(waypoints):
                        continue
                    wp = waypoints[idx]
                    # 수정 제안: 고도 증가 + 속도 감소
                    changes.append({
                        'wp_index': idx,
                        'action': 'altitude_increase_speed_reduce',
                        'before': {'alt_m': wp.get('alt_m', 50), 'speed_mps': wp.get('speed_mps', 5)},
                        'after': {'alt_m': wp.get('alt_m', 50) + 30, 'speed_mps': max(2.0, wp.get('speed_mps', 5) * 0.7)},
                        'reason': f"WP {idx}: 급격한 방위 변화 또는 위험 구간 — 고도 상승 + 감속으로 안전 확보",
                    })

                return self.create_response(True, 'suggest_modification', {
                    'risk_points_identified': identified_risk,
                    'changes': changes,
                    'target_anomaly': target_anomaly,
                    'n_modifications': len(changes),
                })
            except Exception as e:
                return self.create_error(str(e), 'suggest_modification', 'suggest')


if __name__ == '__main__':
    port = int(os.getenv('ROUTE_RECOMMENDER_MCP_PORT', '8052'))
    server = RouteRecommenderMCP(port=port)
    server.run()
