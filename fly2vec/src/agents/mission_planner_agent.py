"""MissionPlannerAgent — 검증-수정 루프 에이전트.

LangGraph 노드:
  parse_input → generate_pg → embed_and_check → [anomaly>0.5?]
    → YES: modify_pg → embed_and_check (max 3회)
    → NO: finalize → output
"""

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from fly2vec.src.base.fly2vec_base_agent import Fly2VecBaseAgent
from fly2vec.src.base.fly2vec_state import MissionPlannerState
from fly2vec.src.llm.fly2vec_llm_factory import build_fly2vec_llm
from fly2vec.src.qdrant.collections import ANOMALY_THRESHOLD, MAX_VERIFICATION_LOOPS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 드론 임무 경로 분석 에이전트입니다.
Planning Graph(PG) JSON을 분석하고, 이상 점수가 높으면 경로를 수정합니다.
anomaly_score > 0.5 또는 conflict가 존재하면 PG를 수정하세요.
최대 3회까지 수정-재검증을 반복합니다."""


def _parse_mcp_result(raw: Any) -> dict:
    """MCP 도구 응답 파싱.

    langchain-mcp-adapters는 list[{'type':'text','text':'<json>'}] 형태로 반환.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict) and 'text' in first:
            try:
                return json.loads(first['text'])
            except (json.JSONDecodeError, TypeError):
                pass
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _parse_json_response(text: str) -> dict:
    """LLM 응답에서 JSON 파싱 (4단계 fallback)."""
    # 1) 순수 JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # 2) ```json 블록
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3) 첫 {...} 패턴
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # 4) fallback
    return {'waypoints': [], 'metadata': {'parse_failed': True}}


class MissionPlannerAgent(Fly2VecBaseAgent):
    """Fly2Vec 검증-수정 루프 에이전트.

    PG 생성 → 임베딩 → anomaly 체크 → 수정 (max 3회) → 최종 PG 출력.
    """

    NODE_NAMES = {
        'PARSE': 'parse_input',
        'GENERATE': 'generate_pg',
        'EMBED_CHECK': 'embed_and_check',
        'MODIFY': 'modify_pg',
        'FINALIZE': 'finalize',
    }

    def __init__(self, **kwargs):
        model = build_fly2vec_llm()
        mcp_servers = [
            {
                'name': 'graph-embedding',
                'url': os.getenv('GRAPH_EMBEDDING_MCP_URL', 'http://graph-embedding-mcp:8050/mcp'),
            },
            {
                'name': 'swarm-analysis',
                'url': os.getenv('SWARM_ANALYSIS_MCP_URL', 'http://swarm-analysis-mcp:8051/mcp'),
            },
            {
                'name': 'route-recommender',
                'url': os.getenv('ROUTE_RECOMMENDER_MCP_URL', 'http://route-recommender-mcp:8052/mcp'),
            },
            {
                'name': 'mapbox-mcp',
                'transport': 'streamable_http',
                'url': os.getenv('MAPBOX_MCP_URL', 'https://mcp.mapbox.com/mcp'),
                'headers': {
                    'Authorization': f'Bearer {os.getenv("MAPBOX_ACCESS_TOKEN", "")}',
                    'Accept': 'application/json, text/event-stream',
                },
            },
        ]
        super().__init__(
            model=model,
            state_schema=MissionPlannerState,
            mcp_servers=mcp_servers,
            agent_name='MissionPlannerAgent',
            lazy_init=True,
            **kwargs,
        )

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.NODE_NAMES['PARSE'], self.node_parse_input)
        graph.add_node(self.NODE_NAMES['GENERATE'], self.node_generate_pg)
        graph.add_node(self.NODE_NAMES['EMBED_CHECK'], self.node_embed_and_check)
        graph.add_node(self.NODE_NAMES['MODIFY'], self.node_modify_pg)
        graph.add_node(self.NODE_NAMES['FINALIZE'], self.node_finalize)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.NODE_NAMES['PARSE'])
        graph.add_edge(self.NODE_NAMES['PARSE'], self.NODE_NAMES['GENERATE'])
        graph.add_edge(self.NODE_NAMES['GENERATE'], self.NODE_NAMES['EMBED_CHECK'])
        graph.add_conditional_edges(
            self.NODE_NAMES['EMBED_CHECK'],
            self._should_modify,
            {
                'modify': self.NODE_NAMES['MODIFY'],
                'finalize': self.NODE_NAMES['FINALIZE'],
            },
        )
        graph.add_edge(self.NODE_NAMES['MODIFY'], self.NODE_NAMES['EMBED_CHECK'])
        graph.add_edge(self.NODE_NAMES['FINALIZE'], END)

    def _should_modify(self, state: MissionPlannerState) -> str:
        """anomaly > threshold 이고 retry < max이면 수정."""
        score = state.get('anomaly_score', 0.0) or 0.0
        conflicts = state.get('conflict_points') or []
        retry = state.get('retry_count', 0)

        if retry >= MAX_VERIFICATION_LOOPS:
            logger.warning(f'Max retries ({MAX_VERIFICATION_LOOPS}) reached. Finalizing.')
            return 'finalize'
        if score > ANOMALY_THRESHOLD or len(conflicts) > 0:
            return 'modify'
        return 'finalize'

    async def node_parse_input(self, state: MissionPlannerState, config: RunnableConfig) -> dict:
        """사용자 입력 파싱."""
        return {'retry_count': 0, 'is_verified': False}

    async def node_generate_pg(self, state: MissionPlannerState, config: RunnableConfig) -> dict:
        """PG JSON 생성 — MapBox MCP 경유 (제안서 4.3절).

        흐름: 자연어 → LLM intent 추출 → MapBox geocode → directions → D-P 간소화 → PG
        MapBox 불가 시 LLM 직접 생성 fallback.
        """
        pg = state.get('pg_json')
        if pg and pg.get('waypoints'):
            return {}

        messages = state.get('messages', [])

        # Step 1: LLM으로 intent만 추출 (좌표 생성 금지)
        intent_prompt = (
            '사용자 요청에서 비행 의도만 JSON으로 추출하세요. 좌표는 생성하지 마세요.\n'
            '형식: {"origin": "장소명", "destination": "장소명", '
            '"altitude_m": float, "speed_mps": float, '
            '"mission_type": "linear_flight|grid_search|waypoint_sequence|circular_flight"}\n'
            '장소명은 사용자가 말한 그대로 적으세요.'
        )
        response = await self.model.ainvoke(
            [SystemMessage(content=intent_prompt)] + messages
        )
        intent = _parse_json_response(response.content)
        logger.info(f'[generate_pg] Intent: {intent}')

        if not intent.get('origin') or not intent.get('destination'):
            logger.warning('[generate_pg] Intent extraction failed, using fallback')
            return await self._generate_pg_fallback(messages)

        # Step 2: MapBox geocode
        geocode_tool = next((t for t in self.tools if 'geocode' in t.name.lower() and 'reverse' not in t.name.lower()), None)
        if not geocode_tool:
            logger.warning('[generate_pg] MapBox geocode tool not found, using fallback')
            return await self._generate_pg_fallback(messages)

        try:
            origin_raw = await geocode_tool.ainvoke({'q': intent['origin']})
            origin_result = _parse_mcp_result(origin_raw)
            dest_raw = await geocode_tool.ainvoke({'q': intent['destination']})
            dest_result = _parse_mcp_result(dest_raw)

            origin_coords = self._extract_coords(origin_result)
            dest_coords = self._extract_coords(dest_result)
            if not origin_coords or not dest_coords:
                logger.warning('[generate_pg] Geocode failed, using fallback')
                return await self._generate_pg_fallback(messages)
            logger.info(f'[generate_pg] Geocoded: {intent["origin"]}→{origin_coords}, {intent["destination"]}→{dest_coords}')
        except Exception as e:
            logger.warning(f'[generate_pg] Geocode error: {e}, using fallback')
            return await self._generate_pg_fallback(messages)

        # Step 3: MapBox directions
        alt = intent.get('altitude_m', 50)
        speed = intent.get('speed_mps', 5.0)
        directions_tool = next((t for t in self.tools if 'direction' in t.name.lower()), None)
        if directions_tool:
            try:
                route_raw = await directions_tool.ainvoke({
                    'origin': f'{origin_coords[1]},{origin_coords[0]}',
                    'destination': f'{dest_coords[1]},{dest_coords[0]}',
                })
                route_result = _parse_mcp_result(route_raw)
                route_coords = self._extract_route_coords(route_result)
                if not route_coords:
                    route_coords = [origin_coords, dest_coords]
                logger.info(f'[generate_pg] Directions: {len(route_coords)} points')
            except Exception as e:
                logger.warning(f'[generate_pg] Directions error: {e}, using direct line')
                route_coords = [origin_coords, dest_coords]
        else:
            route_coords = [origin_coords, dest_coords]

        # Step 4: Douglas-Peucker 간소화
        if len(route_coords) > 20:
            route_coords = self._douglas_peucker(route_coords, epsilon=0.0001)
            logger.info(f'[generate_pg] D-P simplified: {len(route_coords)} points')

        # Step 5: PG JSON 생성
        waypoints = []
        for i, (lat, lon) in enumerate(route_coords):
            waypoints.append({
                'lat_deg': lat, 'lon_deg': lon, 'alt_m': alt,
                'speed_mps': speed, 'seq': i,
                'command': 'NAV_WAYPOINT', 'hold_sec': 0,
            })

        pg_json = {
            'waypoints': waypoints,
            'mission_type': intent.get('mission_type', 'linear_flight'),
            'metadata': {
                'origin': intent.get('origin'),
                'destination': intent.get('destination'),
                'source': 'mapbox_mcp',
            },
        }
        logger.info(f'[generate_pg] MapBox PG: {len(waypoints)} WPs')
        return {'pg_json': pg_json}

    async def _generate_pg_fallback(self, messages: list) -> dict:
        """MapBox 불가 시 LLM 직접 좌표 생성 (fallback)."""
        generation_prompt = (
            '사용자 요청에 맞는 드론 비행 경로를 Planning Graph JSON으로 생성하세요.\n'
            '형식: {"waypoints": [{"lat_deg": float, "lon_deg": float, "alt_m": float, '
            '"speed_mps": float, "seq": int, "command": "NAV_WAYPOINT", "hold_sec": float}], '
            '"mission_type": "grid_search|linear_flight|waypoint_sequence|circular_flight"}\n'
            '최소 4개 이상의 웨이포인트를 포함하세요.'
        )
        response = await self.model.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT + '\n\n' + generation_prompt)] + messages
        )
        pg_json = _parse_json_response(response.content)
        if pg_json.get('metadata') is None:
            pg_json['metadata'] = {}
        pg_json['metadata']['source'] = 'llm_fallback'
        logger.info(f'[generate_pg] Fallback PG: {len(pg_json.get("waypoints", []))} WPs')
        return {'pg_json': pg_json}

    @staticmethod
    def _extract_coords(geocode_result: dict) -> tuple[float, float] | None:
        """MapBox geocode 결과에서 (lat, lon) 추출."""
        if not geocode_result:
            return None
        # MapBox 응답 형식 다양 — 여러 패턴 시도
        for key in ('coordinates', 'center', 'geometry'):
            val = geocode_result.get(key)
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                return (val[1], val[0])  # MapBox는 [lng, lat]
            if isinstance(val, dict) and 'coordinates' in val:
                c = val['coordinates']
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    return (c[1], c[0])
        if 'lat' in geocode_result and 'lon' in geocode_result:
            return (geocode_result['lat'], geocode_result['lon'])
        if 'latitude' in geocode_result and 'longitude' in geocode_result:
            return (geocode_result['latitude'], geocode_result['longitude'])
        return None

    @staticmethod
    def _extract_route_coords(route_result: dict) -> list[tuple[float, float]]:
        """MapBox directions 결과에서 좌표 리스트 추출."""
        coords = []
        if not route_result:
            return coords
        # routes[0].geometry.coordinates
        routes = route_result.get('routes', [])
        if routes:
            geom = routes[0].get('geometry', {})
            for c in geom.get('coordinates', []):
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    coords.append((c[1], c[0]))
        if not coords:
            # flat coordinates list
            for c in route_result.get('coordinates', []):
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    coords.append((c[1], c[0]))
        return coords

    @staticmethod
    def _douglas_peucker(coords: list[tuple], epsilon: float = 0.0001) -> list[tuple]:
        """Douglas-Peucker 좌표 간소화."""
        if len(coords) <= 2:
            return coords
        start, end = coords[0], coords[-1]
        max_dist, max_idx = 0.0, 0
        for i in range(1, len(coords) - 1):
            p = coords[i]
            # 점-직선 거리
            n = abs((end[1] - start[1]) * p[0] - (end[0] - start[0]) * p[1]
                    + end[0] * start[1] - end[1] * start[0])
            d = ((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2) ** 0.5
            dist = n / d if d > 0 else 0
            if dist > max_dist:
                max_dist, max_idx = dist, i
        if max_dist > epsilon:
            left = MissionPlannerAgent._douglas_peucker(coords[:max_idx + 1], epsilon)
            right = MissionPlannerAgent._douglas_peucker(coords[max_idx:], epsilon)
            return left[:-1] + right
        return [start, end]

    async def node_embed_and_check(self, state: MissionPlannerState, config: RunnableConfig) -> dict:
        """PG 임베딩 후 anomaly/conflict 체크."""
        logger.info(f'[embed_and_check] retry={state.get("retry_count", 0)}')
        pg_json = state.get('pg_json', {})
        result: dict[str, Any] = {}

        try:
            # 0. MapBox MCP 보강 (distance/bearing/reverse_geocode)
            try:
                from fly2vec.src.embedding.enrichment import enrich_pg_with_mcp
                enriched_pg = await enrich_pg_with_mcp(pg_json, self.tools)
                logger.info(f'[embed_and_check] enrichment: {enriched_pg.get("route_stats", {}).get("enrichment_source", "unknown")}')
            except Exception as e:
                logger.warning(f'[embed_and_check] MCP enrichment failed, using raw PG: {e}')
                enriched_pg = pg_json

            # 1. embed_mission (보강된 PG 전달)
            embed_tool = next((t for t in self.tools if t.name == 'embed_mission'), None)
            if embed_tool:
                raw = await embed_tool.ainvoke({'pg_json': enriched_pg, 'store': True})
                embed_result = _parse_mcp_result(raw)
                if embed_result.get('success'):
                    vector = embed_result.get('data', {}).get('vector', [])
                    result['mission_vector'] = vector
                else:
                    logger.warning(f'[embed_and_check] embed_mission failed: {embed_result}')
                    vector = []
            else:
                vector = []

            # 2. anomaly_score
            if vector:
                anomaly_tool = next((t for t in self.tools if t.name == 'anomaly_score'), None)
                if anomaly_tool:
                    raw = await anomaly_tool.ainvoke({'vector': vector})
                    anomaly_result = _parse_mcp_result(raw)
                    if anomaly_result.get('success'):
                        result['anomaly_score'] = anomaly_result.get('data', {}).get('score', 0.0)
                    else:
                        result['anomaly_score'] = 0.0
                else:
                    result['anomaly_score'] = 0.0

            # 3. find_similar
            if vector:
                similar_tool = next((t for t in self.tools if t.name == 'find_similar'), None)
                if similar_tool:
                    raw = await similar_tool.ainvoke({'vector': vector, 'top_k': 5})
                    similar_result = _parse_mcp_result(raw)
                    if similar_result.get('success'):
                        result['similar_missions'] = similar_result.get('data', {}).get('similar_missions', [])

            # 4. detect_conflict (군집 분석)
            conflict_tool = next((t for t in self.tools if t.name == 'detect_conflict'), None)
            if conflict_tool:
                raw = await conflict_tool.ainvoke({'missions': [pg_json]})
                conflict_result = _parse_mcp_result(raw)
                if conflict_result.get('success'):
                    result['conflict_points'] = conflict_result.get('data', {}).get('conflicts', [])
                else:
                    result['conflict_points'] = []
            else:
                result['conflict_points'] = []

            # 5. recommend_safe_route (anomaly 높을 때 4축 다기준 추천)
            score = result.get('anomaly_score', 0.0)
            if score > ANOMALY_THRESHOLD and vector:
                rec_tool = next((t for t in self.tools if t.name == 'recommend_safe_route'), None)
                if not rec_tool:
                    # fallback: 기존 recommend_route
                    rec_tool = next((t for t in self.tools if t.name == 'recommend_route'), None)
                if rec_tool:
                    try:
                        raw = await rec_tool.ainvoke({
                            'vector': vector,
                            'mission_type': pg_json.get('mission_type', 'unknown'),
                            'top_k': 3,
                            'max_anomaly': 0.3,
                            'swarm_vectors': [],
                            'weights': {},
                        })
                        rec_result = _parse_mcp_result(raw)
                        if rec_result.get('success'):
                            result['recommended_routes'] = rec_result.get('data', {}).get('recommended_routes', [])
                            logger.info(f'[embed_and_check] {len(result["recommended_routes"])} routes recommended (4-axis)')
                    except Exception as e:
                        logger.warning(f'[embed_and_check] recommend_safe_route failed: {e}')

            # 6. analyze_swarm (패턴 분류)
            analyze_tool = next((t for t in self.tools if t.name == 'analyze_swarm'), None)
            if analyze_tool:
                try:
                    raw = await analyze_tool.ainvoke({'missions': [pg_json]})
                    analyze_result = _parse_mcp_result(raw)
                    if analyze_result.get('success'):
                        result['swarm_analysis'] = analyze_result.get('data', {})
                except Exception as e:
                    logger.warning(f'[embed_and_check] analyze_swarm failed: {e}')

        except Exception as e:
            logger.warning(f'[embed_and_check] Tool call failed (graceful): {e}')
            result.setdefault('anomaly_score', 0.0)
            result.setdefault('conflict_points', [])

        score = result.get('anomaly_score', 0.0)
        conflicts = result.get('conflict_points', [])
        result['is_verified'] = score <= ANOMALY_THRESHOLD and len(conflicts) == 0
        logger.info(f'[embed_and_check] anomaly={score:.3f}, conflicts={len(conflicts)}, verified={result["is_verified"]}')
        return result

    async def node_modify_pg(self, state: MissionPlannerState, config: RunnableConfig) -> dict:
        """LLM에게 PG 수정 요청."""
        retry = state.get('retry_count', 0)
        score = state.get('anomaly_score', 0.0)
        conflicts = state.get('conflict_points', [])
        current_pg = state.get('pg_json', {})

        conflict_desc = ''
        if conflicts:
            conflict_desc = '\n충돌 지점:\n' + '\n'.join(
                f"  - 드론{c.get('drone_a')}↔드론{c.get('drone_b')}: {c.get('distance_m', '?')}m"
                for c in conflicts[:5]
            )

        # 추천 경로 컨텍스트 (RAG)
        rec_desc = ''
        recommended = state.get('recommended_routes', [])
        if recommended:
            rec_desc = '\n추천 정상 경로 (유사하면서 anomaly 낮은 과거 미션):\n' + '\n'.join(
                f"  - {r.get('payload', {}).get('mission_type', '?')} "
                f"({r.get('payload', {}).get('wp_count', '?')} WP, 유사도 {r.get('score', 0):.0%})"
                for r in recommended[:3]
            )

        modification_prompt = (
            f'현재 PG의 anomaly score={score:.3f}, conflicts={len(conflicts)}건.{conflict_desc}{rec_desc}\n\n'
            f'현재 PG JSON:\n```json\n{json.dumps(current_pg, ensure_ascii=False)[:2000]}\n```\n\n'
            f'경로를 수정하여 anomaly를 0.5 이하로 줄이고 충돌을 해소하세요.\n'
            f'위 추천 경로를 참고하여 구조적으로 안전한 경로로 수정하세요.\n'
            f'수정된 전체 PG JSON을 반환하세요. (시도 {retry+1}/{MAX_VERIFICATION_LOOPS})'
        )

        response = await self.model.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=modification_prompt),
        ])

        modified_pg = _parse_json_response(response.content)
        if modified_pg.get('waypoints'):
            logger.info(f'[modify_pg] Modified PG: {len(modified_pg["waypoints"])} WPs (retry {retry+1})')
            return {'pg_json': modified_pg, 'retry_count': retry + 1}
        else:
            logger.warning(f'[modify_pg] Failed to parse modified PG, keeping original')
            return {'retry_count': retry + 1}

    async def node_finalize(self, state: MissionPlannerState, config: RunnableConfig) -> dict:
        """최종 PG 확정."""
        logger.info(
            f'[finalize] verified={state.get("is_verified")}, '
            f'anomaly={state.get("anomaly_score", 0.0):.3f}, '
            f'retries={state.get("retry_count", 0)}'
        )
        return {'is_verified': True}


if __name__ == '__main__':
    import asyncio

    async def main():
        agent = MissionPlannerAgent()
        await agent.initialize()
        result = await agent.graph.ainvoke({
            'messages': [HumanMessage(content='서울역에서 인천공항까지 50m 고도로 비행')],
            'retry_count': 0,
            'is_verified': False,
        })
        print(result)

    asyncio.run(main())
