"""PG JSON 파생변수 보강 모듈 (제안서 4.2절 ① 단계).

MapBox MCP 도구 우선 호출 → 로컬 Haversine/Bearing fallback.
+ 노드 15d 피처 생성.
"""

import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 Haversine 거리 (미터)."""
    R = 6371000
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 방위각 (0~360, 0=North, 90=East)."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(rlat2)
    y = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bearing_change(b1: float, b2: float) -> float:
    """두 방위각 간 변화량 (-180~180)."""
    diff = b2 - b1
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff


def _infer_mission_type(wps: list[dict], bearing_changes: list[float]) -> str:
    """경로 패턴 자동 추론."""
    if len(wps) < 4:
        return 'linear_flight'
    first = wps[0]
    last = wps[-1]
    total_dist = sum(
        haversine_m(wps[i]['lat_deg'], wps[i]['lon_deg'], wps[i + 1]['lat_deg'], wps[i + 1]['lon_deg'])
        for i in range(len(wps) - 1)
    )
    start_end_dist = haversine_m(first['lat_deg'], first['lon_deg'], last['lat_deg'], last['lon_deg'])

    if len(wps) >= 8 and start_end_dist < total_dist * 0.2:
        return 'circular_flight'
    reversals = sum(1 for bc in bearing_changes if abs(bc) > 150)
    if reversals >= 2 and len(wps) >= 6:
        return 'grid_search'
    avg_change = sum(abs(bc) for bc in bearing_changes) / max(len(bearing_changes), 1)
    if avg_change < 15:
        return 'linear_flight'
    return 'waypoint_sequence'


def enrich_pg_features(pg_json: dict) -> dict:
    """PG JSON에 파생변수를 보강.

    엣지 피처, 노드 피처, 경로 통계를 추가하여 반환.
    원본 pg_json을 변경하지 않고 복사본 반환.
    """
    import copy
    enriched = copy.deepcopy(pg_json)
    wps = enriched.get('waypoints', [])
    if len(wps) < 2:
        enriched['route_stats'] = {
            'total_distance_m': 0, 'avg_bearing_change': 0,
            'max_bearing_change': 0, 'mission_type': 'unknown',
        }
        return enriched

    n = len(wps)

    # 엣지 계산: distance, bearing, bearing_change
    distances: list[float] = []
    bearings: list[float] = []
    bearing_changes_list: list[float] = []

    for i in range(n - 1):
        w1, w2 = wps[i], wps[i + 1]
        d = haversine_m(w1['lat_deg'], w1['lon_deg'], w2['lat_deg'], w2['lon_deg'])
        b = bearing_deg(w1['lat_deg'], w1['lon_deg'], w2['lat_deg'], w2['lon_deg'])
        distances.append(d)
        bearings.append(b)

    for i in range(len(bearings) - 1):
        bearing_changes_list.append(bearing_change(bearings[i], bearings[i + 1]))

    # 노드별 피처 추가
    for i, wp in enumerate(wps):
        wp['seq_ratio'] = i / max(n - 1, 1)
        wp['is_start'] = 1.0 if i == 0 else 0.0
        wp['is_end'] = 1.0 if i == n - 1 else 0.0
        wp['is_land_cmd'] = 1.0 if wp.get('command', '') in ('NAV_LAND', 'LAND') else 0.0

        # distance from/to prev/next
        wp['distance_from_prev_m'] = distances[i - 1] if i > 0 else 0.0
        wp['distance_to_next_m'] = distances[i] if i < n - 1 else 0.0

        # bearing
        wp['bearing_from_prev'] = bearings[i - 1] if i > 0 else 0.0
        wp['bearing_change_deg'] = bearing_changes_list[i - 1] if 0 < i < n - 1 and i - 1 < len(bearing_changes_list) else 0.0

        # altitude change
        wp['alt_change_m'] = wp.get('alt_m', 0) - wps[i - 1].get('alt_m', 0) if i > 0 else 0.0

        # terrain_type placeholder (MapBox reverse_geocode 연동 시 채움)
        wp.setdefault('terrain_type', 0.0)

    # 경로 통계
    total_dist = sum(distances)
    avg_bc = sum(abs(bc) for bc in bearing_changes_list) / max(len(bearing_changes_list), 1)
    max_bc = max((abs(bc) for bc in bearing_changes_list), default=0)
    mission_type = enriched.get('mission_type') or _infer_mission_type(wps, bearing_changes_list)

    enriched['route_stats'] = {
        'total_distance_m': round(total_dist, 1),
        'avg_distance_m': round(total_dist / max(len(distances), 1), 1),
        'max_distance_m': round(max(distances, default=0), 1),
        'avg_bearing_change': round(avg_bc, 1),
        'max_bearing_change': round(max_bc, 1),
        'mission_type': mission_type,
        'wp_count': n,
    }

    return enriched


def build_node_features_15d(wp: dict) -> list[float]:
    """단일 WP에서 정규화된 15d 피처 벡터 생성.

    [lat_norm, lon_norm, alt_norm, seq_ratio, speed_norm,
     distance_prev_norm, distance_next_norm, bearing_norm, bearing_change_norm,
     hold_sec_norm, alt_change_norm, is_start, is_end, is_land_cmd, terrain_type_norm]
    """
    def _norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-6)))

    return [
        _norm(wp.get('lat_deg', 35.5), 33.0, 38.0),
        _norm(wp.get('lon_deg', 128.0), 124.0, 132.0),
        _norm(wp.get('alt_m', 0), 0, 500),
        wp.get('seq_ratio', 0.0),
        _norm(wp.get('speed_mps', 2.0), 0, 15),
        _norm(wp.get('distance_from_prev_m', 0), 0, 1000),
        _norm(wp.get('distance_to_next_m', 0), 0, 1000),
        _norm(wp.get('bearing_from_prev', 0), 0, 360),
        _norm(abs(wp.get('bearing_change_deg', 0)), 0, 180),
        _norm(wp.get('hold_sec', 0), 0, 60),
        _norm(abs(wp.get('alt_change_m', 0)), 0, 100),
        wp.get('is_start', 0.0),
        wp.get('is_end', 0.0),
        wp.get('is_land_cmd', 0.0),
        _norm(wp.get('terrain_type', 0), 0, 5),
    ]


# ── MapBox MCP 우선 보강 ─────────────────────────────────────


def _parse_mcp_raw(raw: Any) -> dict:
    """MCP 도구 응답 파싱 (list[{'type':'text','text':'<json>'}] 형태 대응)."""
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


async def enrich_pg_with_mcp(pg_json: dict, tools: list) -> dict:
    """MapBox MCP 도구로 PG를 보강 (제안서 4.2절 ① 단계).

    MCP distance-tool / bearing-tool / reverse-geocode-tool 우선 호출.
    실패 시 로컬 Haversine/Bearing fallback.
    """
    import copy
    enriched = copy.deepcopy(pg_json)
    wps = enriched.get('waypoints', [])
    if len(wps) < 2:
        return enrich_pg_features(pg_json)

    # MCP 도구 탐색
    distance_tool = next((t for t in tools if 'distance' in t.name.lower()), None)
    bearing_tool_mcp = next((t for t in tools if 'bearing' in t.name.lower()), None)
    reverse_geo_tool = next(
        (t for t in tools if 'reverse' in t.name.lower() and 'geocode' in t.name.lower()), None,
    )

    mcp_used = bool(distance_tool or bearing_tool_mcp)
    n = len(wps)
    distances: list[float] = []
    bearings_list: list[float] = []

    for i in range(n - 1):
        w1, w2 = wps[i], wps[i + 1]
        lat1, lon1 = w1['lat_deg'], w1['lon_deg']
        lat2, lon2 = w2['lat_deg'], w2['lon_deg']

        # Distance: MCP 우선 → 로컬 fallback
        d = None
        if distance_tool:
            try:
                raw = await distance_tool.ainvoke({
                    'origin': f'{lon1},{lat1}', 'destination': f'{lon2},{lat2}',
                })
                res = _parse_mcp_raw(raw)
                d = res.get('distance') or res.get('distance_m')
                if isinstance(d, (int, float)):
                    d = float(d)
                else:
                    d = None
            except Exception:
                d = None
        if d is None:
            d = haversine_m(lat1, lon1, lat2, lon2)
        distances.append(d)

        # Bearing: MCP 우선 → 로컬 fallback
        b = None
        if bearing_tool_mcp:
            try:
                raw = await bearing_tool_mcp.ainvoke({
                    'origin': f'{lon1},{lat1}', 'destination': f'{lon2},{lat2}',
                })
                res = _parse_mcp_raw(raw)
                b = res.get('bearing') or res.get('bearing_deg')
                if isinstance(b, (int, float)):
                    b = float(b)
                else:
                    b = None
            except Exception:
                b = None
        if b is None:
            b = bearing_deg(lat1, lon1, lat2, lon2)
        bearings_list.append(b)

    # bearing_change (로컬 계산 — MCP에 없음)
    bc_list: list[float] = []
    for i in range(len(bearings_list) - 1):
        bc_list.append(bearing_change(bearings_list[i], bearings_list[i + 1]))

    # Reverse geocode: 시작/끝 2건만 (API 절약)
    places: dict[str, str] = {}
    if reverse_geo_tool:
        for label, wp in [('start', wps[0]), ('end', wps[-1])]:
            try:
                raw = await reverse_geo_tool.ainvoke({
                    'longitude': wp['lon_deg'], 'latitude': wp['lat_deg'],
                })
                res = _parse_mcp_raw(raw)
                places[label] = res.get('place_name', res.get('name', ''))
            except Exception:
                places[label] = ''

    # 노드별 피처 부여 (enrich_pg_features와 동일 로직)
    for i, wp in enumerate(wps):
        wp['seq_ratio'] = i / max(n - 1, 1)
        wp['is_start'] = 1.0 if i == 0 else 0.0
        wp['is_end'] = 1.0 if i == n - 1 else 0.0
        wp['is_land_cmd'] = 1.0 if wp.get('command', '') in ('NAV_LAND', 'LAND') else 0.0
        wp['distance_from_prev_m'] = distances[i - 1] if i > 0 else 0.0
        wp['distance_to_next_m'] = distances[i] if i < n - 1 else 0.0
        wp['bearing_from_prev'] = bearings_list[i - 1] if i > 0 else 0.0
        wp['bearing_change_deg'] = bc_list[i - 1] if 0 < i < n - 1 and i - 1 < len(bc_list) else 0.0
        wp['alt_change_m'] = wp.get('alt_m', 0) - wps[i - 1].get('alt_m', 0) if i > 0 else 0.0
        wp.setdefault('terrain_type', 0.0)

    total_dist = sum(distances)
    avg_bc = sum(abs(bc) for bc in bc_list) / max(len(bc_list), 1)
    max_bc = max((abs(bc) for bc in bc_list), default=0)
    mission_type = enriched.get('mission_type') or _infer_mission_type(wps, bc_list)

    enriched['route_stats'] = {
        'total_distance_m': round(total_dist, 1),
        'avg_distance_m': round(total_dist / max(len(distances), 1), 1),
        'max_distance_m': round(max(distances, default=0), 1),
        'avg_bearing_change': round(avg_bc, 1),
        'max_bearing_change': round(max_bc, 1),
        'mission_type': mission_type,
        'wp_count': n,
        'start_place': places.get('start', ''),
        'end_place': places.get('end', ''),
        'enrichment_source': 'mapbox_mcp' if mcp_used else 'local',
    }

    logger.info(
        f'[enrich] source={enriched["route_stats"]["enrichment_source"]}, '
        f'wps={n}, dist={total_dist:.0f}m, type={mission_type}'
    )
    return enriched
