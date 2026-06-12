"""공통 군집 시나리오 생성 모듈 v2 + v3 + v4.

v2: 모든 드론이 같은 경로 복사 + 같은 영역 배치.
v3: cooperative/relay는 드론마다 다른 실 경로 + WP 노이즈.

실험 스크립트에서 import하여 사용:
  from swarm_scenario_gen import gen_scenario_v3 as gen_scenario
"""
import math
import random

import numpy as np


def offset_coords(lat, lon, om, bd):
    R = 6371000.0; b = math.radians(bd)
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1)*math.cos(om/R)+math.cos(la1)*math.sin(om/R)*math.cos(b))
    lo2 = lo1 + math.atan2(math.sin(b)*math.sin(om/R)*math.cos(la1), math.cos(om/R)-math.sin(la1)*math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)


# ── v2 (기존, 하위 호환) ──

def gen_scenario_v2(idx, routes, rk):
    """v2: 모든 드론이 같은 영역 내 배치 (같은 경로 복사)."""
    n_drones = random.choice([3, 4, 5])
    typ = random.choices(
        ["cooperative_search", "relay_move", "collision_risk", "coverage_overlap"],
        weights=[40, 35, 10, 15]
    )[0]

    base_key = random.choice(rk[:500])
    base = routes[base_key]
    base_lats = base["wp_lat"].tolist()
    base_lngs = base["wp_lng"].tolist()
    base_alts = base["wp_alt"].tolist()
    center_lat = np.mean(base_lats)
    center_lon = np.mean(base_lngs)
    relay_bearing = random.uniform(0, 360)

    drones = []
    for di in range(n_drones):
        lats = list(base_lats)
        lngs = list(base_lngs)
        alts = list(base_alts)

        if typ == "cooperative_search":
            angle = (360 / n_drones) * di
            dist = random.uniform(300, 500)
            ol, olo = offset_coords(center_lat, center_lon, dist, angle)
            dlat, dlon = ol - center_lat, olo - center_lon
            lats = [l + dlat for l in lats]
            lngs = [l + dlon for l in lngs]
        elif typ == "collision_risk":
            if di > 0:
                dist = random.uniform(30, 150)
                bear = random.uniform(0, 360)
                ol, olo = offset_coords(center_lat, center_lon, dist, bear)
                dlat, dlon = ol - center_lat, olo - center_lon
                lats = [l + dlat for l in lats]
                lngs = [l + dlon for l in lngs]
        elif typ == "coverage_overlap":
            off_lat = random.uniform(-0.0003, 0.0003)
            off_lon = random.uniform(-0.0003, 0.0003)
            lats = [l + off_lat for l in lats]
            lngs = [l + off_lon for l in lngs]
        elif typ == "relay_move":
            dist = random.uniform(200, 400) * di
            if dist > 0:
                ol, olo = offset_coords(center_lat, center_lon, dist, relay_bearing)
                dlat, dlon = ol - center_lat, olo - center_lon
                lats = [l + dlat for l in lats]
                lngs = [l + dlon for l in lngs]

        drones.append({"lats": lats, "lngs": lngs, "alts": alts})
    return drones, typ


# ── v3 유틸리티 ──

def _route_to_lists(df):
    """DataFrame → (lats, lngs, alts) 리스트."""
    return df["wp_lat"].tolist(), df["wp_lng"].tolist(), df["wp_alt"].tolist()


def _translate_to_center(lats, lngs, alts, target_lat, target_lon):
    """경로 평균 중심을 target으로 평행 이동. 형상 보존."""
    dlat = target_lat - float(np.mean(lats))
    dlon = target_lon - float(np.mean(lngs))
    return [l + dlat for l in lats], [l + dlon for l in lngs], list(alts)


def _translate_to_start(lats, lngs, alts, target_lat, target_lon):
    """경로의 첫 WP를 target으로 평행 이동 (relay 핸드오프용)."""
    dlat = target_lat - lats[0]
    dlon = target_lon - lngs[0]
    return [l + dlat for l in lats], [l + dlon for l in lngs], list(alts)


def _add_wp_noise(lats, lngs, sigma_m=20.0):
    """WP마다 독립 가우시안 노이즈. 패턴 과적합 방지."""
    lat_c = float(np.mean(lats))
    dlat_per_m = 1.0 / 111000.0
    dlon_per_m = 1.0 / (111000.0 * max(math.cos(math.radians(lat_c)), 1e-6))
    nl = [l + random.gauss(0, sigma_m) * dlat_per_m for l in lats]
    ng = [g + random.gauss(0, sigma_m) * dlon_per_m for g in lngs]
    return nl, ng


# ── v3 본체 ──

def gen_scenario_v3(idx, routes, rk):
    """v3 군집 시나리오 생성.

    v2 대비 변경:
      - cooperative: 드론마다 다른 실 경로 → Sequential 엣지 차별화
      - relay: 드론마다 다른 경로 + 끝→시작 핸드오프 → Temporal 엣지 의미화
      - collision/coverage: v2 유지 (노이즈 없음 — 신호 보존)
      - WP 노이즈: cooperative/relay만 적용

    Args:
        idx: 시나리오 인덱스
        routes: {key: DataFrame(wp_lat, wp_lng, wp_alt)} 경로 딕셔너리
        rk: 셔플된 경로 키 리스트

    Returns:
        (drones, typ): 드론 리스트, 시나리오 타입
    """
    n_drones = random.choice([3, 4, 5])
    typ = random.choices(
        ["cooperative_search", "relay_move", "collision_risk", "coverage_overlap"],
        weights=[40, 35, 10, 15]
    )[0]

    # 공통 중심점: base route 기준
    pool = rk[:min(500, len(rk))]
    base_key = random.choice(pool)
    base = routes[base_key]
    center_lat = float(np.mean(base["wp_lat"]))
    center_lon = float(np.mean(base["wp_lng"]))

    drones = []

    # ── cooperative_search: 드론마다 다른 경로 + 같은 영역(500-800m 분산) ──
    if typ == "cooperative_search":
        chosen_keys = random.sample(pool, min(n_drones, len(pool)))
        for di in range(n_drones):
            lats, lngs, alts = _route_to_lists(routes[chosen_keys[di]])
            angle = (360.0 / n_drones) * di
            spread = random.uniform(500, 800)
            tgt_lat, tgt_lon = offset_coords(center_lat, center_lon, spread, angle)
            lats, lngs, alts = _translate_to_center(lats, lngs, alts, tgt_lat, tgt_lon)
            lats, lngs = _add_wp_noise(lats, lngs, sigma_m=20.0)
            drones.append({"lats": lats, "lngs": lngs, "alts": alts})

    # ── relay_move: 끝→시작 핸드오프 (200-400m) ──
    elif typ == "relay_move":
        chosen_keys = random.sample(pool, min(n_drones, len(pool)))
        relay_bearing = random.uniform(0, 360)
        for di in range(n_drones):
            lats, lngs, alts = _route_to_lists(routes[chosen_keys[di]])
            if di == 0:
                lats, lngs, alts = _translate_to_center(lats, lngs, alts,
                                                         center_lat, center_lon)
            else:
                prev_end_lat = drones[di - 1]["lats"][-1]
                prev_end_lon = drones[di - 1]["lngs"][-1]
                handoff_dist = random.uniform(200, 400)
                jitter = random.uniform(-30, 30)
                tgt_lat, tgt_lon = offset_coords(
                    prev_end_lat, prev_end_lon,
                    handoff_dist, (relay_bearing + jitter) % 360)
                lats, lngs, alts = _translate_to_start(lats, lngs, alts,
                                                        tgt_lat, tgt_lon)
            lats, lngs = _add_wp_noise(lats, lngs, sigma_m=15.0)
            drones.append({"lats": lats, "lngs": lngs, "alts": alts})

    # ── collision_risk: v2 유지 (노이즈 없음 — 30m 신호 보존) ──
    elif typ == "collision_risk":
        base_lats, base_lngs, base_alts = _route_to_lists(routes[base_key])
        for di in range(n_drones):
            lats, lngs, alts = list(base_lats), list(base_lngs), list(base_alts)
            if di > 0:
                dist = random.uniform(30, 150)
                bear = random.uniform(0, 360)
                ol, olo = offset_coords(center_lat, center_lon, dist, bear)
                dlat, dlon = ol - center_lat, olo - center_lon
                lats = [l + dlat for l in lats]
                lngs = [l + dlon for l in lngs]
            drones.append({"lats": lats, "lngs": lngs, "alts": alts})

    # ── coverage_overlap: v2 개선 (도→미터 통일, 노이즈 없음) ──
    elif typ == "coverage_overlap":
        base_lats, base_lngs, base_alts = _route_to_lists(routes[base_key])
        for di in range(n_drones):
            lats, lngs, alts = list(base_lats), list(base_lngs), list(base_alts)
            off_dist = random.uniform(0, 45)
            off_bear = random.uniform(0, 360)
            ol, olo = offset_coords(center_lat, center_lon, off_dist, off_bear)
            dlat, dlon = ol - center_lat, olo - center_lon
            lats = [l + dlat for l in lats]
            lngs = [l + dlon for l in lngs]
            drones.append({"lats": lats, "lngs": lngs, "alts": alts})

    return drones, typ


# ── v4 경로 유형 분류 ──

def classify_route(df):
    """경로 유형 분류: grid(격자)/linear(선형)/mixed(혼합)/short(짧은).

    방위각 변화량 평균 + 부호 반전 횟수로 격자 패턴을 안정적으로 판별.
    """
    lats = df["wp_lat"].tolist()
    lngs = df["wp_lng"].tolist()
    if len(lats) < 4:
        return "short"

    bearings = []
    for i in range(len(lats) - 1):
        la1, lo1 = math.radians(lats[i]), math.radians(lngs[i])
        la2, lo2 = math.radians(lats[i + 1]), math.radians(lngs[i + 1])
        dlon = lo2 - lo1
        x = math.sin(dlon) * math.cos(la2)
        y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
        bearings.append((math.degrees(math.atan2(x, y)) + 360) % 360)

    signed_changes = []
    for i in range(len(bearings) - 1):
        diff = bearings[i + 1] - bearings[i]
        if diff > 180: diff -= 360
        if diff < -180: diff += 360
        signed_changes.append(diff)

    if not signed_changes:
        return "short"

    avg_abs_change = float(np.mean([abs(c) for c in signed_changes]))
    sign_flips = sum(1 for i in range(len(signed_changes) - 1)
                     if signed_changes[i] * signed_changes[i + 1] < 0)
    flip_ratio = sign_flips / max(len(signed_changes) - 1, 1)

    if avg_abs_change > 90 or (avg_abs_change > 50 and flip_ratio > 0.5):
        return "grid"
    elif avg_abs_change < 20:
        return "linear"
    else:
        return "mixed"


def precompute_route_types(routes):
    """모든 경로의 유형을 사전 계산."""
    return {key: classify_route(df) for key, df in routes.items()}


# ── v4 본체 ──

def gen_scenario_v4(idx, routes, rk, route_types):
    """v4 군집 시나리오 생성.

    v3 대비 변경:
      - cooperative/relay: 같은 유형 경로끼리만 조합
      - collision: 고도 동일 유지 (고도 분리하면 충돌 판별 왜곡)
      - cooperative만 고도 오프셋 (수직 분리 패턴)
    """
    n_drones = random.choice([3, 4, 5])
    typ = random.choices(
        ["cooperative_search", "relay_move", "collision_risk", "coverage_overlap"],
        weights=[40, 35, 10, 15]
    )[0]

    pool = rk[:min(500, len(rk))]
    base_key = random.choice(pool)
    base = routes[base_key]
    center_lat = float(np.mean(base["wp_lat"]))
    center_lon = float(np.mean(base["wp_lng"]))

    drones = []

    if typ == "cooperative_search":
        base_type = route_types.get(base_key, "mixed")
        same_pool = [k for k in pool if route_types.get(k) == base_type]
        if len(same_pool) < n_drones:
            same_pool = [k for k in pool if route_types.get(k) in (base_type, "mixed")]
        if len(same_pool) < n_drones:
            same_pool = list(pool)
        chosen_keys = random.sample(same_pool, min(n_drones, len(same_pool)))
        for di in range(n_drones):
            lats, lngs, alts = _route_to_lists(routes[chosen_keys[di]])
            angle = (360.0 / n_drones) * di
            spread = random.uniform(500, 800)
            tgt_lat, tgt_lon = offset_coords(center_lat, center_lon, spread, angle)
            lats, lngs, alts = _translate_to_center(lats, lngs, alts, tgt_lat, tgt_lon)
            alt_offset = random.uniform(0, 30) * di
            alts = [a + alt_offset for a in alts]
            lats, lngs = _add_wp_noise(lats, lngs, sigma_m=20.0)
            drones.append({"lats": lats, "lngs": lngs, "alts": alts})

    elif typ == "relay_move":
        base_type = route_types.get(base_key, "mixed")
        same_pool = [k for k in pool if route_types.get(k) == base_type]
        if len(same_pool) < n_drones:
            same_pool = [k for k in pool if route_types.get(k) in (base_type, "mixed")]
        if len(same_pool) < n_drones:
            same_pool = list(pool)
        chosen_keys = random.sample(same_pool, min(n_drones, len(same_pool)))
        relay_bearing = random.uniform(0, 360)
        for di in range(n_drones):
            lats, lngs, alts = _route_to_lists(routes[chosen_keys[di]])
            if di == 0:
                lats, lngs, alts = _translate_to_center(lats, lngs, alts,
                                                         center_lat, center_lon)
            else:
                prev_end_lat = drones[di - 1]["lats"][-1]
                prev_end_lon = drones[di - 1]["lngs"][-1]
                handoff_dist = random.uniform(200, 400)
                jitter = random.uniform(-30, 30)
                tgt_lat, tgt_lon = offset_coords(
                    prev_end_lat, prev_end_lon,
                    handoff_dist, (relay_bearing + jitter) % 360)
                lats, lngs, alts = _translate_to_start(lats, lngs, alts,
                                                        tgt_lat, tgt_lon)
            lats, lngs = _add_wp_noise(lats, lngs, sigma_m=15.0)
            drones.append({"lats": lats, "lngs": lngs, "alts": alts})

    elif typ == "collision_risk":
        base_lats, base_lngs, base_alts = _route_to_lists(routes[base_key])
        for di in range(n_drones):
            lats, lngs, alts = list(base_lats), list(base_lngs), list(base_alts)
            if di > 0:
                dist = random.uniform(30, 150)
                bear = random.uniform(0, 360)
                ol, olo = offset_coords(center_lat, center_lon, dist, bear)
                dlat, dlon = ol - center_lat, olo - center_lon
                lats = [l + dlat for l in lats]
                lngs = [l + dlon for l in lngs]
            drones.append({"lats": lats, "lngs": lngs, "alts": alts})

    elif typ == "coverage_overlap":
        base_lats, base_lngs, base_alts = _route_to_lists(routes[base_key])
        for di in range(n_drones):
            lats, lngs, alts = list(base_lats), list(base_lngs), list(base_alts)
            off_dist = random.uniform(0, 45)
            off_bear = random.uniform(0, 360)
            ol, olo = offset_coords(center_lat, center_lon, off_dist, off_bear)
            dlat, dlon = ol - center_lat, olo - center_lon
            lats = [l + dlat for l in lats]
            lngs = [l + dlon for l in lngs]
            drones.append({"lats": lats, "lngs": lngs, "alts": alts})

    return drones, typ
