#!/opt/anaconda3/bin/python
"""
MapBox REST API 기반 드론 미션 웨이포인트 데이터 강화 스크립트

기능:
  1. CSV 로드 + 필터링 (한국 범위, 고도 0-500m, 6+ WP) — test_anomaly_ensemble.py 동일 기준
  2. 로컬 연산 (API 불필요):
     - Haversine 거리 (연속 WP 간) → 엣지 가중치
     - Bearing (연속 WP 간) → 방향 피처
     - Bearing 변화량 → 커브 탐지
  3. MapBox REST API (rate-limited):
     - 역지오코딩: 경로 시작점, 종료점, 중심점 → 장소명
  4. 강화된 데이터 저장: results/enriched_routes.csv

실행: python3 enrich_with_mapbox.py

의존성: pandas, numpy, requests
"""

import os
import sys
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ──────────────────────────────────────────────
# 설정 상수
# ──────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ → 프로젝트 루트
DATA_PATH = PROJECT_ROOT / "fly2vec" / "data" / "dataset_wp_spot.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "report1"
OUTPUT_PATH = OUTPUT_DIR / "enriched_routes.csv"

# MapBox REST API 토큰 (.env에서 가져온 값)
MAPBOX_ACCESS_TOKEN = (
    "pk.eyJ1Ijoic3dqbyIsImEiOiJjbTBtYnh0bzMwMHhzMmtzYXlrYTJhNzhwIn0"
    ".BXGmWZvGxeJjwEI3997Rqw"
)
MAPBOX_GEOCODE_URL = (
    "https://api.mapbox.com/geocoding/v5/mapbox.places/{lng},{lat}.json"
)

# 필터링 조건 (test_anomaly_ensemble.py와 동일)
LAT_MIN, LAT_MAX = 33.0, 38.0   # 한국 위도 범위
LNG_MIN, LNG_MAX = 124.0, 132.0 # 한국 경도 범위
ALT_MIN, ALT_MAX = 0.0, 500.0   # 고도 범위 (m)
MIN_WP_COUNT = 4                  # 최소 웨이포인트 수

# API 호출 제한
MAX_ROUTES_DEMO = 50             # 데모: 상위 50개 경로만 API 호출
API_RATE_LIMIT_SEC = 1.0         # 요청 간 최소 대기 (초)
API_TIMEOUT_SEC = 10             # 요청 타임아웃 (초)
API_MAX_RETRIES = 2              # 실패 시 재시도 횟수

# 지구 반지름 (m) — Haversine 계산용
EARTH_RADIUS_M = 6_371_000


# ──────────────────────────────────────────────
# 1단계: 데이터 로드 + 필터링
# ──────────────────────────────────────────────

def load_and_filter(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """CSV 로드 → 한국 범위 + 고도 + WP 수 필터링 → 경로별 그룹"""
    print("=" * 70)
    print("[1] 데이터 로드 + 필터링")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"  원본 WP 수: {len(df):,}")
    print(f"  원본 경로 수 (wp_spot_id): {df['wp_spot_id'].nunique():,}")

    # 한국 범위 필터
    mask_korea = (
        (df["wp_lat"] >= LAT_MIN) & (df["wp_lat"] <= LAT_MAX) &
        (df["wp_lng"] >= LNG_MIN) & (df["wp_lng"] <= LNG_MAX)
    )
    df = df[mask_korea].copy()
    print(f"  한국 범위 필터 후: {len(df):,} WP")

    # 고도 필터
    mask_alt = (df["wp_alt"] >= ALT_MIN) & (df["wp_alt"] <= ALT_MAX)
    df = df[mask_alt].copy()
    print(f"  고도 필터 (0-500m) 후: {len(df):,} WP")

    # 경로별 그룹 + WP 수 필터
    grouped = {
        name: group.sort_values("wp_seq").reset_index(drop=True)
        for name, group in df.groupby("wp_spot_id")
    }
    routes = {k: v for k, v in grouped.items() if len(v) >= MIN_WP_COUNT}

    total_wp = sum(len(v) for v in routes.values())
    print(f"  WP >= {MIN_WP_COUNT} 필터 후: {len(routes):,} 경로 / {total_wp:,} WP")

    wp_counts = [len(v) for v in routes.values()]
    print(f"  WP 수 통계: min={min(wp_counts)}, median={int(np.median(wp_counts))}, "
          f"mean={np.mean(wp_counts):.1f}, max={max(wp_counts)}")
    print()
    return routes


# ──────────────────────────────────────────────
# 2단계: 로컬 연산 (Haversine / Bearing)
# ──────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이의 Haversine 거리 (미터 단위)"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이의 방위각 (0~360도, 0=North, 90=East)"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)

    x = math.sin(dlam) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2) -
         math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    brng = math.degrees(math.atan2(x, y))
    return brng % 360.0


def bearing_change(b1: float, b2: float) -> float:
    """두 방위각 사이의 변화량 (-180 ~ +180도)"""
    delta = b2 - b1
    # -180 ~ +180 범위로 정규화
    while delta > 180:
        delta -= 360
    while delta < -180:
        delta += 360
    return delta


def compute_route_features(route_df: pd.DataFrame) -> Dict:
    """
    경로 DataFrame에서 로컬 피처 계산

    반환값:
      - wp_count: 웨이포인트 수
      - total_distance_m: 총 거리 (m)
      - avg_distance_m: 평균 엣지 거리 (m)
      - max_distance_m: 최대 엣지 거리 (m)
      - avg_bearing_change: 평균 방위각 변화 (절댓값, deg)
      - max_bearing_change: 최대 방위각 변화 (절댓값, deg)
      - avg_altitude_m: 평균 고도 (m)
      - start_lat, start_lng: 시작점 좌표
      - end_lat, end_lng: 종료점 좌표
      - centroid_lat, centroid_lng: 중심점 좌표
      - distances: 엣지별 거리 리스트 (m)
      - bearings: 엣지별 방위각 리스트 (deg)
      - bearing_changes: 연속 방위각 변화 리스트 (deg)
    """
    lats = route_df["wp_lat"].values
    lngs = route_df["wp_lng"].values
    alts = route_df["wp_alt"].values
    n = len(lats)

    # 엣지별 거리 계산
    distances = []
    for i in range(n - 1):
        d = haversine_m(lats[i], lngs[i], lats[i + 1], lngs[i + 1])
        distances.append(d)

    # 엣지별 방위각 계산
    bearings = []
    for i in range(n - 1):
        b = bearing_deg(lats[i], lngs[i], lats[i + 1], lngs[i + 1])
        bearings.append(b)

    # 연속 방위각 변화량 (커브 탐지용)
    bearing_changes = []
    for i in range(len(bearings) - 1):
        bc = bearing_change(bearings[i], bearings[i + 1])
        bearing_changes.append(abs(bc))

    total_dist = sum(distances)
    avg_dist = total_dist / len(distances) if distances else 0
    max_dist = max(distances) if distances else 0
    avg_bc = np.mean(bearing_changes) if bearing_changes else 0
    max_bc = max(bearing_changes) if bearing_changes else 0

    return {
        "wp_count": n,
        "total_distance_m": round(total_dist, 2),
        "avg_distance_m": round(avg_dist, 2),
        "max_distance_m": round(max_dist, 2),
        "avg_bearing_change": round(avg_bc, 2),
        "max_bearing_change": round(max_bc, 2),
        "avg_altitude_m": round(float(np.mean(alts)), 2),
        "start_lat": lats[0],
        "start_lng": lngs[0],
        "end_lat": lats[-1],
        "end_lng": lngs[-1],
        "centroid_lat": round(float(np.mean(lats)), 6),
        "centroid_lng": round(float(np.mean(lngs)), 6),
        # 상세 리스트 (CSV 저장 시 제외)
        "_distances": distances,
        "_bearings": bearings,
        "_bearing_changes": bearing_changes,
    }


# ──────────────────────────────────────────────
# 3단계: MapBox REST API — 역지오코딩
# ──────────────────────────────────────────────

def reverse_geocode(lat: float, lng: float) -> Optional[str]:
    """
    MapBox REST API로 역지오코딩 수행

    반환: 장소명 문자열 (실패 시 None)
    API: GET /geocoding/v5/mapbox.places/{lng},{lat}.json
    """
    url = MAPBOX_GEOCODE_URL.format(lng=lng, lat=lat)
    params = {
        "access_token": MAPBOX_ACCESS_TOKEN,
        "language": "ko",         # 한국어 결과 우선
        "limit": 1,               # 최상위 결과 1개만
        "types": "place,locality,neighborhood,address",
    }

    for attempt in range(API_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=API_TIMEOUT_SEC)

            # Rate limit 처리 (429)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)  # 지수 백오프
                print(f"    ⚠ Rate limit (429), {wait}초 대기 후 재시도...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            if features:
                return features[0].get("place_name", None)
            return None

        except requests.exceptions.Timeout:
            print(f"    ⚠ 타임아웃 (attempt {attempt + 1}/{API_MAX_RETRIES + 1})")
            if attempt < API_MAX_RETRIES:
                time.sleep(2)
                continue
            return None

        except requests.exceptions.RequestException as e:
            print(f"    ⚠ API 에러: {e}")
            return None

    return None


def enrich_with_geocoding(
    records: List[Dict],
    max_routes: int = MAX_ROUTES_DEMO
) -> List[Dict]:
    """
    경로 레코드에 MapBox 역지오코딩 결과 추가

    API 호출: 경로당 3회 (시작점/종료점/중심점)
    Rate limit: 1초/요청
    """
    print("=" * 70)
    print(f"[3] MapBox 역지오코딩 (상위 {max_routes}개 경로)")
    print("=" * 70)

    total_calls = 0
    for i, rec in enumerate(records[:max_routes]):
        route_id = rec["route_id"]
        print(f"  [{i + 1}/{min(len(records), max_routes)}] {route_id}")

        # 시작점 역지오코딩
        place = reverse_geocode(rec["start_lat"], rec["start_lng"])
        rec["start_place"] = place or "N/A"
        total_calls += 1
        time.sleep(API_RATE_LIMIT_SEC)

        # 종료점 역지오코딩
        place = reverse_geocode(rec["end_lat"], rec["end_lng"])
        rec["end_place"] = place or "N/A"
        total_calls += 1
        time.sleep(API_RATE_LIMIT_SEC)

        # 중심점 역지오코딩
        place = reverse_geocode(rec["centroid_lat"], rec["centroid_lng"])
        rec["centroid_place"] = place or "N/A"
        total_calls += 1
        time.sleep(API_RATE_LIMIT_SEC)

        # 진행 상태 (10개마다)
        if (i + 1) % 10 == 0:
            print(f"    → {i + 1}개 완료, API 호출 {total_calls}회")

    # 나머지 경로 (API 미호출) — 빈 값으로 채움
    for rec in records[max_routes:]:
        rec["start_place"] = ""
        rec["end_place"] = ""
        rec["centroid_place"] = ""

    print(f"\n  총 API 호출: {total_calls}회")
    print()
    return records


# ──────────────────────────────────────────────
# 4단계: 결과 저장 + 통계 출력
# ──────────────────────────────────────────────

def save_and_report(records: List[Dict], output_path: Path):
    """강화된 데이터 CSV 저장 + 요약 통계 출력"""
    print("=" * 70)
    print("[4] 결과 저장 + 통계")
    print("=" * 70)

    # 내부 상세 리스트 필드 제거 (CSV에는 저장하지 않음)
    csv_records = []
    for rec in records:
        csv_rec = {k: v for k, v in rec.items() if not k.startswith("_")}
        csv_records.append(csv_rec)

    df = pd.DataFrame(csv_records)

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {output_path}")
    print(f"  행 수: {len(df):,}")
    print(f"  컬럼: {list(df.columns)}")
    print()

    # ── 요약 통계 ──
    print("─" * 50)
    print("요약 통계")
    print("─" * 50)

    print(f"\n  [경로 수] {len(df):,}")
    print(f"\n  [거리 (m)]")
    print(f"    총 거리  — min: {df['total_distance_m'].min():,.0f}, "
          f"median: {df['total_distance_m'].median():,.0f}, "
          f"mean: {df['total_distance_m'].mean():,.0f}, "
          f"max: {df['total_distance_m'].max():,.0f}")
    print(f"    평균 엣지 — min: {df['avg_distance_m'].min():,.0f}, "
          f"median: {df['avg_distance_m'].median():,.0f}, "
          f"mean: {df['avg_distance_m'].mean():,.0f}, "
          f"max: {df['avg_distance_m'].max():,.0f}")

    print(f"\n  [방위각 변화 (deg)]")
    print(f"    평균 변화 — min: {df['avg_bearing_change'].min():.1f}, "
          f"median: {df['avg_bearing_change'].median():.1f}, "
          f"mean: {df['avg_bearing_change'].mean():.1f}, "
          f"max: {df['avg_bearing_change'].max():.1f}")
    print(f"    최대 변화 — min: {df['max_bearing_change'].min():.1f}, "
          f"median: {df['max_bearing_change'].median():.1f}, "
          f"mean: {df['max_bearing_change'].mean():.1f}, "
          f"max: {df['max_bearing_change'].max():.1f}")

    print(f"\n  [고도 (m)]")
    print(f"    평균 고도 — min: {df['avg_altitude_m'].min():.1f}, "
          f"median: {df['avg_altitude_m'].median():.1f}, "
          f"mean: {df['avg_altitude_m'].mean():.1f}, "
          f"max: {df['avg_altitude_m'].max():.1f}")

    print(f"\n  [WP 수]")
    print(f"    min: {df['wp_count'].min()}, "
          f"median: {int(df['wp_count'].median())}, "
          f"mean: {df['wp_count'].mean():.1f}, "
          f"max: {df['wp_count'].max()}")

    # 역지오코딩 결과 샘플
    geocoded = df[df["start_place"].notna() & (df["start_place"] != "")]
    if not geocoded.empty:
        print(f"\n  [역지오코딩 샘플 (상위 5개)]")
        for _, row in geocoded.head(5).iterrows():
            print(f"    {row['route_id']}:")
            print(f"      시작: {row['start_place']}")
            print(f"      종료: {row['end_place']}")
            print(f"      중심: {row['centroid_place']}")

    # 커브 탐지 — 방위각 변화 상위 10개
    print(f"\n  [고커브 경로 TOP 10 (avg_bearing_change 기준)]")
    top_curves = df.nlargest(10, "avg_bearing_change")
    for _, row in top_curves.iterrows():
        print(f"    {row['route_id']}: avg={row['avg_bearing_change']:.1f}° "
              f"max={row['max_bearing_change']:.1f}° "
              f"dist={row['total_distance_m']:,.0f}m "
              f"wp={row['wp_count']}")

    print()


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  AERION — MapBox 기반 드론 미션 웨이포인트 데이터 강화      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # 1. 데이터 로드 + 필터링
    if not DATA_PATH.exists():
        print(f"ERROR: 데이터 파일 없음 — {DATA_PATH}")
        sys.exit(1)

    routes = load_and_filter(DATA_PATH)

    # 2. 로컬 피처 계산
    print("=" * 70)
    print("[2] 로컬 피처 계산 (Haversine / Bearing / Curve)")
    print("=" * 70)

    records = []
    for route_id, route_df in sorted(routes.items()):
        features = compute_route_features(route_df)
        features["route_id"] = route_id
        records.append(features)

    print(f"  {len(records):,}개 경로 피처 계산 완료")

    # 거리 0인 경로 확인 (동일 좌표 반복)
    zero_dist = [r for r in records if r["total_distance_m"] == 0]
    if zero_dist:
        print(f"  ⚠ 거리=0 경로 {len(zero_dist)}개 (동일 좌표 반복)")
    print()

    # 3. MapBox 역지오코딩 (데모: 상위 N개)
    records = enrich_with_geocoding(records, max_routes=MAX_ROUTES_DEMO)

    # 4. 결과 저장 + 통계
    save_and_report(records, OUTPUT_PATH)

    print("완료!")


if __name__ == "__main__":
    main()
