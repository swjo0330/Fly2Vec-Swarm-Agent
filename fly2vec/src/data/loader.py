"""기존 실험 데이터 → Qdrant 벌크 로드.

기존 파일:
  - data/dataset_mission_plan.csv (2,769건 미션)
  - data/dataset_wp_spot.csv (47,266 WP)
  - results/enriched_routes.csv (MapBox 보강)

통합 플로우:
  CSV 파싱 → 필터링 → PG JSON 변환 → Node2Vec 임베딩 → Qdrant upsert
"""

import asyncio
import csv
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from fly2vec.src.embedding.node2vec_encoder import Node2VecEncoder
from fly2vec.src.qdrant.fly2vec_qdrant import Fly2VecQdrantManager
from fly2vec.src.qdrant.collections import COLLECTION_MISSION

logger = logging.getLogger(__name__)

# 기존 데이터 경로 (proposal/ 기준)
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'  # fly2vec/data/
RESULTS_DIR = Path(__file__).resolve().parents[2] / 'results'


def parse_missions_csv(
    mission_csv: Path,
    wp_csv: Path,
) -> list[dict[str, Any]]:
    """mission_plan + wp_spot CSV를 PG JSON 리스트로 변환.

    필터링:
      - 한국 범위: lat 33~38, lng 124~132
      - 고도 이상치: 0~500m
      - WP 수: >= 4
    """
    # WP 데이터 로드 (wp_spot_id → WP 리스트)
    wp_map: dict[str, list[dict]] = {}
    with open(wp_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            spot_id = row['wp_spot_id']
            try:
                lat = float(row['wp_lat'])
                lon = float(row['wp_lng'])
                alt = float(row['wp_alt'])
                seq = int(row['wp_seq'])
            except (ValueError, KeyError):
                continue

            # 필터: 한국 범위 + 고도
            if not (33 <= lat <= 38 and 124 <= lon <= 132):
                continue
            if not (0 <= alt <= 500):
                continue

            wp_map.setdefault(spot_id, []).append({
                'lat_deg': lat,
                'lon_deg': lon,
                'alt_m': alt,
                'speed_mps': float(row.get('wp_speed', 0)) or 2.0,
                'seq': seq,
                'command': 'NAV_LAND' if row.get('wp_type') == 'LAND' else 'NAV_WAYPOINT',
                'hold_sec': float(row.get('wp_wait', 0)),
            })

    # 미션 데이터 로드
    missions = []
    with open(mission_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            spot_id = row.get('wp_spot_id', '')
            if spot_id not in wp_map:
                continue

            wps = sorted(wp_map[spot_id], key=lambda w: w['seq'])
            if len(wps) < 4:
                continue

            # mission_type 자동 추론
            mission_type = _infer_mission_type(wps)

            missions.append({
                'mission_id': row.get('flying_misn_plan_id', ''),
                'wp_spot_id': spot_id,
                'waypoints': wps,
                'mission_type': mission_type,
                'wp_count': len(wps),
            })

    logger.info(f'[loader] Parsed {len(missions)} missions from CSV (filtered)')
    return missions


def _infer_mission_type(wps: list[dict]) -> str:
    """WP 패턴에서 미션 타입 추론."""
    if len(wps) < 4:
        return 'waypoint_sequence'

    # bearing 변화 계산
    bearings = []
    for i in range(len(wps) - 1):
        dlat = wps[i + 1]['lat_deg'] - wps[i]['lat_deg']
        dlon = wps[i + 1]['lon_deg'] - wps[i]['lon_deg']
        bearing = np.degrees(np.arctan2(dlon, dlat)) % 360
        bearings.append(bearing)

    if len(bearings) < 3:
        return 'waypoint_sequence'

    # 방위 변화
    changes = []
    for i in range(len(bearings) - 1):
        diff = abs(bearings[i + 1] - bearings[i])
        if diff > 180:
            diff = 360 - diff
        changes.append(diff)

    avg_change = np.mean(changes)

    # 격자 패턴: 큰 방위 변화 반복
    if avg_change > 60:
        return 'grid_search'

    # 원형: 시작-끝 근접
    start = wps[0]
    end = wps[-1]
    dist_start_end = np.sqrt(
        (start['lat_deg'] - end['lat_deg'])**2 +
        (start['lon_deg'] - end['lon_deg'])**2
    )
    total_dist = sum(
        np.sqrt(
            (wps[i+1]['lat_deg'] - wps[i]['lat_deg'])**2 +
            (wps[i+1]['lon_deg'] - wps[i]['lon_deg'])**2
        )
        for i in range(len(wps) - 1)
    )
    if total_dist > 0 and dist_start_end / total_dist < 0.2 and len(wps) >= 8:
        return 'circular_flight'

    if avg_change < 15:
        return 'linear_flight'

    return 'waypoint_sequence'


async def bulk_load_to_qdrant(
    missions: list[dict],
    encoder: Node2VecEncoder | None = None,
    qdrant: Fly2VecQdrantManager | None = None,
    batch_size: int = 50,
) -> dict[str, int]:
    """미션 리스트를 임베딩하여 Qdrant에 벌크 로드.

    Returns:
        {'total': N, 'loaded': M, 'failed': F}
    """
    encoder = encoder or Node2VecEncoder()
    qdrant = qdrant or Fly2VecQdrantManager(
        url=os.getenv('QDRANT_URL', 'http://localhost:6340')
    )

    loaded = 0
    failed = 0

    for i in range(0, len(missions), batch_size):
        batch = missions[i:i + batch_size]
        for mission in batch:
            try:
                pg_json = {'waypoints': mission['waypoints']}
                vector = encoder.encode_mission(pg_json)

                await qdrant.store_mission_vector(
                    vector=vector,
                    payload={
                        'mission_id': mission.get('mission_id', ''),
                        'mission_type': mission.get('mission_type', 'unknown'),
                        'wp_count': mission.get('wp_count', 0),
                        'wp_spot_id': mission.get('wp_spot_id', ''),
                    },
                )
                loaded += 1
            except Exception as e:
                logger.warning(f'[loader] Failed to load mission: {e}')
                failed += 1

        logger.info(f'[loader] Progress: {min(i + batch_size, len(missions))}/{len(missions)}')

    result = {'total': len(missions), 'loaded': loaded, 'failed': failed}
    logger.info(f'[loader] Bulk load complete: {result}')
    return result


async def main():
    """메인 벌크 로드 실행."""
    mission_csv = DATA_DIR / 'dataset_mission_plan.csv'
    wp_csv = DATA_DIR / 'dataset_wp_spot.csv'

    if not mission_csv.exists() or not wp_csv.exists():
        logger.error(f'[loader] Data files not found in {DATA_DIR}')
        logger.info(f'[loader] Expected: {mission_csv}, {wp_csv}')
        logger.info('[loader] Create symlinks: ln -s ../../data fly2vec/data')
        return

    logger.info('[loader] Parsing CSV files...')
    missions = parse_missions_csv(mission_csv, wp_csv)

    logger.info(f'[loader] Starting bulk load of {len(missions)} missions...')
    result = await bulk_load_to_qdrant(missions)
    print(f'✓ Bulk load complete: {result}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
