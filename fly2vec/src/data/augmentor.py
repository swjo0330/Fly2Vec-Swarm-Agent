"""Fly2Vec 군집 데이터 증강기.

설계 참조: docs/2026-05-02-swarm-data-augmentation-design.md

6단계 증강 파이프라인:
  Stage 1: 기본 시나리오 (250건)
  Stage 2: 군집 크기 변형 (150건)
  Stage 3: 시간축 변형 (150건)
  Stage 6: 이상 시나리오 주입 (200건)
  → 총 750건+ (목표 1,050건)
"""

import copy
import logging
import math
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 1도 ≈ 111km, 1m ≈ 0.000009도
M_TO_DEG = 1.0 / 111_000.0


class SwarmAugmentor:
    """군집 임무 데이터 증강기.

    실 관제 데이터(2,068건)를 기반으로 R-GAT 학습용 군집 시나리오를 생성합니다.
    """

    SCENARIO_TYPES = ['cooperative_search', 'collision_risk', 'coverage_overlap', 'relay_move']
    ANOMALY_TYPES = ['battery_low', 'gps_drift', 'vehicle_loss', 'comm_outage', 'near_miss', 'scatter']
    TIMING_PATTERNS = ['simultaneous', 'sequential', 'alternating', 'relay']

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._scenario_counter = 0

    def _next_id(self) -> str:
        self._scenario_counter += 1
        return f'aug_{self._scenario_counter:04d}'

    def generate_all(self, missions: list[dict], target_count: int = 1050) -> list[dict]:
        """전체 증강 수행.

        비율 조정:
          - 정상:이상 = 65:35 목표
          - 시간차 비행 25%+ (300건)
          - 이상 유형별 66건
        """
        logger.info(f'[augmentor] Starting augmentation from {len(missions)} base missions')

        scenarios = []
        scenarios.extend(self._stage1_basic(missions, count=350))  # 정상 중심
        scenarios.extend(self._stage2_size_variation(missions, count=200))  # 80% 정상
        scenarios.extend(self._stage3_temporal(missions, count=250))  # 시간차 (정상)
        scenarios.extend(self._stage6_anomaly_injection(missions, count=360))  # 60건/type

        # 목표 미달 시 추가 생성
        while len(scenarios) < target_count:
            extra = self._stage1_basic(missions, count=min(50, target_count - len(scenarios)))
            scenarios.extend(extra)

        logger.info(f'[augmentor] Generated {len(scenarios)} scenarios')
        return scenarios[:target_count]

    def _stage1_basic(self, missions: list[dict], count: int = 250) -> list[dict]:
        """Stage 1: 기본 시나리오 (2~5대, 정상:이상 7:3)."""
        scenarios = []
        normal_count = int(count * 0.7)

        for i in range(count):
            n_drones = self.rng.choice([2, 3, 3, 4, 4, 5])  # 2~5대, 3~4대 중심
            is_normal = i < normal_count

            if is_normal:
                pattern = self.rng.choice(['cooperative_search', 'relay_move'])
            else:
                pattern = self.rng.choice(['collision_risk', 'coverage_overlap'])

            combined = self._combine_missions(missions, n_drones, pattern)
            # 20% 확률로 시간차 비행 부여 (현실적 운용 반영)
            timing = 'simultaneous'
            if self.rng.random() < 0.2:
                timing = self.rng.choice(['sequential', 'alternating'])
                for idx, dm in enumerate(combined):
                    dm['time_offset_min'] = idx * self.rng.randint(3, 8)

            scenarios.append({
                'scenario_id': self._next_id(),
                'type': pattern,
                'label': 'normal' if is_normal else 'anomaly',
                'n_drones': n_drones,
                'timing': timing,
                'anomaly_type': None,
                'missions': combined,
            })

        return scenarios

    def _stage2_size_variation(self, missions: list[dict], count: int = 150) -> list[dict]:
        """Stage 2: 군집 크기 변형 (2~10대, 속도/배터리 이질성)."""
        scenarios = []
        for i in range(count):
            n_drones = self.rng.randint(2, 10)
            # 80% 정상, 20% 이상 (비율 보정)
            if i < int(count * 0.8):
                pattern = self.rng.choice(['cooperative_search', 'relay_move'])
            else:
                pattern = self.rng.choice(['collision_risk', 'coverage_overlap'])
            combined = self._combine_missions(missions, n_drones, pattern)

            # 드론별 속도 다양화
            for drone_mission in combined:
                speed_factor = self.np_rng.uniform(0.5, 2.0)
                for wp in drone_mission.get('waypoints', []):
                    wp['speed_mps'] = round(wp.get('speed_mps', 2.0) * speed_factor, 1)

            label = 'normal' if pattern in ['cooperative_search', 'relay_move'] else 'anomaly'
            scenarios.append({
                'scenario_id': self._next_id(),
                'type': pattern,
                'label': label,
                'n_drones': n_drones,
                'timing': 'simultaneous',
                'anomaly_type': None,
                'missions': combined,
            })

        return scenarios

    def _stage3_temporal(self, missions: list[dict], count: int = 150) -> list[dict]:
        """Stage 3: 시간축 변형 (순차/교대/릴레이)."""
        scenarios = []
        for _ in range(count):
            n_drones = self.rng.randint(3, 6)
            pattern = self.rng.choice(['cooperative_search', 'relay_move'])
            timing = self.rng.choice(['sequential', 'alternating', 'relay'])
            combined = self._combine_missions(missions, n_drones, pattern)

            # 시간 오프셋 부여
            if timing == 'sequential':
                for i, drone_mission in enumerate(combined):
                    drone_mission['time_offset_min'] = i * self.rng.randint(5, 10)
            elif timing == 'alternating':
                for i, drone_mission in enumerate(combined):
                    # 절반만 먼저 출발
                    drone_mission['time_offset_min'] = 0 if i % 2 == 0 else 15
            elif timing == 'relay':
                cycle = self.rng.randint(20, 40)
                for i, drone_mission in enumerate(combined):
                    drone_mission['time_offset_min'] = (i * cycle) % (cycle * 2)

            scenarios.append({
                'scenario_id': self._next_id(),
                'type': pattern,
                'label': 'normal',
                'n_drones': n_drones,
                'timing': timing,
                'anomaly_type': None,
                'missions': combined,
            })

        return scenarios

    def _stage6_anomaly_injection(self, missions: list[dict], count: int = 360) -> list[dict]:
        """Stage 6: 이상 시나리오 주입 (type별 60건)."""
        scenarios = []
        per_type = count // len(self.ANOMALY_TYPES)  # 60건/type

        for anomaly_type in self.ANOMALY_TYPES:
            for _ in range(per_type):
                n_drones = self.rng.randint(2, 5)
                combined = self._combine_missions(missions, n_drones, 'cooperative_search')

                # 이상 주입 대상: 첫 번째 드론
                target = combined[0]
                wps = target.get('waypoints', [])

                if anomaly_type == 'battery_low' and len(wps) > 4:
                    # WP 30~50% 지점에서 절단 + 귀환 WP
                    cut_point = self.rng.randint(len(wps) // 3, len(wps) // 2)
                    return_wp = copy.deepcopy(wps[0])
                    return_wp['command'] = 'NAV_RETURN'
                    target['waypoints'] = wps[:cut_point] + [return_wp]

                elif anomaly_type == 'gps_drift':
                    # 점진적 가우시안 노이즈
                    for i, wp in enumerate(wps):
                        sigma = (i / max(len(wps), 1)) * 20.0 * M_TO_DEG
                        wp['lat_deg'] = wp.get('lat_deg', 0) + self.np_rng.normal(0, sigma)
                        wp['lon_deg'] = wp.get('lon_deg', 0) + self.np_rng.normal(0, sigma)

                elif anomaly_type == 'vehicle_loss' and len(wps) > 3:
                    # 랜덤 시점에서 WP 삭제 (비행 중단)
                    loss_point = self.rng.randint(2, len(wps) - 1)
                    target['waypoints'] = wps[:loss_point]

                elif anomaly_type == 'comm_outage' and len(wps) > 5:
                    # 중간 WP 3~5개 삭제 (통신 손실 구간)
                    gap_start = self.rng.randint(2, len(wps) - 4)
                    gap_len = self.rng.randint(2, 4)
                    target['waypoints'] = wps[:gap_start] + wps[gap_start + gap_len:]

                elif anomaly_type == 'near_miss' and len(combined) > 1:
                    # 두 드론을 극근접 배치 (< 5m)
                    other = combined[1]
                    other_wps = other.get('waypoints', [])
                    if wps and other_wps:
                        mid = min(len(wps) // 2, len(wps) - 1)
                        other_mid = min(len(other_wps) // 2, len(other_wps) - 1)
                        tiny_offset = self.np_rng.uniform(1, 5) * M_TO_DEG
                        other_wps[other_mid]['lat_deg'] = wps[mid].get('lat_deg', 0) + tiny_offset
                        other_wps[other_mid]['lon_deg'] = wps[mid].get('lon_deg', 0) + tiny_offset

                elif anomaly_type == 'scatter':
                    # 점진적 분산 (오프셋 증가)
                    for i, drone_mission in enumerate(combined):
                        spread = (i + 1) * self.rng.randint(500, 2000)
                        angle = self.rng.uniform(0, 2 * math.pi)
                        offset_lat = spread * M_TO_DEG * math.cos(angle)
                        offset_lon = spread * M_TO_DEG * math.sin(angle)
                        drone_mission['waypoints'] = self._apply_offset(
                            drone_mission.get('waypoints', []), offset_lat, offset_lon
                        )

                # 실제 운용에서 발생하는 이상으로 분류 (기존 유형 기반)
                base_type = self.rng.choice(['cooperative_search', 'relay_move'])
                scenarios.append({
                    'scenario_id': self._next_id(),
                    'type': base_type,
                    'label': 'anomaly',
                    'n_drones': n_drones,
                    'timing': 'simultaneous',
                    'anomaly_type': anomaly_type,
                    'missions': combined,
                })

        return scenarios

    def _combine_missions(self, missions: list[dict], n_drones: int, pattern: str) -> list[dict]:
        """N대 드론 조합 생성."""
        if not missions:
            return []

        combined = []
        base_missions = self.rng.sample(missions, min(n_drones, len(missions)))

        # 부족하면 반복 선택
        while len(base_missions) < n_drones:
            base_missions.append(self.rng.choice(missions))

        for i, base in enumerate(base_missions):
            drone_mission = copy.deepcopy(base)
            drone_mission['drone_id'] = f'drone_{i}'
            wps = drone_mission.get('waypoints', [])

            if pattern == 'cooperative_search':
                # 영역 분리: 500m~2km offset
                offset_m = self.rng.randint(500, 2000)
                angle = (2 * math.pi / n_drones) * i
                offset_lat = offset_m * M_TO_DEG * math.cos(angle)
                offset_lon = offset_m * M_TO_DEG * math.sin(angle)
                drone_mission['waypoints'] = self._apply_offset(wps, offset_lat, offset_lon)

            elif pattern == 'collision_risk':
                # 동일 영역 수렴: 30~150m offset
                offset_m = self.rng.randint(30, 150)
                angle = self.rng.uniform(0, 2 * math.pi)
                offset_lat = offset_m * M_TO_DEG * math.cos(angle)
                offset_lon = offset_m * M_TO_DEG * math.sin(angle)
                drone_mission['waypoints'] = self._apply_offset(wps, offset_lat, offset_lon)

            elif pattern == 'coverage_overlap':
                # 같은 중심에 배치: 10~80m offset
                offset_m = self.rng.randint(10, 80)
                angle = self.rng.uniform(0, 2 * math.pi)
                offset_lat = offset_m * M_TO_DEG * math.cos(angle)
                offset_lon = offset_m * M_TO_DEG * math.sin(angle)
                drone_mission['waypoints'] = self._apply_offset(wps, offset_lat, offset_lon)

            elif pattern == 'relay_move':
                # 일렬 간격 배치: 800~1500m
                offset_m = self.rng.randint(800, 1500) * i
                drone_mission['waypoints'] = self._apply_offset(wps, offset_m * M_TO_DEG, 0)

            combined.append(drone_mission)

        return combined

    def _apply_offset(self, waypoints: list[dict], offset_lat: float, offset_lon: float) -> list[dict]:
        """위치 오프셋 적용."""
        result = []
        for wp in waypoints:
            new_wp = copy.deepcopy(wp)
            new_wp['lat_deg'] = new_wp.get('lat_deg', 0) + offset_lat
            new_wp['lon_deg'] = new_wp.get('lon_deg', 0) + offset_lon
            result.append(new_wp)
        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 테스트: 더미 미션으로 증강
    dummy_missions = [
        {
            'mission_id': f'test_{i}',
            'waypoints': [
                {'lat_deg': 37.38 + j * 0.001, 'lon_deg': 126.63 + j * 0.001, 'alt_m': 50, 'speed_mps': 2.0, 'seq': j, 'command': 'NAV_WAYPOINT', 'hold_sec': 0}
                for j in range(8)
            ],
            'mission_type': 'grid_search',
        }
        for i in range(100)
    ]

    augmentor = SwarmAugmentor(seed=42)
    scenarios = augmentor.generate_all(dummy_missions, target_count=100)

    # 통계
    from collections import Counter
    types = Counter(s['type'] for s in scenarios)
    labels = Counter(s['label'] for s in scenarios)
    anomalies = Counter(s['anomaly_type'] for s in scenarios if s['anomaly_type'])

    print(f'Total: {len(scenarios)}')
    print(f'Types: {dict(types)}')
    print(f'Labels: {dict(labels)}')
    print(f'Anomalies: {dict(anomalies)}')
