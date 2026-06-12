"""Fly2Vec 테스트 공통 fixture."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# fly2vec 패키지 import를 위한 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))


@pytest.fixture
def mock_qdrant():
    """Mock Fly2VecQdrantManager."""
    from fly2vec.src.qdrant.fly2vec_qdrant import Fly2VecQdrantManager

    manager = Fly2VecQdrantManager(url="http://fake:6333")
    manager._client = AsyncMock()
    manager._client.query_points = AsyncMock(return_value=MagicMock(points=[]))
    manager._client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    manager._client.create_collection = AsyncMock()
    manager._client.upsert = AsyncMock()
    manager._client.close = AsyncMock()
    return manager


@pytest.fixture
def sample_pg():
    """샘플 PG JSON (6 WP 격자 수색)."""
    return {
        'waypoints': [
            {'lat_deg': 37.38, 'lon_deg': 126.63, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 0, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.38, 'lon_deg': 126.64, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 1, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.37, 'lon_deg': 126.64, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 2, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.37, 'lon_deg': 126.63, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 3, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.37, 'lon_deg': 126.63, 'alt_m': 80, 'speed_mps': 2.0, 'seq': 4, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.37, 'lon_deg': 126.64, 'alt_m': 80, 'speed_mps': 2.0, 'seq': 5, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
        ],
        'mission_type': 'grid_search',
    }


@pytest.fixture
def sample_swarm_cooperative():
    """협력 수색 군집 (3대, 영역 분리)."""
    return [
        {'waypoints': [
            {'lat_deg': 37.38, 'lon_deg': 126.63, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 0, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.38, 'lon_deg': 126.64, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 1, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.37, 'lon_deg': 126.64, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 2, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.37, 'lon_deg': 126.63, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 3, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
        ]},
        {'waypoints': [
            {'lat_deg': 37.40, 'lon_deg': 126.63, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 0, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.40, 'lon_deg': 126.64, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 1, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.39, 'lon_deg': 126.64, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 2, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.39, 'lon_deg': 126.63, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 3, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
        ]},
        {'waypoints': [
            {'lat_deg': 37.42, 'lon_deg': 126.63, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 0, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.42, 'lon_deg': 126.64, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 1, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.41, 'lon_deg': 126.64, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 2, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.41, 'lon_deg': 126.63, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 3, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
        ]},
    ]


@pytest.fixture
def sample_swarm_collision():
    """충돌 위험 군집 (2대, 30m 근접)."""
    return [
        {'waypoints': [
            {'lat_deg': 37.38, 'lon_deg': 126.63, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 0, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.38, 'lon_deg': 126.64, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 1, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
        ]},
        {'waypoints': [
            {'lat_deg': 37.3801, 'lon_deg': 126.6301, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 0, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
            {'lat_deg': 37.3801, 'lon_deg': 126.6401, 'alt_m': 50, 'speed_mps': 2.0, 'seq': 1, 'command': 'NAV_WAYPOINT', 'hold_sec': 0},
        ]},
    ]
