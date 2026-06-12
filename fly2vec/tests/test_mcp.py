"""Fly2Vec MCP 서버 단위 테스트 (Mock 기반)."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest


class TestGraphEmbeddingMCP:
    """GraphEmbeddingMCP 도구 테스트."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch('fly2vec.src.mcp_servers.graph_embedding_mcp.Fly2VecQdrantManager') as MockQdrant:
            mock_qdrant = AsyncMock()
            mock_qdrant.store_mission_vector = AsyncMock(return_value='point_001')
            mock_qdrant.find_similar = AsyncMock(return_value=[])
            mock_qdrant.ensure_collection = AsyncMock()
            MockQdrant.return_value = mock_qdrant

            from fly2vec.src.mcp_servers.graph_embedding_mcp import GraphEmbeddingMCP
            self.server = GraphEmbeddingMCP(port=9999)
            self.server.qdrant = mock_qdrant
            self.mock_qdrant = mock_qdrant

    def _get_tool(self, name: str):
        for tool in self.server.mcp._tool_manager._tools.values():
            if tool.name == name:
                return tool.fn
        raise KeyError(f"Tool '{name}' not found")

    @pytest.mark.asyncio
    async def test_embed_mission_returns_vector(self, sample_pg):
        embed = self._get_tool('embed_mission')
        result = await embed(pg_json=sample_pg, store=True)
        assert result['success'] is True
        assert len(result['data']['vector']) == 128
        assert result['data']['dimension'] == 128

    @pytest.mark.asyncio
    async def test_find_similar_returns_list(self):
        self.mock_qdrant.find_similar = AsyncMock(return_value=[
            {'score': 0.95, 'payload': {'mission_type': 'patrol'}},
            {'score': 0.88, 'payload': {'mission_type': 'search'}},
        ])
        find = self._get_tool('find_similar')
        result = await find(vector=[0.1] * 128, top_k=5)
        assert result['success'] is True
        assert result['data']['count'] == 2

    @pytest.mark.asyncio
    async def test_anomaly_score_range(self):
        self.mock_qdrant.find_similar = AsyncMock(return_value=[
            {'score': 0.7, 'payload': {}},
        ])
        score_fn = self._get_tool('anomaly_score')
        result = await score_fn(vector=[0.1] * 128)
        assert result['success'] is True
        assert 0.0 <= result['data']['score'] <= 1.0

    @pytest.mark.asyncio
    async def test_anomaly_high_when_isolated(self):
        self.mock_qdrant.find_similar = AsyncMock(return_value=[])
        score_fn = self._get_tool('anomaly_score')
        result = await score_fn(vector=[0.99] * 128)
        assert result['data']['score'] >= 0.9
        assert result['data']['is_anomalous'] is True


class TestSwarmAnalysisMCP:
    """SwarmAnalysisMCP 도구 테스트."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch('fly2vec.src.mcp_servers.swarm_analysis_mcp.Fly2VecQdrantManager') as MockQdrant:
            mock_qdrant = AsyncMock()
            mock_qdrant.store_mission_vector = AsyncMock(return_value='sw_001')
            mock_qdrant.ensure_collection = AsyncMock()
            MockQdrant.return_value = mock_qdrant

            from fly2vec.src.mcp_servers.swarm_analysis_mcp import SwarmAnalysisMCP
            self.server = SwarmAnalysisMCP(port=9998)
            self.server.qdrant = mock_qdrant

    def _get_tool(self, name: str):
        for tool in self.server.mcp._tool_manager._tools.values():
            if tool.name == name:
                return tool.fn
        raise KeyError(f"Tool '{name}' not found")

    @pytest.mark.asyncio
    async def test_analyze_swarm_pattern(self, sample_swarm_cooperative):
        analyze = self._get_tool('analyze_swarm')
        result = await analyze(missions=sample_swarm_cooperative)
        assert result['success'] is True
        assert 'pattern' in result['data']
        assert 'edge_stats' in result['data']

    @pytest.mark.asyncio
    async def test_detect_conflict_found(self, sample_swarm_collision):
        detect = self._get_tool('detect_conflict')
        result = await detect(missions=sample_swarm_collision, proximity_threshold_m=500.0)
        assert result['success'] is True
        assert result['data']['has_conflict'] is True
        assert result['data']['conflict_count'] > 0

    @pytest.mark.asyncio
    async def test_detect_conflict_none(self, sample_swarm_cooperative):
        detect = self._get_tool('detect_conflict')
        result = await detect(missions=sample_swarm_cooperative, proximity_threshold_m=100.0)
        assert result['success'] is True
        assert result['data']['has_conflict'] is False

    @pytest.mark.asyncio
    async def test_optimize_assignment(self, sample_swarm_cooperative):
        drones = [{'id': 'd1', 'position': {}, 'battery': 90}, {'id': 'd2', 'position': {}, 'battery': 85}]
        optimize = self._get_tool('optimize_assignment')
        result = await optimize(missions=sample_swarm_cooperative, drones=drones)
        assert result['success'] is True
        assert 'assignments' in result['data']
        assert 'coverage_score' in result['data']
