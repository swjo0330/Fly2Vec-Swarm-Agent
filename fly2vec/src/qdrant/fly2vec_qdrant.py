# Adapted from AERION project (reasoning_agent_lg.py Qdrant patterns)
"""Fly2Vec Qdrant 벡터DB 매니저."""

import logging
import os
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams

from fly2vec.src.qdrant.collections import (
    COLLECTION_MISSION,
    COLLECTION_SWARM,
    DEFAULT_TOP_K,
    DISTANCE_METRIC,
    VECTOR_DIM,
)

logger = logging.getLogger(__name__)


class Fly2VecQdrantManager:
    """Fly2Vec 벡터DB 매니저.

    미션 경로 임베딩과 군집 분석 결과를 저장/검색합니다.
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url or os.getenv('QDRANT_URL', 'http://qdrant-fly2vec:6333')
        self._client: AsyncQdrantClient | None = None
        self._collections_ensured: set[str] = set()

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=self._url)
        return self._client

    async def ensure_collection(self, collection_name: str) -> None:
        """컬렉션이 없으면 생성."""
        if collection_name in self._collections_ensured:
            return
        try:
            collections = await self.client.get_collections()
            names = [c.name for c in collections.collections]
            if collection_name not in names:
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=VECTOR_DIM, distance=DISTANCE_METRIC),
                )
                logger.info(f'[qdrant] Created {collection_name} ({VECTOR_DIM}d, COSINE)')
            self._collections_ensured.add(collection_name)
        except Exception as e:
            logger.warning(f'[qdrant] Collection ensure failed: {e}')

    async def store_mission_vector(
        self,
        vector: list[float],
        payload: dict,
        collection: str = COLLECTION_MISSION,
    ) -> str:
        """미션 벡터 저장. 반환: point_id."""
        await self.ensure_collection(collection)
        point_id = uuid.uuid4().hex
        await self.client.upsert(
            collection_name=collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return point_id

    async def find_similar(
        self,
        vector: list[float],
        top_k: int = DEFAULT_TOP_K,
        collection: str = COLLECTION_MISSION,
    ) -> list[dict]:
        """유사 미션 검색."""
        await self.ensure_collection(collection)
        try:
            response = await self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            return [
                {'score': hit.score, 'payload': hit.payload}
                for hit in response.points
                if hit.payload
            ]
        except Exception as e:
            logger.warning(f'[qdrant] Search failed: {e}')
            return []

    async def recommend_routes(
        self,
        vector: list[float],
        top_k: int = DEFAULT_TOP_K,
        max_anomaly: float = 0.3,
        collection: str = COLLECTION_MISSION,
    ) -> list[dict]:
        """유사하면서 정상인 경로 추천 (anomaly < max_anomaly)."""
        await self.ensure_collection(collection)
        try:
            from qdrant_client.models import FieldCondition, Filter, Range

            response = await self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=top_k,
                with_payload=True,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key='anomaly_score',
                            range=Range(lte=max_anomaly),
                        ),
                    ],
                ),
            )
            return [
                {'score': hit.score, 'payload': hit.payload}
                for hit in response.points
                if hit.payload
            ]
        except Exception as e:
            logger.warning(f'[qdrant] Recommend failed: {e}')
            # fallback: 필터 없이 검색
            return await self.find_similar(vector, top_k, collection)

    async def scroll_vectors(
        self, collection: str = COLLECTION_MISSION, limit: int = 200,
    ) -> list[list[float]]:
        """Qdrant에서 벡터 샘플링 (centroid/IsolationForest 학습용)."""
        await self.ensure_collection(collection)
        try:
            result = await self.client.scroll(
                collection_name=collection, limit=limit, with_vectors=True,
            )
            return [p.vector for p in result[0] if p.vector]
        except Exception as e:
            logger.warning(f'[qdrant] scroll_vectors failed: {e}')
            return []

    # ── RouteRecommenderMCP 확장 (2026-05-25) ──

    async def get_by_id(
        self, point_id: str, collection: str = COLLECTION_MISSION,
    ) -> dict | None:
        """단일 포인트 payload 조회."""
        await self.ensure_collection(collection)
        try:
            result = await self.client.retrieve(
                collection_name=collection, ids=[point_id], with_payload=True,
            )
            return result[0].payload if result else None
        except Exception as e:
            logger.warning(f'[qdrant] get_by_id failed: {e}')
            return None

    async def recommend_multi_criteria(
        self,
        vector: list[float],
        mission_type: str | None = None,
        max_anomaly: float = 0.5,
        terrain_type: int | None = None,
        top_k: int = 20,
        collection: str = COLLECTION_MISSION,
    ) -> list[dict]:
        """다축 필터 적용 후 RAG 검색 (RouteRecommenderMCP용)."""
        await self.ensure_collection(collection)
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

            filters = [FieldCondition(key='anomaly_score', range=Range(lte=max_anomaly))]
            if mission_type:
                filters.append(FieldCondition(key='mission_type', match=MatchValue(value=mission_type)))
            if terrain_type is not None:
                filters.append(FieldCondition(
                    key='terrain_type',
                    range=Range(gte=terrain_type - 0.5, lte=terrain_type + 0.5),
                ))

            response = await self.client.query_points(
                collection_name=collection,
                query=vector,
                limit=top_k,
                with_payload=True,
                query_filter=Filter(must=filters),
            )
            return [
                {'score': hit.score, 'payload': hit.payload, 'id': str(hit.id)}
                for hit in response.points
                if hit.payload
            ]
        except Exception as e:
            logger.warning(f'[qdrant] recommend_multi_criteria failed: {e}')
            return await self.find_similar(vector, top_k, collection)

    async def close(self) -> None:
        """클라이언트 종료."""
        if self._client:
            await self._client.close()
            self._client = None
