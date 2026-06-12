# Adapted from AERION project (Qdrant collection patterns)
"""Fly2Vec Qdrant 컬렉션 상수 정의."""

from qdrant_client.models import Distance

# 컬렉션명
COLLECTION_MISSION = 'fly2vec_mission_paths'
COLLECTION_SWARM = 'fly2vec_swarm_missions'

# 벡터 설정
VECTOR_DIM = 128
DISTANCE_METRIC = Distance.COSINE

# 검색 설정
DEFAULT_TOP_K = 5
ANOMALY_THRESHOLD = 0.5
MAX_VERIFICATION_LOOPS = 3
