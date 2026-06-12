# Adapted from AERION project (base_graph_state.py)
"""Fly2Vec LangGraph 에이전트의 기본 상태 클래스 정의."""

from typing import Any

from langgraph.graph import MessagesState


class Fly2VecInputState(MessagesState):
    """Fly2Vec 워크플로우의 기본 입력 상태 스키마."""


class Fly2VecState(Fly2VecInputState):
    """Fly2Vec 워크플로우의 기본 상태 스키마.

    모든 Fly2Vec 에이전트의 상태는 이 클래스를 확장하여 사용합니다.
    """


class MissionPlannerState(Fly2VecState):
    """MissionPlannerAgent 전용 상태."""

    pg_json: dict[str, Any] | None  # Planning Graph JSON
    mission_vector: list[float] | None  # 128d 임베딩 벡터
    anomaly_score: float | None  # 0~1 이상 점수
    similar_missions: list[dict] | None  # 유사 미션 리스트
    swarm_analysis: dict[str, Any] | None  # 군집 분석 결과
    conflict_points: list[dict] | None  # 충돌 지점
    retry_count: int  # 검증-수정 루프 횟수
    is_verified: bool  # 검증 통과 여부
    error: str | None
    # RouteRecommenderMCP 확장 (2026-05-25)
    recommended_routes: list[dict] | None  # 다축 추천 결과 (score_breakdown 포함)
    recommendation_explanation: str | None  # 추천 이유 자연어
    route_comparison: dict | None  # 경로 비교 + 레이더 차트 데이터
    modification_suggestion: dict | None  # 위험 구간 수정 제안
