# Adapted from AERION project (a2a_interface.py)
"""Fly2Vec A2A 통합 인터페이스."""

from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict

import logging

from a2a.types import AgentCard


logger = logging.getLogger(__name__)


class Fly2VecA2AOutput(TypedDict):
    """Fly2Vec A2A 표준 출력 형식."""

    agent_type: str
    status: Literal['working', 'completed', 'failed', 'input_required']
    text_content: str | None
    data_content: dict[str, Any] | None
    metadata: dict[str, Any]
    stream_event: bool
    final: bool
    error_message: str | None
    requires_approval: bool | None


class Fly2VecA2AAgent(ABC):
    """Fly2Vec A2A 통합 추상 클래스.

    모든 Fly2Vec 에이전트는 이 클래스를 상속하여 A2A 프로토콜 연계를 위한
    표준 인터페이스를 구현합니다.
    """

    def __init__(self) -> None:
        self.agent_type = self.__class__.__name__.replace('Agent', '')
        logger.info(f'Initializing Fly2Vec A2A agent: {self.agent_type}')

    @abstractmethod
    def get_agent_card(self) -> AgentCard:
        """에이전트 메타데이터 반환."""
        ...

    @abstractmethod
    async def execute_for_a2a(
        self, input_dict: dict[str, Any], config: dict[str, Any] | None = None
    ) -> Fly2VecA2AOutput:
        """A2A 표준 입출력으로 에이전트 실행."""
        ...

    @abstractmethod
    def format_stream_event(self, event: dict[str, Any]) -> Fly2VecA2AOutput | None:
        """스트리밍 이벤트 → A2A 표준 출력 변환."""
        ...

    @abstractmethod
    def extract_final_output(self, state: dict[str, Any]) -> Fly2VecA2AOutput:
        """최종 상태 → A2A 출력 추출."""
        ...

    def create_a2a_output(
        self,
        status: Literal['working', 'completed', 'failed', 'input_required'],
        text_content: str | None = None,
        data_content: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        stream_event: bool = False,
        final: bool = False,
        **kwargs: Any,
    ) -> Fly2VecA2AOutput:
        """A2A 표준 출력 생성 헬퍼."""
        return {
            'agent_type': self.agent_type,
            'status': status,
            'text_content': text_content,
            'data_content': data_content,
            'metadata': metadata or {},
            'stream_event': stream_event,
            'final': final,
            'error_message': kwargs.get('error_message'),
            'requires_approval': kwargs.get('requires_approval'),
        }

    def format_error(self, error: Exception, context: str | None = None) -> Fly2VecA2AOutput:
        """예외 → A2A 에러 출력."""
        error_message = f'{type(error).__name__}: {error!s}'
        if context:
            error_message = f'{context}: {error_message}'
        logger.error(f'Fly2Vec A2A Error: {error_message}')
        return self.create_a2a_output(
            status='failed',
            text_content=f'에러 발생: {error!s}',
            metadata={'error_type': type(error).__name__, 'context': context},
            final=True,
            error_message=error_message,
        )

    def is_completion_event(self, event: dict[str, Any]) -> bool:
        """완료 이벤트 여부 판별."""
        event_type = event.get('event', '')
        if event_type == 'on_chain_end':
            node_name = event.get('name', '')
            if node_name in ['__end__', 'final', 'complete']:
                return True
        return bool(event.get('metadata', {}).get('is_final', False))

    def extract_llm_content(self, event: dict[str, Any]) -> str | None:
        """LLM 스트리밍 텍스트 추출."""
        if event.get('event') != 'on_llm_stream':
            return None
        chunk = event.get('data', {}).get('chunk', {})
        if hasattr(chunk, 'content'):
            return chunk.content
        if isinstance(chunk, dict):
            return chunk.get('content', '')
        return None
