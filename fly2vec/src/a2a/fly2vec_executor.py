# Adapted from AERION project (executor.py)
"""Fly2Vec A2A 서버 실행기."""

import logging
from dataclasses import dataclass, field
from typing import Any

from fly2vec.src.base.fly2vec_base_agent import Fly2VecBaseAgent
from fly2vec.src.base.fly2vec_a2a_interface import Fly2VecA2AAgent, Fly2VecA2AOutput

logger = logging.getLogger(__name__)


@dataclass
class Fly2VecExecutorConfig:
    """실행기 설정."""

    enable_streaming: bool = True
    enable_interrupt_handling: bool = False
    task_timeout_seconds: float = 300.0
    max_concurrent_tasks: int = 10


class Fly2VecExecutor:
    """LangGraph 에이전트를 A2A 메시지로 변환하는 실행기.

    Task 라이프사이클 관리 (제출 → 작업중 → 완료/실패).
    """

    def __init__(
        self,
        agent: Fly2VecBaseAgent | Fly2VecA2AAgent,
        config: Fly2VecExecutorConfig | None = None,
    ) -> None:
        self.agent = agent
        self.config = config or Fly2VecExecutorConfig()
        self._active_tasks: dict[str, Any] = {}

    async def execute(self, input_dict: dict[str, Any]) -> Fly2VecA2AOutput:
        """에이전트 실행 및 A2A 출력 반환."""
        try:
            if isinstance(self.agent, Fly2VecA2AAgent):
                return await self.agent.execute_for_a2a(input_dict)

            # BaseAgent인 경우 직접 그래프 실행
            result = await self.agent.graph.ainvoke(input_dict)
            return {
                'agent_type': self.agent.__class__.__name__,
                'status': 'completed',
                'text_content': str(result),
                'data_content': result if isinstance(result, dict) else None,
                'metadata': {},
                'stream_event': False,
                'final': True,
                'error_message': None,
                'requires_approval': None,
            }
        except Exception as e:
            logger.error(f'[Fly2VecExecutor] Execution failed: {e}')
            return {
                'agent_type': self.agent.__class__.__name__,
                'status': 'failed',
                'text_content': None,
                'data_content': None,
                'metadata': {'error_type': type(e).__name__},
                'stream_event': False,
                'final': True,
                'error_message': str(e),
                'requires_approval': None,
            }
