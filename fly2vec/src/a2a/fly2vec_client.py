# Adapted from AERION project (a2a_lg_client_utils.py)
"""Fly2Vec A2A 클라이언트 매니저."""

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.client import A2AClient

logger = logging.getLogger(__name__)


@dataclass
class Fly2VecClientResponse:
    """A2A 클라이언트 응답."""

    text: str = ''
    data: dict[str, Any] | None = None
    status: str = 'unknown'
    event_count: int = 0
    streaming_chunks: list[str] = field(default_factory=list)


class Fly2VecMessageEngine:
    """A2A 메시지 전송 엔진.

    메시지 전송, 이벤트 처리, Task 폴링, 재시도를 관리합니다.
    """

    def __init__(
        self,
        base_url: str,
        streaming: bool = False,
        max_wait: float = 120.0,
        poll_interval: float = 2.0,
    ) -> None:
        self.base_url = base_url
        self.streaming = streaming
        self.max_wait = max_wait
        self.poll_interval = poll_interval
        self._client: A2AClient | None = None
        self._http: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """HTTPX 클라이언트 및 Agent Card 로드."""
        self._http = httpx.AsyncClient(timeout=60.0)
        self._client = await A2AClient.get_client_from_agent_card_url(
            f'{self.base_url}/.well-known/agent.json',
            httpx_client=self._http,
        )

    async def send_text(self, text: str) -> Fly2VecClientResponse:
        """텍스트 메시지 전송."""
        if not self._client:
            await self.initialize()

        response = Fly2VecClientResponse()
        try:
            from a2a.types import MessageSendParams, TextPart, Part

            result = await self._client.send_message(
                MessageSendParams(message={'parts': [Part(root=TextPart(text=text))]})
            )

            # Task 결과 처리
            if hasattr(result, 'result'):
                task = result.result
                if hasattr(task, 'artifacts') and task.artifacts:
                    for artifact in task.artifacts:
                        for part in artifact.parts:
                            if hasattr(part.root, 'text'):
                                response.text += part.root.text
                                response.streaming_chunks.append(part.root.text)
                response.status = task.status.state if hasattr(task, 'status') else 'completed'
            response.event_count = 1
        except Exception as e:
            logger.error(f'[Fly2VecClient] Send failed: {e}')
            response.status = 'failed'
            response.text = str(e)

        return response

    async def close(self) -> None:
        """클라이언트 종료."""
        if self._http:
            await self._http.aclose()
            self._http = None


class Fly2VecClientManager:
    """Fly2Vec A2A 클라이언트 통합 관리.

    Usage:
        async with Fly2VecClientManager(base_url="http://localhost:8080") as mgr:
            resp = await mgr.send_text("분석 요청")
            print(resp.text)
    """

    def __init__(self, base_url: str, streaming: bool = False) -> None:
        self._engine = Fly2VecMessageEngine(base_url=base_url, streaming=streaming)

    async def __aenter__(self) -> 'Fly2VecClientManager':
        await self._engine.initialize()
        return self

    async def __aexit__(self, *args) -> None:
        await self._engine.close()

    async def send_text(self, text: str) -> Fly2VecClientResponse:
        """텍스트 메시지 전송."""
        return await self._engine.send_text(text)
