# Adapted from AERION project (base_mcp_server.py)
"""Fly2Vec MCP 서버 베이스 클래스."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


class Fly2VecResponse(BaseModel):
    """Fly2Vec MCP 표준 응답."""

    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    success: bool = Field(True, description='성공 여부')
    query: str = Field(..., description='원본 쿼리')
    data: Any | None = Field(None, description='응답 데이터')


class Fly2VecErrorResponse(BaseModel):
    """Fly2Vec MCP 에러 응답."""

    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    success: bool = Field(False, description='항상 False')
    query: str = Field(..., description='원본 쿼리')
    error: str = Field(..., description='에러 메시지')
    func_name: str | None = Field(None, description='에러 발생 함수명')


class Fly2VecMCPBase(ABC):
    """Fly2Vec MCP 서버 베이스 클래스.

    하위 클래스에서 _initialize_clients()와 _register_tools()를 구현합니다.
    """

    MCP_PATH = '/mcp'

    def __init__(
        self,
        server_name: str,
        port: int,
        host: str = '0.0.0.0',
        debug: bool = False,
        transport: Literal['streamable-http', 'stdio'] = 'streamable-http',
        server_instructions: str = '',
    ):
        self.server_name = server_name
        self.host = host
        self.port = port
        self.debug = debug
        self.transport = transport
        self.server_instructions = server_instructions

        # FastMCP 인스턴스 생성
        self.mcp = FastMCP(
            name=self.server_name,
            instructions=self.server_instructions,
        )

        # 초기화
        self._initialize_clients()
        self._register_tools()

    @abstractmethod
    def _initialize_clients(self) -> None:
        """외부 클라이언트 초기화 (Qdrant, 모델 등)."""
        ...

    @abstractmethod
    def _register_tools(self) -> None:
        """MCP 도구 등록 (@self.mcp.tool() 사용)."""
        ...

    def create_response(
        self, success: bool, query: str, data: Any = None, **kwargs
    ) -> dict:
        """표준 응답 생성."""
        if success:
            return Fly2VecResponse(success=True, query=query, data=data, **kwargs).model_dump()
        return Fly2VecErrorResponse(
            success=False, query=query, error=str(data), **kwargs
        ).model_dump()

    def create_error(self, error: str, query: str, func_name: str) -> dict:
        """에러 응답 생성."""
        return Fly2VecErrorResponse(
            success=False, query=query, error=error, func_name=func_name
        ).model_dump()

    def run(self) -> None:
        """MCP 서버 실행."""
        self.mcp.run(transport=self.transport, host=self.host, port=self.port)
