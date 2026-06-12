# Adapted from AERION project (base_graph_agent.py)
"""Fly2Vec LangGraph 에이전트의 추상 기본 클래스."""

from typing import Any, ClassVar

try:
    from typing import Never
except ImportError:
    from typing_extensions import Never

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy


class Fly2VecBaseAgent:
    """Fly2Vec LangGraph 에이전트의 추상 기본 클래스.

    주요 기능:
    - StateGraph 구축 및 컴파일
    - 노드/엣지 초기화 추상 메서드
    - MCP 도구 통합
    - 상태 검증 및 스키마 관리
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {'DEFAULT': 'default'}

    def __init__(
        self,
        model: BaseChatModel | None = None,
        state_schema: type | None = None,
        config_schema: type | None = None,
        input_state: type | None = None,
        output_state: type | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        tools: list[BaseTool] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.checkpointer = checkpointer
        self.store = store
        self.tools = tools or []
        self.mcp_servers = mcp_servers
        self.state_schema = state_schema
        self.config_schema = config_schema
        self.input_state = input_state
        self.output_state = output_state

        self.max_retry_attempts = kwargs.get('max_retry_attempts', 2)
        self.agent_name = kwargs.get('agent_name')
        self.is_debug = kwargs.get('is_debug', False)
        self.lazy_init = kwargs.get('lazy_init', False)
        self._graph: CompiledStateGraph | None = None

        self.retry_policy = (
            RetryPolicy(max_attempts=self.max_retry_attempts)
            if self.max_retry_attempts > 0
            else None
        )
        if not self.lazy_init:
            self.graph = self.build_graph()

    async def mcp_tools_init(self) -> list[BaseTool] | None:
        """MCP 서버 연결 및 도구 로드."""
        import logging
        _logger = logging.getLogger(__name__)
        connections = {}
        for server in self.mcp_servers:
            headers = server.get('headers', {})
            # 빈 Authorization 헤더가 있으면 해당 서버만 스킵
            auth = headers.get('Authorization', '')
            if auth and auth.strip() in ('Bearer', 'Bearer '):
                _logger.warning(f'[mcp] Skipping {server["name"]}: empty auth token')
                continue
            conn = {
                'url': server['url'],
                'transport': server.get('transport', 'streamable_http'),
            }
            if headers:
                conn['headers'] = headers
            connections[server['name']] = conn
        if not connections:
            _logger.warning('[mcp] No MCP servers configured')
            return self.tools
        try:
            self.mcp_client = MultiServerMCPClient(connections=connections)
            mcp_tools: list[BaseTool] = await self.mcp_client.get_tools()
            self.tools = (self.tools or []) + mcp_tools
        except Exception as e:
            _logger.warning(f'[mcp] Connection failed (graceful): {e}')
            self.tools = self.tools or []
        return self.tools

    async def initialize(self) -> 'Fly2VecBaseAgent':
        """비동기 초기화."""
        if self.mcp_servers:
            await self.mcp_tools_init()
        if self.lazy_init and self._graph is None:
            self._graph = self.build_graph()
            self.graph = self._graph
        return self

    @classmethod
    async def create(cls, **kwargs) -> 'Fly2VecBaseAgent':
        """비동기 팩토리 메서드."""
        instance = cls(**kwargs, lazy_init=True)
        await instance.initialize()
        return instance

    @property
    def graph(self) -> CompiledStateGraph | None:
        if self.lazy_init and self._graph is None:
            raise RuntimeError('Agent not initialized. Call await agent.initialize() first.')
        return self._graph if self.lazy_init else self._internal_graph

    @graph.setter
    def graph(self, value: CompiledStateGraph | None) -> None:
        if self.lazy_init:
            self._graph = value
        else:
            self._internal_graph = value

    def init_nodes(self, graph: StateGraph) -> Never:
        """서브클래스에서 구현: 그래프 노드 등록."""
        raise NotImplementedError('Subclasses must implement init_nodes')

    def init_edges(self, graph: StateGraph) -> Never:
        """서브클래스에서 구현: 그래프 엣지 정의."""
        raise NotImplementedError('Subclasses must implement init_edges')

    def build_graph(self) -> CompiledStateGraph:
        """StateGraph 구축 및 컴파일."""
        _graph = StateGraph(
            state_schema=self.state_schema,
            context_schema=self.config_schema,
            input_schema=self.input_state,
            output_schema=self.output_state,
        )
        self.init_nodes(_graph)
        self.init_edges(_graph)
        return _graph.compile(
            checkpointer=self.checkpointer,
            store=self.store,
            debug=self.is_debug,
            name=f'{self.agent_name or self.__class__.__name__}',
        )
