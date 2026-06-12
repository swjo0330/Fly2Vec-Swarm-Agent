"""Fly2Vec MissionPlanner A2A 서버.

자연어 입력을 A2A message/send로 수신하여 PG 생성 + 검증-수정 루프 실행.

실행:
  cd fly2vec
  PYTHONPATH=.. uv run python -m fly2vec.src.agents.run_server

테스트:
  curl -X POST http://localhost:8060/a2a/message/send \
    -H "Content-Type: application/json" \
    -d '{"text": "서울역에서 인천공항까지 50m 고도로 비행 계획"}'
"""

import asyncio
import json
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 로깅
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Fly2Vec MissionPlanner", version="1.0")

# 글로벌 Agent (lazy init)
_agent = None
_executor = None


async def get_agent():
    global _agent, _executor
    if _agent is None:
        from fly2vec.src.agents.mission_planner_agent import MissionPlannerAgent
        from fly2vec.src.a2a.fly2vec_executor import Fly2VecExecutor

        _agent = MissionPlannerAgent()
        await _agent.initialize()
        _executor = Fly2VecExecutor(_agent)
        logger.info('[server] Agent initialized')
    return _agent, _executor


@app.get('/')
async def health():
    return {'status': 'ok', 'agent': 'MissionPlannerAgent'}


@app.get('/.well-known/agent.json')
async def agent_card():
    """A2A Agent Card."""
    return {
        'name': 'Fly2Vec MissionPlanner',
        'description': 'R-GAT 기반 군집 드론 임무 그래프 분석 에이전트',
        'url': f'http://localhost:{os.getenv("AGENT_PORT", "8060")}',
        'version': '1.0',
        'capabilities': {
            'streaming': False,
            'tools': [
                'embed_mission', 'find_similar', 'anomaly_score', 'recommend_route',
                'analyze_swarm', 'detect_conflict', 'optimize_assignment',
                'recommend_safe_route', 'explain_recommendation', 'compare_routes', 'suggest_modification',
            ],
        },
    }


@app.post('/a2a/message/send')
async def message_send(request: Request):
    """A2A message/send — 자연어 → PG 생성 + 검증.

    입력: {"text": "서울역에서 인천공항까지 50m 비행"}
    또는: {"pg_json": {...}}  (PG 직접 입력)
    """
    body = await request.json()
    agent, executor = await get_agent()

    # 입력 파싱
    text = body.get('text', '')
    pg_json = body.get('pg_json', None)

    from langchain_core.messages import HumanMessage

    input_state = {
        'messages': [HumanMessage(content=text)] if text else [],
        'pg_json': pg_json or {},
        'retry_count': 0,
        'is_verified': False,
    }

    logger.info(f'[server] Request: {text[:50] if text else "PG direct"}...')

    # Agent 실행
    result = await executor.execute(input_state)

    # 응답 정리
    if result.get('status') == 'completed':
        data = result.get('data_content', {})
        response = {
            'status': 'completed',
            'pg_json': data.get('pg_json'),
            'anomaly_score': data.get('anomaly_score'),
            'is_verified': data.get('is_verified'),
            'similar_missions': data.get('similar_missions', [])[:3],
            'recommended_routes': data.get('recommended_routes', [])[:3],
            'swarm_analysis': data.get('swarm_analysis'),
            'retry_count': data.get('retry_count', 0),
        }
    else:
        response = {
            'status': 'failed',
            'error': result.get('error_message'),
        }

    logger.info(f'[server] Response: verified={response.get("is_verified")}, '
                f'anomaly={response.get("anomaly_score")}')
    return JSONResponse(content=response)


def main():
    port = int(os.getenv('AGENT_PORT', '8060'))
    logger.info(f'[server] Starting Fly2Vec MissionPlanner on port {port}')
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')


if __name__ == '__main__':
    main()
