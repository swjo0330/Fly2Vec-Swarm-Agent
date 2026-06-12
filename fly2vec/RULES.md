# Fly2Vec 프로젝트 규칙

> AERION 파생 독립 프로젝트. AERION 코드를 직접 수정하지 않는다.

## 1. 독립성 원칙

- AERION 소스 코드를 **절대 직접 수정 금지** — 참조/복사만 허용
- 복사한 코드는 반드시 **Fly2Vec 네이밍**으로 리네이밍
- import 경로는 `fly2vec.*` 으로 통일 (AERION import 금지)
- 파일명/클래스명이 AERION과 **겹치면 안됨**

## 2. 네이밍 규칙

| AERION 원본 | Fly2Vec 명칭 | 비고 |
|------------|-------------|------|
| BaseGraphAgent | Fly2VecBaseAgent | LangGraph 기반 |
| BaseGraphState | Fly2VecState | 상태 스키마 |
| BaseA2AAgent | Fly2VecA2AAgent | A2A 인터페이스 |
| A2AClientManager | Fly2VecClientManager | A2A 클라이언트 |
| LangGraphAgentExecutor | Fly2VecExecutor | A2A 서버 |
| BaseMCPServer | Fly2VecMCPBase | MCP 서버 기반 |
| DroneToolsMCPServer | GraphEmbeddingMCP | 임베딩 MCP |
| — (신규) | SwarmAnalysisMCP | Fly2Vec-Swarm MCP |
| — (신규) | MissionPlannerAgent | 검증-수정 루프 에이전트 |

## 3. 프로젝트 구조

```
fly2vec/
├── RULES.md              ← 이 파일
├── README.md             ← 프로젝트 개요
├── docs/                 ← 설계 문서
├── src/
│   ├── __init__.py
│   ├── base/             ← LangGraph 기반 클래스
│   │   ├── fly2vec_base_agent.py
│   │   ├── fly2vec_state.py
│   │   └── fly2vec_a2a_interface.py
│   ├── a2a/              ← A2A 통신
│   │   ├── fly2vec_client.py
│   │   └── fly2vec_executor.py
│   ├── agents/           ← LangGraph 에이전트
│   │   ├── mission_planner_agent.py   (검증-수정 루프)
│   │   └── analysis_agent.py          (임베딩+이상탐지)
│   ├── mcp_servers/      ← MCP 도구 서버
│   │   ├── fly2vec_mcp_base.py
│   │   ├── graph_embedding_mcp.py     (embed/similar/anomaly)
│   │   └── swarm_analysis_mcp.py      (analyze_swarm/detect_conflict)
│   ├── embedding/        ← GNN 모델
│   │   ├── node2vec_encoder.py
│   │   ├── gcn_encoder.py
│   │   └── rgat_encoder.py
│   ├── qdrant/           ← 벡터DB 연동
│   │   ├── fly2vec_qdrant.py
│   │   └── collections.py
│   └── llm/              ← LLM 백엔드
│       └── fly2vec_llm_factory.py
├── tests/
│   ├── test_embedding.py
│   ├── test_mcp.py
│   └── test_agent.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
└── pyproject.toml
```

## 4. 기술 스택

- **Python**: 3.12
- **패키지 관리**: uv + pyproject.toml
- **Agent**: LangGraph + LangChain
- **통신**: A2A SDK (Google A2A) + FastMCP
- **GNN**: PyTorch + PyTorch Geometric
- **임베딩**: gensim (Node2Vec) + NetworkX
- **벡터DB**: Qdrant (128d, COSINE)
- **이상탐지**: scikit-learn, hdbscan, umap-learn
- **LLM**: GPT-4o-mini / Claude Haiku / Gemini 2.5 Flash

## 5. MCP 서버 도구 명세

### Graph Embedding MCP (graph_embedding_mcp.py)
- `embed_mission(pg_json)` → 128d 벡터
- `find_similar(vector, top_k)` → 유사 미션 리스트
- `anomaly_score(vector)` → 0~1 이상 점수

### Swarm Analysis MCP (swarm_analysis_mcp.py)
- `analyze_swarm(missions)` → 패턴/이상/추천
- `detect_conflict(missions)` → 충돌 지점 리스트
- `optimize_assignment(missions, drones)` → 최적 배치

## 6. 에이전트 검증-수정 루프

```
MissionPlannerAgent (LangGraph):
  parse_input → generate_pg → embed_and_check → [anomaly>0.5?]
    → YES: modify_pg → embed_and_check (max 3회)
    → NO: finalize → output
```

## 7. Docker 구성

- 프로젝트명: `fly2vec`
- 네트워크: `fly2vec-net`
- 서비스: fly2vec-agent, graph-embedding-mcp, swarm-analysis-mcp, qdrant-fly2vec
- 볼륨: fly2vec_qdrant_storage

## 8. 테스트 전략

- 프레임워크: `pytest` + `pytest-asyncio`
- 최소 커버리지: 80% (핵심 모듈: embedding, mcp_servers, qdrant)
- 테스트 구분:
  - `tests/unit/` — 단위 테스트 (Qdrant mock, LLM mock)
  - `tests/integration/` — 통합 테스트 (실제 Qdrant 컨테이너)
- 실행: `pytest tests/` 또는 `docker compose --profile test up`
- MCP 도구별 최소 1개 테스트 케이스 필수

## 9. 환경 변수/시크릿

- `.env`는 `.gitignore`에 포함 (커밋 금지)
- `.env.example`에 필수/선택 구분 명시
- 필수 환경변수:
  - `QDRANT_URL` (기본: http://qdrant-fly2vec:6333)
  - `LLM_BACKEND` (openai | claude | gemini, 기본: openai)
  - `OPENAI_API_KEY` 또는 해당 백엔드 키
- 선택 환경변수:
  - `MAPBOX_ACCESS_TOKEN` (경로 A 자연어 입력 시)
  - `FLY2VEC_LOG_LEVEL` (기본: INFO)
- fallback 순서: openai → gemini → claude

## 10. 데이터 경로

- 실험 데이터: `data/` (CSV, 모델 가중치)
- 실험 결과: `results/` (report, UMAP 이미지)
- 위 디렉토리는 `.gitignore`에 포함 (대용량)

## 11. 문서화/기록 규칙

- **모든 구현 단계는 `docs/`에 기록** (기말 보고서 작성 근거)
- 기록 형식: `docs/YYYY-MM-DD-<키워드>.md`
- 기록 에이전트 필수 포함 내용:
  - 작업 일시, 완료한 Step, 변경된 파일 목록
  - 설계안 대비 구현 차이점 (있으면)
  - 테스트 결과 요약
  - 다음 작업 TODO
- 세션 종료 시 반드시 `docs/` 문서 업데이트
- 보고서용 스크린샷, 실험 결과는 `results/`에 별도 보관

## 12. 커밋/배포

- git 커밋은 사용자 명시 요청 시에만
- AERION 메인 브랜치에 영향 주지 않음
- etc/proposal/fly2vec/ 경로 내에서만 작업
