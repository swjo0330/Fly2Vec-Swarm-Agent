# Fly2Vec

**R-GAT 기반 군집 드론 임무 그래프의 공간적 의미 학습 및 지능형 임무 계획 분석 에이전트 체계**

조성원, 이서정 | AI개론 기말 프로젝트 | 2026

## Overview

드론 비행 경로(Planning Graph)를 그래프로 모델링하고, Node2Vec → GCN → R-GAT 3단계 진화적 접근으로 128d 벡터 임베딩을 생성한다. Qdrant RAG + LLM Agent 검증-수정 루프로 구조적 이상 탐지, 유사 미션 검색, 군집 충돌 사전 감지를 수행하는 **에이전트 시스템**.

## Quick Start

```bash
cp .env.example .env
# .env에 OPENAI_API_KEY 등 설정

cd docker
docker compose up -d

# 기존 데이터 벌크 로드
python -m fly2vec.src.data.loader
```

## Architecture

```
MissionPlannerAgent (LangGraph, 검증-수정 루프)
  │
  ├─ Graph Embedding MCP (port 8050)
  │    ├─ embed_mission(pg_json) → 128d
  │    ├─ find_similar(vector) → top-K
  │    └─ anomaly_score(vector) → 0~1
  │
  ├─ Swarm Analysis MCP (port 8051)
  │    ├─ analyze_swarm(missions) → 패턴/이상
  │    ├─ detect_conflict(missions) → 충돌 지점
  │    └─ optimize_assignment(missions, drones)
  │
  └─ Qdrant (128d, COSINE, port 6340)
       ├─ fly2vec_mission_paths (단일 경로)
       └─ fly2vec_swarm_missions (군집 분석)

서버 내 검증-수정 루프:
  PG 생성 → 임베딩 → anomaly > 0.5? → YES: 수정 (max 3회) → NO: 최종 PG
```

## Project Structure

```
fly2vec/
├── RULES.md              # 프로젝트 규칙 (AERION 독립성, 네이밍)
├── pyproject.toml        # Python 3.12, uv 패키지 관리
├── .env.example          # 환경 변수 템플릿
├── docker/               # Dockerfile + docker-compose.yml
├── docs/                 # 설계 문서
├── data/                 # 실 관제 데이터 (2,769건 미션, 47,266 WP)
├── results/              # 실험 결과
└── src/
    ├── base/             # LangGraph 기반 (Fly2VecBaseAgent, Fly2VecState)
    ├── a2a/              # A2A 통신 (Fly2VecClientManager, Fly2VecExecutor)
    ├── llm/              # LLM 팩토리 (openai/claude/gemini)
    ├── qdrant/           # 벡터DB (Fly2VecQdrantManager, collections)
    ├── mcp_servers/      # MCP 도구 서버 (GraphEmbeddingMCP, SwarmAnalysisMCP)
    ├── embedding/        # GNN 인코더 (Node2Vec, GCN, R-GAT)
    ├── agents/           # MissionPlannerAgent (검증-수정 루프)
    └── data/             # CSV → Qdrant 벌크 로더
```

## Tech Stack

| 범주 | 기술 |
|------|------|
| Agent | LangGraph, A2A SDK, FastMCP |
| GNN | PyTorch Geometric (GCNConv, GATConv) |
| Embedding | gensim (Node2Vec), NetworkX |
| Vector DB | Qdrant (128d, COSINE, HNSW) |
| LLM | GPT-4o-mini / Claude Haiku / Gemini 2.5 Flash |
| Infra | Docker, uv, Python 3.12 |

## Status (2026-05-02)

| Phase | 내용 | 상태 |
|-------|------|------|
| 1 | 인프라 (Docker, pyproject.toml) | Done |
| 2 | Base Layer (LangGraph, A2A, LLM) | Done |
| 3 | Qdrant + RAG | Done |
| 4 | Graph Embedding MCP | Done |
| 5 | Fly2Vec-Swarm MCP | Done |
| 6 | MissionPlannerAgent | Done |
| 7 | 기존 데이터 통합 (loader) | Done |
| 8 | 통합 테스트 + E2E | **TODO** |

## References

Node2Vec[KDD'16], GAT[ICLR'18], R-GCN[ESWC'18], MAGNNET[2025], HIPPO-MAT[2025], MAGAT[2021], D-GATAD[2025], GCBF+[MIT'24], AttentionSwarm[2025] 외 7편
