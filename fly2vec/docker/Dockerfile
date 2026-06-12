FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first (saves ~2GB vs full CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric CPU-only
RUN pip install --no-cache-dir torch-geometric \
    --extra-index-url https://data.pyg.org/whl/torch-2.7.0+cpu.html 2>/dev/null || \
    pip install --no-cache-dir torch-geometric

# Copy pyproject.toml and install remaining deps with pip
COPY pyproject.toml ./
RUN pip install --no-cache-dir . 2>/dev/null || \
    pip install --no-cache-dir \
    "langgraph>=0.6.11" \
    "langchain>=0.3.27" \
    "langchain-core>=1.2.0" \
    "langchain-openai>=0.3.35" \
    "langchain-anthropic>=0.3.0" \
    "langchain-google-genai>=2.0.0" \
    "langchain-mcp-adapters>=0.2.0" \
    "a2a-sdk[grpc,http-server]>=0.3.22" \
    "fastmcp>=2.14.1,<3" \
    "gensim>=4.3.0" \
    "node2vec>=0.4.0" \
    "networkx>=3.2" \
    "qdrant-client>=1.12.0" \
    "scikit-learn>=1.4.0" \
    "hdbscan>=0.8.33" \
    "umap-learn>=0.5.5" \
    "numpy>=1.26.0" \
    "scipy>=1.13.0" \
    "python-dotenv>=1.2.0" \
    "structlog>=25.0.0" \
    "pydantic>=2.10.0" \
    "pydantic-settings>=2.0.0" \
    "httpx>=0.28.0" \
    "fastapi>=0.127.0" \
    "uvicorn>=0.40.0"

# Create fly2vec package structure so 'fly2vec.src.*' imports work
# Import paths: fly2vec.src.mcp_servers.*, fly2vec.src.agents.*, etc.
RUN mkdir -p /app/fly2vec
COPY src/ /app/fly2vec/src/
COPY data/ /app/fly2vec/data/
RUN touch /app/fly2vec/__init__.py
COPY .env.example ./.env.example

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "fly2vec.src.agents.mission_planner_agent"]
