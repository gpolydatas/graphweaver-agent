# graphweaver-agent/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps (cached unless base image changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl build-essential && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# LAYER 1: Dependencies only (cached unless pyproject.toml changes)
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync && uv pip install streamlit

# LAYER 2: Download ML model (cached unless Layer 1 changes)
RUN --mount=type=cache,target=/root/.cache/huggingface \
    uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# LAYER 3: Source code (rebuilds on any code change - but fast!)
COPY src/ ./src/
COPY mcp_servers/ ./mcp_servers/
COPY agent.py streamlit_app.py business_rules.yaml ./

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

CMD ["python", "agent.py"]
