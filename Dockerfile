# ---- Base image ----
FROM python:3.12-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---- App setup ----
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install uv && uv pip install --system --no-cache -r requirements.txt

# Copy application source
COPY . .

# FastAPI / uvicorn port
EXPOSE 8000

# Health-check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

