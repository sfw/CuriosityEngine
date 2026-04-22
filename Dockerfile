# Curiosity Engine — containerized so the whole runtime (including code_execution)
# sits inside an isolation boundary. The engine is CLI-driven; run it interactively
# via `docker compose run --rm engine ...` or the `./curiosity` wrapper script.

FROM python:3.13-slim-bookworm

# System deps needed by lxml / trafilatura and for the scientific Python stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for defense in depth.
ARG ENGINE_UID=1000
ARG ENGINE_GID=1000
RUN groupadd --gid ${ENGINE_GID} engine \
    && useradd --uid ${ENGINE_UID} --gid ${ENGINE_GID} --create-home --shell /bin/bash engine

WORKDIR /app

# Install deps first so they cache across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        numpy \
        scipy \
        pandas \
        scikit-learn \
        matplotlib

# Copy the rest of the project. .dockerignore scrubs venv / git / journal / caches.
COPY . .

RUN chown -R engine:engine /app
USER engine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg \
    HOME=/home/engine

# Journal + register live in /workspace (bind-mounted from host so state persists).
# Config lives in /home/engine/.CuriosityEngine (bind-mounted from host too).
WORKDIR /workspace

ENTRYPOINT ["python", "/app/curiosity_engine.py"]
