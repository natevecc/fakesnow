FROM python:3.13-slim@sha256:a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra server
RUN /app/.venv/bin/python -c "import uvicorn, starlette"

FROM python:3.13-slim@sha256:a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d
LABEL org.opencontainers.image.source="https://github.com/natevecc/fakesnow"
LABEL org.opencontainers.image.description="fakesnow Snowflake-protocol HTTP server (natevecc fork with snowflake-integration patches)"
LABEL org.opencontainers.image.licenses="MIT"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
COPY fakesnow ./fakesnow
COPY scripts/healthcheck.py /app/scripts/healthcheck.py
RUN useradd --create-home --shell /bin/bash --uid 1001 fakesnow \
    && chown -R fakesnow:fakesnow /app
USER fakesnow
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "/app/scripts/healthcheck.py"]
ENTRYPOINT ["tini", "--", "python", "-m", "uvicorn", "fakesnow.server:app", "--host", "0.0.0.0", "--port", "8000"]
