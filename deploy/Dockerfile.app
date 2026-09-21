FROM node:24.12.0-bookworm-slim AS web-build

WORKDIR /build
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY web/package.json web/package.json
RUN pnpm install --frozen-lockfile
COPY web web
RUN pnpm --dir web build

FROM python:3.12.12-slim-bookworm AS application

ENV PATH=/opt/gods-watching/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/gods-watching/.venv

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

WORKDIR /opt/gods-watching
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY alembic.ini ./
COPY assets assets
COPY server server
RUN uv sync --locked --no-dev
COPY --from=web-build /build/web/dist /opt/gods-watching/web/dist

EXPOSE 8000
CMD ["uvicorn", "gods_watching.api.production:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
