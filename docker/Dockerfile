# syntax=docker/dockerfile:1

# Build:  docker build -f docker/Dockerfile -t easyrag .
# Extras: docker build -f docker/Dockerfile --build-arg EXTRAS="anthropic pdf" -t easyrag .
#         (any of: openai anthropic huggingface pdf all; empty = offline defaults only)

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app
ARG EXTRAS=""

# Dependencies first, from the lockfile only, so code changes don't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project \
        $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
        $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)


FROM python:${PYTHON_VERSION}-slim

RUN useradd --create-home --uid 1000 easyrag

COPY --from=builder /opt/venv /opt/venv
COPY configs /app/configs

# The index lives on the mounted data volume so it survives the container. Model downloads
# (huggingface extra) go to ~/.cache, which compose mounts as a named volume.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    EASYRAG_PIPELINE_INDEX_DIR=/app/data/.rag_index

WORKDIR /app
RUN mkdir -p /app/data /home/easyrag/.cache \
    && chown easyrag:easyrag /app/data /home/easyrag/.cache
USER easyrag
VOLUME ["/app/data"]

ENTRYPOINT ["easyrag"]
CMD ["--help"]
