FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-cache

RUN find /app/.venv -type f -name "libctranslate2*.so*" -print0 \
    | xargs -0 -r -I{} patchelf --clear-execstack "{}"

COPY app/ ./app/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7007

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7007"]
