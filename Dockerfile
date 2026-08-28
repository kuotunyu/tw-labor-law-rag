# Shared image for both the `api` and `ui` docker-compose services — they
# differ only in the CMD/command override. On Linux, PyPI's default `torch`
# wheel already bundles CUDA (unlike Windows, which needs the separate cu128
# index configured in pyproject.toml for local dev), so no extra index is
# needed here as long as the host has nvidia-container-toolkit installed.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
# Not installed as a package (see pyproject.toml [tool.uv] package = false);
# src/ is put on PYTHONPATH instead, same as local dev.
RUN uv sync --frozen --no-dev

COPY src ./src
COPY scripts ./scripts
COPY ui ./ui

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    API_URL="http://127.0.0.1:8000"

EXPOSE 7860

CMD ["python", "scripts/run_space.py"]
