# syntax=docker/dockerfile:1.7
FROM python:3.11.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

WORKDIR /app
ARG REQUIREMENTS_LOCK=pseudonymizer/requirements.lock.txt
COPY ${REQUIREMENTS_LOCK} /app/requirements.lock.txt
RUN --network=none --mount=type=bind,source=.deps/wheels,target=/wheels,readonly \
    pip install --no-cache-dir --no-index --find-links=/wheels \
    --only-binary=:all: --require-hashes -r /app/requirements.lock.txt
COPY pseudonymizer /app/pseudonymizer

EXPOSE 8000
CMD ["uvicorn", "pseudonymizer.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
