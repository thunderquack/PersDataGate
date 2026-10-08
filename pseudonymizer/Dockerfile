FROM python:3.11.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    PIP_CACHE_DIR=/pip-cache

WORKDIR /app
COPY pseudonymizer/requirements.txt /app/pseudonymizer/requirements.txt
COPY pseudonymizer /app/pseudonymizer

EXPOSE 8000
CMD ["python", "-m", "pseudonymizer.bootstrap"]
