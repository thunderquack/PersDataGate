FROM python:3.11.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    PIP_CACHE_DIR=/pip-cache

WORKDIR /app
COPY pseudonymizer/requirements.cpu.txt /app/pseudonymizer/requirements.cpu.txt
COPY pseudonymizer/requirements.cuda.txt /app/pseudonymizer/requirements.cuda.txt
COPY pseudonymizer /app/pseudonymizer

EXPOSE 8000
CMD ["python", "-m", "pseudonymizer.bootstrap"]
