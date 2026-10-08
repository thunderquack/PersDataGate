FROM python:3.11.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

WORKDIR /app
COPY pseudonymizer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY pseudonymizer /app/pseudonymizer

EXPOSE 8000
CMD ["uvicorn", "pseudonymizer.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
