FROM persdatagate-pseudonymizer-deps:py311-cu128-v1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

WORKDIR /app
COPY pseudonymizer /app/pseudonymizer

EXPOSE 8000
CMD ["uvicorn", "pseudonymizer.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
