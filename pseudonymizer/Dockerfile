FROM python:3.11.12-slim

ARG DEVICE=cuda
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    BASE_DEVICE=${DEVICE}

WORKDIR /app

# Install the large, device-specific dependency set in a reusable image layer.
COPY pseudonymizer/dependencies.py /app/pseudonymizer/dependencies.py
COPY pseudonymizer/__init__.py /app/pseudonymizer/__init__.py
COPY pseudonymizer/requirements.${DEVICE}.txt /app/base-requirements.txt
RUN python -m pseudonymizer.dependencies --device "${DEVICE}" --requirements /app/base-requirements.txt \
    && python -m pip check \
    && python -m pip freeze --all > /app/base-constraints.txt

# The small, optional delta can be supplied at build time or mounted by Compose.
COPY pseudonymizer/requirements.txt /app/pseudonymizer/requirements.txt
COPY pseudonymizer /app/pseudonymizer
COPY common /app/common

EXPOSE 8000
CMD ["python", "-m", "pseudonymizer.bootstrap"]
