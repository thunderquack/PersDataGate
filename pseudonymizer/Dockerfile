FROM python:3.11.12-slim

ARG DEVICE=cuda
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    BASE_DEVICE=${DEVICE}

WORKDIR /app

# Install the selected PyTorch build first; its matching CUDA runtime wheels
# are resolved from PyPI when needed, while the exact torch variant comes from
# the official device-specific index.
RUN if [ "${DEVICE}" = "cpu" ]; then \
        python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==2.7.1+cpu"; \
    elif [ "${DEVICE}" = "cuda" ]; then \
        python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple "torch==2.7.1+cu128"; \
    else \
        echo "Unsupported DEVICE: ${DEVICE}" >&2; exit 1; \
    fi

# Keep direct dependencies in individual layers so Docker can reuse the
# unchanged package installs when one dependency changes.
RUN python -m pip install --no-cache-dir "fastapi==0.115.12"
RUN python -m pip install --no-cache-dir "uvicorn[standard]==0.34.2"
RUN python -m pip install --no-cache-dir "pydantic==2.11.4"
RUN python -m pip install --no-cache-dir "transformers==4.51.3"
RUN python -m pip install --no-cache-dir "huggingface-hub==0.30.2"
RUN python -m pip install --no-cache-dir "tokenizers==0.21.1"
RUN python -m pip install --no-cache-dir "hf-xet==1.3.0"
RUN python -m pip install --no-cache-dir "pymorphy3==2.0.3"
RUN python -m pip install --no-cache-dir "peft==0.15.2"
RUN python -m pip install --no-cache-dir "gliner2[local]==2.0.0"

# Freeze the actual CPU/CUDA baseline to constrain optional runtime packages.
RUN python -m pip check \
    && python -m pip freeze --all > /app/base-constraints.txt

# The small, optional delta can be supplied at build time or mounted by Compose.
COPY pseudonymizer/requirements.txt /app/pseudonymizer/requirements.txt
COPY pseudonymizer /app/pseudonymizer
COPY common /app/common

EXPOSE 8000
CMD ["python", "-m", "pseudonymizer.bootstrap"]
