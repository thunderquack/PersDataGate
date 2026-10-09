from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_device
from .service import MAX_TEXT_LENGTH, MappingStore, anonymize_text, expires_iso, restore_text

MAX_RESPONSE_LENGTH = 3_000_000
GLINER_MODEL_ID = "urchade/gliner_multi-v2.1"
GLINER_REVISION = "443d26d654e0324125a96bebd8e796c14ff2efe6"

logger = logging.getLogger("uvicorn.error")


class AnonymizeRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)
    include_steps: bool = False


class DeanonymizeRequest(BaseModel):
    mapping_id: str
    text: str = Field(max_length=MAX_RESPONSE_LENGTH)
    include_steps: bool = False


class GLiNERDetector:
    def __init__(self, model: Any, inference_mode: Any):
        self.model = model
        self.tokenizer = model.data_processor.transformer_tokenizer
        self.lock = threading.Lock()
        self.inference_mode = inference_mode

    def predict_entities(self, *args: Any, **kwargs: Any) -> Any:
        with self.lock, self.inference_mode():
            return self.model.predict_entities(*args, **kwargs)


def create_app(detector: Any | None = None, store: MappingStore | None = None,
               pipeline: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        device = get_device()
        logger.info("Initializing GLiNER on %s", device)
        if detector is not None:
            app.state.detector = detector
            app.state.pipeline = pipeline
        else:
            try:
                import torch
                from gliner import GLiNER
                from huggingface_hub import snapshot_download

                if device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is false")
                cache_dir = os.getenv("HF_HOME", "/models/huggingface")
                model_snapshot = snapshot_download(
                    repo_id=GLINER_MODEL_ID,
                    revision=GLINER_REVISION,
                    cache_dir=cache_dir,
                    allow_patterns=["gliner_config.json", "model.safetensors"],
                )
                model = GLiNER.from_pretrained(
                    model_snapshot,
                    max_length=384,
                    cache_dir=cache_dir,
                )
                model.to(device)
                model.eval()
                app.state.detector = GLiNERDetector(model, torch.inference_mode)
                if pipeline is None:
                    from .local_llm import QwenGenerator
                    from .persons import PersonPipeline

                    generator = QwenGenerator(device=device, cache_dir=cache_dir)
                    app.state.pipeline = PersonPipeline(generator)
                else:
                    app.state.pipeline = pipeline
            except Exception as exc:
                logger.error("Local model startup failed (%s)", type(exc).__name__)
                raise RuntimeError(f"Could not initialize local models on {device}: {exc}") from exc
        app.state.store = store or MappingStore()
        app.state.inference_lock = threading.Lock()
        app.state.device = device
        app.state.ready = True
        yield
        app.state.ready = False
        close = getattr(app.state.store, "close", None)
        if close:
            close()

    app = FastAPI(title="PersDataGate Pseudonymizer", lifespan=lifespan)

    @app.get("/health/ready")
    async def readiness() -> dict[str, bool | str]:
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="model_not_ready")
        return {"ready": True, "device": app.state.device}

    @app.post("/anonymize")
    def anonymize(request: AnonymizeRequest) -> dict[str, Any]:
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="model_not_ready")
        try:
            steps: dict[str, Any] | None = {} if request.include_steps else None
            with app.state.inference_lock:
                safe_text, mapping = anonymize_text(
                    request.text, app.state.detector, getattr(app.state, "pipeline", None),
                    steps=steps,
                )
        except Exception as exc:
            logger.error("Anonymization failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="anonymization_failed") from exc
        mapping_id, expiration = app.state.store.create(mapping)
        response = {
            "mapping_id": mapping_id,
            "text": safe_text,
            "expires_at": expires_iso(expiration),
            "warnings": list(mapping.warnings),
        }
        if steps is not None:
            response["steps"] = steps
        return response

    @app.post("/deanonymize")
    def deanonymize(request: DeanonymizeRequest) -> dict[str, Any]:
        mapping = app.state.store.get(request.mapping_id)
        if mapping is None:
            raise HTTPException(status_code=404, detail="mapping_not_found")
        try:
            steps: dict[str, Any] | None = {} if request.include_steps else None
            with app.state.inference_lock:
                text, warnings = restore_text(
                    request.text, mapping, getattr(app.state, "pipeline", None),
                    steps=steps,
                )
        except Exception as exc:
            logger.error("De-anonymization failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="deanonymization_failed") from exc
        response = {"text": text, "warnings": warnings}
        if steps is not None:
            response["steps"] = steps
        return response

    return app


app = create_app()
