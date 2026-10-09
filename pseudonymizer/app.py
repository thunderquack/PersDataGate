from __future__ import annotations

import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.pipeline_log import PipelineLogError, event, install_pipeline_logging
from .config import get_device
from .local_llm import ContextLimitError
from .service import MAX_TEXT_LENGTH, MappingStore, anonymize_text, expires_iso, restore_text

MAX_RESPONSE_LENGTH = 3_000_000
GLINER_MODEL_ID = "fastino/gliner2.5-multi-v1"
GLINER_REVISION = "cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3"

logger = logging.getLogger("uvicorn.error")


class AnonymizeRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)
    include_steps: bool = False


class DeanonymizeRequest(BaseModel):
    mapping_id: str
    text: str = Field(max_length=MAX_RESPONSE_LENGTH)
    include_steps: bool = False


class GLiNER2Detector:
    def __init__(self, model: Any, inference_mode: Any,
                 model_id: str = GLINER_MODEL_ID, revision: str = GLINER_REVISION):
        self.model = model
        self.tokenizer = model.processor.tokenizer
        self.lock = threading.Lock()
        self.inference_mode = inference_mode
        self.model_id = model_id
        self.model_revision = revision

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        with self.lock, self.inference_mode():
            response = self.model.extract_entities(
                text,
                labels,
                threshold=threshold,
                include_confidence=True,
                include_spans=True,
            )
        if not isinstance(response, dict) or not isinstance(response.get("entities"), dict):
            raise ValueError("GLiNER2 returned an invalid entity response")

        predictions: list[dict[str, Any]] = []
        entities = response["entities"]
        for label in labels:
            rows = entities.get(label, [])
            if not isinstance(rows, list):
                raise ValueError("GLiNER2 returned an invalid entity list")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("GLiNER2 returned a malformed entity")
                value = row.get("text")
                start, end = row.get("start"), row.get("end")
                score = row.get("confidence")
                try:
                    normalized_score = float(score)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError("GLiNER2 returned a malformed entity") from None
                if (
                    not isinstance(value, str)
                    or not value
                    or not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or start < 0
                    or end <= start
                    or end > len(text)
                    or text[start:end] != value
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not math.isfinite(normalized_score)
                    or not 0.0 <= normalized_score <= 1.0
                ):
                    raise ValueError("GLiNER2 returned a malformed entity")
                predictions.append({
                    "text": value,
                    "label": label,
                    "start": start,
                    "end": end,
                    "score": normalized_score,
                })
        return predictions


def create_app(detector: Any | None = None, store: MappingStore | None = None,
               pipeline: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        device = get_device()
        logger.info("Initializing GLiNER2.5 on %s", device)
        if detector is not None:
            app.state.detector = detector
            app.state.pipeline = pipeline
        else:
            try:
                import torch
                from gliner2 import AutoExtractor
                from huggingface_hub import snapshot_download

                if device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA GPU is required but torch.cuda.is_available() is false")
                cache_dir = os.getenv("HF_HOME", "/models/huggingface")
                model_snapshot = snapshot_download(
                    repo_id=GLINER_MODEL_ID,
                    revision=GLINER_REVISION,
                    cache_dir=cache_dir,
                    allow_patterns=[
                        "config.json",
                        "encoder_config/config.json",
                        "model.safetensors",
                        "tokenizer.json",
                        "tokenizer_config.json",
                    ],
                )
                model = AutoExtractor.from_pretrained(
                    model_snapshot,
                    map_location=device,
                )
                model.eval()
                app.state.detector = GLiNER2Detector(
                    model, torch.inference_mode, GLINER_MODEL_ID, GLINER_REVISION
                )
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
    install_pipeline_logging(app, service="pseudonymizer")

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
        except PipelineLogError:
            raise
        except ContextLimitError as exc:
            event("anonymization.context_limit", {
                "input_tokens": exc.input_tokens,
                "maximum_input_tokens": exc.maximum_input_tokens,
                "context_window": exc.context_window,
            })
            raise HTTPException(status_code=413, detail={
                "code": "local_model_context_exceeded",
                "input_tokens": exc.input_tokens,
                "maximum_input_tokens": exc.maximum_input_tokens,
                "context_window": exc.context_window,
            }) from None
        except Exception as exc:
            event("anonymization.error", {"type": type(exc).__name__})
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
        except PipelineLogError:
            raise
        except Exception as exc:
            event("deanonymization.error", {"type": type(exc).__name__})
            logger.error("De-anonymization failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="deanonymization_failed") from exc
        response = {"text": text, "warnings": warnings}
        if steps is not None:
            response["steps"] = steps
        return response

    return app


app = create_app()
