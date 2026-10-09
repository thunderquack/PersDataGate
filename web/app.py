from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import (
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)
from pydantic import BaseModel, Field

MAX_INPUT_CHARS = 50_000
TIMEOUT_SECONDS = 120.0
OPENAI_INSTRUCTIONS = (
    "Answer the user's request using the supplied document and prompt. "
    "Preserve each pseudonymization marker you use exactly as written; do not translate, "
    "modify, or invent markers."
)


class ProcessRequest(BaseModel):
    document: str = Field(default="")
    prompt: str = Field(default="")


class ApiError(Exception):
    def __init__(self, stage: str, code: str, message: str, *, prepared: str = "", received: str = "",
                 sent_request: dict[str, Any] | None = None):
        self.stage, self.code, self.message = stage, code, message
        self.prepared, self.received = prepared, received
        self.sent_request = sent_request


class PseudonymizerClient:
    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(
            timeout=float(os.getenv("PSEUDONYMIZER_TIMEOUT_SECONDS", "600"))
        )
        self._owns_client = client is None

    async def anonymize(self, text: str) -> dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/anonymize", json={"text": text, "include_steps": True}
        )
        response.raise_for_status()
        return response.json()

    async def deanonymize(self, mapping_id: str, text: str) -> dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/deanonymize",
            json={"mapping_id": mapping_id, "text": text, "include_steps": True}
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()


def create_app(*, pseudonymizer: Any | None = None, openai_client: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.owns_pseudonymizer = pseudonymizer is None
        app.state.pseudonymizer = pseudonymizer or PseudonymizerClient(
            os.getenv("PSEUDONYMIZER_URL", "http://pseudonymizer:8000")
        )
        app.state.openai_client = openai_client
        app.state.owns_openai = openai_client is None
        if app.state.openai_client is None and os.getenv("OPENAI_API_KEY"):
            app.state.openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
        yield
        if app.state.owns_pseudonymizer:
            await app.state.pseudonymizer.close()
        if app.state.owns_openai and app.state.openai_client is not None:
            await app.state.openai_client.close()

    app = FastAPI(title="PersDataGate Web", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return JSONResponse(status_code=503 if exc.stage == "configuration" else 502, content={
            "error": {"stage": exc.stage, "code": exc.code, "message": exc.message},
            "prepared_text": exc.prepared, "received_text": exc.received,
            "sent_request": exc.sent_request,
            "steps": getattr(request.state, "processing_steps", {}),
            "warnings": getattr(request.state, "processing_warnings", []),
        })

    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), encoding="utf-8") as page:
            return page.read()

    @app.get("/health")
    async def health():
        return {"status": "ok", "openai_configured": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL")),
                "openai_model_configured": bool(os.getenv("OPENAI_MODEL")),
                "pseudonymizer_url_configured": bool(os.getenv("PSEUDONYMIZER_URL", "http://pseudonymizer:8000"))}

    @app.post("/api/process")
    async def process(body: ProcessRequest, request: Request):
        request.state.processing_steps = {}
        request.state.processing_warnings = []
        combined = f"[ПРОМПТ]\n{body.prompt}\n\n[ДОКУМЕНТ]\n{body.document}"
        if len(body.prompt) + len(body.document) > MAX_INPUT_CHARS or len(combined) > MAX_INPUT_CHARS:
            return JSONResponse(status_code=413, content={"error": {
                "stage": "validation", "code": "input_too_large",
                "message": f"Суммарный объём документа и промпта должен быть не больше {MAX_INPUT_CHARS} символов."},
                "prepared_text": "", "received_text": ""})
        model = os.getenv("OPENAI_MODEL")
        if not os.getenv("OPENAI_API_KEY") or not model or request.app.state.openai_client is None:
            raise ApiError("configuration", "openai_not_configured", "Для обработки задайте OPENAI_API_KEY и OPENAI_MODEL.")
        prepared = ""
        try:
            anonymized = await request.app.state.pseudonymizer.anonymize(combined)
            mapping_id, prepared = anonymized["mapping_id"], anonymized["text"]
            request.state.processing_steps.update(anonymized.get("steps", {}))
            request.state.processing_steps["anonymized_text"] = prepared
            request.state.processing_warnings = anonymized.get("warnings", [])
        except Exception:
            raise ApiError("anonymize", "pseudonymizer_unavailable",
                           "Не удалось псевдонимизировать текст. Запрос в OpenAI не отправлен.") from None
        sent_request = {"model": model, "instructions": OPENAI_INSTRUCTIONS,
                        "input": prepared, "store": False}
        try:
            result = await request.app.state.openai_client.responses.create(
                model=model, instructions=OPENAI_INSTRUCTIONS, input=prepared, store=False,
                timeout=TIMEOUT_SECONDS,
            )
            received = result.output_text
        except AuthenticationError:
            raise ApiError("openai", "openai_authentication_failed",
                           "OpenAI отклонил ключ API. Проверьте OPENAI_API_KEY.",
                           prepared=prepared, sent_request=sent_request) from None
        except NotFoundError:
            raise ApiError("openai", "openai_model_not_found",
                           "OpenAI не нашёл указанную модель. Проверьте OPENAI_MODEL.",
                           prepared=prepared, sent_request=sent_request) from None
        except BadRequestError:
            raise ApiError("openai", "openai_request_rejected",
                           "OpenAI отклонил запрос. Проверьте модель и параметры.",
                           prepared=prepared, sent_request=sent_request) from None
        except APITimeoutError:
            raise ApiError("openai", "openai_timeout", "OpenAI не ответил за отведённое время.",
                           prepared=prepared, sent_request=sent_request) from None
        except RateLimitError:
            raise ApiError("openai", "openai_rate_limited",
                           "Превышен лимит запросов или квота OpenAI. Проверьте проект API и повторите попытку позже.",
                           prepared=prepared, sent_request=sent_request) from None
        except Exception:
            raise ApiError("openai", "openai_request_failed", "Не удалось получить ответ OpenAI.",
                           prepared=prepared, sent_request=sent_request) from None
        try:
            restored = await request.app.state.pseudonymizer.deanonymize(mapping_id, received)
            restored_text = restored["text"]
            request.state.processing_steps.update(restored.get("steps", {}))
            warnings = list(dict.fromkeys(anonymized.get("warnings", []) + restored.get("warnings", [])))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ApiError("deanonymize", "mapping_not_found",
                               "Таблица замен истекла или потеряна после перезапуска сервиса.",
                               prepared=prepared, received=received, sent_request=sent_request) from None
            raise ApiError("deanonymize", "deanonymization_failed",
                           "Ответ получен, но восстановить исходные значения не удалось.",
                           prepared=prepared, received=received, sent_request=sent_request) from None
        except Exception:
            raise ApiError("deanonymize", "deanonymization_failed",
                           "Ответ получен, но восстановить исходные значения не удалось.",
                           prepared=prepared, received=received, sent_request=sent_request) from None
        return {"sent_request": sent_request, "received_text": received,
                "restored_text": restored_text, "warnings": warnings,
                "steps": request.state.processing_steps}

    return app


app = create_app()
