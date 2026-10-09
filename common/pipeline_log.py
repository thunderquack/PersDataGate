from __future__ import annotations

import contextvars
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

_pipeline_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("pipeline_id", default=None)
_service: contextvars.ContextVar[str | None] = contextvars.ContextVar("pipeline_service", default=None)
_write_lock = threading.RLock()
_SERVICES = {"web", "pseudonymizer"}
_PIPELINE_ROUTES = {
    "web": {"/api/process"},
    "pseudonymizer": {"/anonymize", "/deanonymize"},
}
logger = logging.getLogger("uvicorn.error")


class PipelineLogError(RuntimeError):
    """The pipeline event could not be serialized or persisted."""


def parse_pipeline_id(value: str | None) -> str | None:
    """Return a normalized canonical UUID, or None for missing/malformed input."""
    if not isinstance(value, str):
        return None
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None
    return str(parsed)


def bind(pipeline_id: str, service: str | None = None) -> tuple[contextvars.Token, contextvars.Token]:
    normalized = parse_pipeline_id(pipeline_id)
    if normalized is None:
        raise ValueError("pipeline_id must be a UUID")
    id_token = _pipeline_id.set(normalized)
    service_token = _service.set(service if service in _SERVICES else None)
    return id_token, service_token


def reset(tokens: tuple[contextvars.Token, contextvars.Token]) -> None:
    id_token, service_token = tokens
    _service.reset(service_token)
    _pipeline_id.reset(id_token)


def current_id() -> str | None:
    return _pipeline_id.get()


def _log_dir() -> Path:
    configured = os.getenv("PIPELINE_LOG_DIR")
    return Path(configured) if configured else Path(__file__).resolve().parent.parent / "logs" / "pipelines"


def _lock_file(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_file(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


def event(stage: str, data: Any) -> None:
    """Append one JSONL record for the currently bound request; otherwise do nothing."""
    pipeline_id = current_id()
    if pipeline_id is None:
        return
    try:
        service = _service.get() or os.getenv("PIPELINE_SERVICE", "web")
        if service not in _SERVICES:
            raise ValueError("PIPELINE_SERVICE must be web or pseudonymizer")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "pipeline_id": pipeline_id,
            "service": service,
            "stage": str(stage),
            "data": data,
        }
        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        root = _log_dir()
        directory = root / pipeline_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            directory.chmod(0o700)
        except OSError:
            # Keep the log usable on filesystems that do not implement POSIX modes.
            pass
        target = directory / f"{service}.jsonl"
        # Keep both process-local threads and multiple ASGI workers from interleaving records.
        with _write_lock:
            descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            locked = False
            try:
                _lock_file(descriptor)
                locked = True
                written = os.write(descriptor, line)
                if written != len(line):
                    raise OSError("incomplete pipeline log append")
                os.fsync(descriptor)
            finally:
                if locked:
                    _unlock_file(descriptor)
                os.close(descriptor)
        logger.info("pipeline_id=%s stage=%s", pipeline_id, stage)
    except PipelineLogError:
        raise
    except Exception as exc:
        raise PipelineLogError("Could not persist pipeline event") from exc


async def _body_json(request: Request) -> Any:
    raw = await request.body()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw.decode("utf-8", errors="replace")


async def _response_json(response: Response) -> tuple[bytes, Any]:
    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        raw = getattr(response, "body", b"") or b""
    else:
        raw = b"".join([chunk async for chunk in body_iterator])
    try:
        return raw, json.loads(raw) if raw else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw, None


def _pipeline_error(pipeline_id: str, *, status_code: int, code: str) -> JSONResponse:
    from fastapi.responses import JSONResponse
    messages = {
        "pipeline_log_failed": "Не удалось записать этапы обработки в журнал.",
        "pipeline_failed": "Внутренняя ошибка обработки запроса.",
        "invalid_pipeline_id": "Передан некорректный идентификатор пайплайна.",
    }
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": messages.get(code, "Ошибка обработки запроса.")},
                 "pipeline_id": pipeline_id},
        headers={"X-Pipeline-ID": pipeline_id},
    )


def _reserve_pipeline_dir(pipeline_id: str) -> bool:
    root = _log_dir()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = root / pipeline_id
    try:
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        try:
            directory.chmod(0o700)
        except OSError:
            pass
        return True
    except FileExistsError:
        return False


def install_pipeline_logging(app: FastAPI, service: str) -> None:
    """Attach correlation IDs and full JSON request/response logging to pipeline routes."""
    from fastapi.responses import JSONResponse, Response
    if service not in _SERVICES:
        raise ValueError("service must be web or pseudonymizer")
    routes = _PIPELINE_ROUTES[service]

    @app.middleware("http")
    async def pipeline_logging(request: Request, call_next):
        if request.method != "POST" or request.url.path not in routes:
            return await call_next(request)

        supplied = request.headers.get("X-Pipeline-ID")
        parsed = parse_pipeline_id(supplied)
        invalid = supplied is not None and parsed is None
        pipeline_id = parsed or str(uuid4())
        if service == "web" or parsed is None:
            # The web entry point owns GUID uniqueness. A repeated client ID is
            # replaced so every submitted pipeline has its own log directory.
            try:
                while not _reserve_pipeline_dir(pipeline_id):
                    pipeline_id = str(uuid4())
            except Exception:
                return _pipeline_error(pipeline_id, status_code=503, code="pipeline_log_failed")
        elif not (_log_dir() / pipeline_id).is_dir():
            try:
                _reserve_pipeline_dir(pipeline_id)
            except Exception:
                return _pipeline_error(pipeline_id, status_code=503, code="pipeline_log_failed")
        tokens = bind(pipeline_id, service)
        try:
            if invalid:
                try:
                    event("request_error", {"code": "invalid_pipeline_id"})
                except Exception:
                    return _pipeline_error(pipeline_id, status_code=503, code="pipeline_log_failed")
                return _pipeline_error(pipeline_id, status_code=400, code="invalid_pipeline_id")

            try:
                body = await _body_json(request)
                event("request", {"method": request.method, "path": request.url.path, "body": body})
            except Exception:
                return _pipeline_error(pipeline_id, status_code=503, code="pipeline_log_failed")

            try:
                response = await call_next(request)
            except PipelineLogError:
                return _pipeline_error(pipeline_id, status_code=503, code="pipeline_log_failed")
            except Exception as exc:
                try:
                    event("error", {"type": type(exc).__name__})
                except Exception:
                    return _pipeline_error(pipeline_id, status_code=503, code="pipeline_log_failed")
                response = _pipeline_error(pipeline_id, status_code=500, code="pipeline_failed")

            try:
                raw, content = await _response_json(response)
                if isinstance(content, dict):
                    content["pipeline_id"] = pipeline_id
                    response = JSONResponse(
                        status_code=response.status_code,
                        content=content,
                        headers={k: v for k, v in response.headers.items()
                                 if k.lower() not in {"content-length", "content-type", "x-pipeline-id"}},
                        media_type=response.media_type or "application/json",
                    )
                else:
                    response = Response(
                        content=raw,
                        status_code=response.status_code,
                        headers={k: v for k, v in response.headers.items()
                                 if k.lower() not in {"content-length", "x-pipeline-id"}},
                        media_type=response.media_type,
                    )
                response.headers["X-Pipeline-ID"] = pipeline_id
                event("response", {"status_code": response.status_code, "body": content if content is not None else raw.decode("utf-8", "replace")})
            except Exception:
                return _pipeline_error(pipeline_id, status_code=503, code="pipeline_log_failed")
            return response
        finally:
            reset(tokens)
