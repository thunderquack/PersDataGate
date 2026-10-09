from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.pipeline_log import (
    PipelineLogError,
    bind,
    current_id,
    event,
    install_pipeline_logging,
    parse_pipeline_id,
    reset,
)


def _records(directory, pipeline_id, service):
    path = directory / pipeline_id / f"{service}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parse_and_context_binding_are_canonical_and_resettable():
    value = str(uuid4())
    assert parse_pipeline_id(value.upper()) == value
    assert parse_pipeline_id("bad-guid") is None
    assert parse_pipeline_id(None) is None
    assert current_id() is None
    tokens = bind(value, "web")
    try:
        assert current_id() == value
    finally:
        reset(tokens)
    assert current_id() is None


def test_event_without_bound_context_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path))
    event("startup", {"status": "ready"})
    assert list(tmp_path.iterdir()) == []


def test_event_serialization_failure_has_distinct_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path))
    tokens = bind(str(uuid4()), "pseudonymizer")
    try:
        try:
            event("model.result", {"not_json": object()})
        except PipelineLogError as exc:
            assert isinstance(exc.__cause__, TypeError)
        else:
            raise AssertionError("event should report serialization failure")
    finally:
        reset(tokens)


def test_concurrent_events_are_complete_jsonl_records(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path))
    pipeline_id = str(uuid4())

    def write(index):
        tokens = bind(pipeline_id, "web")
        try:
            event("step", {"index": index, "value": "персональные данные"})
        finally:
            reset(tokens)

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(write, range(120)))

    records = _records(tmp_path, pipeline_id, "web")
    assert len(records) == 120
    assert {record["data"]["index"] for record in records} == set(range(120))
    assert all(record["pipeline_id"] == pipeline_id for record in records)
    assert all(record["service"] == "web" for record in records)
    assert all("." in record["timestamp"] and record["timestamp"].endswith("Z") for record in records)


def test_middleware_logs_full_json_and_adds_pipeline_id_even_for_422(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path))
    app = FastAPI()
    install_pipeline_logging(app, "web")

    @app.post("/api/process")
    async def process(body: dict):
        return {"echo": body["document"]}

    with TestClient(app) as client:
        response = client.post("/api/process", json={"document": "Иван Петров"})
        invalid = client.post("/api/process", json=[])

    pipeline_id = response.headers["X-Pipeline-ID"]
    assert response.json() == {"echo": "Иван Петров", "pipeline_id": pipeline_id}
    assert invalid.status_code == 422
    assert invalid.json()["pipeline_id"] == invalid.headers["X-Pipeline-ID"]
    records = _records(tmp_path, pipeline_id, "web")
    assert [record["stage"] for record in records] == ["request", "response"]
    assert records[0]["data"]["body"] == {"document": "Иван Петров"}
    assert records[1]["data"]["body"]["echo"] == "Иван Петров"


def test_supplied_id_is_used_and_invalid_id_returns_new_id(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path))
    app = FastAPI()
    install_pipeline_logging(app, "pseudonymizer")

    @app.post("/anonymize")
    async def anonymize(body: dict):
        return {"text": body["text"]}

    with TestClient(app) as client:
        expected = str(uuid4())
        response = client.post("/anonymize", json={"text": "secret"}, headers={"X-Pipeline-ID": expected})
        invalid = client.post("/anonymize", json={"text": "secret"}, headers={"X-Pipeline-ID": "../bad"})

    assert response.headers["X-Pipeline-ID"] == expected
    assert response.json()["pipeline_id"] == expected
    assert invalid.status_code == 400
    assert invalid.json()["pipeline_id"] == invalid.headers["X-Pipeline-ID"]
    assert invalid.json()["pipeline_id"] != "../bad"
    assert _records(tmp_path, expected, "pseudonymizer")[0]["data"]["body"] == {"text": "secret"}


def test_web_replaces_reused_client_id_to_keep_pipeline_ids_unique(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path))
    app = FastAPI()
    install_pipeline_logging(app, "web")

    @app.post("/api/process")
    async def process():
        return {"ok": True}

    supplied = str(uuid4())
    with TestClient(app) as client:
        first = client.post("/api/process", json={}, headers={"X-Pipeline-ID": supplied})
        second = client.post("/api/process", json={}, headers={"X-Pipeline-ID": supplied})

    first_id = first.headers["X-Pipeline-ID"]
    second_id = second.headers["X-Pipeline-ID"]
    assert first_id != second_id
    assert first.json()["pipeline_id"] == first_id
    assert second.json()["pipeline_id"] == second_id
    assert len(_records(tmp_path, first_id, "web")) == 2
    assert len(_records(tmp_path, second_id, "web")) == 2


def test_unexpected_endpoint_error_returns_safe_500_and_is_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path))
    app = FastAPI()
    install_pipeline_logging(app, "web")

    @app.post("/api/process")
    async def process():
        raise RuntimeError("private error detail")

    with TestClient(app) as client:
        response = client.post("/api/process", json={})

    pipeline_id = response.headers["X-Pipeline-ID"]
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "pipeline_failed"
    assert "private error detail" not in response.text
    assert response.json()["pipeline_id"] == pipeline_id
    records = _records(tmp_path, pipeline_id, "web")
    assert [record["stage"] for record in records] == ["request", "error", "response"]
    assert records[1]["data"] == {"type": "RuntimeError"}
    assert records[-1]["data"]["status_code"] == 500


def test_pipeline_log_write_failure_is_visible(tmp_path, monkeypatch):
    app = FastAPI()
    install_pipeline_logging(app, "web")

    @app.post("/api/process")
    async def process():
        return {"ok": True}

    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path / "missing-parent" / "file"))
    # A file at the configured root makes creation of the pipeline directory fail.
    configured = tmp_path / "not-a-directory"
    configured.write_text("", encoding="utf-8")
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(configured))
    with TestClient(app) as client:
        response = client.post("/api/process", json={})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "pipeline_log_failed"
    assert response.json()["pipeline_id"] == response.headers["X-Pipeline-ID"]


def test_async_context_does_not_bleed_between_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path))
    app = FastAPI()
    install_pipeline_logging(app, "web")

    @app.post("/api/process")
    async def process(body: dict):
        await asyncio.sleep(body["delay"])
        event("handler", {"value": body["value"]})
        return {"value": body["value"]}

    async def run_requests():
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await asyncio.gather(
                client.post("/api/process", json={"value": "first", "delay": 0.02}),
                client.post("/api/process", json={"value": "second", "delay": 0}),
            )

    first, second = asyncio.run(run_requests())
    for response, value in ((first, "first"), (second, "second")):
        pipeline_id = response.headers["X-Pipeline-ID"]
        records = _records(tmp_path, pipeline_id, "web")
        handler = next(record for record in records if record["stage"] == "handler")
        assert handler["data"]["value"] == value
        assert all(record["pipeline_id"] == pipeline_id for record in records)
