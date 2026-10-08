from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI
from fastapi.testclient import TestClient

from pseudonymizer.app import create_app as create_pseudonymizer_app
from web.app import OPENAI_INSTRUCTIONS, TIMEOUT_SECONDS, create_app
from web.app import PseudonymizerClient


class FakePseudonymizer:
    def __init__(self, fail_anonymize=False, fail_restore=False):
        self.fail_anonymize = fail_anonymize
        self.fail_restore = fail_restore
        self.calls = []

    async def anonymize(self, text):
        self.calls.append(("anonymize", text))
        if self.fail_anonymize:
            raise RuntimeError("private source should not be included in error")
        return {"mapping_id": "opaque-map-id", "text": "[masked document]"}

    async def deanonymize(self, mapping_id, text):
        self.calls.append(("deanonymize", mapping_id, text))
        if self.fail_restore:
            raise RuntimeError("restore unavailable")
        return {"text": "[restored answer]", "warnings": []}

    async def close(self):
        pass


class FakeResponses:
    def __init__(self, output_text="[response with marker]", fail=False):
        self.output_text = output_text
        self.fail = fail
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("synthetic OpenAI failure")
        return SimpleNamespace(output_text=self.output_text)


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.responses = FakeResponses(**kwargs)


def test_dispatch_sends_only_masked_text_and_restores_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    pseudonymizer = FakePseudonymizer()
    openai = FakeOpenAI()
    with TestClient(create_app(pseudonymizer=pseudonymizer, openai_client=openai)) as client:
        response = client.post("/api/process", json={"prompt": "Summarize", "document": "Alice Smith lives at 123 Main St"})

    assert response.status_code == 200
    body = response.json()
    expected_raw = "[ПРОМПТ]\nSummarize\n\n[ДОКУМЕНТ]\nAlice Smith lives at 123 Main St"
    assert pseudonymizer.calls[0] == ("anonymize", expected_raw)
    assert openai.responses.calls == [{
        "model": "test-model", "instructions": OPENAI_INSTRUCTIONS,
        "input": "[masked document]", "store": False, "timeout": TIMEOUT_SECONDS,
    }]
    serialized_payload = repr(openai.responses.calls)
    assert "Alice Smith" not in serialized_payload
    assert "opaque-map-id" not in serialized_payload
    assert body["received_text"] == "[response with marker]"
    assert body["restored_text"] == "[restored answer]"
    assert pseudonymizer.calls[1] == ("deanonymize", "opaque-map-id", "[response with marker]")


def test_openai_failure_preserves_prepared_text_without_source(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    pseudonymizer = FakePseudonymizer()
    openai = FakeOpenAI(fail=True)
    with TestClient(create_app(pseudonymizer=pseudonymizer, openai_client=openai)) as client:
        response = client.post("/api/process", json={"prompt": "p", "document": "secret raw value"})

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["stage"] == "openai"
    assert body["prepared_text"] == "[masked document]"
    assert body["received_text"] == ""
    assert "secret raw value" not in response.text
    assert len(openai.responses.calls) == 1


def test_pseudonymizer_failure_does_not_call_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    pseudonymizer = FakePseudonymizer(fail_anonymize=True)
    openai = FakeOpenAI()
    with TestClient(create_app(pseudonymizer=pseudonymizer, openai_client=openai)) as client:
        response = client.post("/api/process", json={"prompt": "p", "document": "secret raw value"})

    assert response.status_code == 502
    assert response.json()["error"]["stage"] == "anonymize"
    assert openai.responses.calls == []
    assert "secret raw value" not in response.text


def test_restore_failure_keeps_both_masked_request_and_received_answer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    pseudonymizer = FakePseudonymizer(fail_restore=True)
    openai = FakeOpenAI(output_text="answer containing marker")
    with TestClient(create_app(pseudonymizer=pseudonymizer, openai_client=openai)) as client:
        response = client.post("/api/process", json={"prompt": "p", "document": "raw"})

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["stage"] == "deanonymize"
    assert body["prepared_text"] == "[masked document]"
    assert body["received_text"] == "answer containing marker"


def test_input_limit_and_missing_openai_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    pseudo = FakePseudonymizer()
    openai = FakeOpenAI()
    with TestClient(create_app(pseudonymizer=pseudo, openai_client=openai)) as client:
        oversized = client.post("/api/process", json={"prompt": "", "document": "x" * 50_001})
        assert oversized.status_code == 413
        assert pseudo.calls == []
        assert openai.responses.calls == []

    monkeypatch.delenv("OPENAI_API_KEY")
    with TestClient(create_app(pseudonymizer=FakePseudonymizer(), openai_client=FakeOpenAI())) as client:
        response = client.post("/api/process", json={"prompt": "p", "document": "d"})
    assert response.status_code == 503
    assert response.json()["error"]["stage"] == "configuration"


def test_real_http_and_openai_sdk_pipeline_stays_pseudonymized(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    class PersonDetector:
        def predict_entities(self, text, labels, threshold):
            match = re.search(r"Alice Example", text)
            if not match:
                return []
            return [{"text": match.group(), "label": "person name", "start": match.start(),
                     "end": match.end(), "score": 0.99}]

    async def scenario():
        pseudo_app = create_pseudonymizer_app(detector=PersonDetector())
        captured = []
        sdk_errors = []
        async with pseudo_app.router.lifespan_context(pseudo_app):
            pseudo_http = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=pseudo_app), base_url="http://pseudonymizer"
            )
            pseudo_client = PseudonymizerClient("http://pseudonymizer", client=pseudo_http)

            async def openai_wire(request):
                request_bytes = await request.aread()
                request_body = json.loads(request_bytes)
                captured.append({"path": request.url.path, **request_body})
                assert "Alice Example" not in request_body["input"]
                assert "alice@example.org" not in request_body["input"]
                assert "opaque-map-id" not in request_bytes.decode("utf-8")
                marker = re.search(r"⟦[^⟦⟧]+⟧", request_body["input"]).group()
                response_body = {
                    "id": "resp_synthetic", "object": "response", "created_at": 1,
                    "status": "completed", "error": None, "incomplete_details": None,
                    "instructions": request_body["instructions"], "max_output_tokens": None,
                    "model": "gpt-4o-mini", "output": [{
                        "id": "msg_synthetic", "type": "message", "status": "completed",
                        "role": "assistant", "content": [{
                            "type": "output_text", "text": f"Hello {marker}", "annotations": []
                        }],
                    }],
                    "parallel_tool_calls": True, "temperature": 1.0, "tool_choice": "auto",
                    "tools": [], "top_p": 1.0, "truncation": "disabled",
                    "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0},
                              "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 0},
                              "total_tokens": 13}, "metadata": {},
                }
                return httpx.Response(200, json=response_body, request=request)

            sdk_http = httpx.AsyncClient(transport=httpx.MockTransport(openai_wire))
            sdk_client = AsyncOpenAI(api_key="synthetic-test-key", max_retries=0, http_client=sdk_http)
            original_create = sdk_client.responses.create

            async def traced_create(**kwargs):
                try:
                    return await original_create(**kwargs)
                except Exception as exc:
                    sdk_errors.append(f"{type(exc).__name__}: {exc}")
                    raise

            sdk_client.responses.create = traced_create
            web_app = create_app(pseudonymizer=pseudo_client, openai_client=sdk_client)
            async with web_app.router.lifespan_context(web_app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=web_app), base_url="http://web"
                ) as browser:
                    response = await browser.post("/api/process", json={
                        "prompt": "Summarize for Alice Example", "document": "Alice Example: alice@example.org",
                    })
            await sdk_client.close()
            await pseudo_http.aclose()
        return response, sdk_errors

    response, sdk_errors = asyncio.run(scenario())
    assert response.status_code == 200, f"{response.text} SDK errors: {sdk_errors}"
    body = response.json()
    wire = captured[0]
    assert wire["path"] == "/v1/responses"
    assert wire["model"] == "gpt-4o-mini"
    assert wire["store"] is False
    assert "temperature" not in wire
    assert "Alice Example" not in json.dumps(wire)
    assert "alice@example.org" not in json.dumps(wire)
    assert "mapping_id" not in wire
    assert body["sent_request"] == {
        "model": wire["model"], "instructions": wire["instructions"],
        "input": wire["input"], "store": False,
    }
    assert body["received_text"].startswith("Hello ⟦PERSON_")
    assert body["restored_text"] == "Hello Alice Example"
