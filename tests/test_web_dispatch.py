from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.responses import JSONResponse

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


def test_link_and_restore_warnings_are_combined_without_sending_them(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    class WarningClient(FakePseudonymizer):
        async def anonymize(self, text):
            return {"mapping_id": "opaque", "text": "masked", "warnings": ["ambiguous"]}

        async def deanonymize(self, mapping_id, text):
            return {"text": "restored", "warnings": ["ambiguous", "uncertain form"]}

    openai = FakeOpenAI()
    with TestClient(create_app(pseudonymizer=WarningClient(), openai_client=openai)) as client:
        response = client.post("/api/process", json={"document": "private", "prompt": "summarize"})
    assert response.json()["warnings"] == ["ambiguous", "uncertain form"]
    assert "ambiguous" not in repr(openai.responses.calls)


def test_local_model_timeout_is_separate_from_openai(monkeypatch):
    monkeypatch.setenv("PSEUDONYMIZER_TIMEOUT_SECONDS", "780")
    client = PseudonymizerClient("http://example.test")
    assert client.client.timeout.read == 780
    assert TIMEOUT_SECONDS == 120
    asyncio.run(client.close())


def test_intermediate_steps_are_local_and_survive_openai_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    steps = {"entities": [{"value": "Private Name", "code": "PERSON"}],
             "persons": [{"canonical": "Private Name", "aliases": ["Private Name"]}]}

    class StepClient(FakePseudonymizer):
        async def anonymize(self, text):
            return {"mapping_id": "opaque", "text": "masked", "steps": steps, "warnings": ["uncertain link"]}

        async def deanonymize(self, mapping_id, text):
            return {"text": "restored", "steps": {"restorations": [{"value": "Private Name", "status": "restored"}]}}

    for failure in (False, True):
        openai = FakeOpenAI(fail=failure)
        with TestClient(create_app(pseudonymizer=StepClient(), openai_client=openai)) as client:
            response = client.post("/api/process", json={"document": "Private Name"})
        body = response.json()
        assert body["steps"]["entities"] == steps["entities"]
        assert body["steps"]["anonymized_text"] == "masked"
        assert "Private Name" not in repr(openai.responses.calls)
        assert "steps" not in repr(openai.responses.calls)
        if failure:
            assert body["warnings"] == ["uncertain link"]
            assert "restorations" not in body["steps"]
        else:
            assert body["steps"]["restorations"][0]["value"] == "Private Name"


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


def test_downstream_mapping_404_preserves_received_and_sent_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    async def scenario():
        pseudonymizer_api = FastAPI()

        @pseudonymizer_api.post("/anonymize")
        async def anonymize(_body: dict):
            return {"mapping_id": "expired-map", "text": "[masked]"}

        @pseudonymizer_api.post("/deanonymize")
        async def deanonymize(_body: dict):
            return JSONResponse(status_code=404, content={"detail": "mapping_not_found"})

        pseudo_http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=pseudonymizer_api), base_url="http://pseudonymizer"
        )
        pseudo_client = PseudonymizerClient("http://pseudonymizer", client=pseudo_http)
        web_app = create_app(
            pseudonymizer=pseudo_client,
            openai_client=FakeOpenAI(output_text="[received answer]"),
        )
        async with web_app.router.lifespan_context(web_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=web_app), base_url="http://web"
            ) as browser:
                response = await browser.post("/api/process", json={"prompt": "p", "document": "private source"})
        await pseudo_http.aclose()
        return response

    response = asyncio.run(scenario())
    assert response.status_code == 502
    body = response.json()
    assert body["error"] == {
        "stage": "deanonymize", "code": "mapping_not_found",
        "message": "Таблица замен истекла или потеряна после перезапуска сервиса.",
    }
    assert body["prepared_text"] == "[masked]"
    assert body["received_text"] == "[received answer]"
    assert body["sent_request"] == {
        "model": "test-model", "instructions": OPENAI_INSTRUCTIONS,
        "input": "[masked]", "store": False,
    }
    assert "private source" not in response.text


def test_actual_sdk_auth_401_returns_safe_message_without_upstream_error_or_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-secret-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    async def scenario():
        async def unauthorized(request):
            await request.aread()
            return httpx.Response(401, json={"error": {
                "message": "Incorrect API key provided: sk-test-private-value",
                "type": "invalid_request_error", "param": None, "code": "invalid_api_key",
            }}, request=request)

        sdk_http = httpx.AsyncClient(transport=httpx.MockTransport(unauthorized))
        sdk_client = AsyncOpenAI(
            api_key="synthetic-secret-key", max_retries=0, http_client=sdk_http
        )
        web_app = create_app(pseudonymizer=FakePseudonymizer(), openai_client=sdk_client)
        async with web_app.router.lifespan_context(web_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=web_app), base_url="http://web"
            ) as browser:
                response = await browser.post("/api/process", json={"prompt": "p", "document": "private source"})
        await sdk_client.close()
        return response

    response = asyncio.run(scenario())
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["stage"] == "openai"
    assert body["error"]["code"] == "openai_authentication_failed"
    assert body["prepared_text"] == "[masked document]"
    assert body["received_text"] == ""
    assert body["sent_request"]["input"] == "[masked document]"
    assert "synthetic-secret-key" not in response.text
    assert "sk-test-private-value" not in response.text
    assert "Incorrect API key provided" not in response.text
    assert "private source" not in response.text


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
            return [{"text": match.group(), "label": "person name", "start": match.start(),
                     "end": match.end(), "score": 0.99}
                    for match in re.finditer(r"Alice Example", text)]

    async def scenario():
        pseudo_app = create_pseudonymizer_app(detector=PersonDetector())
        captured = []
        wire_errors = []
        async with pseudo_app.router.lifespan_context(pseudo_app):
            pseudo_http = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=pseudo_app), base_url="http://pseudonymizer"
            )
            mapping_ids = []

            class RecordingPseudonymizerClient(PseudonymizerClient):
                async def anonymize(self, text):
                    result = await super().anonymize(text)
                    mapping_ids.append(result["mapping_id"])
                    return result

            pseudo_client = RecordingPseudonymizerClient("http://pseudonymizer", client=pseudo_http)

            async def openai_wire(request):
                try:
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
                except Exception as exc:
                    wire_errors.append(f"{type(exc).__name__}: {exc}")
                    raise

            sdk_http = httpx.AsyncClient(transport=httpx.MockTransport(openai_wire))
            sdk_client = AsyncOpenAI(api_key="synthetic-test-key", max_retries=0, http_client=sdk_http)
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
        return response, captured, mapping_ids, wire_errors

    response, captured, mapping_ids, wire_errors = asyncio.run(scenario())
    assert response.status_code == 200, f"{response.text}; wire errors: {wire_errors}"
    body = response.json()
    wire = captured[0]
    assert wire["path"] == "/v1/responses"
    assert wire["model"] == "gpt-4o-mini"
    assert wire["store"] is False
    assert "temperature" not in wire
    assert "Alice Example" not in json.dumps(wire)
    assert "alice@example.org" not in json.dumps(wire)
    assert "mapping_id" not in wire
    assert all(mapping_id not in json.dumps(wire) for mapping_id in mapping_ids)
    assert body["sent_request"] == {
        "model": wire["model"], "instructions": wire["instructions"],
        "input": wire["input"], "store": False,
    }
    assert body["received_text"].startswith("Hello ⟦PERSON_")
    assert body["restored_text"] == "Hello Alice Example"
