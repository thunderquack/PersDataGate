from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from web.app import OPENAI_INSTRUCTIONS, TIMEOUT_SECONDS, create_app


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

