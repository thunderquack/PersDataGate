from __future__ import annotations

from fastapi.testclient import TestClient

from pseudonymizer.local_llm import ContextLimitError
from pseudonymizer.app import create_app
from pseudonymizer.service import MappingStore, anonymize_text, restore_text


class FakeDetector:
    """Return scripted spans using GLiNER's public prediction shape."""

    def __init__(self, entities_by_chunk=None):
        self.entities_by_chunk = entities_by_chunk or {}
        self.calls = []

    def predict_entities(self, text, labels, threshold):
        self.calls.append((text, labels, threshold))
        return list(self.entities_by_chunk.get(text, []))


def entity(text: str, label: str, in_text: str, score=0.9):
    start = in_text.index(text)
    return {"text": text, "label": label, "start": start, "end": start + len(text), "score": score}


def test_mapping_store_expires_at_creation_and_unknown_ids_return_none():
    now = [100.0]
    store = MappingStore(ttl_seconds=30, clock=lambda: now[0], wall_clock=lambda: now[0])
    mapping_id, expires = store.create({"marker": "value"})
    assert expires == 130
    assert store.get(mapping_id) == {"marker": "value"}
    now[0] = 129.9
    assert store.get(mapping_id) is not None
    now[0] = 130
    assert store.get(mapping_id) is None
    assert store.get("unknown") is None
    store.close()


def test_person_context_limit_is_a_safe_client_error_without_raw_text():
    source = "Мария Иванова попросила подготовить подробную справку."
    detector = FakeDetector({source: [entity("Мария Иванова", "person name", source)]})

    class OversizedPersonPipeline:
        def link(self, _text, _spans):
            raise ContextLimitError(45_123, 40_704, 40_960)

    with TestClient(create_app(detector=detector, pipeline=OversizedPersonPipeline())) as client:
        response = client.post("/anonymize", json={"text": source})

    assert response.status_code == 413
    body = response.json()
    assert body["detail"] == {
        "code": "local_model_context_exceeded", "input_tokens": 45_123,
        "maximum_input_tokens": 40_704, "context_window": 40_960,
    }
    assert source not in response.text


def test_repeated_unicode_values_share_marker_and_restore_once():
    source = "Юлия Müller, email julia@example.org; снова Юлия Müller"
    first_start = source.index("Юлия Müller")
    second_start = source.rindex("Юлия Müller")
    detector = FakeDetector({source: [
        {"text": "Юлия Müller", "label": "person name", "start": first_start,
         "end": first_start + len("Юлия Müller"), "score": 0.9},
        {"text": "Юлия Müller", "label": "person name", "start": second_start,
         "end": second_start + len("Юлия Müller"), "score": 0.9},
    ]})
    masked, mapping = anonymize_text(source, detector)
    marker = next(iter(mapping))
    assert masked.count(marker) == 2
    assert mapping[marker] == "Юлия Müller"
    restored, warnings = restore_text(f"Ответ: {marker}; вложенный ⟦OTHER_" + "a" * 32 + "_1⟧", mapping)
    assert restored == "Ответ: Юлия Müller; вложенный ⟦OTHER_" + "a" * 32 + "_1⟧"
    assert len(warnings) == 1


def test_restoration_is_one_pass_when_original_value_looks_like_marker():
    marker = "⟦PERSON_" + "a" * 32 + "_1⟧"
    nested_marker = "⟦PERSON_" + "b" * 32 + "_2⟧"
    text, warnings = restore_text(marker, {marker: nested_marker})
    assert text == nested_marker
    assert warnings == []


def test_format_rule_wins_over_overlapping_model_span():
    source = "Contact julia@example.org today"
    detector = FakeDetector({source: [entity("julia@example.org", "person name", source, score=1.0)]})
    masked, mapping = anonymize_text(source, detector)
    assert "julia@example.org" not in masked
    assert len(mapping) == 1
    assert next(iter(mapping.values())) == "julia@example.org"
    assert next(iter(mapping)).startswith("⟦EMAIL_")


def test_long_text_is_chunked_with_overlap_and_preserves_global_spans():
    lead = "x" * 690
    source = lead + " Иван Петров " + "y" * 700
    chunk_start = source[580:1280]
    detector = FakeDetector({chunk_start: [entity("Иван Петров", "person name", chunk_start)]})
    masked, mapping = anonymize_text(source, detector)
    assert len(detector.calls) > 1
    assert "Иван Петров" not in masked
    assert next(iter(mapping.values())) == "Иван Петров"
    assert masked.startswith(lead)
    assert masked.endswith("y" * 700)


class FakeTokenizer:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def __call__(self, text, **kwargs):
        self.calls += 1
        if self.fail:
            raise ValueError("bad tokenizer offsets")
        # One token per character makes the 250/300-token chunk boundaries
        # deterministic without importing the GLiNER/PyTorch stack.
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def test_tokenizer_controls_chunking_and_offsets_are_mapped_to_original_text():
    source = "a" * 275 + " Иван Петров " + "b" * 650
    tokenizer = FakeTokenizer()
    chunk_start, chunk_end = 250, 550
    selected_chunk = source[chunk_start:chunk_end]
    detector = FakeDetector({selected_chunk: [entity("Иван Петров", "person name", selected_chunk)]})
    detector.tokenizer = tokenizer
    masked, mapping = anonymize_text(source, detector)
    assert tokenizer.calls == 1
    assert len(detector.calls) > 1
    assert "Иван Петров" not in masked
    assert next(iter(mapping.values())) == "Иван Петров"


def test_tokenizer_failure_fails_closed_at_api_boundary():
    class BrokenDetector:
        tokenizer = FakeTokenizer(fail=True)

        def predict_entities(self, text, labels, threshold):
            raise AssertionError("must not infer after tokenizer failure")

    with TestClient(create_app(detector=BrokenDetector())) as client:
        response = client.post("/anonymize", json={"text": "raw private data"})
    assert response.status_code == 503
    assert response.json()["detail"] == "anonymization_failed"


def test_api_round_trip_ttl_and_mapping_isolation():
    clock = [10.0]
    store = MappingStore(ttl_seconds=5, clock=lambda: clock[0])
    source = "Contact a@example.org"
    detector = FakeDetector({source: []})
    with TestClient(create_app(detector=detector, store=store)) as client:
        first = client.post("/anonymize", json={"text": source})
        second = client.post("/anonymize", json={"text": source})
        assert first.status_code == second.status_code == 200
        first_body, second_body = first.json(), second.json()
        assert first_body["mapping_id"] != second_body["mapping_id"]
        marker = "⟦EMAIL_" in first_body["text"]
        assert marker
        roundtrip = client.post("/deanonymize", json={"mapping_id": first_body["mapping_id"], "text": first_body["text"]})
        assert roundtrip.json()["text"] == source
        isolated = client.post("/deanonymize", json={"mapping_id": second_body["mapping_id"], "text": first_body["text"]})
        assert isolated.json()["text"] == first_body["text"]
        assert isolated.json()["warnings"]
        clock[0] = 15.0
        expired = client.post("/deanonymize", json={"mapping_id": first_body["mapping_id"], "text": first_body["text"]})
        assert expired.status_code == 404
        unknown = client.post("/deanonymize", json={"mapping_id": "missing", "text": "hello"})
        assert unknown.status_code == 404
    store.close()


def test_person_pipeline_failures_and_incomplete_assignments_fail_closed():
    source = "Иван Петров"
    detector = FakeDetector({source: [entity(source, "person name", source)]})

    class IncompletePipeline:
        def link(self, text, spans):
            return {}, []

    class FailedPipeline:
        def link(self, text, spans):
            raise RuntimeError("sensitive local prompt")

    for pipeline in (IncompletePipeline(), FailedPipeline()):
        with TestClient(create_app(detector=detector, pipeline=pipeline)) as client:
            response = client.post("/anonymize", json={"text": source})
        assert response.status_code == 503
        assert response.json()["detail"] == "anonymization_failed"
        assert response.json()["pipeline_id"] == response.headers["X-Pipeline-ID"]


def test_rich_person_mapping_api_restoration_warning_and_ttl():
    from pseudonymizer.persons import PersonRecord

    source = "Иван Петров"
    detector = FakeDetector({source: [entity(source, "person name", source)]})
    clock = [0.0]
    store = MappingStore(ttl_seconds=5, clock=lambda: clock[0])

    class Pipeline:
        def link(self, text, spans):
            return {spans[0].start: PersonRecord("one", source, (source,))}, ["link warning"]

        def restore(self, *args):
            return None

    with TestClient(create_app(detector=detector, store=store, pipeline=Pipeline())) as client:
        masked = client.post("/anonymize", json={"text": source}).json()
        assert masked["warnings"] == ["link warning"]
        assert source not in str(masked)
        result = client.post("/deanonymize", json={"mapping_id": masked["mapping_id"], "text": masked["text"]})
        assert result.json()["text"] == masked["text"]
        assert len(result.json()["warnings"]) == 2
        clock[0] = 5
        assert client.post("/deanonymize", json={"mapping_id": masked["mapping_id"], "text": masked["text"]}).status_code == 404


def test_opt_in_steps_capture_entities_clusters_and_occurrence_forms():
    from pseudonymizer.persons import PersonRecord

    source = "Иван Петров; Ивана Петрова; a@example.org"
    detector = FakeDetector({source: [entity(value, "person name", source)
                                     for value in ("Иван Петров", "Ивана Петрова")]})

    class Pipeline:
        def link(self, text, spans):
            record = PersonRecord("one", "Иван Петров", tuple(span.value for span in spans))
            return {span.start: record for span in spans}, []

        def restore(self, text, marker, record, start, end):
            return "Ивану Петрову" if start == 0 else None

    with TestClient(create_app(detector=detector, pipeline=Pipeline())) as client:
        default = client.post("/anonymize", json={"text": source}).json()
        assert "steps" not in default
        traced = client.post("/anonymize", json={"text": source, "include_steps": True}).json()
        steps = traced["steps"]
        assert [item["source"] for item in steps["entities"]] == ["gliner", "gliner", "rule"]
        assert steps["persons"][0]["aliases"] == ["Иван Петров", "Ивана Петрова"]
        assert len(steps["persons"][0]["mentions"]) == 2
        assert steps["anonymized_text"] == traced["text"]
        marker = steps["persons"][0]["marker"]
        answer = f"{marker} и {marker} ⟦unknown⟧"
        restored = client.post("/deanonymize", json={
            "mapping_id": traced["mapping_id"], "text": answer, "include_steps": True
        }).json()
        occurrences = restored["steps"]["restorations"]
        assert [item["status"] for item in occurrences] == ["restored", "uncertain", "unknown"]
        assert occurrences[0]["value"] == "Ивану Петрову"
        assert occurrences[1]["value"] is None
        # Each response trace is fresh; the original private mapping stays usable.
        repeated = client.post("/deanonymize", json={"mapping_id": traced["mapping_id"], "text": marker}).json()
        assert "steps" not in repeated
        assert repeated["text"] == "Ивану Петрову"


def test_address_restoration_trace_reports_reason_without_calling_qwen():
    from pseudonymizer.persons import PersonPipeline, PersonRecord
    from pseudonymizer.service import PseudonymMapping

    class ForbiddenGenerator:
        def choose(self, *args):
            raise AssertionError("Address field needs no model selection")

    marker = "⟦PERSON_local_1⟧"
    mapping = PseudonymMapping()
    mapping[marker] = "Egor Ivanov"
    mapping.persons[marker] = PersonRecord("one", "Egor Ivanov", ("Egor Ivanov", "EGOR IVANOV"))
    text = f"An\n{marker}\n⟦ADDRESS_local_1⟧\n12051 Berlin"
    trace = {}
    restored, warnings = restore_text(text, mapping, PersonPipeline(ForbiddenGenerator()), steps=trace)
    assert restored.startswith("An\nEgor Ivanov\n")
    assert trace["restorations"][0]["reason"] == "recipient_address"
    assert trace["restorations"][0]["status"] == "restored"
    assert all("inflect" not in warning for warning in warnings)
