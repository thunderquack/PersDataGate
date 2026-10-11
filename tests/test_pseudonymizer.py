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
    assert marker == "⟦PERSON_1⟧"
    assert masked.count(marker) == 2
    assert mapping[marker] == "Юлия Müller"
    restored, warnings = restore_text(f"Ответ: {marker}; вложенный ⟦OTHER_" + "a" * 32 + "_1⟧", mapping)
    assert restored == "Ответ: Юлия Müller; вложенный ⟦OTHER_" + "a" * 32 + "_1⟧"
    assert len(warnings) == 1


def test_inline_qwen_mention_ids_never_leak_into_anonymized_api_text(monkeypatch):
    from pseudonymizer.persons import PersonPipeline
    from pseudonymizer.service import Span

    monkeypatch.setenv("DEVICE", "cpu")
    source = "ÉvaÉva"

    class AdjacentUnicodeDetector:
        def predict_entities(self, text, labels, threshold):
            return [
                {"text": "Éva", "label": "person name", "start": 0, "end": 3, "score": 0.99},
                {"text": "Éva", "label": "person name", "start": 3, "end": 6, "score": 0.99},
            ]

    class SamePersonGenerator:
        def __init__(self):
            self.payloads = []

        def check_input(self, system, payload):
            self.payloads.append(payload)
            return 10

        def choose(self, system, payload):
            self.payloads.append(payload)
            return {"same_person_ids": ["m2"]}

    generator = SamePersonGenerator()
    with TestClient(create_app(
        detector=AdjacentUnicodeDetector(), pipeline=PersonPipeline(generator)
    )) as client:
        response = client.post("/anonymize", json={"text": source})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "⟦PERSON_1⟧⟦PERSON_1⟧"
    assert body["text"].count("⟦m") == 0
    assert "⟦m1⟧" not in response.text and "⟦m2⟧" not in response.text
    assert "Éva" not in response.text
    assert generator.payloads[0]["document"] == "⟦m1⟧ Éva⟦m2⟧ Éva"
    rows = generator.payloads[0]["mentions"]
    assert [(row["start"], row["end"], row["source_start"], row["source_end"])
            for row in rows] == [(5, 8, 0, 3), (13, 16, 3, 6)]


def test_short_marker_does_not_collide_with_existing_source_marker():
    source = "Literal ⟦PERSON_1⟧; Иван Петров"
    detector = FakeDetector({source: [entity("Иван Петров", "person", source)]})
    masked, mapping = anonymize_text(source, detector)
    assert masked == "Literal ⟦PERSON_1⟧; ⟦PERSON_2⟧"
    assert mapping == {"⟦PERSON_2⟧": "Иван Петров"}
    restored, warnings = restore_text(masked, mapping)
    assert restored == source
    assert len(warnings) == 1


def test_restoration_is_one_pass_when_original_value_looks_like_marker():
    marker = "⟦PERSON_" + "a" * 32 + "_1⟧"
    nested_marker = "⟦PERSON_" + "b" * 32 + "_2⟧"
    text, warnings = restore_text(marker, {marker: nested_marker, nested_marker: "must not be inserted"})
    assert text == nested_marker
    assert warnings == []


def test_all_known_markers_restore_saved_values_without_calling_pipeline_and_report_reason():
    from pseudonymizer.persons import PersonRecord
    from pseudonymizer.service import PseudonymMapping

    class ForbiddenPipeline:
        def restore(self, *args):
            raise AssertionError("Known source values must not be sent to a model")

        def restore_with_details(self, *args):
            raise AssertionError("Known source values must not be sent to a model")

    person = "⟦PERSON_demo_1⟧"
    phone = "⟦PHONE_demo_1⟧"
    address = "⟦ADDRESS_demo_1⟧"
    unknown = "⟦PERSON_unknown_1⟧"
    mapping = PseudonymMapping()
    mapping[person] = "Élise Müller"
    mapping.persons[person] = PersonRecord("p1", "Élise Müller", ("Élise Müller",))
    mapping[phone] = "+49 30 555 01904"
    mapping[address] = "Sonnenallee 118, 12059 Berlin"
    text = f"{person} / {person} / {phone} / {phone} / {address} / {unknown}"
    steps = {}

    restored, warnings = restore_text(text, mapping, ForbiddenPipeline(), steps=steps)

    assert restored == (
        "Élise Müller / Élise Müller / +49 30 555 01904 / +49 30 555 01904 / "
        "Sonnenallee 118, 12059 Berlin / ⟦PERSON_unknown_1⟧"
    )
    occurrences = steps["restorations"]
    assert [item["status"] for item in occurrences] == ["restored"] * 5 + ["unknown"]
    assert [item["reason"] for item in occurrences[:5]] == ["saved_value"] * 5
    assert len(warnings) == 1


def test_format_rule_wins_over_overlapping_model_span():
    source = "Contact julia@example.org today"
    detector = FakeDetector({source: [entity("julia@example.org", "person name", source, score=1.0)]})
    masked, mapping = anonymize_text(source, detector)
    assert "julia@example.org" not in masked
    assert len(mapping) == 1
    assert next(iter(mapping.values())) == "julia@example.org"
    assert next(iter(mapping)).startswith("⟦EMAIL_")


def test_person_and_other_entity_confidence_thresholds(monkeypatch):
    import math
    from pseudonymizer.service import (
        ACCEPTED_THRESHOLDS, LABELS, MODEL_INFERENCE_THRESHOLD, find_spans,
    )

    source = (
        "Irene Wirth; I. Wirth; Low Address; High Address; Low DOB; High DOB; "
        "Low Person; Invalid Score"
    )
    predictions = [
        entity("Irene Wirth", "person", source, score=0.145),
        entity("I. Wirth", "person", source, score=0.145),
        entity("Low Address", "street address", source, score=0.44),
        entity("High Address", "street address", source, score=0.45),
        entity("Low DOB", "date of birth", source, score=0.44),
        entity("High DOB", "date of birth", source, score=0.45),
        entity("Low Person", "person", source, score=0.099),
        entity("Invalid Score", "person", source, score=math.nan),
    ]
    detector = FakeDetector({source: predictions})
    events = []
    monkeypatch.setattr("pseudonymizer.service.event", lambda name, data: events.append((name, data)))

    spans = find_spans(source, detector)

    assert [span.value for span in spans] == [
        "Irene Wirth", "I. Wirth", "High Address", "High DOB"
    ]
    assert detector.calls == [(source, LABELS, MODEL_INFERENCE_THRESHOLD)]
    assert "person" in detector.calls[0][1]
    assert "person name" not in detector.calls[0][1]
    chunk_log = next(data for name, data in events if name == "ner.chunk")
    assert chunk_log["threshold"] == 0.10
    assert chunk_log["accepted_thresholds"] == ACCEPTED_THRESHOLDS
    rejected = chunk_log["rejected_low_confidence"]
    assert {(item["value"], item["threshold"], item["reason"]) for item in rejected} == {
        ("Low Address", 0.45, "low_confidence"),
        ("Low DOB", 0.45, "low_confidence"),
        ("Low Person", 0.10, "low_confidence"),
        ("Invalid Score", 0.10, "non_finite_score"),
    }
    assert all(item["score"] is None or math.isfinite(item["score"])
               for item in chunk_log["returned_entities"])


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
        second = client.post("/anonymize", json={"text": "Contact b@example.org"})
        assert first.status_code == second.status_code == 200
        first_body, second_body = first.json(), second.json()
        assert first_body["mapping_id"] != second_body["mapping_id"]
        marker = "⟦EMAIL_" in first_body["text"]
        assert marker
        roundtrip = client.post("/deanonymize", json={"mapping_id": first_body["mapping_id"], "text": first_body["text"]})
        assert roundtrip.json()["text"] == source
        isolated = client.post("/deanonymize", json={"mapping_id": second_body["mapping_id"], "text": first_body["text"]})
        assert first_body["text"] == second_body["text"] == "Contact ⟦EMAIL_1⟧"
        assert isolated.json()["text"] == "Contact b@example.org"
        assert isolated.json()["warnings"] == []
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
            raise AssertionError("Known source values must be restored literally")

    with TestClient(create_app(detector=detector, store=store, pipeline=Pipeline())) as client:
        masked = client.post("/anonymize", json={"text": source}).json()
        assert masked["warnings"] == ["link warning"]
        assert source not in str(masked)
        result = client.post("/deanonymize", json={"mapping_id": masked["mapping_id"], "text": masked["text"]})
        assert result.json()["text"] == source
        assert result.json()["warnings"] == ["link warning"]
        clock[0] = 5
        assert client.post("/deanonymize", json={"mapping_id": masked["mapping_id"], "text": masked["text"]}).status_code == 404


def test_opt_in_steps_capture_entities_clusters_and_saved_value_restorations():
    from pseudonymizer.persons import PersonRecord

    source = "Иван Петров; Ивана Петрова; a@example.org"
    detector = FakeDetector({source: [entity(value, "person name", source)
                                     for value in ("Иван Петров", "Ивана Петрова")]})

    class Pipeline:
        def link(self, text, spans):
            record = PersonRecord("one", "Иван Петров", tuple(span.value for span in spans))
            return {span.start: record for span in spans}, []

        def restore(self, text, marker, record, start, end):
            raise AssertionError("Known source values must be restored literally")

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
        assert [item["status"] for item in occurrences] == ["restored", "restored", "unknown"]
        assert [item["value"] for item in occurrences[:2]] == ["Иван Петров", "Иван Петров"]
        assert [item["reason"] for item in occurrences[:2]] == ["saved_value", "saved_value"]
        # Each response trace is fresh; the original private mapping stays usable.
        repeated = client.post("/deanonymize", json={"mapping_id": traced["mapping_id"], "text": marker}).json()
        assert "steps" not in repeated
        assert repeated["text"] == "Иван Петров"


def test_person_restoration_uses_saved_value_even_in_german_recipient_address():
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
    assert trace["restorations"][0]["reason"] == "saved_value"
    assert trace["restorations"][0]["status"] == "restored"
    assert all("inflect" not in warning for warning in warnings)
