from __future__ import annotations

from fastapi.testclient import TestClient

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
