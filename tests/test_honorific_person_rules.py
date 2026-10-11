from __future__ import annotations

from pseudonymizer.persons import PersonRecord
from pseudonymizer.service import anonymize_text, find_spans


class NoEntities:
    model_id = "fake"
    model_revision = "test"

    def predict_entities(self, text, labels, threshold):
        return []


class PredictedPerson:
    def __init__(self, full_name: str, source: str):
        self.source = source
        self.full_name = full_name

    def predict_entities(self, text, labels, threshold):
        start = self.source.index(self.full_name)
        return [{
            "text": self.full_name,
            "label": "person",
            "start": start,
            "end": start + len(self.full_name),
            "score": 0.99,
        }]


def test_honorific_fallback_finds_four_missing_mentions_and_uses_existing_linker():
    source = "Sehr geehrter Herr Meyer, Herr Schmidt, Frau Weber und Frau Schulz."
    detector = NoEntities()
    found = find_spans(source, detector)
    assert [span.value for span in found] == [
        "Herr Meyer", "Herr Schmidt", "Frau Weber", "Frau Schulz",
    ]
    assert all(span.code == "PERSON" and span.rule for span in found)

    class RecordingLinker:
        def __init__(self):
            self.mentions = []

        def link(self, document, mentions):
            self.mentions = list(mentions)
            return {
                span.start: PersonRecord(str(span.start), span.value, (span.value,))
                for span in mentions
            }, []

    linker = RecordingLinker()
    steps = {}
    anonymized, mapping = anonymize_text(source, detector, linker, steps=steps)

    assert [span.value for span in linker.mentions] == [span.value for span in found]
    assert len(mapping) == 4
    assert all(value in mapping.values() for value in (
        "Herr Meyer", "Herr Schmidt", "Frau Weber", "Frau Schulz",
    ))
    assert all(row["source"] == "rule" for row in steps["entities"])
    assert all(marker in anonymized for marker in mapping)


def test_honorific_fallback_supports_hyphenated_unicode_surnames():
    source = "Frau Müller-Lüdenscheidt schrieb an Herrn O’Neill."
    spans = find_spans(source, NoEntities())
    assert [span.value for span in spans] == [
        "Frau Müller-Lüdenscheidt", "Herrn O’Neill",
    ]


def test_honorific_fallback_skips_existing_gliner_person_overlap():
    source = "Herrn Alexander Hoffmann wird erwartet."
    start = source.index("Alexander Hoffmann")
    detector = PredictedPerson("Alexander Hoffmann", source)

    spans = find_spans(source, detector)

    assert [(span.value, span.start, span.end, span.rule) for span in spans] == [
        ("Alexander Hoffmann", start, start + len("Alexander Hoffmann"), False),
    ]


def test_honorific_fallback_requires_one_capitalized_surname_token():
    source = "Herr and Frau meier; Frau Dr. Lenz"
    assert find_spans(source, NoEntities()) == []
