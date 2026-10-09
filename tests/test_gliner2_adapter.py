from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@contextmanager
def _inference_mode():
    yield


class FakeExtractor:
    def __init__(self, output):
        self.output = output
        self.calls = []
        self.processor = SimpleNamespace(tokenizer=object())

    def extract_entities(self, text, labels, **kwargs):
        self.calls.append((text, labels, kwargs))
        return self.output


def test_gliner2_adapter_normalizes_spans_and_keeps_repeated_mentions():
    from pseudonymizer.app import GLiNER2Detector

    extractor = FakeExtractor({
        "entities": {
            "person": [
                {"text": "Ivan Petrov", "confidence": 0.91, "start": 0, "end": 11},
                {"text": "Ivan Petrov", "confidence": 0.88, "start": 16, "end": 27},
            ],
        },
    })
    detector = GLiNER2Detector(extractor, _inference_mode)

    actual = detector.predict_entities("Ivan Petrov met Ivan Petrov", ["person"], threshold=0.42)

    assert actual == [
        {"text": "Ivan Petrov", "label": "person", "start": 0, "end": 11, "score": 0.91},
        {"text": "Ivan Petrov", "label": "person", "start": 16, "end": 27, "score": 0.88},
    ]
    assert extractor.calls == [("Ivan Petrov met Ivan Petrov", ["person"], {
        "threshold": 0.42, "include_confidence": True, "include_spans": True,
    })]


@pytest.mark.parametrize(
    "output",
    [
        None,
        [],
        {"entities": []},
        {"entities": {"person": {"text": "Ivan", "confidence": 0.9, "start": 0, "end": 4}}},
        {"entities": {"person": [{"text": "Ivan", "confidence": 0.9, "start": 0}]}},
        {"entities": {"person": [{"text": "Ivan", "confidence": "high", "start": 0, "end": 4}]}},
        {"entities": {"person": [{"text": "Eve", "confidence": 0.9, "start": 0, "end": 3}]}},
    ],
)
def test_gliner2_adapter_rejects_malformed_extractor_output(output):
    from pseudonymizer.app import GLiNER2Detector

    detector = GLiNER2Detector(FakeExtractor(output), _inference_mode)
    with pytest.raises(ValueError, match="GLiNER2"):
        detector.predict_entities("Ivan", ["person"], threshold=0.4)
