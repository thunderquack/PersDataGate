from __future__ import annotations

import pytest

from pseudonymizer.local_llm import PersonGroupingError
from pseudonymizer.persons import PersonPipeline
from pseudonymizer.service import Span


class ScriptedGenerator:
    def __init__(self, selections=None, *, preflight_error=None):
        self.selections = list(selections or [])
        self.preflight_error = preflight_error
        self.preflight_calls = []
        self.calls = []

    def check_input(self, system, payload):
        self.preflight_calls.append((system, payload))
        if self.preflight_error:
            raise self.preflight_error
        return 17

    def choose(self, system, payload):
        self.calls.append((system, payload))
        if self.selections:
            return {"same_person_ids": self.selections.pop(0)}
        return {"same_person_ids": []}


def person_spans(text, *values):
    spans = []
    cursor = 0
    for value in values:
        start = text.index(value, cursor)
        spans.append(Span(start, start + len(value), value, "PERSON", 0.95))
        cursor = start + len(value)
    return spans


def test_anchor_grouping_uses_source_order_full_document_offsets_and_only_unassigned_ids():
    text = ("I. Müller; " + "historischer Kontext. " * 50
            + "Elisabeth Müller; Élisabeth Müller; Mrs Example")
    spans = person_spans(text, "I. Müller", "Elisabeth Müller", "Élisabeth Müller", "Mrs Example")
    generator = ScriptedGenerator(selections=[["m2"], []])

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert len(generator.preflight_calls) == 1
    assert len(generator.calls) == 2
    for system, payload in generator.calls:
        assert "person" in system.lower()
        assert payload["task"] == "link_person_mentions"
        assert payload["document"] == text
        assert payload["mentions"] == [
            {"id": f"m{index}", "value": span.value,
             "start": span.start, "end": span.end}
            for index, span in enumerate(spans, 1)
        ]
        assert all(text[row["start"]:row["end"]] == row["value"]
                   for row in payload["mentions"])
        assert payload["remaining_candidates"] == [
            row for row in payload["mentions"]
            if row["id"] in payload["allowed_same_person_ids"]
        ]
    first, second = [payload for _, payload in generator.calls]
    preflight_payload = generator.preflight_calls[0][1]
    assert preflight_payload == first
    assert first["anchor_id"] == "m1"
    assert first["anchor"] == first["mentions"][0]
    assert first["allowed_same_person_ids"] == ["m2", "m3", "m4"]
    assert first["previously_processed_ids"] == []
    assert second["anchor_id"] == "m3"
    assert second["anchor"] == second["mentions"][2]
    assert second["allowed_same_person_ids"] == ["m4"]
    assert second["previously_processed_ids"] == ["m1", "m2"]
    assert second["remaining_candidates"] == [second["mentions"][3]]
    # All original mentions remain in the document table; the prior group is context only.
    assert second["mentions"] == first["mentions"]
    assert len({record.identifier for record in records.values()}) == 3
    assert records[spans[0].start].identifier == records[spans[1].start].identifier
    assert records[spans[2].start].identifier != records[spans[0].start].identifier


def test_each_anchor_call_is_bounded_by_mentions_and_uncertain_choices_remain_singletons():
    values = [f"Person{i} Surname{i}" for i in range(66)]
    text = "; ".join(values)
    spans = person_spans(text, *values)
    generator = ScriptedGenerator()

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert len(generator.calls) == len(spans) - 1
    assert len(generator.calls) < len(spans)
    assert len(records) == len(spans)
    assert len({record.identifier for record in records.values()}) == len(spans)
    payloads = [payload for _, payload in generator.calls]
    assert [payload["anchor_id"] for payload in payloads] == [
        f"m{index}" for index in range(1, len(spans))
    ]
    for index, payload in enumerate(payloads):
        assert payload["allowed_same_person_ids"] == [
            f"m{candidate}" for candidate in range(index + 2, len(spans) + 1)
        ]


def test_repeated_spellings_can_be_separate_and_name_similarity_does_not_override_model():
    text = "John Smith; John Smith; Иван Иванович Петров; Иван Петрович Петров"
    spans = person_spans(
        text, "John Smith", "John Smith", "Иван Иванович Петров", "Иван Петрович Петров"
    )
    # First pair split; last two joined to the third source-order anchor.
    generator = ScriptedGenerator(selections=[[], [], ["m4"]])

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert records[spans[0].start].identifier != records[spans[1].start].identifier
    assert records[spans[2].start].identifier == records[spans[3].start].identifier
    assert len(generator.calls) == 3


@pytest.mark.parametrize("response", [
    {},
    {"same_person_ids": None},
    {"same_person_ids": "m2"},
    {"same_person_ids": ["m99"]},        # unknown or already-processed ID
    {"same_person_ids": ["m1"]},         # anchor cannot be selected as its own candidate
    {"same_person_ids": ["m2", "m2"]},   # duplicate ID
    {"same_person_ids": [2]},
    {"same_person_ids": [], "extra": 1}, # malformed exact schema
])
def test_bad_anchor_choice_raises_safe_grouping_error(response):
    class BadGenerator(ScriptedGenerator):
        def choose(self, system, payload):
            self.calls.append((system, payload))
            return response

    text = "Alice Example; Bob Example; Casey Example"
    spans = person_spans(text, "Alice Example", "Bob Example", "Casey Example")
    generator = BadGenerator()

    with pytest.raises(PersonGroupingError):
        PersonPipeline(generator).link(text, spans)
    assert len(generator.preflight_calls) == 1
    assert len(generator.calls) == 1


def test_already_processed_id_is_rejected_as_unallowed():
    text = "Alice Example; Bob Example; Casey Example; Drew Example"
    spans = person_spans(text, "Alice Example", "Bob Example", "Casey Example", "Drew Example")
    generator = ScriptedGenerator(selections=[["m2"], ["m2"]])

    with pytest.raises(PersonGroupingError):
        PersonPipeline(generator).link(text, spans)
    assert len(generator.calls) == 2
    assert generator.calls[1][1]["anchor_id"] == "m3"
    assert generator.calls[1][1]["allowed_same_person_ids"] == ["m4"]


@pytest.mark.parametrize("person_values", [(), ("Irene Blake",)])
def test_zero_or_one_person_preflights_complete_input_and_skips_generation(person_values):
    text = "; ".join(person_values) if person_values else "No names here."
    spans = person_spans(text, *person_values)
    generator = ScriptedGenerator()

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert len(generator.preflight_calls) == 1
    assert generator.preflight_calls[0][1]["document"] == text
    assert generator.preflight_calls[0][1]["mentions"] == [
        {"id": f"m{index}", "value": span.value,
         "start": span.start, "end": span.end}
        for index, span in enumerate(spans, 1)
    ]
    assert generator.calls == []
    if spans:
        assert records[spans[0].start].canonical == spans[0].value
    else:
        assert records == {}


def test_non_person_spans_are_excluded_from_grouping():
    text = "Irene Blake; 0300 555 0100; 10 Oak Road; Alex Blake"
    spans = person_spans(text, "Irene Blake", "Alex Blake")
    phone_start = text.index("0300 555 0100")
    address_start = text.index("10 Oak Road")
    spans.extend([
        Span(phone_start, phone_start + len("0300 555 0100"), "0300 555 0100", "PHONE", 0.99),
        Span(address_start, address_start + len("10 Oak Road"), "10 Oak Road", "ADDRESS", 0.99),
    ])
    generator = ScriptedGenerator()

    PersonPipeline(generator).link(text, spans)

    assert len(generator.calls) == 1
    assert [row["value"] for row in generator.calls[0][1]["mentions"]] == [
        "Irene Blake", "Alex Blake",
    ]


def test_longest_observed_value_is_canonical_and_equal_length_uses_earliest():
    text = "Elisabeth Müller; Élisabeth Müller; Christopher Alexander de la Cruz"
    spans = person_spans(
        text, "Elisabeth Müller", "Élisabeth Müller", "Christopher Alexander de la Cruz"
    )
    # Keep first two together and third separate; then exercise longest selection separately.
    generator = ScriptedGenerator(selections=[["m2"], []])
    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert records[spans[0].start].canonical == "Elisabeth Müller"
    assert records[spans[1].start].canonical == "Elisabeth Müller"
    assert records[spans[2].start].canonical == "Christopher Alexander de la Cruz"


@pytest.mark.parametrize("person_values", [(), ("Irene Blake",)])
def test_preflight_overflow_stops_before_generation_for_zero_or_one_person(person_values):
    text = "; ".join(person_values) if person_values else "No names here."
    spans = person_spans(text, *person_values)
    generator = ScriptedGenerator(preflight_error=RuntimeError("context too large"))

    with pytest.raises(RuntimeError, match="context too large"):
        PersonPipeline(generator).link(text, spans)
    assert len(generator.preflight_calls) == 1
    assert generator.calls == []
