from __future__ import annotations

from collections import defaultdict

import pytest

from pseudonymizer.persons import PersonPipeline
from pseudonymizer.service import Span


class ScriptedGenerator:
    """Return controlled finite choices while recording every group snapshot."""

    def __init__(self, *, assign=None, validate=None, preflight=None):
        self.assign = assign or (lambda payload: payload["groups"][0]["id"])
        self.validate = validate or (lambda payload: "same")
        self.preflight = preflight
        self.preflight_calls = []
        self.calls = []

    def check_input(self, system, payload):
        self.preflight_calls.append((system, payload))
        if isinstance(self.preflight, Exception):
            raise self.preflight
        return self.preflight if isinstance(self.preflight, int) else 12

    def choose(self, system, payload):
        self.calls.append((system, payload))
        if payload["task"] in {"assign_person", "retry_person"}:
            assignment = self.assign(payload)
            return {"assignment": assignment}
        if payload["task"] == "validate_group_member":
            relation = self.validate(payload)
            return {"relation": relation}
        raise AssertionError(f"Unexpected task: {payload.get('task')}")


def person_spans(text, *values):
    spans = []
    cursor = 0
    for value in values:
        start = text.index(value, cursor)
        spans.append(Span(start, start + len(value), value, "PERSON", 0.95))
        cursor = start + len(value)
    return spans


def value_for(payload, mention_id):
    return next(row["value"] for row in payload["mentions"] if row["id"] == mention_id)


def records_by_value(spans, records):
    return {span.value: records[span.start] for span in spans}


def test_groups_incrementally_with_complete_exact_offset_context_and_canonical_name():
    text = "I. Müller; " + "historischer Kontext. " * 50 + "Elisabeth Müller; Élisabeth Müller"
    spans = person_spans(text, "I. Müller", "Elisabeth Müller", "Élisabeth Müller")
    generator = ScriptedGenerator()

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert len(generator.preflight_calls) == 1
    assert generator.preflight_calls[0][1]["document"] == text
    assert [row["value"] for row in generator.preflight_calls[0][1]["mentions"]] == [
        "I. Müller", "Elisabeth Müller", "Élisabeth Müller",
    ]
    assert all(
        text[row["start"]:row["end"]] == row["value"]
        for row in generator.preflight_calls[0][1]["mentions"]
    )
    assignments = [payload for _, payload in generator.calls
                   if payload["task"] in {"assign_person", "retry_person"}]
    assert [value_for(payload, payload["current_mention_id"]) for payload in assignments] == [
        "Élisabeth Müller", "I. Müller",
    ]
    assert all(payload["document"] == text for _, payload in generator.calls)
    assert all(
        text[row["start"]:row["end"]] == row["value"]
        for _, payload in generator.calls for row in payload["mentions"]
    )
    assert len({record.identifier for record in records.values()}) == 1
    assert {record.canonical for record in records.values()} == {"Elisabeth Müller"}
    assert set(next(iter(records.values())).aliases) == {
        "I. Müller", "Elisabeth Müller", "Élisabeth Müller",
    }


def test_ambiguous_assignment_is_retried_once_after_other_groups_are_known():
    text = "Alexander Smith; Alice Smith; Bo Li"
    spans = person_spans(text, "Alexander Smith", "Alice Smith", "Bo Li")
    calls_by_value = defaultdict(int)

    def assign(payload):
        value = value_for(payload, payload["current_mention_id"])
        calls_by_value[value] += 1
        if value == "Bo Li":
            return "new"
        if value == "Alice Smith" and calls_by_value[value] == 1:
            return "unknown"
        return "entity_1"

    generator = ScriptedGenerator(assign=assign)
    records, warnings = PersonPipeline(generator).link(text, spans)

    assignment_calls = [payload for _, payload in generator.calls
                        if payload["task"] in {"assign_person", "retry_person"}]
    retry = [payload for payload in assignment_calls
             if value_for(payload, payload["current_mention_id"]) == "Alice Smith"]
    assert len(retry) == 2
    assert [payload["task"] for payload in retry] == ["assign_person", "retry_person"]
    assert retry[0]["allowed_assignments"] == ["entity_1", "new", "unknown"]
    assert retry[1]["allowed_assignments"] == ["entity_1", "entity_2", "new", "unknown"]
    assert retry[0]["groups"] == [
        {"id": "entity_1", "member_ids": ["m1"], "values": ["Alexander Smith"]},
    ]
    assert retry[1]["groups"] == [
        {"id": "entity_1", "member_ids": ["m1"], "values": ["Alexander Smith"]},
        {"id": "entity_2", "member_ids": ["m3"], "values": ["Bo Li"]},
    ]
    by_value = records_by_value(spans, records)
    assert by_value["Alice Smith"].identifier == by_value["Alexander Smith"].identifier
    assert by_value["Bo Li"].identifier != by_value["Alexander Smith"].identifier
    assert warnings == []


def test_unresolved_unknown_is_retried_once_then_kept_separate_with_warning():
    text = "Alexander Smith; A. Smith"
    spans = person_spans(text, "Alexander Smith", "A. Smith")
    generator = ScriptedGenerator(assign=lambda payload: "unknown")

    records, warnings = PersonPipeline(generator).link(text, spans)

    assignments = [payload for _, payload in generator.calls
                   if payload["task"] in {"assign_person", "retry_person"}]
    assert len(assignments) == 2
    assert assignments[0]["task"] == "assign_person"
    assert assignments[1]["task"] == "retry_person"
    assert value_for(assignments[0], assignments[0]["current_mention_id"]) == "A. Smith"
    assert value_for(assignments[1], assignments[1]["current_mention_id"]) == "A. Smith"
    by_value = records_by_value(spans, records)
    assert by_value["Alexander Smith"].identifier != by_value["A. Smith"].identifier
    assert warnings


def test_final_validation_splits_a_conflicting_member_from_immutable_group_snapshot():
    text = "Alexander Smith; Alex Smith; A. Smith"
    spans = person_spans(text, "Alexander Smith", "Alex Smith", "A. Smith")

    def validate(payload):
        return "different" if value_for(payload, payload["target_mention_id"]) == "A. Smith" else "same"

    generator = ScriptedGenerator(validate=validate)
    records, warnings = PersonPipeline(generator).link(text, spans)

    validations = [payload for _, payload in generator.calls
                   if payload["task"] == "validate_group_member"]
    assert len(validations) == 2
    assert {tuple(payload["other_member_ids"]) for payload in validations} == {
        ("m1", "m2"), ("m1", "m3"),
    }
    # Both checks see the same immutable pre-split group membership.
    assert all(payload["groups"][0]["member_ids"] == ["m1", "m2", "m3"]
               for payload in validations)
    by_value = records_by_value(spans, records)
    assert by_value["Alexander Smith"].identifier == by_value["Alex Smith"].identifier
    assert by_value["A. Smith"].identifier != by_value["Alexander Smith"].identifier
    assert warnings


def test_repeated_identical_spellings_can_be_assigned_to_distinct_people():
    text = "John Smith; John Smith"
    spans = person_spans(text, "John Smith", "John Smith")
    seen = 0

    def assign(payload):
        nonlocal seen
        seen += 1
        return "new"

    records, _ = PersonPipeline(ScriptedGenerator(assign=assign)).link(text, spans)
    assert records[spans[0].start].identifier != records[spans[1].start].identifier


def test_model_can_link_linguistically_dissimilar_names_without_a_python_name_filter():
    text = "Иван Иванович Петров; Иван Петрович Петров"
    spans = person_spans(text, "Иван Иванович Петров", "Иван Петрович Петров")
    records, warnings = PersonPipeline(ScriptedGenerator()).link(text, spans)

    assert records[spans[0].start].identifier == records[spans[1].start].identifier
    assert warnings == []


@pytest.mark.parametrize("relation", ["different", "unknown"])
def test_group_validation_splits_non_same_member_outcome(relation):
    text = "Alexander Smith; Alex Smith"
    spans = person_spans(text, "Alexander Smith", "Alex Smith")
    records, warnings = PersonPipeline(
        ScriptedGenerator(validate=lambda payload: relation)
    ).link(text, spans)
    assert records[spans[0].start].identifier != records[spans[1].start].identifier
    assert warnings


def test_invalid_assignment_to_unoffered_group_is_kept_separate_and_warns():
    text = "Alexander Smith; Alex Smith"
    spans = person_spans(text, "Alexander Smith", "Alex Smith")
    records, warnings = PersonPipeline(
        ScriptedGenerator(assign=lambda payload: "not-an-offered-group")
    ).link(text, spans)
    assert records[spans[0].start].identifier != records[spans[1].start].identifier
    assert warnings


def test_no_person_mentions_still_preflights_complete_input_and_makes_no_generation_calls():
    text = "No names here."
    phone_start = text.index("No")
    non_person = Span(phone_start, phone_start + 2, "No", "PHONE", 0.99)
    generator = ScriptedGenerator(preflight=17)

    records, warnings = PersonPipeline(generator).link(text, [non_person])

    assert records == {}
    assert warnings == []
    assert len(generator.preflight_calls) == 1
    assert generator.preflight_calls[0][1] == {
        "task": "person_grouping_preflight", "document": text, "mentions": [], "groups": [],
    }
    assert generator.calls == []


def test_anonymize_text_preflights_even_when_detector_finds_no_person():
    from pseudonymizer.service import anonymize_text

    class NoPersonDetector:
        def predict_entities(self, text, labels, threshold):
            return []

    text = "A document with no detected person names."
    generator = ScriptedGenerator(preflight=23)
    pipeline = PersonPipeline(generator)

    masked, mapping = anonymize_text(text, NoPersonDetector(), pipeline)

    assert masked == text
    assert mapping == {}
    assert len(generator.preflight_calls) == 1
    assert generator.preflight_calls[0][1]["document"] == text
    assert generator.calls == []


def test_single_person_mention_still_preflights_and_gets_complete_assignment():
    text = "Irene Blake"
    spans = person_spans(text, "Irene Blake")
    generator = ScriptedGenerator()

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert len(generator.preflight_calls) == 1
    assert generator.calls == []
    assert set(records) == {spans[0].start}
    assert records[spans[0].start].canonical == "Irene Blake"
    assert warnings == []


def test_context_preflight_error_aborts_before_any_assignment():
    text = "Irene Blake"
    spans = person_spans(text, "Irene Blake")
    error = RuntimeError("context too large")
    generator = ScriptedGenerator(preflight=error)

    with pytest.raises(RuntimeError, match="context too large"):
        PersonPipeline(generator).link(text, spans)
    assert len(generator.preflight_calls) == 1
    assert generator.calls == []


def test_incremental_calls_scale_linearly_for_66_mentions():
    values = [f"Person{i} Surname{i}" for i in range(66)]
    text = "; ".join(values)
    spans = person_spans(text, *values)
    generator = ScriptedGenerator()

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert len(generator.calls) <= 3 * (len(spans) - 1)
    assert len(records) == len(spans)
    assert len({record.identifier for record in records.values()}) == 1
    assert warnings == []


def test_nonperson_spans_are_not_added_to_group_mention_table():
    text = "Irene Blake; 0300 555 0100; 10 Oak Road"
    spans = person_spans(text, "Irene Blake")
    phone_start = text.index("0300 555 0100")
    address_start = text.index("10 Oak Road")
    spans.extend([
        Span(phone_start, phone_start + len("0300 555 0100"), "0300 555 0100", "PHONE", 0.99),
        Span(address_start, address_start + len("10 Oak Road"), "10 Oak Road", "ADDRESS", 0.99),
    ])
    generator = ScriptedGenerator()

    records, _ = PersonPipeline(generator).link(text, spans)

    assert len(records) == 1
    assert all(row["value"] == "Irene Blake"
               for row in generator.preflight_calls[0][1]["mentions"])


def test_preflight_input_is_not_truncated_by_pipeline():
    text = "Irene Blake; " + ("контекст " * 1000)
    spans = person_spans(text, "Irene Blake")
    generator = ScriptedGenerator()

    PersonPipeline(generator).link(text, spans)

    assert generator.preflight_calls[0][1]["document"] == text
    assert len(generator.preflight_calls[0][1]["document"]) == len(text)
