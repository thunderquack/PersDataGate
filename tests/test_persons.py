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


def expected_annotated_document(text, spans):
    parts = []
    rows = []
    source_cursor = 0
    shift = 0
    rows_by_value = {}
    for span in spans:
        normalization_key = span.value.strip().casefold()
        row = rows_by_value.get(normalization_key)
        if row is None:
            mention_id = f"m{len(rows) + 1}"
            row = {
                "id": mention_id,
                "value": span.value,
                "observed_values": [span.value],
                "start": None,
                "end": None,
                "source_start": None,
                "source_end": None,
                "occurrences": [],
            }
            rows_by_value[normalization_key] = row
            rows.append(row)
        else:
            mention_id = row["id"]
            if span.value not in row["observed_values"]:
                row["observed_values"].append(span.value)
        tag = f"⟦{mention_id}⟧ "
        parts.extend((text[source_cursor:span.start], tag, span.value))
        annotated_start = span.start + shift + len(tag)
        occurrence = {
            "value": span.value,
            "start": annotated_start,
            "end": annotated_start + len(span.value),
            "source_start": span.start,
            "source_end": span.end,
        }
        row["occurrences"].append(occurrence)
        if row["start"] is None:
            row.update(occurrence)
        shift += len(tag)
        source_cursor = span.end
    parts.append(text[source_cursor:])
    return "".join(parts), rows


def test_anchor_grouping_uses_source_order_full_document_offsets_and_only_unassigned_ids():
    text = ("I. Müller; " + "historischer Kontext. " * 50
            + "Elisabeth Müller; Élisabeth Müller; Mrs Example")
    spans = person_spans(text, "I. Müller", "Elisabeth Müller", "Élisabeth Müller", "Mrs Example")
    generator = ScriptedGenerator(selections=[["m2"], []])
    annotated_text, expected_mentions = expected_annotated_document(text, spans)

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert len(generator.preflight_calls) == 1
    assert len(generator.calls) == 2
    for system, payload in generator.calls:
        assert "person" in system.lower()
        assert payload["task"] == "link_person_mentions"
        assert payload["document"] == annotated_text
        assert payload["document_annotation"]
        assert payload["mentions"] == expected_mentions
        assert all(payload["document"][row["start"]:row["end"]] == row["value"]
                   for row in payload["mentions"])
        assert all(text[row["source_start"]:row["source_end"]] == row["value"]
                   for row in payload["mentions"])
        assert payload["remaining_candidates"] == [
            row for row in payload["mentions"]
            if row["id"] in payload["allowed_same_person_ids"]
        ]
    first, second = [payload for _, payload in generator.calls]
    preflight_payload = generator.preflight_calls[0][1]
    assert preflight_payload == first
    assert preflight_payload["document"] == annotated_text
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


def test_inline_ids_bind_repeated_adjacent_unicode_occurrences_without_changing_source_offsets():
    text = "ÉvaÉva"
    spans = person_spans(text, "Éva", "Éva")
    generator = ScriptedGenerator()

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert generator.calls == []
    payload = generator.preflight_calls[0][1]
    assert payload["document"] == "⟦m1⟧ Éva⟦m1⟧ Éva"
    assert payload["mentions"] == [
        {"id": "m1", "value": "Éva", "start": 5, "end": 8,
         "source_start": 0, "source_end": 3, "occurrences": [
             {"start": 5, "end": 8, "source_start": 0, "source_end": 3},
             {"start": 13, "end": 16, "source_start": 3, "source_end": 6},
         ]},
    ]
    row = payload["mentions"][0]
    assert [payload["document"][occ["start"]:occ["end"]]
            for occ in row["occurrences"]] == ["Éva", "Éva"]
    assert [text[occ["source_start"]:occ["source_end"]]
            for occ in row["occurrences"]] == ["Éva", "Éva"]
    assert records[0].identifier == records[3].identifier


def test_exact_raw_repeats_share_an_id_before_model_and_keep_every_source_start():
    text = "John Smith; John Smith"
    spans = person_spans(text, "John Smith", "John Smith")
    generator = ScriptedGenerator()

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert generator.calls == []  # one unique raw value needs no model choice
    payload = generator.preflight_calls[0][1]
    assert payload["document"] == "⟦m1⟧ John Smith; ⟦m1⟧ John Smith"
    assert payload["mentions"] == [{
        "id": "m1", "value": "John Smith", "start": 5, "end": 15,
        "source_start": 0, "source_end": 10,
        "occurrences": [
            {"start": 5, "end": 15, "source_start": 0, "source_end": 10},
            {"start": 22, "end": 32, "source_start": 12, "source_end": 22},
        ],
    }]
    assert set(records) == {0, 12}
    assert records[0].identifier == records[12].identifier


def test_trim_and_unicode_casefold_pregroup_but_internal_space_nfc_and_initials_stay_distinct():
    values = [
        " ÄNNA ", "änna",                    # trim + Unicode casefold
        "Straße", "STRASSE",                 # German sharp-s casefold expansion
        " ИВАН ", "иван",                    # Cyrillic casefold
        "A  B", "A B",                       # internal whitespace preserved
        "é", "e\u0301",                      # no Unicode normalization
        "J. Smith", "John Smith",            # no initials or linguistic heuristic
    ]
    text = "; ".join(values)
    spans = person_spans(text, *values)
    generator = ScriptedGenerator(selections=[[], [], [], [], [], [], [], []])

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert len(generator.calls) == 8
    payload = generator.calls[0][1]
    assert [row["value"] for row in payload["mentions"]] == [
        " ÄNNA ", "Straße", " ИВАН ", "A  B", "A B", "é", "e\u0301", "J. Smith", "John Smith",
    ]
    assert [row["id"] for row in payload["mentions"]] == [f"m{i}" for i in range(1, 10)]
    assert len({records[spans[index].start].identifier for index in (0, 1)}) == 1
    assert len({records[spans[index].start].identifier for index in (2, 3)}) == 1
    assert len({records[spans[index].start].identifier for index in (4, 5)}) == 1
    assert records[spans[6].start].identifier != records[spans[7].start].identifier
    assert records[spans[8].start].identifier != records[spans[9].start].identifier
    assert records[spans[10].start].identifier != records[spans[11].start].identifier
    assert payload["mentions"][0]["observed_values"] == [" ÄNNA ", "änna"]
    assert payload["mentions"][1]["observed_values"] == ["Straße", "STRASSE"]
    assert payload["mentions"][2]["observed_values"] == [" ИВАН ", "иван"]


def test_model_selection_of_one_unique_id_maps_all_of_its_exact_occurrences():
    text = "John Smith; John Smith; Johnny Smith"
    spans = person_spans(text, "John Smith", "John Smith", "Johnny Smith")
    generator = ScriptedGenerator(selections=[["m2"]])

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert warnings == []
    assert len(generator.calls) == 1
    payload = generator.calls[0][1]
    assert [row["id"] for row in payload["mentions"]] == ["m1", "m2"]
    assert payload["allowed_same_person_ids"] == ["m2"]
    assert payload["remaining_candidates"] == [payload["mentions"][1]]
    assert payload["mentions"][0]["occurrences"] == [
        {"start": 5, "end": 15, "source_start": 0, "source_end": 10},
        {"start": 22, "end": 32, "source_start": 12, "source_end": 22},
    ]
    assert set(records) == {0, 12, 24}
    assert len({record.identifier for record in records.values()}) == 1


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
    rows = payloads[0]["mentions"]
    for index in (8, 9):  # m9 -> m10 changes tag length; later offsets include that shift.
        row = rows[index]
        inserted_prefix_length = sum(len(f"⟦m{tag_index}⟧ ")
                                     for tag_index in range(1, index + 2))
        assert row["start"] == row["source_start"] + inserted_prefix_length
        assert row["end"] == row["start"] + len(row["value"])
        assert payloads[0]["document"][row["start"]:row["end"]] == row["value"]


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
    annotated_text, expected_mentions = expected_annotated_document(text, spans)
    assert generator.preflight_calls[0][1]["document"] == annotated_text
    assert generator.preflight_calls[0][1]["mentions"] == expected_mentions
    assert all(annotated_text[row["start"]:row["end"]] == row["value"]
               for row in expected_mentions)
    assert all(text[row["source_start"]:row["source_end"]] == row["value"]
               for row in expected_mentions)
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
