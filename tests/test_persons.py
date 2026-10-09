from __future__ import annotations

import pytest

from pseudonymizer.persons import PersonPipeline, PersonRecord
from pseudonymizer.service import Span


class ScriptedGenerator:
    def __init__(self, *, link="same"):
        self.link = link
        self.link_calls = []

    def choose(self, system, payload):
        if "allowed_relations" in payload:
            self.link_calls.append(payload)
            return {"relation": self.link(payload) if callable(self.link) else self.link}
        raise AssertionError("PersonPipeline no longer makes form-selection calls")


def person_spans(text, *values):
    spans = []
    cursor = 0
    for value in values:
        start = text.index(value, cursor)
        spans.append(Span(start, start + len(value), value, "PERSON", 0.95))
        cursor = start + len(value)
    return spans


def test_declined_name_and_initial_link_globally_and_expand_to_full_name():
    text = "Иван Петров подписал договор. " + "описание " * 100 + "Ивана Петрова назначили. И. Петров подтвердил."
    spans = person_spans(text, "Иван Петров", "Ивана Петрова", "И. Петров")
    generator = ScriptedGenerator(link="same")
    pipeline = PersonPipeline(generator)

    records, warnings = pipeline.link(text, spans)

    assert warnings == []
    assert set(records) == {span.start for span in spans}
    assert len({record.identifier for record in records.values()}) == 1
    record = records[spans[0].start]
    assert record.canonical == "Ивана Петрова"
    assert set(record.aliases) == {"Иван Петров", "Ивана Петрова", "И. Петров"}
    assert generator.link_calls  # distant mentions are still compared globally
    link_payload = generator.link_calls[0]
    assert link_payload["document"] == text
    left = link_payload["left"]
    right = link_payload["right"]
    assert text[left["start"]:left["end"]] == left["mention"]
    assert text[right["start"]:right["end"]] == right["mention"]
    assert "before" not in left and "after" not in left
    assert "before" not in right and "after" not in right
    assert all(len(call["left"]["mention"]) >= len(call["right"]["mention"])
               for call in generator.link_calls)


def test_short_earlier_mention_is_swapped_after_the_fuller_spelling():
    text = "A. Smith was first mentioned. Later, Alexander Smith replied."
    spans = person_spans(text, "A. Smith", "Alexander Smith")
    generator = ScriptedGenerator(link="same")

    PersonPipeline(generator).link(text, spans)

    payload = generator.link_calls[0]
    assert payload["left"]["mention"] == "Alexander Smith"
    assert payload["left"]["start"] == text.index("Alexander Smith")
    assert payload["right"]["mention"] == "A. Smith"
    assert payload["right"]["start"] == text.index("A. Smith")
    assert text[payload["left"]["start"]:payload["left"]["end"]] == "Alexander Smith"
    assert text[payload["right"]["start"]:payload["right"]["end"]] == "A. Smith"


def test_equal_length_pair_preserves_original_order():
    text = "John Smith, then John Smith"
    spans = person_spans(text, "John Smith", "John Smith")
    generator = ScriptedGenerator(link="same")

    PersonPipeline(generator).link(text, spans)

    payload = generator.link_calls[0]
    assert len(payload["left"]["mention"]) == len(payload["right"]["mention"])
    assert payload["left"]["start"] == spans[0].start
    assert payload["right"]["start"] == spans[1].start


def test_ambiguous_initial_is_kept_separate_from_two_possible_people():
    text = "Иван Петров и Игорь Петров. И. Петров направил письмо."
    spans = person_spans(text, "Иван Петров", "Игорь Петров", "И. Петров")
    generator = ScriptedGenerator(link=lambda payload: "unknown")
    records, warnings = PersonPipeline(generator).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 3
    assert warnings


def test_ambiguous_initial_does_not_join_two_aliases_of_one_candidate():
    text = "Иван Петров встретил Ивана Петрова, затем Игоря Петрова. И. Петров ушёл."
    spans = person_spans(
        text, "Иван Петров", "Ивана Петрова", "Игоря Петрова", "И. Петров"
    )

    def relation(payload):
        pair = {payload["left"]["mention"], payload["right"]["mention"]}
        if pair == {"Иван Петров", "Ивана Петрова"}:
            return "same"
        return "unknown"

    records, warnings = PersonPipeline(
        ScriptedGenerator(link=relation)
    ).link(text, spans)

    assert records[spans[0].start].identifier == records[spans[1].start].identifier
    assert records[spans[3].start].identifier != records[spans[0].start].identifier
    assert records[spans[3].start].identifier != records[spans[2].start].identifier
    assert warnings


def test_model_same_links_ambiguous_initial_without_linguistic_filter():
    text = "Иван Петров, Игорь Петров и И. Петров"
    spans = person_spans(text, "Иван Петров", "Игорь Петров", "И. Петров")
    generator = ScriptedGenerator(link="same")
    records, warnings = PersonPipeline(generator).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 1
    assert warnings == []
    assert len(generator.link_calls) == 3


def test_model_same_links_surname_only_mentions_without_linguistic_filter():
    text = "Иван Петров, Игорь Петров, Петров"
    spans = person_spans(text, "Иван Петров", "Игорь Петров", "Петров")
    generator = ScriptedGenerator(link="same")
    records, _ = PersonPipeline(generator).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 1
    assert len(generator.link_calls) == 3


def test_model_same_links_initial_with_different_first_letter():
    text = "Иван Петров, А. Петров"
    spans = person_spans(text, "Иван Петров", "А. Петров")
    generator = ScriptedGenerator(link="same")
    records, _ = PersonPipeline(generator).link(text, spans)

    assert records[spans[0].start].identifier == records[spans[1].start].identifier
    assert len(generator.link_calls) == 1


def test_conflicting_patronymics_are_offered_and_model_same_is_respected():
    text = "Иван Иванович Петров и Иван Петрович Петров"
    spans = person_spans(text, "Иван Иванович Петров", "Иван Петрович Петров")
    generator = ScriptedGenerator(link="same")
    records, _ = PersonPipeline(generator).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 1
    assert len(generator.link_calls) == 1


def test_family_name_first_conflicting_patronymics_are_offered_to_model():
    text = "Петров Иван Иванович и Петров Иван Петрович"
    spans = person_spans(text, "Петров Иван Иванович", "Петров Иван Петрович")
    generator = ScriptedGenerator(link="same")
    records, _ = PersonPipeline(generator).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 1
    assert len(generator.link_calls) == 1


def test_explicit_same_spelling_can_represent_different_people():
    text = "Иван Петров из отдела A; Иван Петров из отдела B"
    spans = person_spans(text, "Иван Петров", "Иван Петров")
    records, warnings = PersonPipeline(
        ScriptedGenerator(link="different")
    ).link(text, spans)

    assert warnings == []
    assert records[spans[0].start].identifier != records[spans[1].start].identifier


def test_all_unordered_person_pairs_reach_model_once_including_initials_surnames_and_repeats():
    text = "Irene Blake; I. Blake; Blake; Irene Blake; Ada Lovelace; 0300 555 0100; 10 Oak Road"
    values = ["Irene Blake", "I. Blake", "Blake", "Irene Blake", "Ada Lovelace"]
    spans = person_spans(text, *values)
    phone_start = text.index("0300 555 0100")
    address_start = text.index("10 Oak Road")
    spans.extend([
        Span(phone_start, phone_start + len("0300 555 0100"), "0300 555 0100", "PHONE", 0.99),
        Span(address_start, address_start + len("10 Oak Road"), "10 Oak Road", "ADDRESS", 0.99),
    ])
    generator = ScriptedGenerator(link="unknown")

    PersonPipeline(generator).link(text, spans)

    expected = {
        tuple(sorted((spans[i].start, spans[j].start)))
        for i in range(len(values)) for j in range(i + 1, len(values))
    }
    actual = [
        tuple(sorted((payload["left"]["start"], payload["right"]["start"])))
        for payload in generator.link_calls
    ]
    assert len(actual) == len(expected)
    assert len(set(actual)) == len(actual)
    assert set(actual) == expected
    assert all(
        payload["document"] == text
        and text[payload["left"]["start"]:payload["left"]["end"]] == payload["left"]["mention"]
        and text[payload["right"]["start"]:payload["right"]["end"]] == payload["right"]["mention"]
        for payload in generator.link_calls
    )


def test_dissimilar_names_are_merged_when_model_says_same():
    text = "Irene Blake and Ada Lovelace"
    spans = person_spans(text, "Irene Blake", "Ada Lovelace")
    generator = ScriptedGenerator(link="same")

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert records[spans[0].start].identifier == records[spans[1].start].identifier
    assert warnings == []
    assert len(generator.link_calls) == 1


def test_canonical_is_longest_exact_observed_unicode_spelling_ties_by_document_offset():
    text = "Elise Müller; Élisabeth Müller; Elisabeth Müller; E. Müller"
    values = ["Elise Müller", "Élisabeth Müller", "Elisabeth Müller", "E. Müller"]
    spans = list(reversed(person_spans(text, *values)))
    records, warnings = PersonPipeline(
        ScriptedGenerator(link="same")
    ).link(text, spans)

    assert warnings == []
    assert {record.canonical for record in records.values()} == {"Élisabeth Müller"}
    assert set(records[spans[0].start].aliases) == set(values)


def test_all_person_mentions_are_uncapped_beyond_previous_64_candidate_limit():
    values = [f"Person{i} Surname{i}" for i in range(66)]
    text = "; ".join(values)
    spans = person_spans(text, *values)
    generator = ScriptedGenerator(link="same")

    records, warnings = PersonPipeline(generator).link(text, spans)

    assert len(generator.link_calls) == 66 * 65 // 2
    assert len(records) == 66
    assert len({record.identifier for record in records.values()}) == 1
    assert warnings == []


@pytest.mark.parametrize("barrier", ["different", "unknown"])
@pytest.mark.parametrize("reverse_spans", [False, True])
def test_conflicting_same_component_is_order_independent(barrier, reverse_spans):
    text = "Irene Jones; Robert Smith; I. Jones"
    spans = person_spans(text, "Irene Jones", "Robert Smith", "I. Jones")
    if reverse_spans:
        spans = list(reversed(spans))
    barrier_pair = frozenset(("Irene Jones", "Robert Smith"))

    def relation(payload):
        pair = frozenset((payload["left"]["mention"], payload["right"]["mention"]))
        return barrier if pair == barrier_pair else "same"

    generator = ScriptedGenerator(link=relation)
    records, warnings = PersonPipeline(generator).link(text, spans)

    by_value = {span.value: records[span.start].identifier for span in spans}
    assert by_value["Irene Jones"] != by_value["Robert Smith"]
    assert len(generator.link_calls) == 3
    assert warnings


def test_unknown_local_link_relation_keeps_mentions_separate_and_warns():
    text = "Иван Петров и Ивана Петрова"
    spans = person_spans(text, "Иван Петров", "Ивана Петрова")
    records, warnings = PersonPipeline(
        ScriptedGenerator(link="unknown")
    ).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 2
    assert warnings


def test_patronymic_variants_are_all_offered_to_model():
    text = "Ивана Петрова Ивановича; Иван Иванович Петров; Иван Петрович Петров"
    values = ["Ивана Петрова Ивановича", "Иван Иванович Петров", "Иван Петрович Петров"]
    generator = ScriptedGenerator(link="unknown")
    pipeline = PersonPipeline(generator)
    pipeline.link(text, person_spans(text, *values))

    assert len(generator.link_calls) == 3
    assert {
        frozenset((payload["left"]["mention"], payload["right"]["mention"]))
        for payload in generator.link_calls
    } == {
        frozenset((values[0], values[1])),
        frozenset((values[0], values[2])),
        frozenset((values[1], values[2])),
    }
