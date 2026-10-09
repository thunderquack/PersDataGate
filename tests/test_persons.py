from __future__ import annotations

from dataclasses import dataclass

import pytest

from pseudonymizer.persons import PersonPipeline
from pseudonymizer.service import PseudonymMapping, Span, restore_text


class ScriptedGenerator:
    def __init__(self, *, link="same", form=None):
        self.link = link
        self.form = form
        self.link_calls = []
        self.form_calls = []

    def choose(self, system, payload):
        if "allowed_relations" in payload:
            self.link_calls.append(payload)
            return {"relation": self.link(payload) if callable(self.link) else self.link}
        self.form_calls.append(payload)
        return {"form": self.form(payload) if callable(self.form) else self.form}


@dataclass
class FakeParse:
    normal_form: str
    forms: dict[str, str]

    @property
    def word(self):
        return self.normal_form

    def inflect(self, tags):
        form = self.forms.get(next(iter(tags)))
        return FakeParse(form, self.forms) if form else None


class FakeMorph:
    """Small deterministic substitute for pymorphy3's parse/inflect API."""

    LEMMAS = {
        "иван": "иван", "ивана": "иван", "ивану": "иван",
        "петров": "петров", "петрова": "петров", "петрову": "петров",
        "иванович": "иванович", "ивановича": "иванович",
        "петрович": "петрович", "петровича": "петрович",
    }
    FORMS = {
        "иван": {"nomn": "Иван", "gent": "Ивана", "datv": "Ивану", "accs": "Ивана", "ablt": "Иваном", "loct": "Иване"},
        "петров": {"nomn": "Петров", "gent": "Петрова", "datv": "Петрову", "accs": "Петрова", "ablt": "Петровым", "loct": "Петрове"},
    }

    def parse(self, token):
        lemma = self.LEMMAS.get(token.casefold(), token.casefold())
        forms = self.FORMS.get(lemma, {"nomn": token})
        return [FakeParse(lemma, forms)]


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
    pipeline = PersonPipeline(generator, morph=FakeMorph())

    records, warnings = pipeline.link(text, spans)

    assert warnings == []
    assert set(records) == {span.start for span in spans}
    assert len({record.identifier for record in records.values()}) == 1
    record = records[spans[0].start]
    assert record.canonical == "Иван Петров"
    assert set(record.aliases) == {"Иван Петров", "Ивана Петрова", "И. Петров"}
    assert generator.link_calls  # distant mentions are still compared globally


def test_ambiguous_initial_is_kept_separate_from_two_possible_people():
    text = "Иван Петров и Игорь Петров. И. Петров направил письмо."
    spans = person_spans(text, "Иван Петров", "Игорь Петров", "И. Петров")
    generator = ScriptedGenerator(link=lambda payload: "unknown")
    records, warnings = PersonPipeline(generator, morph=FakeMorph()).link(text, spans)

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
        ScriptedGenerator(link=relation), morph=FakeMorph()
    ).link(text, spans)

    assert records[spans[0].start].identifier == records[spans[1].start].identifier
    assert records[spans[3].start].identifier != records[spans[0].start].identifier
    assert records[spans[3].start].identifier != records[spans[2].start].identifier
    assert warnings


def test_ambiguous_initial_stays_separate_even_if_generator_says_same():
    text = "Иван Петров, Игорь Петров и И. Петров"
    spans = person_spans(text, "Иван Петров", "Игорь Петров", "И. Петров")
    records, warnings = PersonPipeline(
        ScriptedGenerator(link="same"), morph=FakeMorph()
    ).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 3
    assert warnings


def test_surname_only_ambiguous_between_two_full_names_stays_separate():
    text = "Иван Петров, Игорь Петров, Петров"
    spans = person_spans(text, "Иван Петров", "Игорь Петров", "Петров")
    generator = ScriptedGenerator(link="same")
    records, _ = PersonPipeline(generator, morph=FakeMorph()).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 3
    assert records[spans[0].start].identifier != records[spans[1].start].identifier
    assert records[spans[2].start].identifier not in {
        records[spans[0].start].identifier, records[spans[1].start].identifier
    }


def test_initial_with_wrong_first_letter_does_not_link_to_full_name():
    text = "Иван Петров, А. Петров"
    spans = person_spans(text, "Иван Петров", "А. Петров")
    generator = ScriptedGenerator(link="same")
    records, _ = PersonPipeline(generator, morph=FakeMorph()).link(text, spans)

    assert records[spans[0].start].identifier != records[spans[1].start].identifier
    assert generator.link_calls == []


def test_conflicting_patronymics_are_never_offered_as_link_candidates():
    text = "Иван Иванович Петров и Иван Петрович Петров"
    spans = person_spans(text, "Иван Иванович Петров", "Иван Петрович Петров")
    generator = ScriptedGenerator(link="same")
    records, _ = PersonPipeline(generator, morph=FakeMorph()).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 2
    assert generator.link_calls == []


def test_conflicting_patronymics_are_checked_in_family_name_first_order_too():
    text = "Петров Иван Иванович и Петров Иван Петрович"
    spans = person_spans(text, "Петров Иван Иванович", "Петров Иван Петрович")
    generator = ScriptedGenerator(link="same")
    records, _ = PersonPipeline(generator, morph=FakeMorph()).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 2
    assert generator.link_calls == []


def test_explicit_same_spelling_can_represent_different_people():
    text = "Иван Петров из отдела A; Иван Петров из отдела B"
    spans = person_spans(text, "Иван Петров", "Иван Петров")
    records, warnings = PersonPipeline(
        ScriptedGenerator(link="different"), morph=FakeMorph()
    ).link(text, spans)

    assert warnings == []
    assert records[spans[0].start].identifier != records[spans[1].start].identifier


def test_conflicting_pair_prevents_a_transitive_same_relation_from_merging():
    text = "Иван Петров, Ивана Петрова, И. Петров"
    spans = person_spans(text, "Иван Петров", "Ивана Петрова", "И. Петров")

    def relation(payload):
        pair = {payload["left"]["mention"], payload["right"]["mention"]}
        return "different" if pair == {"Ивана Петрова", "И. Петров"} else "same"

    records, warnings = PersonPipeline(
        ScriptedGenerator(link=relation), morph=FakeMorph()
    ).link(text, spans)

    assert records[spans[0].start].identifier == records[spans[1].start].identifier
    assert records[spans[2].start].identifier != records[spans[0].start].identifier
    assert warnings


@pytest.mark.parametrize(
    ("context", "selected"),
    [
        ("Позвонил ⟦PERSON_x_1⟧", "Иван Петров"),
        ("Документ передан ⟦PERSON_x_1⟧", "Ивану Петрову"),
    ],
)
def test_russian_restoration_selects_only_supplied_finite_form(context, selected):
    canonical = "Иван Петров"
    generator = ScriptedGenerator(form=selected)
    pipeline = PersonPipeline(generator, morph=FakeMorph())
    record = pipeline.link(canonical, person_spans(canonical, canonical))[0][0]
    marker = "⟦PERSON_x_1⟧"
    start = context.index(marker)

    restored = pipeline.restore(context, marker, record, start, start + len(marker))

    assert restored == selected
    assert selected in generator.form_calls[0]["allowed_forms"]


def test_real_morphology_normalizes_ru_case_without_losing_full_identity():
    pytest.importorskip("pymorphy3")
    pipeline = PersonPipeline(generator=object())

    assert pipeline._canonical(["Иван Петров", "Ивана Петрова"]) == "Иван Петров"
    assert pipeline._canonical(["Анна Петрова", "Анне Петровой"]) == "Анна Петрова"


def test_real_morphology_does_not_change_female_surname_gender_in_forms():
    pytest.importorskip("pymorphy3")
    options = PersonPipeline(generator=object())._forms("Анна Петрова")

    assert "Анна Петрова" in options
    assert "Анна Петров" not in options


def test_invalid_or_uncertain_form_leaves_marker_for_caller_warning():
    canonical = "Иван Петров"
    pipeline = PersonPipeline(ScriptedGenerator(form="Неизвестное Имя"), morph=FakeMorph())
    record = pipeline.link(canonical, person_spans(canonical, canonical))[0][0]
    marker = "⟦PERSON_x_1⟧"
    assert pipeline.restore(marker, marker, record, 0, len(marker)) is None


def test_german_person_form_allows_only_canonical_or_finite_possessive():
    canonical = "Max Müller"
    pipeline = PersonPipeline(ScriptedGenerator(form="Max Müllers"))
    record = pipeline.link(canonical, person_spans(canonical, canonical))[0][0]
    marker = "⟦PERSON_x_1⟧"

    assert pipeline.restore(marker, marker, record, 0, len(marker)) == "Max Müllers"


def test_repeated_person_marker_is_inflected_separately_for_each_occurrence():
    canonical = "Иван Петров"
    record = PersonPipeline(ScriptedGenerator(), morph=FakeMorph()).link(
        canonical, person_spans(canonical, canonical)
    )[0][0]
    marker = "⟦PERSON_x_1⟧"
    source = f"Позвонил {marker}. {'текст ' * 110}Документ передан {marker}."

    def choose_form(payload):
        if "передан" in payload["before"].lower():
            return "Ивану Петрову"
        return "Иван Петров"

    pipeline = PersonPipeline(ScriptedGenerator(form=choose_form), morph=FakeMorph())
    mapping = PseudonymMapping()
    mapping[marker] = canonical
    mapping.persons[marker] = record
    restored, warnings = restore_text(source, mapping, pipeline)

    assert restored == f"Позвонил Иван Петров. {'текст ' * 110}Документ передан Ивану Петрову."
    assert warnings == []
    assert len(pipeline.generator.form_calls) == 2
    first_payload, second_payload = pipeline.generator.form_calls
    assert "Позвонил" in first_payload["before"]
    assert "Документ передан" not in first_payload["before"] + first_payload["after"]
    assert "Документ передан" in second_payload["before"]
    assert "Позвонил" not in second_payload["before"] + second_payload["after"]
    for payload in (first_payload, second_payload):
        assert payload["target_start"] == len(payload["before"])
        assert payload["target_end"] - payload["target_start"] == len(marker)


def test_unknown_local_link_relation_keeps_mentions_separate_and_warns():
    text = "Иван Петров и Ивана Петрова"
    spans = person_spans(text, "Иван Петров", "Ивана Петрова")
    records, warnings = PersonPipeline(
        ScriptedGenerator(link="unknown"), morph=FakeMorph()
    ).link(text, spans)

    assert len({record.identifier for record in records.values()}) == 2
    assert warnings


def test_adjacent_identical_markers_have_distinct_target_contexts():
    from pseudonymizer.persons import PersonRecord

    marker = "⟦PERSON_x_1⟧"
    text = f"Нет {marker}. Передано {marker}."
    generator = ScriptedGenerator(form=lambda payload:
        "Ивана Петрова" if payload["before"] == "Нет " else "Ивану Петрову")
    pipeline = PersonPipeline(generator, morph=FakeMorph())
    mapping = PseudonymMapping()
    mapping[marker] = "Иван Петров"
    mapping.persons[marker] = PersonRecord("one", "Иван Петров", ("Иван Петров",))
    result, warnings = restore_text(text, mapping, pipeline)
    assert result == "Нет Ивана Петрова. Передано Ивану Петрову."
    assert warnings == []
    assert generator.form_calls[0]["target_start"] != generator.form_calls[1]["target_start"]


def test_declined_patronymic_order_and_full_alias_preference():
    pytest.importorskip("pymorphy3")
    pipeline = PersonPipeline(ScriptedGenerator())
    assert pipeline._candidate("Ивана Петрова Ивановича", "Иван Иванович Петров")
    assert not pipeline._candidate("Ивана Петрова Ивановича", "Иван Петрович Петров")
    assert pipeline._canonical(["И. И. Петров", "Иван Петров"]) == "Иван Петров"
