"""Conservative local linking and contextual restoration for person mentions.

The language model used here is deliberately a selector: it can link two
observed mentions or choose among forms generated locally, but it cannot add
names or rewrite response text.
"""

from __future__ import annotations

import itertools
import re
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from common.pipeline_log import PipelineLogError, event


class LocalGenerator(Protocol):
    def choose(self, system: str, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PersonRecord:
    identifier: str
    canonical: str
    aliases: tuple[str, ...]


_TOKEN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*\.?", re.UNICODE)
_INITIAL = re.compile(r"^[A-Za-zА-Яа-яЁё]$")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
_ADDRESS_MARKER_LINE = re.compile(r"^\s*[⟦\[]ADDRESS_[^⟧\]]+[⟧\]]\s*$")
_GERMAN_STREET_LINE = re.compile(
    r"^[A-ZÄÖÜ][\wÄÖÜäöüß.'’\-]*(?:\s+[\wÄÖÜäöüß.'’\-]+){0,4}"
    r"\s+\d+[a-zA-Z]?(?:\s*[-/]\s*\d+[a-zA-Z]?)?$"
)
_GERMAN_POSTAL_CITY_LINE = re.compile(r"^\s*\d{5}\s+[^\W\d_][^\r\n]*$", re.UNICODE)
_CASE_OPTIONS = ("nomn", "gent", "datv", "accs", "ablt", "loct")
_SYSTEM_LINK = (
    "You link observed person-name mentions in a document. Return one JSON object "
    'with relation exactly "same", "different", or "unknown". Use same only when '
    "the context supports identity. Names with different explicit given names or "
    "patronymics are different. An initial that could refer to multiple people is "
    "unknown. Do not infer names or use outside knowledge."
)
_SYSTEM_FORM = (
    "Choose the grammatically appropriate form of this already-known person name "
    "for the target marker occurrence identified by its offsets. Use only that "
    "occurrence's surrounding context. Return JSON with form equal to exactly one supplied "
    "option, or null if uncertain. Never invent, edit, or translate a name."
)


def _tokens(value: str) -> list[str]:
    return [token.rstrip(".") for token in _TOKEN.findall(value)]


def _is_initial(token: str) -> bool:
    return len(token) == 1 and bool(_INITIAL.fullmatch(token))


class PersonPipeline:
    """Link PERSON spans and select safe inflections from finite local options."""

    def __init__(self, generator: LocalGenerator, morph: Any | None = None):
        self.generator = generator
        self._morph = morph
        self._morph_loaded = morph is not None

    def _get_morph(self) -> Any | None:
        if not self._morph_loaded:
            self._morph_loaded = True
            try:
                import pymorphy3

                self._morph = pymorphy3.MorphAnalyzer()
            except ImportError:
                self._morph = None
        return self._morph

    @staticmethod
    def _match_case(value: str, source: str) -> str:
        if source.isupper():
            return value.upper()
        if source[:1].isupper():
            return value[:1].upper() + value[1:]
        return value

    def _lemmas(self, token: str) -> set[str]:
        morph = self._get_morph()
        if morph is None or not _CYRILLIC.search(token):
            return {token.casefold()}
        try:
            parses = morph.parse(token)
            if parses:
                return {str(parse.normal_form).casefold() for parse in parses}
        except (AttributeError, TypeError, ValueError):
            pass
        return {token.casefold()}

    def _patronymic(self, token: str) -> bool:
        value = token.casefold()
        suffixes = ("ович", "евич", "ич", "овна", "евна", "ична", "инична")
        if value.endswith(suffixes):
            return True
        morph = self._get_morph()
        if morph is None:
            return False
        try:
            return any(
                "Patr" in self._grammemes(parse) or
                str(parse.normal_form).casefold().endswith(suffixes)
                for parse in morph.parse(token)
            )
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _grammemes(parse: Any) -> set[str]:
        tag = getattr(parse, "tag", None)
        return set(getattr(tag, "grammemes", ()))

    def _roles(self, tokens: list[str]) -> list[str | None]:
        if len(tokens) < 2 or not all(_CYRILLIC.search(token) for token in tokens):
            return [None] * len(tokens)
        roles: list[str | None] = ["Name"] + [None] * (len(tokens) - 1)
        if len(tokens) == 2:
            roles[1] = "Surn"
        elif self._patronymic(tokens[1]):
            roles[1] = "Patr"
            roles[-1] = "Surn"
        elif self._patronymic(tokens[-1]):
            roles[1] = "Surn"
            roles[-1] = "Patr"
        else:
            roles[1] = "Patr"
            roles[-1] = "Surn"
        return roles

    def _role_parses(self, token: str, role: str | None,
                     gender: str | None = None) -> list[Any]:
        morph = self._get_morph()
        if morph is None:
            return []
        parses = list(morph.parse(token))
        if role:
            role_parses = [parse for parse in parses if role in self._grammemes(parse)]
            if role_parses:
                parses = role_parses
            elif any(self._grammemes(parse) & {"Name", "Surn", "Patr"} for parse in parses):
                return []
        if gender:
            gender_parses = [parse for parse in parses
                             if not getattr(getattr(parse, "tag", None), "gender", None)
                             or getattr(parse.tag, "gender", None) == gender]
            if gender_parses:
                parses = gender_parses
            elif any(getattr(getattr(parse, "tag", None), "gender", None)
                     for parse in parses):
                return []
        return parses

    def _nominative_forms(self, token: str, role: str | None,
                          gender: str | None) -> set[str]:
        try:
            return {parse.inflect({"nomn"}).word for parse in self._role_parses(token, role, gender)
                    if parse.inflect({"nomn"}) is not None}
        except (AttributeError, TypeError, ValueError):
            return set()

    def _parts(self, value: str) -> tuple[set[str], set[str] | None, set[str] | None, bool]:
        """Return given-name forms, surname forms, patronymic forms, initial flag."""
        tokens = _tokens(value)
        if not tokens:
            return set(), None, None, False
        initial = _is_initial(tokens[0])
        given = {tokens[0][0].casefold()} if initial else self._lemmas(tokens[0])
        if len(tokens) == 1:
            # Surname-only mentions are candidates but never sufficient by
            # themselves to establish identity.
            return given, self._lemmas(tokens[0]), None, initial
        surname_index = len(tokens) - 1
        patronymic_index: int | None = None
        if len(tokens) >= 3:
            if self._patronymic(tokens[1]):
                patronymic_index = 1
            elif self._patronymic(tokens[-1]):
                patronymic_index = len(tokens) - 1
                surname_index = 1
            else:
                # Russian full-name order is normally given, patronymic,
                # surname; this fallback handles names without recognizable
                # patronymic endings conservatively.
                patronymic_index = 1
        surname = self._lemmas(tokens[surname_index])
        patronymic = self._lemmas(tokens[patronymic_index]) if patronymic_index is not None else None
        return given, surname, patronymic, initial

    def _candidate(self, left: str, right: str) -> bool:
        given_a, surname_a, patronymic_a, initial_a = self._parts(left)
        given_b, surname_b, patronymic_b, initial_b = self._parts(right)
        if not given_a or not given_b or not surname_a or not surname_b:
            return False
        if surname_a.isdisjoint(surname_b):
            return False
        raw_a, raw_b = _tokens(left), _tokens(right)
        short_a, short_b = len(raw_a) == 1, len(raw_b) == 1
        if not short_a and not short_b and not initial_a and not initial_b and given_a.isdisjoint(given_b):
            return False
        if (patronymic_a and patronymic_b and patronymic_a.isdisjoint(patronymic_b)):
            return False
        # A one-token surname is compared to the family-name component. A
        # one-letter initial must match the first letter of the explicit name.
        for raw, initial, expanded, other_short in (
                (left, initial_a, given_b, short_b), (right, initial_b, given_a, short_a)):
            first = _tokens(raw)[0]
            if initial and not other_short and not all(
                    value.startswith(first[0].casefold()) for value in expanded):
                return False
        return True

    def _compatible_cluster_names(self, left: str, right: str) -> bool:
        given_a, surname_a, patronymic_a, initial_a = self._parts(left)
        given_b, surname_b, patronymic_b, initial_b = self._parts(right)
        if not surname_a or not surname_b or surname_a.isdisjoint(surname_b):
            return False
        raw_a, raw_b = _tokens(left), _tokens(right)
        short_a, short_b = len(raw_a) == 1, len(raw_b) == 1
        if not short_a and not short_b and not initial_a and not initial_b and given_a.isdisjoint(given_b):
            return False
        if patronymic_a and patronymic_b and patronymic_a.isdisjoint(patronymic_b):
            return False
        return True

    def _candidate_pairs(self, mentions: list[Any], limit: int = 64) -> tuple[set[tuple[int, int]], set[int]]:
        """Build a bounded surname/initial index instead of comparing all pairs."""
        buckets: dict[tuple[str, str], list[int]] = {}
        surname_buckets: dict[str, list[int]] = {}
        part_cache: list[tuple[set[str], set[str] | None, set[str] | None, bool]] = []
        surname_only: list[bool] = []
        for index, mention in enumerate(mentions):
            parts = self._parts(mention.value)
            part_cache.append(parts)
            tokens = _tokens(mention.value)
            only_surname = len(tokens) == 1
            surname_only.append(only_surname)
            given, surnames, _, initial = parts
            if not surnames:
                continue
            initials = ({tokens[0][0].casefold()} if initial else
                        {value[0] for value in given if value})
            for surname in surnames:
                surname_buckets.setdefault(surname, []).append(index)
                for first in initials:
                    bucket = buckets.setdefault((surname, first), [])
                    if not bucket or bucket[-1] != index:
                        bucket.append(index)

        pairs: set[tuple[int, int]] = set()
        overflow: set[int] = set()
        for index, mention in enumerate(mentions):
            given, surnames, _, initial = part_cache[index]
            if not surnames:
                continue
            tokens = _tokens(mention.value)
            only_surname = surname_only[index]
            initials = ({tokens[0][0].casefold()} if initial else
                        {value[0] for value in given if value})
            candidates: set[int] = set()
            capped = False

            def add_limited(values: list[int]) -> None:
                nonlocal capped
                for value in values:
                    candidates.add(value)
                    if len(candidates) > limit:
                        capped = True
                        return

            for surname in surnames:
                if only_surname:
                    add_limited(surname_buckets.get(surname, []))
                else:
                    for first in initials:
                        add_limited(buckets.get((surname, first), []))
                        if capped:
                            break
                    # Single-letter mentions look up every given-name initial
                    # sharing their surname; their ambiguity is assessed below.
                    if initial and not capped:
                        add_limited(surname_buckets.get(surname, []))
                if capped:
                    break
            candidates.discard(index)
            accepted: list[int] = []
            for other in sorted(candidates, key=lambda value: (abs(value - index), value)):
                if self._candidate(mention.value, mentions[other].value):
                    accepted.append(other)
            if capped or len(accepted) > limit:
                overflow.add(index)
                accepted = accepted[:limit]
            for other in accepted:
                pairs.add((min(index, other), max(index, other)))
        return pairs, overflow

    def link(self, text: str, spans: list[Any]) -> tuple[dict[int, PersonRecord], list[str]]:
        mentions = [span for span in spans if getattr(span, "code", None) == "PERSON"]
        if not mentions:
            return {}, []

        # Union-find over mentions; only blocked candidates are sent to the model.
        parent = list(range(len(mentions)))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> bool:
            ra, rb = root(a), root(b)
            if ra != rb:
                members_a = [index for index in range(len(mentions)) if root(index) == ra]
                members_b = [index for index in range(len(mentions)) if root(index) == rb]
                for index_a in members_a:
                    for index_b in members_b:
                        if not self._compatible_cluster_names(
                                mentions[index_a].value, mentions[index_b].value):
                            return False
                parent[rb] = ra
            return True

        warnings: list[str] = []
        candidate_pairs, overflow = self._candidate_pairs(mentions)
        event("person.candidates", {
            "mentions": [{"value": mention.value, "start": mention.start,
                          "end": mention.end} for mention in mentions],
            "candidate_pairs": [
                {"left": mentions[i].value, "left_start": mentions[i].start,
                 "right": mentions[j].value, "right_start": mentions[j].start}
                for i, j in sorted(candidate_pairs)
            ],
            "overflow_indices": sorted(overflow),
        })
        if overflow:
            warnings.append("Some person mentions had too many candidates and were kept separate.")
        cannot_link: set[tuple[int, int]] = set()

        def full_name(index: int) -> bool:
            tokens = _tokens(mentions[index].value)
            return len(tokens) >= 2 and not _is_initial(tokens[0])

        def decide(i: int, j: int) -> str:
            left, right = mentions[i], mentions[j]
            left_window_start = max(0, left.start - 160)
            left_window_end = min(len(text), left.end + 160)
            right_window_start = max(0, right.start - 160)
            right_window_end = min(len(text), right.end + 160)
            payload = {
                "left": {
                    "mention": left.value,
                    "before": text[left_window_start:left.start],
                    "after": text[left.end:left_window_end],
                    "start": left.start,
                    "end": left.end,
                },
                "right": {
                    "mention": right.value,
                    "before": text[right_window_start:right.start],
                    "after": text[right.end:right_window_end],
                    "start": right.start,
                    "end": right.end,
                },
                "allowed_relations": ["same", "different", "unknown"],
            }
            # A local-model failure blocks pseudonymization and must not allow
            # the caller to proceed to OpenAI with unresolved identities.
            try:
                result = self.generator.choose(_SYSTEM_LINK, payload)
            except PipelineLogError:
                raise
            except Exception as exc:
                event("person.link_decision", {
                    "left": payload["left"], "right": payload["right"],
                    "decision": "error", "error_type": type(exc).__name__,
                })
                raise
            relation = result.get("relation") if isinstance(result, dict) else None
            decision = relation if relation in ("same", "different", "unknown") else "invalid"
            event("person.link_decision", {
                "left": payload["left"], "right": payload["right"],
                "decision": decision, "model_result": result,
            })
            return decision

        def merge_same_pairs(same_pairs: list[tuple[int, int]]) -> None:
            for i, j in same_pairs:
                left_root, right_root = root(i), root(j)
                if left_root == right_root:
                    continue
                blocked = any((min(a, b), max(a, b)) in cannot_link
                              for a in range(len(mentions)) if root(a) == left_root
                              for b in range(len(mentions)) if root(b) == right_root)
                if blocked:
                    event("person.link_conflict", {
                        "left": mentions[i].value, "right": mentions[j].value,
                        "reason": "contradicts_prior_different_or_unknown_decision",
                    })
                    warnings.append("Conflicting person identities were kept separate.")
                elif not union(i, j):
                    event("person.link_conflict", {
                        "left": mentions[i].value, "right": mentions[j].value,
                        "reason": "incompatible_name_components",
                    })
                    warnings.append("Conflicting person-name details were kept separate.")

        def store_decision(i: int, j: int, relation: str,
                           same_pairs: list[tuple[int, int]]) -> None:
            pair = (min(i, j), max(i, j))
            if relation == "same":
                same_pairs.append(pair)
            else:
                # Both explicit-different and uncertain pairs constrain later
                # transitive merges; an unknown decision must not become a link
                # through a third mention.
                cannot_link.add(pair)
            if relation == "invalid":
                warnings.append("Person mentions could not be linked with confidence.")
            elif relation == "unknown":
                warnings.append("A person-name mention was ambiguous and kept separate.")

        # Establish complete-name clusters before considering initials and
        # surname-only mentions, so a short form cannot attach to Ivan first and
        # then bridge Ivan to Igor through a shared initial.
        ordered_pairs = sorted(candidate_pairs)
        full_same: list[tuple[int, int]] = []
        for i, j in ordered_pairs:
            if full_name(i) and full_name(j) and i not in overflow and j not in overflow:
                store_decision(i, j, decide(i, j), full_same)
            elif full_name(i) and full_name(j):
                event("person.link_decision", {
                    "left": {"mention": mentions[i].value, "start": mentions[i].start},
                    "right": {"mention": mentions[j].value, "start": mentions[j].start},
                    "decision": "kept_separate", "reason": "candidate_overflow",
                })
        merge_same_pairs(full_same)

        anchored_roots: dict[int, set[int]] = {}
        quarantined: set[int] = set(overflow)
        for index in range(len(mentions)):
            if full_name(index):
                continue
            anchors = {
                root(other) for other in range(len(mentions))
                if full_name(other) and other not in overflow and
                (min(index, other), max(index, other)) in candidate_pairs
            }
            anchored_roots[index] = anchors
            if len(anchors) > 1 or index in overflow:
                quarantined.add(index)
                event("person.link_decision", {
                    "mention": mentions[index].value,
                    "start": mentions[index].start,
                    "decision": "kept_separate",
                    "reason": "ambiguous_abbreviation" if len(anchors) > 1 else "candidate_overflow",
                    "anchor_mentions": [
                        {"value": mentions[other].value, "start": mentions[other].start}
                        for other in range(len(mentions)) if root(other) in anchors
                    ],
                })
                warnings.append("An ambiguous abbreviated person-name mention matched multiple people and was kept separate.")

        short_same: list[tuple[int, int]] = []
        for i, j in ordered_pairs:
            if full_name(i) and full_name(j):
                continue
            if i in quarantined or j in quarantined:
                event("person.link_decision", {
                    "left": {"mention": mentions[i].value, "start": mentions[i].start},
                    "right": {"mention": mentions[j].value, "start": mentions[j].start},
                    "decision": "kept_separate", "reason": "ambiguous_abbreviation",
                })
                continue
            roots_i, roots_j = anchored_roots.get(i, set()), anchored_roots.get(j, set())
            if full_name(i) and roots_j and root(i) not in roots_j:
                event("person.link_decision", {
                    "left": {"mention": mentions[i].value, "start": mentions[i].start},
                    "right": {"mention": mentions[j].value, "start": mentions[j].start},
                    "decision": "kept_separate", "reason": "not_linked_to_candidate_anchor",
                })
                continue
            if full_name(j) and roots_i and root(j) not in roots_i:
                event("person.link_decision", {
                    "left": {"mention": mentions[i].value, "start": mentions[i].start},
                    "right": {"mention": mentions[j].value, "start": mentions[j].start},
                    "decision": "kept_separate", "reason": "not_linked_to_candidate_anchor",
                })
                continue
            # Two abbreviated mentions tied to different complete identities
            # cannot act as a bridge between those identities.
            if roots_i and roots_j and roots_i.isdisjoint(roots_j):
                event("person.link_decision", {
                    "left": {"mention": mentions[i].value, "start": mentions[i].start},
                    "right": {"mention": mentions[j].value, "start": mentions[j].start},
                    "decision": "kept_separate", "reason": "different_candidate_anchors",
                })
                continue
            store_decision(i, j, decide(i, j), short_same)
        merge_same_pairs(short_same)

        groups: dict[int, list[int]] = {}
        for index in range(len(mentions)):
            groups.setdefault(root(index), []).append(index)

        records_by_start: dict[int, PersonRecord] = {}
        logged_groups: list[dict[str, Any]] = []
        for members in groups.values():
            observed = [mentions[index].value for index in members]
            canonical = self._canonical(observed)
            record = PersonRecord(
                identifier=f"person_{uuid.uuid4().hex}",
                canonical=canonical,
                aliases=tuple(dict.fromkeys(observed)),
            )
            logged_groups.append({
                "mention_values": observed,
                "mention_offsets": [{"start": mentions[index].start,
                                     "end": mentions[index].end} for index in members],
                "canonical": record.canonical,
                "aliases": list(record.aliases),
                "identifier": record.identifier,
            })
            for index in members:
                records_by_start[mentions[index].start] = record
        event("person.groups", {"groups": logged_groups})
        return records_by_start, list(dict.fromkeys(warnings))

    def _canonical(self, observed: list[str]) -> str:
        # Prefer an observed full nominative form when one is available.
        selected = max(observed, key=lambda value: (
            not any(_is_initial(token) for token in _tokens(value)),
            len(_tokens(value)), self._nominative_score(value), len(value)))
        tokens = _tokens(selected)
        if (not tokens or any(_is_initial(token) for token in tokens) or
                not _CYRILLIC.search(selected)):
            return selected
        morph = self._get_morph()
        if morph is None:
            return selected
        roles = self._roles(tokens)
        gender = self._given_gender(tokens[0])
        normalized: list[str] = []
        for token, role in zip(tokens, roles):
            if _is_initial(token):
                normalized.append(token)
                continue
            try:
                forms = self._nominative_forms(token, role, gender)
                # Do not guess when the analyzer permits multiple name readings.
                nominative = next(iter(forms)) if len(forms) == 1 else token
                normalized.append(self._match_case(nominative, token))
            except (AttributeError, TypeError, ValueError):
                normalized.append(token)
        return " ".join(normalized)

    def _nominative_score(self, name: str) -> int:
        morph = self._get_morph()
        if morph is None:
            return 0
        tokens = _tokens(name)
        roles = self._roles(tokens)
        gender = self._given_gender(tokens[0]) if tokens else None
        score = 0
        for token, role in zip(tokens, roles):
            if _is_initial(token):
                continue
            try:
                forms = self._nominative_forms(token, role, gender)
                if len(forms) == 1 and next(iter(forms)).casefold() == token.casefold():
                    score += 1
            except (AttributeError, TypeError, ValueError):
                continue
        return score

    def restore(self, text: str, marker: str, record: PersonRecord,
                start: int, end: int) -> str | None:
        """Return a selected finite form for one marker occurrence, or None."""
        return self.restore_with_details(text, marker, record, start, end)[0]

    @staticmethod
    def _is_german_recipient_address(text: str, marker: str,
                                     start: int, end: int) -> bool:
        """Recognize a name in a compact German recipient-address block."""
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end < 0:
            line_end = len(text)
        target_line = text[line_start:line_end].strip(" \t\r")
        if target_line != marker:
            return False

        prior_lines = text[:line_start].splitlines()
        if not prior_lines or not re.fullmatch(r"An\s*:?(?:\s+(?:den|die|das|dem|der))?", prior_lines[-1].strip(), re.IGNORECASE):
            return False

        following = [line for line in text[line_end + 1:line_end + 801].splitlines()
                     if line.strip()]
        # The recipient's address may itself be pseudonymized or remain visible.
        if len(following) < 2 or not (
                _ADDRESS_MARKER_LINE.fullmatch(following[0])
                or _GERMAN_STREET_LINE.fullmatch(following[0].strip())):
            return False
        return bool(_GERMAN_POSTAL_CITY_LINE.fullmatch(following[1]))

    def restore_with_details(self, text: str, marker: str, record: PersonRecord,
                             start: int, end: int) -> tuple[str | None, str]:
        """Restore one occurrence and explain the conservative decision."""
        canonical = record.canonical
        # In German correspondence the recipient line is conventionally the
        # uninflected nominative. Restrict this shortcut to a clear address
        # block and a Latin canonical name.
        if (_LATIN.search(canonical) and not _CYRILLIC.search(canonical)
                and self._is_german_recipient_address(text, marker, start, end)):
            event("restore.choice", {
                "marker": marker, "canonical": canonical, "options": [canonical],
                "start": start, "end": end, "chosen": canonical,
                "reason": "recipient_address",
            })
            return canonical, "recipient_address"

        options = self._forms(record.canonical)
        if not options:
            event("restore.choice", {
                "marker": marker, "canonical": canonical, "options": [],
                "start": start, "end": end, "chosen": None, "reason": "no_forms",
            })
            return None, "no_forms"
        window_start = max(0, start - 180)
        window_end = min(len(text), end + 180)
        payload = {
            "marker": marker,
            "canonical": record.canonical,
            "before": text[window_start:start],
            "after": text[end:window_end],
            "target_start": start - window_start,
            "target_end": end - window_start,
            "context_start": window_start,
            "allowed_forms": options,
        }
        event("restore.options", {"marker": marker, "canonical": canonical,
                                   "options": options, "payload": payload})
        try:
            result = self.generator.choose(_SYSTEM_FORM, payload)
        except PipelineLogError:
            raise
        except Exception as exc:
            event("restore.choice", {
                "marker": marker, "canonical": canonical, "options": options,
                "start": start, "end": end, "chosen": None,
                "reason": "local_model_error", "error_type": type(exc).__name__,
            })
            return None, "local_model_error"
        if not isinstance(result, dict) or "form" not in result:
            event("restore.choice", {
                "marker": marker, "canonical": canonical, "options": options,
                "start": start, "end": end, "model_result": result,
                "chosen": None, "reason": "invalid_model_choice",
            })
            return None, "invalid_model_choice"
        form = result["form"]
        if form is None:
            event("restore.choice", {
                "marker": marker, "canonical": canonical, "options": options,
                "start": start, "end": end, "model_result": result,
                "chosen": None, "reason": "model_uncertain",
            })
            return None, "model_uncertain"
        if not isinstance(form, str) or form not in options:
            event("restore.choice", {
                "marker": marker, "canonical": canonical, "options": options,
                "start": start, "end": end, "model_result": result,
                "chosen": None, "reason": "invalid_model_choice",
            })
            return None, "invalid_model_choice"
        event("restore.choice", {
            "marker": marker, "canonical": canonical, "options": options,
            "start": start, "end": end, "model_result": result,
            "chosen": form, "reason": "model_selected",
        })
        return form, "model_selected"

    def _forms(self, canonical: str) -> list[str]:
        tokens = _tokens(canonical)
        if not tokens:
            return []
        forms = {canonical}
        if _CYRILLIC.search(canonical):
            morph = self._get_morph()
            if morph is None:
                return []
            if len(tokens) == 1:
                return [canonical]
            roles = self._roles(tokens)
            gender = self._given_gender(tokens[0])
            # Full Cartesian combinations make implausible mixed cases; use only
            # forms where all name components share a grammatical case. Preserve
            # alternative pymorphy parses, with a strict cap on generated forms.
            for case in _CASE_OPTIONS:
                token_choices: list[list[str]] = []
                complete_case = True
                for token, role in zip(tokens, roles):
                    if _is_initial(token):
                        token_choices.append([token + "."])
                        continue
                    choices: set[str] = set()
                    try:
                        for parse in self._role_parses(token, role, gender):
                            inflected = parse.inflect({case})
                            if inflected is not None:
                                choices.add(self._match_case(inflected.word, token))
                    except (AttributeError, TypeError, ValueError):
                        pass
                    if not choices:
                        complete_case = False
                        break
                    token_choices.append(sorted(choices))
                if not complete_case:
                    continue
                for combination in itertools.islice(itertools.product(*token_choices), 64):
                    forms.add(" ".join(combination))
        elif _LATIN.search(canonical):
            last = tokens[-1]
            if last[-1:].casefold() in {"s", "x", "z"}:
                forms.add(canonical + "’")
            else:
                forms.add(canonical + "s")
        return sorted(forms)

    def _given_gender(self, given_name: str) -> str | None:
        morph = self._get_morph()
        if morph is None:
            return None
        try:
            parses = [parse for parse in morph.parse(given_name)
                      if "Name" in self._grammemes(parse)]
            nominative = [parse for parse in parses
                          if "nomn" in self._grammemes(parse)]
            if nominative:
                parses = nominative
            genders = {getattr(parse.tag, "gender", None) for parse in parses}
            genders.discard(None)
            return next(iter(genders)) if len(genders) == 1 else None
        except (AttributeError, TypeError, ValueError):
            return None
