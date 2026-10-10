from __future__ import annotations

import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from common.pipeline_log import event

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
MAX_TEXT_LENGTH = 50_000
LABELS = [
    "person", "street address", "date of birth", "passport number",
    "identity document number", "bank account number",
]
MODEL_INFERENCE_THRESHOLD = 0.10
PERSON_THRESHOLD = 0.10
OTHER_ENTITY_THRESHOLD = 0.45
ACCEPTED_THRESHOLDS = {
    "PERSON": PERSON_THRESHOLD,
    "ADDRESS": OTHER_ENTITY_THRESHOLD,
    "DOB": OTHER_ENTITY_THRESHOLD,
    "PASSPORT": OTHER_ENTITY_THRESHOLD,
    "ID": OTHER_ENTITY_THRESHOLD,
    "BANK_ACCOUNT": OTHER_ENTITY_THRESHOLD,
    "EMAIL": OTHER_ENTITY_THRESHOLD,
    "PHONE": OTHER_ENTITY_THRESHOLD,
}

# Format based detectors have precedence over model spans.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)")),
    ("IBAN", re.compile(r"(?<![A-Z0-9])(?P<value>[A-Z]{2}\d{2}(?: ?[A-Z0-9]){10,32})", re.I)),
    ("PHONE", re.compile(r"(?i)\b(?:phone|telefon|телефон)\s*[:№]?\s*(?P<value>(?:\+|00)?\d[\d ()/.-]{5,}\d)|(?<!\w)(?P<international>\+\d[\d ()/.-]{5,}\d)(?!\w)")),
    ("PASSPORT", re.compile(r"(?i)\bpassport\s+(?:no\.?|number)\s+(?P<value>[\w-]{5,14})|\bпаспорт\s+(?:№|номер)\s*(?P<value_ru>[\w-]{5,14})|\bReisepassnummer\s*[:№]?\s*(?P<value_de>[\w-]{5,14})")),
    ("ID", re.compile(r"(?i)\b(?:id|identity|document)\s+(?:no\.?|number)\s+(?P<value>[\w-]{5,20})|\b(?:номер документа|идентификационный номер)\s+(?P<value_ru>[\w-]{5,20})|\b(?:Ausweisnummer|Dokumentennummer)\s*[:№]?\s*(?P<value_de>[\w-]{5,20})")),
)
IBAN_LENGTHS = {
    "AL": 28, "AD": 24, "AT": 20, "AZ": 28, "BH": 22, "BY": 28,
    "BE": 16, "BA": 20, "BR": 29, "BG": 22, "CR": 22, "HR": 21,
    "CY": 28, "CZ": 24, "DK": 18, "DO": 28, "EE": 20, "FO": 18,
    "FI": 18, "FR": 27, "GE": 22, "DE": 22, "GI": 23, "GR": 27,
    "GL": 18, "GT": 28, "HU": 28, "IS": 26, "IQ": 23, "IE": 22,
    "IL": 23, "IT": 27, "JO": 30, "KZ": 20, "KW": 30, "LV": 21,
    "LB": 28, "LI": 21, "LT": 20, "LU": 20, "MK": 19, "MT": 31,
    "MR": 27, "MU": 30, "MC": 27, "MD": 24, "ME": 22, "NL": 18,
    "NO": 15, "PK": 24, "PS": 29, "PL": 28, "PT": 25, "QA": 29,
    "RO": 24, "LC": 32, "SM": 27, "SA": 24, "RS": 22, "SC": 31,
    "SK": 24, "SI": 19, "ES": 24, "SE": 24, "CH": 21, "TL": 23,
    "TN": 24, "TR": 26, "UA": 29, "AE": 23, "GB": 22, "VA": 22,
    "VG": 24,
}


def _valid_iban_prefix(candidate: str) -> int | None:
    compact = "".join(candidate.split()).upper()
    length = IBAN_LENGTHS.get(compact[:2])
    if length is None or len(compact) < length:
        return None
    iban = compact[:length]
    rearranged = iban[4:] + iban[:4]
    remainder = 0
    for char in rearranged:
        digits = str(ord(char) - 55) if char.isalpha() else char
        for digit in digits:
            remainder = (remainder * 10 + int(digit)) % 97
    return length if remainder == 1 else None


def _iban_span(match: re.Match[str]) -> tuple[str, int, int] | None:
    candidate = match.group("value")
    compact_length = _valid_iban_prefix(candidate)
    if compact_length is None:
        return None
    count = 0
    end = 0
    for end, char in enumerate(candidate, start=1):
        if char.isalnum():
            count += 1
            if count == compact_length:
                break
    start = match.start("value")
    return candidate[:end], start, start + end
LABEL_CODES = {
    "person name": "PERSON", "person": "PERSON", "street address": "ADDRESS",
    "address": "ADDRESS", "date of birth": "DOB", "date of birth date": "DOB",
    "passport number": "PASSPORT", "identity document number": "ID",
    "bank account number": "BANK_ACCOUNT", "email": "EMAIL", "phone number": "PHONE",
}
_ADDRESS_CONTEXT = re.compile(r"(?i)\b(?:по\s+адресу|адрес(?:а|е|у)?|wohnhaft\s+in|adresse|address)\s*:?\s*")
_CITY_PREFIX = re.compile(r"(?i)(?:г\.?\s+[A-ZА-ЯЁ][\w-]*,\s*)$")
_POSTCODE_CITY_SUFFIX = re.compile(r"(?i)(?:,\s*|\s+)\d{5}\s+[A-ZÄÖÜ][\wÄÖÜäöüß-]*")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    value: str
    code: str
    confidence: float
    rule: bool = False


class PseudonymMapping(dict[str, str]):
    """Legacy marker mapping with private person records and safe warnings."""

    def __init__(self) -> None:
        super().__init__()
        self.persons: dict[str, Any] = {}
        self.warnings: list[str] = []


def _expand_contextual_address(text: str, span: Span) -> Span:
    """Include a nearby city/postcode only inside an explicit address phrase."""
    if span.code != "ADDRESS":
        return span
    before_start = max(0, span.start - 120)
    before = text[before_start : span.start]
    contexts = list(_ADDRESS_CONTEXT.finditer(before))
    if not contexts:
        return span
    context_end = contexts[-1].end()
    context_tail = re.sub(r"(?i)\bг\.", "г", before[context_end:])
    if re.search(r"[.!?;\n]", context_tail):
        return span

    start = span.start
    city_prefix = _CITY_PREFIX.search(before)
    if city_prefix:
        start = before_start + city_prefix.start()
    end = span.end
    city_suffix = _POSTCODE_CITY_SUFFIX.match(text[end:])
    if city_suffix:
        end += city_suffix.end()
    return Span(start, end, text[start:end], span.code, span.confidence, span.rule)


class MappingStore:
    """Thread-safe in-memory mappings with a sliding-independent creation TTL."""

    def __init__(self, ttl_seconds: int = 1800, clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time):
        self._ttl = ttl_seconds
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._items: dict[str, tuple[float, dict[str, str]]] = {}
        self._stop = threading.Event()
        self._cleaner = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleaner.start()

    def create(self, mapping: dict[str, str]) -> tuple[str, float]:
        now = self._clock()
        mapping_id = str(uuid.uuid4())
        with self._lock:
            self._purge(now)
            self._items[mapping_id] = (now + self._ttl, mapping)
        return mapping_id, self._wall_clock() + self._ttl

    def get(self, mapping_id: str) -> dict[str, str] | None:
        now = self._clock()
        with self._lock:
            self._purge(now)
            entry = self._items.get(mapping_id)
            return entry[1] if entry else None

    def _purge(self, now: float) -> None:
        for key in [key for key, (expires, _) in self._items.items() if expires <= now]:
            del self._items[key]

    def _cleanup_loop(self) -> None:
        while not self._stop.wait(60):
            with self._lock:
                self._purge(self._clock())

    def close(self) -> None:
        self._stop.set()
        if self._cleaner is not threading.current_thread():
            self._cleaner.join(timeout=1)


def _chunks(text: str, tokenizer: Any | None = None) -> list[tuple[int, str]]:
    if tokenizer is not None:
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            if offsets and len(offsets) <= 300:
                return [(0, text)]
            if offsets:
                chunks = []
                step = 250
                for token_start in range(0, len(offsets), step):
                    token_end = min(token_start + 300, len(offsets))
                    char_start = 0 if token_start == 0 else offsets[token_start][0]
                    char_end = len(text) if token_end == len(offsets) else offsets[token_end - 1][1]
                    chunks.append((char_start, text[char_start:char_end]))
                    if token_end == len(offsets):
                        break
                return chunks
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise RuntimeError("Tokenizer failed to split input safely") from exc
    if len(text) <= CHUNK_SIZE:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(text), step):
        chunks.append((start, text[start : start + CHUNK_SIZE]))
        if start + CHUNK_SIZE >= len(text):
            break
    return chunks


def find_spans(text: str, model: Any) -> list[Span]:
    spans: list[Span] = []
    # Include only high-risk entities; do not ask GLiNER to extract generic dates,
    # organizations, or geographic names.
    model_id = getattr(model, "model_id", None)
    model_revision = getattr(model, "model_revision", None)
    for chunk_index, (start, chunk) in enumerate(_chunks(text, getattr(model, "tokenizer", None))):
        try:
            predictions = model.predict_entities(
                chunk, LABELS, threshold=MODEL_INFERENCE_THRESHOLD
            )
        except Exception as exc:
            event("ner.error", {
                "chunk_index": chunk_index, "chunk_offset": start,
                "chunk_text": chunk, "model": model_id,
                "revision": model_revision, "error_type": type(exc).__name__,
                "threshold": MODEL_INFERENCE_THRESHOLD,
                "accepted_thresholds": ACCEPTED_THRESHOLDS,
            })
            raise
        rejected_low_confidence: list[dict[str, Any]] = []
        logged_predictions: list[dict[str, Any]] = []
        for item in predictions:
            if not isinstance(item, dict):
                continue
            value = item.get("text", "")
            local_start = item.get("start")
            local_end = item.get("end")
            if not value or not isinstance(local_start, int) or not isinstance(local_end, int):
                continue
            code = LABEL_CODES.get(str(item.get("label", "")).lower())
            raw_score = item.get("score", 0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError, OverflowError):
                score = math.nan
            score_is_finite = math.isfinite(score)
            logged_predictions.append({
                "text": value,
                "label": item.get("label"),
                "start": local_start,
                "end": local_end,
                "score": score if score_is_finite else None,
            })
            if code and 0 <= local_start < local_end <= len(chunk):
                entity_start, entity_end = start + local_start, start + local_end
                threshold = ACCEPTED_THRESHOLDS[code]
                if not score_is_finite or score < threshold:
                    rejected_low_confidence.append({
                        "value": value,
                        "label": item.get("label"),
                        "code": code,
                        "start": entity_start,
                        "end": entity_end,
                        "score": score if score_is_finite else None,
                        "threshold": threshold,
                        "reason": "low_confidence" if score_is_finite else "non_finite_score",
                    })
                    continue
                entity_text = text[entity_start:entity_end]
                if entity_text.strip().casefold() in LABEL_CODES:
                    continue
                span = Span(entity_start, entity_end, entity_text, code, score)
                spans.append(_expand_contextual_address(text, span))
        event("ner.chunk", {
            "chunk_index": chunk_index,
            "chunk_offset": start,
            "chunk_text": chunk,
            "labels": LABELS,
            "threshold": MODEL_INFERENCE_THRESHOLD,
            "accepted_thresholds": ACCEPTED_THRESHOLDS,
            "model": model_id,
            "revision": model_revision,
            "returned_entities": logged_predictions,
            "rejected_low_confidence": rejected_low_confidence,
        })
    rule_detections: list[dict[str, Any]] = []
    for code, pattern in PATTERNS:
        for match in pattern.finditer(text):
            if code == "IBAN":
                iban = _iban_span(match)
                if iban is None:
                    rule_detections.append({"code": code, "start": match.start(),
                                            "end": match.end(), "value": match.group(),
                                            "accepted": False, "reason": "invalid_iban"})
                    continue
                value, start, end = iban
                spans.append(Span(start, end, value, code, 1.0, True))
                rule_detections.append({"code": code, "start": start, "end": end,
                                        "value": value, "accepted": True})
                continue
            group = next((name for name, value in match.groupdict().items() if value is not None), None)
            value = match.group(group) if group else match.group()
            start, end = match.span(group) if group else match.span()
            if code == "PHONE":
                # A date immediately following a phone number can otherwise be
                # absorbed by the permissive international-number separators.
                date = re.search(r"\s+\d{1,2}[./]\d{1,2}[./]\d{4}\b", value)
                if date:
                    value = value[:date.start()].rstrip()
                    end = start + len(value)
                digits = sum(char.isdigit() for char in value)
                if digits > 15 or digits < 7:
                    rule_detections.append({"code": code, "start": start, "end": end,
                                            "value": value, "accepted": False,
                                            "reason": "invalid_digit_count"})
                    continue
            spans.append(Span(start, end, value, code, 1.0, True))
            rule_detections.append({"code": code, "start": start, "end": end,
                                    "value": value, "accepted": True})
    event("ner.rules", {"detections": rule_detections})

    # Deduplicate overlapping-window matches, then resolve overlap by rule,
    # longer span, and confidence in that order.
    unique: dict[tuple[int, int, str], Span] = {}
    for span in spans:
        key = (span.start, span.end, span.code)
        if key not in unique or span.confidence > unique[key].confidence:
            unique[key] = span
    candidates = list(unique.values())
    selected: list[Span] = []
    rejected: list[dict[str, Any]] = []
    candidates.sort(key=lambda s: (not s.rule, -(s.end - s.start), -s.confidence, s.start))
    for candidate in candidates:
        overlaps = [other for other in selected
                    if candidate.start < other.end and other.start < candidate.end]
        if not overlaps:
            selected.append(candidate)
        else:
            rejected.append({"start": candidate.start, "end": candidate.end,
                             "value": candidate.value, "code": candidate.code,
                             "source": "rule" if candidate.rule else "gliner",
                             "overlaps": [{"start": other.start, "end": other.end,
                                           "value": other.value, "code": other.code}
                                          for other in overlaps]})
    event("ner.selected", {
        "candidates": [{"start": span.start, "end": span.end, "value": span.value,
                        "code": span.code, "confidence": span.confidence,
                        "source": "rule" if span.rule else "gliner"} for span in candidates],
        "selected": [{"start": span.start, "end": span.end, "value": span.value,
                      "code": span.code, "confidence": span.confidence,
                      "source": "rule" if span.rule else "gliner"} for span in selected],
        "rejected_overlaps": rejected,
    })
    return sorted(selected, key=lambda s: s.start)


def anonymize_text(text: str, model: Any, pipeline: Any | None = None, *,
                   steps: dict[str, Any] | None = None) -> tuple[str, PseudonymMapping]:
    spans = find_spans(text, model)
    if steps is not None:
        steps["entities"] = [
            {
                "start": span.start,
                "end": span.end,
                "value": span.value,
                "code": span.code,
                "source": "rule" if span.rule else "gliner",
            }
            for span in spans
        ]
    mapping = PseudonymMapping()
    value_markers: dict[tuple[str, str], str] = {}
    pieces: list[str] = []
    cursor = 0
    counters: dict[str, int] = {}
    reserved_markers = set(_MARKER.findall(text))

    def next_marker(code: str) -> str:
        while True:
            counters[code] = counters.get(code, 0) + 1
            marker = f"⟦{code}_{counters[code]}⟧"
            if marker not in reserved_markers:
                return marker

    assignments: dict[int, Any] = {}
    if pipeline is not None:
        person_spans = [span for span in spans if span.code == "PERSON"]
        if person_spans:
            linked, warnings = pipeline.link(text, person_spans)
            if set(linked) != {span.start for span in person_spans}:
                raise RuntimeError("Incomplete person assignments")
            assignments = linked
            mapping.warnings.extend(warnings)
    for span in spans:
        record = assignments.get(span.start) if span.code == "PERSON" else None
        if record is not None:
            marker = value_markers.get(("PERSON_ID", record.identifier))
            if marker is None:
                marker = next_marker("PERSON")
                value_markers[("PERSON_ID", record.identifier)] = marker
                mapping[marker] = record.canonical
                mapping.persons[marker] = record
        else:
            key = (span.code, span.value)
            if key in value_markers:
                marker = value_markers[key]
            else:
                marker = next_marker(span.code)
                value_markers[key] = marker
                mapping[marker] = span.value
        pieces.extend((text[cursor : span.start], marker))
        cursor = span.end
    pieces.append(text[cursor:])
    anonymized_text = "".join(pieces)
    event("anonymization.result", {
        "anonymized_text": anonymized_text,
        "mapping": dict(mapping),
        "persons": [
            {"marker": marker, "canonical": record.canonical,
             "aliases": list(record.aliases)}
            for marker, record in mapping.persons.items()
        ],
        "warnings": list(mapping.warnings),
    })
    if steps is not None:
        mentions_by_marker: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            record = assignments.get(span.start) if span.code == "PERSON" else None
            if record is None:
                continue
            marker = value_markers.get(("PERSON_ID", record.identifier))
            if marker is not None:
                mentions_by_marker.setdefault(marker, []).append({
                    "start": span.start, "end": span.end, "value": span.value,
                })
        steps["persons"] = [
            {
                "marker": marker,
                "canonical": record.canonical,
                "aliases": list(record.aliases),
                "mentions": mentions_by_marker.get(marker, []),
            }
            for marker, record in mapping.persons.items()
        ]
        steps["anonymized_text"] = anonymized_text
    return anonymized_text, mapping


_MARKER = re.compile(r"⟦[^⟦⟧]{1,120}⟧")


def restore_text(text: str, mapping: dict[str, str], pipeline: Any | None = None, *,
                 steps: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    warnings: list[str] = list(getattr(mapping, "warnings", []))
    known = set(mapping)
    occurrences: list[dict[str, Any]] = []
    for match in _MARKER.finditer(text):
        marker = match.group(0)
        if marker not in known:
            warnings.append(f"Unknown marker left unchanged: {marker}")
            occurrences.append({
                "marker": marker, "start": match.start(), "end": match.end(),
                "value": None, "status": "unknown",
            })
    # Single pass avoids interpreting text restored from a source value. The
    # compatibility `pipeline` argument is intentionally unused: restoration
    # is always the exact value recorded during anonymization.
    def replace(match: re.Match[str]) -> str:
        marker = match.group(0)
        if marker not in known:
            return marker
        value = mapping[marker]
        occurrences.append({
            "marker": marker, "start": match.start(), "end": match.end(),
            "value": value, "status": "restored", "reason": "saved_value",
        })
        return value

    restored = _MARKER.sub(replace, text)
    event("restore.result", {
        "input_text": text,
        "restored_text": restored,
        "occurrences": sorted(occurrences, key=lambda item: item["start"]),
        "warnings": warnings,
    })
    if steps is not None:
        # Each invocation produces a fresh trace. Caller mappings and stored
        # restoration state remain untouched.
        steps["restorations"] = sorted(occurrences, key=lambda item: item["start"])
    return restored, warnings


def expires_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
