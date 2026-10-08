from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
MAX_TEXT_LENGTH = 50_000
LABELS = [
    "person name", "street address", "date of birth", "passport number",
    "identity document number", "bank account number",
]

# Format based detectors have precedence over model spans.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)")),
    ("IBAN", re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9][ ]?){11,30})(?![A-Za-z0-9])", re.I)),
    ("PHONE", re.compile(r"(?i)\b(?:phone|telefon|телефон)\s*[:№]?\s*(?P<value>(?:\+|00)?\d[\d ()/.-]{5,}\d)|(?<!\w)(?P<international>(?:\+|00)\d[\d ()/.-]{5,}\d)(?!\w)")),
    ("PASSPORT", re.compile(r"(?i)\bpassport\s+(?:no\.?|number)\s+(?P<value>[\w-]{5,14})|\bпаспорт\s+(?:№|номер)\s*(?P<value_ru>[\w-]{5,14})|\bReisepassnummer\s*[:№]?\s*(?P<value_de>[\w-]{5,14})")),
    ("ID", re.compile(r"(?i)\b(?:id|identity|document)\s+(?:no\.?|number)\s+(?P<value>[\w-]{5,20})|\b(?:номер документа|идентификационный номер)\s+(?P<value_ru>[\w-]{5,20})|\b(?:Ausweisnummer|Dokumentennummer)\s*[:№]?\s*(?P<value_de>[\w-]{5,20})")),
)
LABEL_CODES = {
    "person name": "PERSON", "person": "PERSON", "street address": "ADDRESS",
    "address": "ADDRESS", "date of birth": "DOB", "date of birth date": "DOB",
    "passport number": "PASSPORT", "identity document number": "ID",
    "bank account number": "BANK_ACCOUNT", "email": "EMAIL", "phone number": "PHONE",
}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    value: str
    code: str
    confidence: float
    rule: bool = False


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
        except (KeyError, TypeError, ValueError, NotImplementedError):
            pass
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
    for start, chunk in _chunks(text, getattr(model, "tokenizer", None)):
        for item in model.predict_entities(chunk, LABELS, threshold=0.45):
            value = item.get("text", "")
            local_start = item.get("start")
            local_end = item.get("end")
            if not value or not isinstance(local_start, int) or not isinstance(local_end, int):
                continue
            code = LABEL_CODES.get(str(item.get("label", "")).lower())
            if code and 0 <= local_start < local_end <= len(chunk):
                spans.append(Span(start + local_start, start + local_end, text[start + local_start : start + local_end], code, float(item.get("score", 0))))
    for code, pattern in PATTERNS:
        for match in pattern.finditer(text):
            group = next((name for name, value in match.groupdict().items() if value is not None), None)
            value = match.group(group) if group else match.group()
            start, end = match.span(group) if group else match.span()
            spans.append(Span(start, end, value, code, 1.0, True))

    # Deduplicate overlapping-window matches, then resolve overlap by rule,
    # longer span, and confidence in that order.
    candidates = list({(s.start, s.end, s.code): s for s in spans}.values())
    candidates.sort(key=lambda s: (not s.rule, -(s.end - s.start), -s.confidence, s.start))
    selected: list[Span] = []
    for candidate in candidates:
        if not any(candidate.start < other.end and other.start < candidate.end for other in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda s: s.start)


def anonymize_text(text: str, model: Any) -> tuple[str, dict[str, str]]:
    spans = find_spans(text, model)
    mapping: dict[str, str] = {}
    value_markers: dict[str, str] = {}
    nonce = uuid.uuid4().hex
    pieces: list[str] = []
    cursor = 0
    counters: dict[str, int] = {}
    for span in spans:
        if span.value in value_markers:
            marker = value_markers[span.value]
        else:
            counters[span.code] = counters.get(span.code, 0) + 1
            marker = f"⟦{span.code}_{nonce}_{counters[span.code]}⟧"
            value_markers[span.value] = marker
            mapping[marker] = span.value
        pieces.extend((text[cursor : span.start], marker))
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces), mapping


_MARKER = re.compile(r"⟦[^⟦⟧]{1,120}⟧")


def restore_text(text: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    known = set(mapping)
    for marker in _MARKER.findall(text):
        if marker not in known:
            warnings.append(f"Unknown marker left unchanged: {marker}")
    # Single pass avoids interpreting text restored from a source value.
    restored = _MARKER.sub(lambda match: mapping.get(match.group(0), match.group(0)), text)
    return restored, warnings


def expires_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
