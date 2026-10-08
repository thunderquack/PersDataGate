#!/usr/bin/env python3
"""Report anonymizer misses and extra detections on synthetic RU/DE examples.

Run inside the Compose network, for example:
  docker compose exec -T web python - < scripts/evaluate_model.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = os.environ.get("PSEUDONYMIZER_URL", "http://pseudonymizer:8000").rstrip("/")
TIMEOUT_SECONDS = 180
MARKER = re.compile(r"⟦[A-Z_]+_[0-9a-f]{32}_\d+⟧")
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "model_cases.json"
SYNTHETIC_CASES = [
    {
        "id": "ru_basic", "language": "ru",
        "text": "Заявитель Анна Смирнова, дата рождения 14.03.1985, проживает по адресу: г. Казань, ул. Лесная, д. 12. Телефон +7 912 345-67-89, почта anna.smirnova@example.org. План 01.09.2025 утвердило Министерство примеров в Казани.",
        "expected": ["Анна Смирнова", "14.03.1985", "г. Казань, ул. Лесная, д. 12", "+7 912 345-67-89", "anna.smirnova@example.org"],
        "expected_retained": ["01.09.2025", "Министерство примеров", "Казани"],
    },
    {
        "id": "de_basic", "language": "de",
        "text": "Antragstellerin Anna Müller, geboren am 14.03.1985, wohnhaft in Lindenstraße 12, 10115 Berlin. Telefon +49 30 12345678, E-Mail anna.mueller@example.org. Der Plan vom 01.09.2025 wurde vom Beispielministerium in Berlin beschlossen.",
        "expected": ["Anna Müller", "14.03.1985", "Lindenstraße 12, 10115 Berlin", "+49 30 12345678", "anna.mueller@example.org"],
        "expected_retained": ["01.09.2025", "Beispielministerium", "Berlin"],
    },
    {
        "id": "ru_financial_and_document_ids", "language": "ru",
        "text": "Письмо от ООО Ромашка, Москва, от 01.09.2025. Паспорт № 4510-123456, IBAN DE89 3704 0044 0532 0130 00.",
        "expected": ["4510-123456", "DE89 3704 0044 0532 0130 00"],
        "expected_retained": ["01.09.2025", "ООО Ромашка", "Москва"],
    },
    {
        "id": "de_financial_and_document_ids", "language": "de",
        "text": "Schreiben der Beispiel GmbH aus Berlin vom 01.09.2025. Passport number AB1234567, IBAN DE89 3704 0044 0532 0130 00.",
        "expected": ["AB1234567", "DE89 3704 0044 0532 0130 00"],
        "expected_retained": ["01.09.2025", "Beispiel GmbH", "Berlin"],
    },
]


def post(path: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc


def evaluate(case: dict) -> dict:
    anonymized = post("/anonymize", {"text": case["text"]})
    safe_text = anonymized["text"]
    mapping_id = anonymized["mapping_id"]
    markers = list(dict.fromkeys(MARKER.findall(safe_text)))
    recovered = {}
    for marker in markers:
        response = post("/deanonymize", {"mapping_id": mapping_id, "text": marker})
        recovered[marker] = response["text"]

    expected = case["expected"]
    missed = [value for value in expected if value in safe_text]
    mapped_values = list(recovered.values())
    found_expected = [value for value in expected if value in mapped_values]
    partial_expected = [
        value for value in expected
        if value not in mapped_values and any(mapped in value or value in mapped for mapped in mapped_values)
    ]
    extras = [value for value in mapped_values if value not in expected]
    retained = case.get("expected_retained", [])
    incorrectly_masked_retained = [value for value in retained if value not in safe_text]
    round_trip = post("/deanonymize", {"mapping_id": mapping_id, "text": safe_text})
    return {
        "id": case["id"], "language": case["language"],
        "expected_count": len(expected), "detected_expected_count": len(found_expected),
        "partial_detection": partial_expected,
        "missed": missed, "unexpected_mappings": extras,
        "retained_values_masked": incorrectly_masked_retained,
        "round_trip_exact": round_trip["text"] == case["text"],
        "warnings": round_trip.get("warnings", []),
    }


def main() -> int:
    try:
        cases = json.loads(FIXTURES.read_text(encoding="utf-8")) if FIXTURES.is_file() else SYNTHETIC_CASES
        reports = [evaluate(case) for case in cases]
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"service": BASE_URL, "cases": reports}, ensure_ascii=False, indent=2))
    return 0 if all(case["round_trip_exact"] and not case["warnings"] for case in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
