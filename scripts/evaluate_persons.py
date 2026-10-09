#!/usr/bin/env python3
"""Evaluate local person linking and contextual restoration against a running service.

The curated synthetic cases live in ``docs/person-evaluation.json``. Each source
mention has unique plain-text anchors so the evaluator can find its marker in
the anonymized response without relying on model-specific offsets. It then
checks expected same-person and different-person assignments, sends marked
response contexts to ``/deanonymize``, and verifies unknown markers remain
untouched with a warning.

Run after starting the Compose services. The supported command sends this
script to the web container, which already has network access and the service
URL configured:

    docker compose exec -T web python - < scripts/evaluate_persons.py

The script uses only the standard library and makes no OpenAI calls. It does
not start containers or download model weights. Set ``PSEUDONYMIZER_URL`` to
the pseudonymizer's reachable base URL. Requests allow up to 600 seconds for
local inference. Exit status is 0 only when every assertion passes, 1 when an
evaluation assertion fails, or 2 when the service/fixture cannot be evaluated.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = os.environ.get("PSEUDONYMIZER_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_SECONDS = 600
MARKER = re.compile(r"⟦[^⟦⟧]{1,120}⟧")
FIXTURES = Path(__file__).resolve().parent.parent / "docs" / "person-evaluation.json"
# Keep a copy in the script because the documented stdin invocation runs inside
# the web image, which does not contain the repository's docs directory.
DEFAULT_CASES = [
    {
        "id": "ru_declensions_initials_full_name", "language": "ru",
        "source": "Участник A: Иван Петров; конец A. Участник B: Ивана Петрова; конец B. Участник C: И. Петров; конец C. Фамилия участника A: Петров; конец F.",
        "mentions": [
            {"id": "ivan_nom", "before": "Участник A: ", "after": "; конец A"},
            {"id": "ivan_gen", "before": "Участник B: ", "after": "; конец B"},
            {"id": "ivan_initial", "before": "Участник C: ", "after": "; конец C"},
            {"id": "ivan_surname_only", "before": "Фамилия участника A: ", "after": "; конец F"},
        ],
        "links": [
            {"left": "ivan_nom", "right": "ivan_gen", "relation": "same"},
            {"left": "ivan_nom", "right": "ivan_initial", "relation": "same"},
            {"left": "ivan_nom", "right": "ivan_surname_only", "relation": "same"},
        ],
        "restorations": [
            {"id": "ru_dative", "mention": "ivan_nom", "template": "Передали {marker} документы.", "expected": "Передали Ивану Петрову документы."},
            {"id": "ru_genitive", "mention": "ivan_nom", "template": "Нет {marker} в списке.", "expected": "Нет Ивана Петрова в списке."},
        ],
    },
    {
        "id": "ru_patronymic_orders", "language": "ru",
        "source": "Форма A: Иван Петров Иванович; конец patronymic-last. Форма B: Иван Иванович Петров; конец patronymic-middle.",
        "mentions": [
            {"id": "patronymic_last", "before": "Форма A: ", "after": "; конец patronymic-last"},
            {"id": "patronymic_middle", "before": "Форма B: ", "after": "; конец patronymic-middle"},
        ],
        "links": [{"left": "patronymic_last", "right": "patronymic_middle", "relation": "same"}],
    },
    {
        "id": "ru_ambiguous_initial", "language": "ru",
        "source": "Первый: Иван Петров; конец first. Второй: Игорь Петров; конец second. Ссылка: И. Петров; конец initial.",
        "mentions": [
            {"id": "ivan", "before": "Первый: ", "after": "; конец first"},
            {"id": "igor", "before": "Второй: ", "after": "; конец second"},
            {"id": "initial", "before": "Ссылка: ", "after": "; конец initial"},
        ],
        "links": [
            {"left": "ivan", "right": "igor", "relation": "different"},
            {"left": "ivan", "right": "initial", "relation": "different"},
            {"left": "igor", "right": "initial", "relation": "different"},
        ],
        "warning_contains": ["ambiguous"],
    },
    {
        "id": "ru_same_spelling_different_departments", "language": "ru",
        "source": "Отдел A: сотрудник 1 Иван Петров. Отдел B: другой сотрудник 2 Иван Петров. Это разные люди; конец A и конец B.",
        "mentions": [
            {"id": "dept_a", "before": "Отдел A: сотрудник 1 ", "after": ". Отдел B"},
            {"id": "dept_b", "before": "другой сотрудник 2 ", "after": ". Это разные люди"},
        ],
        "links": [{"left": "dept_a", "right": "dept_b", "relation": "different"}],
    },
    {
        "id": "ru_woman_inflected_forms", "language": "ru",
        "source": "Заявительница: Мария Иванова; конец applicant. Повторно о заявительнице: Марии Ивановой; конец repeat.",
        "mentions": [
            {"id": "maria_one", "before": "Заявительница: ", "after": "; конец applicant"},
            {"id": "maria_two", "before": "Повторно о заявительнице: ", "after": "; конец repeat"},
        ],
        "links": [{"left": "maria_one", "right": "maria_two", "relation": "same"}],
        "restorations": [
            {"id": "ru_woman_dative", "mention": "maria_one", "template": "Письмо адресовано {marker}.", "expected": "Письмо адресовано Марии Ивановой."},
            {"id": "ru_woman_genitive", "mention": "maria_one", "template": "Документы {marker} приложены.", "expected": "Документы Марии Ивановой приложены."},
        ],
    },
    {
        "id": "de_initial_full_and_possessive", "language": "de",
        "source": "Vollname: Max Müller; Ende full. Kurzform: M. Müller; Ende initial.",
        "mentions": [
            {"id": "max_full", "before": "Vollname: ", "after": "; Ende full"},
            {"id": "max_initial", "before": "Kurzform: ", "after": "; Ende initial"},
        ],
        "links": [{"left": "max_full", "right": "max_initial", "relation": "same"}],
        "restorations": [
            {"id": "de_possessive", "mention": "max_full", "template": "Das ist {marker} Bericht.", "expected": "Das ist Max Müllers Bericht."},
        ],
        "unknown_marker": {
            "marker": "⟦PERSON_unknown_999⟧",
            "text": "Unbekannt: ⟦PERSON_unknown_999⟧.",
        },
    },
]


def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
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


def locate_marker(safe_text: str, mention: dict[str, str]) -> str | None:
    """Find a mention's marker by its unmasked, unique surrounding anchors."""
    before = mention["before"]
    after = mention["after"]
    before_pos = safe_text.find(before)
    if before_pos < 0:
        return None
    start = before_pos + len(before)
    after_pos = safe_text.find(after, start)
    if after_pos < 0:
        return None
    between = safe_text[start:after_pos]
    markers = MARKER.findall(between)
    if len(markers) != 1:
        return None
    return markers[0]


def evaluate(case: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    anonymized = post("/anonymize", {"text": case["source"]})
    mapping_id = anonymized["mapping_id"]
    safe_text = anonymized["text"]
    warnings = list(anonymized.get("warnings", []))

    markers: dict[str, str] = {}
    for mention in case["mentions"]:
        marker = locate_marker(safe_text, mention)
        if marker is None:
            issues.append(f"mention {mention['id']}: expected one marker between anchors")
        else:
            markers[mention["id"]] = marker

    false_merges = 0
    false_splits = 0
    link_assertions = 0
    for assertion in case.get("links", []):
        left = markers.get(assertion["left"])
        right = markers.get(assertion["right"])
        if left is None or right is None:
            continue
        link_assertions += 1
        expected_same = assertion["relation"] == "same"
        actual_same = left == right
        if expected_same and not actual_same:
            false_splits += 1
            issues.append(f"{assertion['left']} and {assertion['right']}: expected shared marker")
        elif not expected_same and actual_same:
            false_merges += 1
            issues.append(f"{assertion['left']} and {assertion['right']}: false merge")

    restoration_passes = 0
    restoration_failures = 0
    for restoration in case.get("restorations", []):
        marker = markers.get(restoration["mention"])
        if marker is None:
            continue
        response_text = restoration["template"].replace("{marker}", marker)
        expected = restoration["expected"].replace("{marker}", marker)
        result = post("/deanonymize", {"mapping_id": mapping_id, "text": response_text})
        warnings.extend(result.get("warnings", []))
        actual = result["text"]
        if actual == expected:
            restoration_passes += 1
        else:
            restoration_failures += 1
            issues.append(
                f"restoration {restoration['id']}: expected {expected!r}, got {actual!r}"
            )
        if restoration.get("expect_marker") and marker not in actual:
            issues.append(f"restoration {restoration['id']}: uncertain form marker was not retained")
            restoration_failures += 1

    unknown = case.get("unknown_marker")
    unknown_preserved = None
    if unknown:
        unknown_text = unknown["text"]
        result = post("/deanonymize", {"mapping_id": mapping_id, "text": unknown_text})
        unknown_preserved = result["text"] == unknown_text
        warnings.extend(result.get("warnings", []))
        if not unknown_preserved:
            issues.append("unknown marker was changed")
        if not any(unknown["marker"] in warning for warning in result.get("warnings", [])):
            issues.append("unknown marker was not reported in warnings")

    if case.get("warning_contains"):
        combined = " ".join(warnings).casefold()
        for expected_warning in case["warning_contains"]:
            if expected_warning.casefold() not in combined:
                issues.append(f"expected warning containing {expected_warning!r}")

    return {
        "id": case["id"],
        "language": case["language"],
        "link_assertions": link_assertions,
        "false_merges": false_merges,
        "false_splits": false_splits,
        "restoration_passes": restoration_passes,
        "restoration_failures": restoration_failures,
        "unknown_marker_preserved": unknown_preserved,
        "warnings": list(dict.fromkeys(warnings)),
        "issues": issues,
    }


def main() -> int:
    try:
        cases = json.loads(FIXTURES.read_text(encoding="utf-8")) if FIXTURES.is_file() else DEFAULT_CASES
        reports = [evaluate(case) for case in cases]
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    totals = {
        key: sum(report[key] for report in reports)
        for key in ("link_assertions", "false_merges", "false_splits",
                    "restoration_passes", "restoration_failures")
    }
    totals["unknown_markers_preserved"] = sum(
        report["unknown_marker_preserved"] is True for report in reports
    )
    totals["issues"] = sum(len(report["issues"]) for report in reports)
    print(json.dumps({"service": BASE_URL, "totals": totals, "cases": reports},
                     ensure_ascii=False, indent=2))
    return 0 if totals["issues"] == 0 and totals["false_merges"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
