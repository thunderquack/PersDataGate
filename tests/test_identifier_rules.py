from fastapi.testclient import TestClient

from pseudonymizer.app import create_app
from pseudonymizer.service import MappingStore, anonymize_text


class NoEntities:
    def predict_entities(self, text, labels, threshold):
        return []


def test_iban_checksum_bounds_match_before_following_text():
    source = "IBAN DE89 3704 0044 0532 0130 00 und weitere Angaben"
    masked, mapping = anonymize_text(source, NoEntities())

    assert "und weitere Angaben" in masked
    assert next(iter(mapping.values())) == "DE89 3704 0044 0532 0130 00"


def test_invalid_iban_like_number_is_not_masked_as_iban_or_phone():
    source = "IBAN DE89 3704 0044 0532 0130 01 und weitere Angaben"
    masked, mapping = anonymize_text(source, NoEntities())

    assert masked == source
    assert mapping == {}


def test_phone_does_not_absorb_adjacent_date():
    source = "Telefon: +49 30 123456 12.05.2024"
    masked, mapping = anonymize_text(source, NoEntities())

    assert "12.05.2024" in masked
    assert next(iter(mapping.values())) == "+49 30 123456"


def test_deanonymize_accepts_large_openai_response_after_marker_expansion():
    marker = "⟦PERSON_0123456789abcdef0123456789abcdef_1⟧"
    store = MappingStore()
    mapping_id, _ = store.create({marker: "I"})
    masked_response = (marker + " ") * 2_000

    with TestClient(create_app(detector=NoEntities(), store=store)) as client:
        response = client.post(
            "/deanonymize",
            json={"mapping_id": mapping_id, "text": masked_response},
        )

    assert response.status_code == 200
    assert response.json()["text"] == ("I ") * 2_000
    store.close()
