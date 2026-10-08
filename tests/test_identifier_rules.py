from pseudonymizer.service import anonymize_text


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
