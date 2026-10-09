from __future__ import annotations

import json


def test_gliner2_tokenizer_config_maps_list_without_mutating_input():
    from pseudonymizer.app import _normalize_gliner2_tokenizer_config

    original = {
        "tokenizer_class": "DebertaV2Tokenizer",
        "additional_special_tokens": ["[BASE]"],
        "extra_special_tokens": ["[SEP_STRUCT]", "[SEP_TEXT]"],
        "bos_token": "[BOS]",
    }

    normalized = _normalize_gliner2_tokenizer_config(original)

    assert normalized == {
        "tokenizer_class": "DebertaV2Tokenizer",
        "additional_special_tokens": ["[BASE]", "[SEP_STRUCT]", "[SEP_TEXT]"],
        "bos_token": "[BOS]",
    }
    assert original["extra_special_tokens"] == ["[SEP_STRUCT]", "[SEP_TEXT]"]
    assert original["additional_special_tokens"] == ["[BASE]"]


def test_gliner2_compatible_snapshot_preserves_source_config_and_token_ids(tmp_path, monkeypatch):
    from pathlib import Path
    import shutil

    from pseudonymizer.app import _compatible_gliner2_snapshot

    source = tmp_path / "snapshot"
    (source / "encoder_config").mkdir(parents=True)
    source_config = {
        "tokenizer_class": "DebertaV2Tokenizer",
        "extra_special_tokens": ["[SEP_STRUCT]", "[SEP_TEXT]"],
    }
    config_path = source / "tokenizer_config.json"
    config_path.write_text(json.dumps(source_config), encoding="utf-8")
    tokenizer_contents = json.dumps({"added_tokens": [
        {"id": 250102, "content": "[SEP_STRUCT]"},
        {"id": 250103, "content": "[SEP_TEXT]"},
    ]})
    (source / "tokenizer.json").write_text(tokenizer_contents, encoding="utf-8")
    for relative_path in ("config.json", "encoder_config/config.json", "model.safetensors"):
        (source / relative_path).write_bytes(b"fixture")

    # Copying stands in for symlinks so the test is portable on Windows without
    # developer-mode symlink privileges. The production helper still receives
    # the same snapshot tree and config structure.
    monkeypatch.setattr(Path, "symlink_to", lambda destination, target: shutil.copyfile(target, destination))
    original_config_bytes = config_path.read_bytes()

    with _compatible_gliner2_snapshot(str(source)) as compatible_path:
        compatible = Path(compatible_path)
        normalized = json.loads((compatible / "tokenizer_config.json").read_text(encoding="utf-8"))
        tokens = json.loads((compatible / "tokenizer.json").read_text(encoding="utf-8"))["added_tokens"]

    assert normalized["additional_special_tokens"] == ["[SEP_STRUCT]", "[SEP_TEXT]"]
    assert "extra_special_tokens" not in normalized
    assert [(token["id"], token["content"]) for token in tokens] == [
        (250102, "[SEP_STRUCT]"),
        (250103, "[SEP_TEXT]"),
    ]
    assert config_path.read_bytes() == original_config_bytes
    assert json.loads(config_path.read_text(encoding="utf-8")) == source_config
