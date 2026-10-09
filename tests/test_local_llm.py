from __future__ import annotations

import json
import sys
import types

import pytest

from pseudonymizer.local_llm import (
    MAX_NEW_TOKENS,
    MAX_TOTAL_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    GenerationError,
    QwenGenerator,
)


class FakeTensor:
    def __init__(self, size: int):
        self.shape = (1, size)
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, item):
        return self


class FakeBatch:
    def __getitem__(self, item):
        return FakeTensor(1)


def mocked_transformers(monkeypatch, *, response: str, input_tokens: int = 8):
    calls = {"tokenizer_load": [], "model_load": [], "template": [], "encoded": [], "generate": []}
    input_ids = FakeTensor(input_tokens)

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["tokenizer_load"].append((args, kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            calls["template"].append((messages, kwargs))
            return "rendered prompt"

        def __call__(self, rendered, **kwargs):
            calls["encoded"].append((rendered, kwargs))
            return {"input_ids": input_ids, "attention_mask": FakeTensor(input_tokens)}

        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return [ord(char) + 100 for char in text]

        eos_token_id = 2

        def decode(self, tokens, **kwargs):
            return response

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["model_load"].append((args, kwargs))
            return cls()

        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return FakeBatch()

    class InferenceMode:
        def __enter__(self):
            calls["inference_entered"] = True

        def __exit__(self, *args):
            return False

    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.inference_mode = InferenceMode
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForCausalLM = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


@pytest.mark.parametrize(("device", "dtype"), [("cpu", "float32"), ("cuda", "float16")])
def test_loads_pinned_model_on_selected_device_and_uses_nonthinking_chat_template(
    monkeypatch, tmp_path, device, dtype
):
    calls = mocked_transformers(monkeypatch, response='{"relation":"same"}')
    generator = QwenGenerator(device=device, cache_dir=tmp_path)

    assert generator.choose("Return relation", {"id": "p1"}) == {"relation": "same"}
    assert calls["tokenizer_load"][0] == (
        (MODEL_ID,), {"revision": MODEL_REVISION, "cache_dir": str(tmp_path)}
    )
    assert calls["model_load"][0][0] == (MODEL_ID,)
    assert calls["model_load"][0][1]["revision"] == MODEL_REVISION
    assert calls["model_load"][0][1]["torch_dtype"] == dtype
    assert calls["model_device"] == device
    assert calls["eval"] is True
    assert calls["template"][0][1]["enable_thinking"] is False
    assert calls["generate"][0]["max_new_tokens"] == MAX_NEW_TOKENS
    assert calls["generate"][0]["do_sample"] is False
    assert calls["inference_entered"] is True


def test_does_not_truncate_or_generate_when_encoded_prompt_exceeds_budget(monkeypatch, tmp_path):
    calls = mocked_transformers(
        monkeypatch,
        response='{"relation":"same"}',
        input_tokens=MAX_TOTAL_TOKENS - MAX_NEW_TOKENS + 1,
    )
    generator = QwenGenerator(device="cpu", cache_dir=tmp_path)

    with pytest.raises(GenerationError, match="maximum is 7936 input tokens"):
        generator.choose("instructions", {"large": "payload"})
    assert calls["generate"] == []
    assert calls["encoded"][0][1].get("truncation") is None


@pytest.mark.parametrize(
    "response",
    [
        "Here is the answer: {\"relation\":\"same\"}",
        '{"relation":"same","relation":"different"}',
        '[{"relation":"same"}]',
    ],
)
def test_rejects_prose_duplicate_keys_and_non_object_json(monkeypatch, tmp_path, response):
    mocked_transformers(monkeypatch, response=response)
    generator = QwenGenerator(device="cpu", cache_dir=tmp_path)

    with pytest.raises(GenerationError):
        generator.choose("instructions", {"id": "p1"})


def test_returns_json_values_for_caller_semantic_validation(monkeypatch, tmp_path):
    # The generator parses JSON only; the person pipeline must check that this
    # choice belongs to the candidate IDs it supplied.
    mocked_transformers(monkeypatch, response=json.dumps({"person_id": "not-an-offered-id"}))
    generator = QwenGenerator(device="cpu", cache_dir=tmp_path)

    assert generator.choose("Choose an offered person", {"allowed_ids": ["p1"]}) == {
        "person_id": "not-an-offered-id"
    }


@pytest.mark.parametrize(
    ("payload", "permitted"),
    [
        ({"allowed_relations": ["same", "different"]}, ['{"relation":"same"}', '{"relation":"different"}']),
        ({"allowed_forms": ["Ивана Иванова", "Ivanov" ]}, ['{"form":"Ивана Иванова"}', '{"form":"Ivanov"}', '{"form":null}']),
    ],
)
def test_constrains_generation_to_exact_unicode_choices(monkeypatch, tmp_path, payload, permitted):
    calls = mocked_transformers(monkeypatch, response=permitted[0])
    generator = QwenGenerator(device="cpu", cache_dir=tmp_path)

    generator.choose("Choose an allowed value", payload)
    generation = calls["generate"][0]
    callback = generation["prefix_allowed_tokens_fn"]
    prompt = [0] * 8
    sequences = [[ord(char) + 100 for char in text] for text in permitted]
    first_allowed = callback(0, prompt)
    assert first_allowed == sorted({sequence[0] for sequence in sequences})

    for tokens in sequences:
        for offset in range(len(tokens)):
            allowed = callback(0, prompt + tokens[:offset])
            assert tokens[offset] in allowed
        terminal = callback(0, prompt + tokens)
        assert terminal == [2]

    # An out-of-trie token cannot lead to another name or relation.
    assert callback(0, prompt + [999]) == [2]


@pytest.mark.parametrize(
    "payload",
    [
        {"allowed_relations": []},
        {"allowed_forms": ["valid", 3]},
    ],
)
def test_rejects_invalid_finite_choice_lists(monkeypatch, tmp_path, payload):
    calls = mocked_transformers(monkeypatch, response='{"relation":"same"}')
    generator = QwenGenerator(device="cpu", cache_dir=tmp_path)

    with pytest.raises(GenerationError):
        generator.choose("Choose", payload)
    assert calls["generate"] == []
