"""Small, local Qwen-backed JSON generator used by person processing."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from common.pipeline_log import event


MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MAX_TOTAL_TOKENS = 8192
MAX_NEW_TOKENS = 256

# Generation is serialized because one model instance is shared by the API's
# request handlers and may share a single GPU.
_GENERATION_LOCK = threading.Lock()


class GenerationError(RuntimeError):
    """Safe, bounded error from model input validation or generation."""


def _finite_choices(payload: Any) -> list[dict[str, Any]] | None:
    """Return the only permitted JSON outputs for production choice payloads."""
    if not isinstance(payload, dict):
        return None
    key = "allowed_relations" if "allowed_relations" in payload else (
        "allowed_forms" if "allowed_forms" in payload else None
    )
    if key is None:
        return None
    values = payload[key]
    if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
        raise GenerationError(f"{key} must be a non-empty list of strings")
    output_key = "relation" if key == "allowed_relations" else "form"
    choices = [{output_key: value} for value in dict.fromkeys(values)]
    if key == "allowed_forms":
        choices.append({"form": None})
    return choices


def _choice_trie(tokenizer: Any, choices: list[dict[str, Any]]) -> dict[int, Any]:
    """Build a token trie containing only serialized candidate objects."""
    trie: dict[int, Any] = {}
    for choice in choices:
        text = json.dumps(choice, ensure_ascii=False, separators=(",", ":"))
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens or len(tokens) + 1 > MAX_NEW_TOKENS:
            raise GenerationError("A permitted model choice exceeds the output token limit")
        node = trie
        for token in tokens:
            node = node.setdefault(int(token), {})
        node[None] = True
    return trie


def _allowed_next_tokens(trie: dict[int, Any], input_length: int, eos_token_id: int):
    """Create Transformers' callback for constrained, deterministic decoding."""
    def callback(_batch_id: int, sequence: Any) -> list[int]:
        if hasattr(sequence, "tolist"):
            sequence = sequence.tolist()
        if sequence and isinstance(sequence[0], list):
            sequence = sequence[0]
        node: Any = trie
        for token in sequence[input_length:]:
            node = node.get(int(token)) if isinstance(node, dict) else None
            if node is None:
                return [eos_token_id]
        allowed = [token for token in node if token is not None]
        if node.get(None):
            allowed.append(eos_token_id)
        return allowed or [eos_token_id]
    return callback


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


class QwenGenerator:
    """Generate one strict JSON object with a pinned local Qwen model."""

    def __init__(self, device: str, cache_dir: str | Path):
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")

        # Keep heavy imports inside initialization so importing the service
        # module does not load PyTorch or Transformers eagerly.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir),
        )
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir),
            torch_dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()
        self._torch = torch

    def choose(self, system: str, payload: Any) -> dict[str, Any]:
        """Return a parsed JSON object; never accept prose or invented names."""
        event("llm.input", {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "system": system,
            "payload": payload,
        })
        if not isinstance(system, str):
            raise GenerationError("Model instructions must be text")
        try:
            payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            event("llm.error", {"type": type(exc).__name__, "phase": "serialize_payload"})
            raise GenerationError("Model input must be JSON-serializable") from exc

        messages = [
            {
                "role": "system",
                "content": (
                    f"{system}\n\n"
                    "Return exactly one valid JSON object and no other text. "
                    "Use only IDs and values explicitly present in the input; "
                    "never invent, expand, or rewrite names."
                ),
            },
            {"role": "user", "content": payload_text},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"]
        except Exception as exc:
            event("llm.error", {"type": type(exc).__name__, "phase": "encode_input"})
            raise GenerationError("Could not encode model input") from exc

        input_tokens = int(input_ids.shape[-1])
        event("llm.input", {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "rendered_prompt": rendered,
            "input_tokens": input_tokens,
        })
        input_budget = MAX_TOTAL_TOKENS - MAX_NEW_TOKENS
        if input_tokens > input_budget:
            event("llm.error", {
                "type": "GenerationError",
                "phase": "input_limit",
                "input_tokens": input_tokens,
                "maximum_input_tokens": input_budget,
            })
            raise GenerationError(
                f"Model input is too large ({input_tokens} tokens); maximum is "
                f"{input_budget} input tokens"
            )

        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        try:
            choices = _finite_choices(payload)
            generation_options: dict[str, Any] = {}
            if choices is not None:
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is None:
                    raise GenerationError("Tokenizer has no EOS token for constrained output")
                trie = _choice_trie(self.tokenizer, choices)
                generation_options["prefix_allowed_tokens_fn"] = _allowed_next_tokens(
                    trie, input_tokens, int(eos_token_id)
                )
            with _GENERATION_LOCK, self._torch.inference_mode():
                output = self.model.generate(
                    **model_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    **generation_options,
                )
            completion_ids = output[0, input_ids.shape[-1] :]
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            event("llm.error", {"type": type(exc).__name__, "phase": "generation"})
            raise GenerationError("Local model generation failed") from exc

        event("llm.output", {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "raw_completion": completion,
        })

        try:
            result = json.loads(completion, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, ValueError) as exc:
            event("llm.error", {"type": type(exc).__name__, "phase": "parse_json",
                                 "raw_completion": completion})
            raise GenerationError("Local model did not return a strict JSON object") from exc
        if not isinstance(result, dict):
            event("llm.error", {"type": "GenerationError", "phase": "validate_json",
                                 "parsed": result})
            raise GenerationError("Local model response must be a JSON object")
        event("llm.output", {"parsed": result})
        return result
