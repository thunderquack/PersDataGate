"""Small, local Qwen-backed JSON generator used by person processing."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MAX_TOTAL_TOKENS = 8192
MAX_NEW_TOKENS = 256

# Generation is serialized because one model instance is shared by the API's
# request handlers and may share a single GPU.
_GENERATION_LOCK = threading.Lock()


class GenerationError(RuntimeError):
    """Safe, bounded error from model input validation or generation."""


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
        if not isinstance(system, str):
            raise GenerationError("Model instructions must be text")
        try:
            payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
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
            raise GenerationError("Could not encode model input") from exc

        input_tokens = int(input_ids.shape[-1])
        input_budget = MAX_TOTAL_TOKENS - MAX_NEW_TOKENS
        if input_tokens > input_budget:
            raise GenerationError(
                f"Model input is too large ({input_tokens} tokens); maximum is "
                f"{input_budget} input tokens"
            )

        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        try:
            with _GENERATION_LOCK, self._torch.inference_mode():
                output = self.model.generate(
                    **model_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            completion_ids = output[0, input_ids.shape[-1] :]
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            raise GenerationError("Local model generation failed") from exc

        try:
            result = json.loads(completion, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise GenerationError("Local model did not return a strict JSON object") from exc
        if not isinstance(result, dict):
            raise GenerationError("Local model response must be a JSON object")
        return result
