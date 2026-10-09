"""Runtime configuration shared by bootstrap and the API app."""

from __future__ import annotations

import os


SUPPORTED_DEVICES = ("cpu", "cuda")


def get_device() -> str:
    device = os.getenv("DEVICE", "cuda").strip().lower()
    if device not in SUPPORTED_DEVICES:
        choices = ", ".join(SUPPORTED_DEVICES)
        raise ValueError(f"DEVICE must be one of {choices}; got {device!r}")
    return device
