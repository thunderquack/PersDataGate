"""Install the CUDA PyTorch wheel separately from packages served by PyPI."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
PYPI_INDEX = "https://pypi.org/simple"
PIP_CACHE = os.environ.get("PIP_CACHE_DIR", "/pip-cache")
TORCH_LINE = re.compile(r"^\s*torch(?:\[[^]]+\])?\s*(?P<spec>[<>=!~].*)$", re.IGNORECASE)
INDEX_LINE = re.compile(r"^\s*--(?:extra-)?index-url(?:\s|=)|^\s*--find-links(?:\s|=)", re.IGNORECASE)


def _pip(*args: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "--cache-dir",
            PIP_CACHE,
            *args,
        ],
        check=True,
    )


def install_dependencies(requirements: Path, upgrade: bool = False) -> str:
    """Install CUDA 12.8 torch from its index, then resolve everything else on PyPI.

    Returns the selected torch version. Requirements may contain legacy index
    directives; these are omitted from the PyPI-only requirements passed to pip.
    """
    lines = requirements.read_text(encoding="utf-8").splitlines()
    torch_line = next((line.strip() for line in lines if TORCH_LINE.match(line)), None)
    if torch_line is None:
        raise ValueError(f"{requirements} must contain a torch version requirement")

    torch_args = ["install", "--no-deps", "--only-binary=:all:", "--index-url", CUDA_INDEX]
    if upgrade:
        torch_args.append("--upgrade")
    torch_args.append(torch_line)
    _pip(*torch_args)

    torch_version = importlib.metadata.version("torch")
    if not torch_version.endswith("+cu128"):
        raise RuntimeError(f"Expected a CUDA 12.8 PyTorch build; got {torch_version}")

    clean_lines: list[str] = []
    for line in lines:
        if INDEX_LINE.match(line):
            continue
        match = TORCH_LINE.match(line)
        if match:
            # Keep pip's PyPI resolver pinned to the CUDA build just installed.
            clean_lines.append(f"torch=={torch_version}")
        else:
            clean_lines.append(line)

    with tempfile.TemporaryDirectory(prefix="persdatagate-requirements-") as temp_dir:
        clean_requirements = Path(temp_dir) / "requirements.txt"
        clean_requirements.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")
        args = ["install", "--only-binary=:all:", "--index-url", PYPI_INDEX]
        recovered_wheels = Path(PIP_CACHE) / "recovered-wheels"
        if recovered_wheels.is_dir():
            # Optional salvage of already-downloaded NVIDIA wheels. This path is
            # deliberately excluded from the CUDA-index PyTorch install above.
            args.extend(["--find-links", str(recovered_wheels)])
        if upgrade:
            args.append("--upgrade")
        args.extend(["-r", str(clean_requirements)])
        _pip(*args)

    return torch_version
