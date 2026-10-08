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
        if not TORCH_LINE.match(line):
            clean_lines.append(line)
    # Torch is already installed. Resolve its dependencies explicitly on PyPI
    # without requesting the CUDA-only torch distribution from that index.
    torch_dependencies = importlib.metadata.requires("torch") or []
    clean_lines.extend(torch_dependencies)

    with tempfile.TemporaryDirectory(prefix="persdatagate-requirements-") as temp_dir:
        clean_requirements = Path(temp_dir) / "requirements.txt"
        clean_requirements.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")
        constraints = Path(temp_dir) / "constraints.txt"
        constraints.write_text(f"torch=={torch_version}\n", encoding="utf-8")
        args = ["install", "--only-binary=:all:", "--index-url", PYPI_INDEX]
        if upgrade:
            args.append("--upgrade")
        args.extend(["-r", str(clean_requirements), "-c", str(constraints)])
        _pip(*args)

    return torch_version
