"""Install pinned service dependencies from the persistent pip cache, then start the API."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

from pseudonymizer.dependencies import install_dependencies
from pseudonymizer.config import get_device


APP_DIR = Path(__file__).resolve().parent
REQUIREMENTS = {
    "cpu": APP_DIR / "requirements.cpu.txt",
    "cuda": APP_DIR / "requirements.cuda.txt",
}
INSTALL_MARKER = Path("/tmp/pseudonymizer-requirements.sha256")


def main() -> None:
    device = get_device()
    requirements = REQUIREMENTS[device]
    requirements_hash = f"{device}:{hashlib.sha256(requirements.read_bytes()).hexdigest()}"
    if not INSTALL_MARKER.exists() or INSTALL_MARKER.read_text().strip() != requirements_hash:
        print(
            f"Installing {device} dependencies: PyTorch from its selected index, other packages from PyPI; "
            "pip uses /pip-cache.",
            flush=True,
        )
        install_dependencies(requirements, device=device)
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        INSTALL_MARKER.write_text(requirements_hash)
    else:
        print("Pseudonymizer dependencies already installed for this container.", flush=True)

    os.execvpe(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "pseudonymizer.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--workers",
            "1",
            "--no-access-log",
        ],
        os.environ,
    )


if __name__ == "__main__":
    main()
