"""Install optional runtime dependencies and start the API."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from pseudonymizer.config import get_device
from pseudonymizer.dependencies import install_runtime_dependencies


APP_DIR = Path(__file__).resolve().parent
REQUIREMENTS = APP_DIR / "requirements.txt"
BASE_CONSTRAINTS = Path("/app/base-constraints.txt")


def main() -> None:
    device = get_device()
    base_device = os.environ.get("BASE_DEVICE", "cuda").strip().lower()
    if device != base_device:
        raise RuntimeError(
            f"This image contains the {base_device} dependency set, but DEVICE={device} was requested. "
            "Rebuild the pseudonymizer image for the selected device."
        )

    print(
        f"Checking optional runtime dependencies from {REQUIREMENTS}; "
        "the image dependency versions are pinned.",
        flush=True,
    )
    install_runtime_dependencies(REQUIREMENTS, BASE_CONSTRAINTS)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

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
