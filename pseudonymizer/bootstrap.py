"""Install pinned service dependencies from the persistent pip cache, then start the API."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


APP_DIR = Path(__file__).resolve().parent
REQUIREMENTS = APP_DIR / "requirements.txt"
INSTALL_MARKER = Path("/tmp/pseudonymizer-requirements.sha256")


def main() -> None:
    requirements_hash = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not INSTALL_MARKER.exists() or INSTALL_MARKER.read_text().strip() != requirements_hash:
        print("Installing pseudonymizer dependencies (pip downloads are cached in /pip-cache)...", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--cache-dir",
                os.environ.get("PIP_CACHE_DIR", "/pip-cache"),
                "--only-binary=:all:",
                "-r",
                str(REQUIREMENTS),
            ],
            check=True,
        )
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
