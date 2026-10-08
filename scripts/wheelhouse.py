#!/usr/bin/env python3
"""Manage the Linux/Python 3.11 wheelhouse used by the pseudonymizer image."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path


ROOT = Path('/repo')
DEPS = Path('/deps')
ACTIVE_WHEELS = DEPS / 'wheels'
ACTIVE_LOCK = ROOT / 'pseudonymizer' / 'requirements.lock.txt'
CANDIDATE = DEPS / 'candidate'
CANDIDATE_WHEELS = CANDIDATE / 'wheels'
CANDIDATE_LOCK = CANDIDATE / 'requirements.lock.txt'
PINNED_INPUT = ROOT / 'pseudonymizer' / 'requirements.txt'
UPDATE_INPUT = ROOT / 'pseudonymizer' / 'requirements.in'


def canonical_name(value: str) -> str:
    return re.sub(r'[-_.]+', '-', value).lower()


def wheel_metadata(path: Path) -> tuple[str, str]:
    if path.suffix != '.whl':
        raise ValueError(f'Not a wheel file: {path.name}')
    with zipfile.ZipFile(path) as archive:
        metadata_paths = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metadata_paths) != 1:
            raise ValueError(f'Expected one METADATA file in {path.name}')
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode('utf-8'))
    name, version = metadata.get('Name'), metadata.get('Version')
    if not name or not version:
        raise ValueError(f'Missing Name or Version in {path.name}')
    return canonical_name(name), version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def wheel_inventory(paths: list[Path]) -> dict[tuple[str, str], list[tuple[Path, str]]]:
    inventory: dict[tuple[str, str], list[tuple[Path, str]]] = {}
    for path in paths:
        name, version = wheel_metadata(path)
        inventory.setdefault((name, version), []).append((path, sha256(path)))
    return inventory


def wheels_in(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob('*.whl') if path.is_file())


def write_lock(wheels: list[Path], destination: Path) -> None:
    inventory = wheel_inventory(wheels)
    packages: dict[str, tuple[str, list[str]]] = {}
    for (name, version), candidates in inventory.items():
        if name in packages:
            prior_version = packages[name][0]
            raise ValueError(f'Multiple versions for {name}: {prior_version} and {version}')
        packages[name] = (version, sorted({digest for _, digest in candidates}))

    lines = [
        '# Generated from downloaded Linux cp311 wheels by scripts/wheelhouse.py.',
        '# Do not edit by hand. Every artifact is pinned by SHA-256.',
    ]
    for name in sorted(packages):
        version, hashes = packages[name]
        rendered_name = name.replace('-', '_')
        lines.append(f'{rendered_name}=={version} \\\n+    ' + ' \\\n+    '.join(f'--hash=sha256:{digest}' for digest in hashes))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(destination)


def parse_lock(path: Path) -> dict[str, tuple[str, set[str]]]:
    requirements: dict[str, tuple[str, set[str]]] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '==' in line and not line.startswith('--hash='):
            requirement = line.split('\\', 1)[0].strip()
            name, version = requirement.split('==', 1)
            current = canonical_name(name)
            if current in requirements:
                raise ValueError(f'Duplicate package in lock: {current}')
            requirements[current] = (version.strip(), set())
        elif line.startswith('--hash=sha256:') and current:
            requirements[current][1].add(line.removeprefix('--hash=sha256:'))
        else:
            raise ValueError(f'Unsupported lock line: {raw_line}')
    if not requirements or any(not hashes for _, hashes in requirements.values()):
        raise ValueError(f'Lock is empty or has entries without hashes: {path}')
    return requirements


def validate_wheelhouse(lock_path: Path, wheel_paths: list[Path]) -> None:
    expected = parse_lock(lock_path)
    actual = wheel_inventory(wheel_paths)
    missing: list[str] = []
    for name, (version, hashes) in expected.items():
        available = actual.get((name, version), [])
        if not any(digest in hashes for _, digest in available):
            missing.append(f'{name}=={version}')
    if missing:
        raise ValueError(f'Wheelhouse does not contain lock-matching wheels: {", ".join(missing)}')


def pip_download(requirements: Path, destination: Path, find_links: list[Path], *, locked: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-m', 'pip', 'download', '--disable-pip-version-check',
        '--only-binary=:all:', '--dest', str(destination),
    ]
    for directory in find_links:
        if directory.exists():
            command.extend(['--find-links', str(directory)])
    if locked:
        command.extend(['--require-hashes', '--no-deps'])
    command.extend(['-r', str(requirements)])
    subprocess.run(command, check=True)


def merge_wheels(source: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    for wheel in wheels_in(source):
        target = destination / wheel.name
        if target.exists():
            if sha256(target) != sha256(wheel):
                raise ValueError(f'Wheel filename collision with different content: {wheel.name}')
            continue
        shutil.copy2(wheel, target)
    return wheels_in(destination)


def existing_satisfactions(lock: Path, wheel_dir: Path) -> set[str]:
    expected = parse_lock(lock)
    actual = wheel_inventory(wheels_in(wheel_dir))
    return {
        name for name, (version, hashes) in expected.items()
        if any(digest in hashes for _, digest in actual.get((name, version), []))
    }


def requirements_missing_from_lock(lock: Path, wheel_dir: Path, destination: Path) -> Path | None:
    """Write a temporary locked input containing only packages absent locally."""
    expected = parse_lock(lock)
    satisfied = existing_satisfactions(lock, wheel_dir)
    missing = set(expected) - satisfied
    if not missing:
        print('All hash-locked wheels are already present in .deps/wheels.')
        return None
    lines: list[str] = []
    for name in sorted(missing):
        version, hashes = expected[name]
        lines.append(f'{name}=={version} \\\n+    ' + ' \\\n+    '.join(f'--hash=sha256:{digest}' for digest in sorted(hashes)))
    requirement_file = destination / 'missing-requirements.txt'
    requirement_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return requirement_file


def prepare() -> None:
    ACTIVE_WHEELS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prepare-', dir=DEPS) as temp:
        staging = Path(temp) / 'wheels'
        if ACTIVE_LOCK.exists():
            requirement_file = requirements_missing_from_lock(ACTIVE_LOCK, ACTIVE_WHEELS, Path(temp))
            if requirement_file:
                pip_download(requirement_file, staging, [ACTIVE_WHEELS], locked=True)
            merged = merge_wheels(staging, ACTIVE_WHEELS)
            validate_wheelhouse(ACTIVE_LOCK, merged)
        else:
            if not PINNED_INPUT.exists():
                raise FileNotFoundError(f'Pinned initial dependencies not found: {PINNED_INPUT}')
            pip_download(PINNED_INPUT, staging, [ACTIVE_WHEELS], locked=False)
            staged = wheels_in(staging)
            if not staged:
                raise ValueError('pip did not produce any wheels from the pinned requirements.')
            assert_torch_cuda(staged)
            write_lock(staged, ACTIVE_LOCK)
            merged = merge_wheels(staging, ACTIVE_WHEELS)
            validate_wheelhouse(ACTIVE_LOCK, merged)
    print(f'Active lock ready: {ACTIVE_LOCK.relative_to(ROOT)}')
    print(f'Wheel files retained in {ACTIVE_WHEELS}')


def assert_torch_cuda(wheels: list[Path]) -> None:
    torch_wheels = [(path, wheel_metadata(path)[1]) for path in wheels if wheel_metadata(path)[0] == 'torch']
    if len(torch_wheels) != 1 or '+cu128' not in torch_wheels[0][1]:
        selected = ', '.join(version for _, version in torch_wheels) or 'none'
        raise ValueError(f'Expected exactly one CUDA 12.8 PyTorch wheel (version containing +cu128); selected: {selected}')


def update() -> None:
    if not UPDATE_INPUT.exists():
        raise FileNotFoundError(f'Compatible update constraints not found: {UPDATE_INPUT}')
    CANDIDATE_WHEELS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='update-', dir=DEPS) as temp:
        staging = Path(temp) / 'wheels'
        pip_download(UPDATE_INPUT, staging, [ACTIVE_WHEELS], locked=False)
        downloaded = wheels_in(staging)
        if not downloaded:
            raise ValueError('pip did not produce any candidate wheels.')
        assert_torch_cuda(downloaded)
        # Retain only the artifacts selected for this candidate. The active wheelhouse
        # remains unchanged until the candidate lock is explicitly promoted.
        for wheel in downloaded:
            destination = CANDIDATE_WHEELS / wheel.name
            if destination.exists() and sha256(destination) != sha256(wheel):
                raise ValueError(f'Candidate wheel filename collision: {wheel.name}')
            shutil.copy2(wheel, destination)
        write_lock(downloaded, CANDIDATE_LOCK)
        validate_wheelhouse(CANDIDATE_LOCK, wheels_in(CANDIDATE_WHEELS))
    print(f'Candidate lock ready: {CANDIDATE_LOCK.relative_to(DEPS)}')
    print('Build and validate this candidate image before running Promote.')


def promote() -> None:
    if not CANDIDATE_LOCK.exists():
        raise FileNotFoundError('No candidate lock exists. Run Update and validate the candidate first.')
    validate_wheelhouse(CANDIDATE_LOCK, wheels_in(CANDIDATE_WHEELS))
    ACTIVE_WHEELS.mkdir(parents=True, exist_ok=True)
    for wheel in wheels_in(CANDIDATE_WHEELS):
        destination = ACTIVE_WHEELS / wheel.name
        if destination.exists() and sha256(destination) != sha256(wheel):
            raise ValueError(f'Active wheel filename collision: {wheel.name}')
        if not destination.exists():
            shutil.copy2(wheel, destination)
    validate_wheelhouse(CANDIDATE_LOCK, wheels_in(ACTIVE_WHEELS))
    ACTIVE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    temporary = ACTIVE_LOCK.with_suffix(ACTIVE_LOCK.suffix + '.partial')
    shutil.copy2(CANDIDATE_LOCK, temporary)
    temporary.replace(ACTIVE_LOCK)
    print(f'Promoted candidate lock to {ACTIVE_LOCK.relative_to(ROOT)}')
    print('The next Compose build will install the promoted versions from local wheels.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('Prepare', 'Update', 'Promote'), nargs='?', default='Prepare')
    args = parser.parse_args()
    DEPS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('PIP_CACHE_DIR', str(DEPS / 'pip-cache'))
    try:
        {'Prepare': prepare, 'Update': update, 'Promote': promote}[args.mode]()
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(f'Wheelhouse error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
