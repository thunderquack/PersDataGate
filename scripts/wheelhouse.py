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
# Docker reads one immutable-by-hash wheel pool; locks choose exact versions.
CANDIDATE_LOCK = CANDIDATE / 'requirements.lock.txt'
PREVIOUS_LOCK = DEPS / 'previous' / 'requirements.lock.txt'
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
        parts = [f'{rendered_name}=={version}']
        parts.extend(f'--hash=sha256:{digest}' for digest in hashes)
        lines.append(' '.join(parts))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(destination)


def parse_lock(path: Path) -> dict[str, tuple[str, set[str]]]:
    requirements: dict[str, tuple[str, set[str]]] = {}
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        tokens = line.split()
        if '==' not in tokens[0]:
            raise ValueError(f'Unsupported lock line: {raw_line}')
        name, version = tokens[0].split('==', 1)
        key = canonical_name(name)
        if key in requirements:
            raise ValueError(f'Duplicate package in lock: {key}')
        hashes: set[str] = set()
        for option in tokens[1:]:
            if not option.startswith('--hash=sha256:'):
                raise ValueError(f'Unsupported lock option: {option}')
            hashes.add(option.removeprefix('--hash=sha256:'))
        if not hashes:
            raise ValueError(f'Lock entry has no hash: {key}')
        requirements[key] = (version, hashes)
    if not requirements:
        raise ValueError(f'Lock is empty: {path}')
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


def pip_download(
    requirements: Path,
    destination: Path,
    find_links: list[Path],
    *,
    locked: bool,
    constraints: Path | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, '-m', 'pip', 'download', '--disable-pip-version-check',
        '--only-binary=:all:', '--dest', str(destination),
        '--extra-index-url', 'https://download.pytorch.org/whl/cu128',
    ]
    for directory in find_links:
        if directory.exists():
            command.extend(['--find-links', str(directory)])
    if locked:
        command.extend(['--require-hashes', '--no-deps'])
    if constraints:
        command.extend(['-c', str(constraints)])
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
        parts = [f"{name}=={version}"]
        parts.extend(f"--hash=sha256:{digest}" for digest in sorted(hashes))
        lines.append(' '.join(parts))
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
            merged = merge_wheels(staging, ACTIVE_WHEELS)
            temporary_lock = ACTIVE_LOCK.with_suffix('.prepare')
            write_lock(staged, temporary_lock)
            validate_wheelhouse(temporary_lock, merged)
            ACTIVE_LOCK.parent.mkdir(parents=True, exist_ok=True)
            temporary_lock.replace(ACTIVE_LOCK)
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
    with tempfile.TemporaryDirectory(prefix='update-', dir=DEPS) as temp:
        staging = Path(temp) / 'wheels'
        torch_staging = Path(temp) / 'torch'
        torch_staging.mkdir(parents=True, exist_ok=True)
        torch_constraint = read_torch_constraint(UPDATE_INPUT)
        torch_download = [
            sys.executable, '-m', 'pip', 'download', '--disable-pip-version-check',
            '--index-url', 'https://download.pytorch.org/whl/cu128',
            '--only-binary=:all:', '--no-deps', '--dest', str(torch_staging),
        ]
        if ACTIVE_WHEELS.exists():
            torch_download.extend(['--find-links', str(ACTIVE_WHEELS)])
        torch_download.append(torch_constraint)
        subprocess.run(torch_download, check=True)
        torch_only = wheels_in(torch_staging)
        assert_torch_cuda(torch_only)
        torch_version = wheel_metadata(torch_only[0])[1]
        constraints = Path(temp) / 'constraints.txt'
        constraints.write_text(f'torch=={torch_version}\n', encoding='utf-8')
        pip_download(UPDATE_INPUT, staging, [ACTIVE_WHEELS, torch_staging], locked=False, constraints=constraints)
        downloaded = wheels_in(staging)
        if not downloaded:
            raise ValueError('pip did not produce any candidate wheels.')
        assert_torch_cuda(downloaded)
        # Candidate artifacts are additive; old wheels remain available for rollback.
        merged = merge_wheels(staging, ACTIVE_WHEELS)
        write_lock(downloaded, CANDIDATE_LOCK)
        validate_wheelhouse(CANDIDATE_LOCK, merged)
    print(f'Candidate lock ready: {CANDIDATE_LOCK.relative_to(DEPS)}')
    print('Build and validate this candidate image before running Promote.')


def promote() -> None:
    if not CANDIDATE_LOCK.exists():
        raise FileNotFoundError('No candidate lock exists. Run Update and validate the candidate first.')
    validate_wheelhouse(CANDIDATE_LOCK, wheels_in(ACTIVE_WHEELS))
    if ACTIVE_LOCK.exists() and ACTIVE_LOCK.read_bytes() == CANDIDATE_LOCK.read_bytes():
        print('Candidate is already the active lock; nothing to promote.')
        return
    ACTIVE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    temporary = ACTIVE_LOCK.with_suffix(ACTIVE_LOCK.suffix + '.partial')
    if ACTIVE_LOCK.exists():
        PREVIOUS_LOCK.parent.mkdir(parents=True, exist_ok=True)
        previous_temporary = PREVIOUS_LOCK.with_suffix(PREVIOUS_LOCK.suffix + '.partial')
        shutil.copy2(ACTIVE_LOCK, previous_temporary)
        previous_temporary.replace(PREVIOUS_LOCK)
    shutil.copy2(CANDIDATE_LOCK, temporary)
    temporary.replace(ACTIVE_LOCK)
    print(f'Promoted candidate lock to {ACTIVE_LOCK.relative_to(ROOT)}')
    print('The next Compose build will install the promoted versions from local wheels.')


def read_torch_constraint(path: Path) -> str:
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or line.startswith('--'):
            continue
        if re.match(r'^torch\s*[<>=!~]', line):
            return re.sub(r'\s+', '', line)
    raise ValueError(f'No torch version bounds found in {path}')


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
