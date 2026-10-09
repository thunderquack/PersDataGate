from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys
import types

import pytest

from pseudonymizer import bootstrap, dependencies
from pseudonymizer.config import get_device


def _fake_model(monkeypatch, *, cuda_available: bool):
    calls = {"cuda_checks": 0, "to": [], "eval": 0}

    class FakeModel:
        data_processor = types.SimpleNamespace(transformer_tokenizer=object())

        def to(self, device):
            calls["to"].append(device)
            return self

        def eval(self):
            calls["eval"] += 1
            return self

    model = FakeModel()

    def is_available():
        calls["cuda_checks"] += 1
        return cuda_available

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=is_available)

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    torch.inference_mode = InferenceMode

    gliner = types.ModuleType("gliner")
    gliner.GLiNER = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: "/fake/model-snapshot"
    from pseudonymizer.local_llm import QwenGenerator

    class FakeGenerator:
        def __init__(self, device, cache_dir):
            calls["generator"] = (device, cache_dir)

    monkeypatch.setattr("pseudonymizer.local_llm.QwenGenerator", FakeGenerator)
    persons = types.ModuleType("pseudonymizer.persons")
    persons.PersonPipeline = lambda generator: types.SimpleNamespace(generator=generator)
    monkeypatch.setitem(sys.modules, "pseudonymizer.persons", persons)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "gliner", gliner)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def test_device_configuration_defaults_and_rejects_invalid_values(monkeypatch):
    monkeypatch.delenv("DEVICE", raising=False)
    assert get_device() == "cuda"
    monkeypatch.setenv("DEVICE", " CPU ")
    assert get_device() == "cpu"
    monkeypatch.setenv("DEVICE", "metal")
    with pytest.raises(ValueError, match="DEVICE must be one of"):
        get_device()


def test_cpu_startup_skips_cuda_probe_and_reports_device(monkeypatch):
    from fastapi.testclient import TestClient
    from pseudonymizer.app import create_app

    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("HF_HOME", "/fake/cache")
    calls = _fake_model(monkeypatch, cuda_available=False)
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").json() == {"ready": True, "device": "cpu"}
    assert calls == {
        "cuda_checks": 0, "to": ["cpu"], "eval": 1,
        "generator": ("cpu", "/fake/cache"),
    }


def test_cuda_startup_uses_cuda_and_reports_device(monkeypatch):
    from fastapi.testclient import TestClient
    from pseudonymizer.app import create_app

    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("HF_HOME", "/fake/cache")
    calls = _fake_model(monkeypatch, cuda_available=True)
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").json() == {"ready": True, "device": "cuda"}
    assert calls == {
        "cuda_checks": 1, "to": ["cuda"], "eval": 1,
        "generator": ("cuda", "/fake/cache"),
    }


def test_cuda_startup_fails_when_gpu_is_unavailable(monkeypatch):
    from fastapi.testclient import TestClient
    from pseudonymizer.app import create_app

    monkeypatch.setenv("DEVICE", "cuda")
    calls = _fake_model(monkeypatch, cuda_available=False)
    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        with TestClient(create_app()):
            pass
    assert calls == {"cuda_checks": 1, "to": [], "eval": 0}


def test_invalid_device_fails_before_loading_model(monkeypatch):
    from fastapi.testclient import TestClient
    from pseudonymizer.app import create_app

    monkeypatch.setenv("DEVICE", "tpu")
    with pytest.raises(ValueError, match="DEVICE must be one of"):
        with TestClient(create_app()):
            pass


@pytest.mark.parametrize(
    ("device", "expected_torch", "expected_index", "upgrade"),
    [
        ("cpu", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", False),
        ("cuda", "2.7.1+cu128", "https://download.pytorch.org/whl/cu128", False),
        ("cpu", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu", True),
        ("cuda", "2.7.1+cu128", "https://download.pytorch.org/whl/cu128", True),
    ],
)
def test_dependency_installer_selects_torch_build_and_keeps_other_deps_on_pypi(
    monkeypatch, tmp_path: Path, device, expected_torch, expected_index, upgrade
):
    requirements = tmp_path / f"requirements.{device}.txt"
    requirements.write_text(
        "--extra-index-url https://pypi.nvidia.com\n"
        f"torch=={expected_torch}\nfastapi==0.115.12\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dependencies, "PIP_CACHE", str(tmp_path / "pip-cache"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: expected_torch)
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: ["filelock>=3"])
    pip_calls = []

    def fake_pip(*args):
        pip_calls.append(args)
        if "-r" in args:
            requirements_file = Path(args[args.index("-r") + 1])
            constraints_file = Path(args[args.index("-c") + 1])
            contents = requirements_file.read_text(encoding="utf-8")
            assert "fastapi==0.115.12" in contents
            assert "filelock>=3" in contents
            assert "torch==" not in contents
            assert "pypi.nvidia.com" not in contents
            assert constraints_file.read_text(encoding="utf-8") == f"torch=={expected_torch}\n"

    monkeypatch.setattr(dependencies, "_pip", fake_pip)
    assert dependencies.install_dependencies(requirements, device=device, upgrade=upgrade) == expected_torch
    assert len(pip_calls) == 2
    assert pip_calls[0][pip_calls[0].index("--index-url") + 1] == expected_index
    assert pip_calls[0][-1] == f"torch=={expected_torch}"
    assert pip_calls[1][pip_calls[1].index("--index-url") + 1] == dependencies.PYPI_INDEX
    assert ("--upgrade" in pip_calls[0]) is upgrade
    assert ("--upgrade" in pip_calls[1]) is upgrade


@pytest.mark.parametrize(("device", "wrong_version"), [("cpu", "2.7.1+cu128"), ("cuda", "2.7.1+cpu")])
def test_dependency_installer_rejects_wrong_torch_build_before_installing_other_packages(
    monkeypatch, tmp_path: Path, device, wrong_version
):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch>=2.7.1,<3\nfastapi>=0.115\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(dependencies, "_pip", lambda *args: calls.append(args))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: wrong_version)
    with pytest.raises(RuntimeError, match="Expected a PyTorch"):
        dependencies.install_dependencies(requirements, device=device)
    assert len(calls) == 1


def test_dependency_installer_rejects_invalid_device_before_pip(monkeypatch, tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.7.1\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(dependencies, "_pip", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="Unsupported device"):
        dependencies.install_dependencies(requirements, device="metal")
    assert calls == []


def test_cpu_requirements_do_not_request_cuda_or_nvidia_packages():
    requirements = Path(__file__).parents[1] / "pseudonymizer/requirements.cpu.txt"
    contents = requirements.read_text(encoding="utf-8").lower()
    assert "torch==" in contents and "+cpu" in contents
    assert "+cu" not in contents
    assert "nvidia-" not in contents


def test_bootstrap_install_fingerprint_includes_device_and_skips_same_device(
    monkeypatch, tmp_path: Path
):
    cpu_requirements = tmp_path / "requirements.cpu.txt"
    cuda_requirements = tmp_path / "requirements.cuda.txt"
    cpu_requirements.write_text("torch==2.7.1+cpu\n", encoding="utf-8")
    # Identical requirements make this prove the device prefix itself changes
    # the install fingerprint, rather than only the selected file contents.
    cuda_requirements.write_text("torch==2.7.1+cpu\n", encoding="utf-8")
    marker = tmp_path / "installed.sha256"
    monkeypatch.setattr(bootstrap, "REQUIREMENTS", {"cpu": cpu_requirements, "cuda": cuda_requirements})
    monkeypatch.setattr(bootstrap, "INSTALL_MARKER", marker)
    installs = []
    monkeypatch.setattr(bootstrap, "install_dependencies", lambda path, device: installs.append((path, device)))
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: None)

    class StopAfterStart(Exception):
        pass

    monkeypatch.setattr(bootstrap.os, "execvpe", lambda *args: (_ for _ in ()).throw(StopAfterStart()))
    for device in ("cpu", "cpu", "cuda"):
        monkeypatch.setenv("DEVICE", device)
        with pytest.raises(StopAfterStart):
            bootstrap.main()

    assert installs == [(cpu_requirements, "cpu"), (cuda_requirements, "cuda")]
    assert cpu_requirements.read_bytes() == cuda_requirements.read_bytes()
    assert marker.read_text(encoding="utf-8") == (
        "cuda:" + __import__("hashlib").sha256(cuda_requirements.read_bytes()).hexdigest()
    )
