from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

from pseudonymizer import bootstrap, dependencies
from pseudonymizer.config import get_device


def _fake_model(monkeypatch, *, cuda_available: bool, snapshot_path: Path):
    calls = {"cuda_checks": 0, "to": [], "eval": 0, "loads": []}
    snapshot_path.mkdir()
    source_config = {
        "tokenizer_class": "DebertaV2Tokenizer",
        "additional_special_tokens": ["[BASE]"],
        "extra_special_tokens": ["[SEP_STRUCT]", "[SEP_TEXT]"],
    }
    (snapshot_path / "tokenizer_config.json").write_text(json.dumps(source_config), encoding="utf-8")
    (snapshot_path / "config.json").write_text("{}", encoding="utf-8")
    (snapshot_path / "encoder_config").mkdir()
    (snapshot_path / "encoder_config" / "config.json").write_text("{}", encoding="utf-8")
    token_ids = [250102, 250103]
    (snapshot_path / "tokenizer.json").write_text(json.dumps({
        "added_tokens": [
            {"id": token_id, "content": token}
            for token_id, token in zip(token_ids, source_config["extra_special_tokens"])
        ],
    }), encoding="utf-8")
    (snapshot_path / "model.safetensors").write_bytes(b"test fixture")

    class FakeModel:
        processor = types.SimpleNamespace(tokenizer=object())

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

    def from_pretrained(*args, **kwargs):
        compatible_path = Path(args[0])
        compatible_config = json.loads((compatible_path / "tokenizer_config.json").read_text(encoding="utf-8"))
        compatible_tokenizer = json.loads((compatible_path / "tokenizer.json").read_text(encoding="utf-8"))
        calls["loads"].append((args, kwargs, compatible_config, compatible_tokenizer))
        return model

    gliner2 = types.ModuleType("gliner2")
    gliner2.AutoExtractor = types.SimpleNamespace(from_pretrained=from_pretrained)
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: str(snapshot_path)
    from pseudonymizer.local_llm import QwenGenerator

    class FakeGenerator:
        def __init__(self, device, cache_dir):
            calls["generator"] = (device, cache_dir)

    monkeypatch.setattr("pseudonymizer.local_llm.QwenGenerator", FakeGenerator)
    persons = types.ModuleType("pseudonymizer.persons")
    persons.PersonPipeline = lambda generator: types.SimpleNamespace(generator=generator)
    monkeypatch.setitem(sys.modules, "pseudonymizer.persons", persons)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "gliner2", gliner2)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(Path, "symlink_to", lambda target_file, source_file: shutil.copyfile(source_file, target_file))
    return calls


def test_device_configuration_defaults_and_rejects_invalid_values(monkeypatch):
    monkeypatch.delenv("DEVICE", raising=False)
    assert get_device() == "cuda"
    monkeypatch.setenv("DEVICE", " CPU ")
    assert get_device() == "cpu"
    monkeypatch.setenv("DEVICE", "metal")
    with pytest.raises(ValueError, match="DEVICE must be one of"):
        get_device()


def test_cpu_startup_skips_cuda_probe_and_reports_device(monkeypatch, tmp_path: Path):
    from fastapi.testclient import TestClient
    from pseudonymizer.app import create_app

    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("HF_HOME", "/fake/cache")
    snapshot = tmp_path / "snapshot"
    calls = _fake_model(monkeypatch, cuda_available=False, snapshot_path=snapshot)
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").json() == {"ready": True, "device": "cpu"}
    loads = calls.pop("loads")
    assert calls == {
        "cuda_checks": 0, "to": [], "eval": 1,
        "generator": ("cpu", "/fake/cache"),
    }
    assert len(loads) == 1
    args, kwargs, normalized_config, tokenizer_json = loads[0]
    assert Path(args[0]) != snapshot
    assert kwargs == {"map_location": "cpu"}
    assert normalized_config["additional_special_tokens"] == ["[BASE]", "[SEP_STRUCT]", "[SEP_TEXT]"]
    assert "extra_special_tokens" not in normalized_config
    assert tokenizer_json["added_tokens"] == [
        {"id": 250102, "content": "[SEP_STRUCT]"},
        {"id": 250103, "content": "[SEP_TEXT]"},
    ]
    assert json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))["extra_special_tokens"] == ["[SEP_STRUCT]", "[SEP_TEXT]"]


def test_cuda_startup_uses_cuda_and_reports_device(monkeypatch, tmp_path: Path):
    from fastapi.testclient import TestClient
    from pseudonymizer.app import create_app

    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("HF_HOME", "/fake/cache")
    snapshot = tmp_path / "snapshot"
    calls = _fake_model(monkeypatch, cuda_available=True, snapshot_path=snapshot)
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").json() == {"ready": True, "device": "cuda"}
    loads = calls.pop("loads")
    assert calls == {
        "cuda_checks": 1, "to": [], "eval": 1,
        "generator": ("cuda", "/fake/cache"),
    }
    assert len(loads) == 1
    args, kwargs, normalized_config, tokenizer_json = loads[0]
    assert Path(args[0]) != snapshot
    assert kwargs == {"map_location": "cuda"}
    assert normalized_config["additional_special_tokens"] == ["[BASE]", "[SEP_STRUCT]", "[SEP_TEXT]"]
    assert "extra_special_tokens" not in normalized_config
    assert tokenizer_json["added_tokens"] == [
        {"id": 250102, "content": "[SEP_STRUCT]"},
        {"id": 250103, "content": "[SEP_TEXT]"},
    ]
    assert json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))["extra_special_tokens"] == ["[SEP_STRUCT]", "[SEP_TEXT]"]


def test_cuda_startup_fails_when_gpu_is_unavailable(monkeypatch, tmp_path: Path):
    from fastapi.testclient import TestClient
    from pseudonymizer.app import create_app

    monkeypatch.setenv("DEVICE", "cuda")
    calls = _fake_model(monkeypatch, cuda_available=False, snapshot_path=tmp_path / "snapshot")
    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        with TestClient(create_app()):
            pass
    assert calls == {"cuda_checks": 1, "to": [], "eval": 0, "loads": []}


def test_invalid_device_fails_before_loading_model(monkeypatch):
    from fastapi.testclient import TestClient
    from pseudonymizer.app import create_app

    monkeypatch.setenv("DEVICE", "tpu")
    with pytest.raises(ValueError, match="DEVICE must be one of"):
        with TestClient(create_app()):
            pass


@pytest.mark.parametrize(
    ("device", "expected_torch", "expected_index"),
    [
        ("cpu", "2.7.1+cpu", "https://download.pytorch.org/whl/cpu"),
        ("cuda", "2.7.1+cu128", "https://download.pytorch.org/whl/cu128"),
    ],
)
def test_base_dependency_installer_selects_torch_build_and_keeps_other_deps_on_pypi(
    monkeypatch, tmp_path: Path, device, expected_torch, expected_index
):
    requirements = tmp_path / f"requirements.{device}.txt"
    requirements.write_text(
        "--extra-index-url https://pypi.nvidia.com\n"
        f"torch=={expected_torch}\nfastapi==0.115.12\n",
        encoding="utf-8",
    )
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
    assert dependencies.install_dependencies(requirements, device=device) == expected_torch
    assert len(pip_calls) == 2
    assert pip_calls[0][pip_calls[0].index("--index-url") + 1] == expected_index
    assert pip_calls[0][-1] == f"torch=={expected_torch}"
    assert pip_calls[1][pip_calls[1].index("--index-url") + 1] == dependencies.PYPI_INDEX


def test_pip_invocations_disable_download_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(dependencies.subprocess, "run", lambda command, **kwargs: calls.append(command))
    dependencies._pip("install", "sample==1")
    assert "--no-cache-dir" in calls[0]


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


def test_pseudonymizer_dockerfile_pins_cpu_and_cuda_torch_builds():
    dockerfile = Path(__file__).parents[1] / "pseudonymizer/Dockerfile"
    contents = dockerfile.read_text(encoding="utf-8").lower()
    assert '"torch==2.7.1+cpu"' in contents
    assert '"torch==2.7.1+cu128"' in contents
    assert "download.pytorch.org/whl/cpu" in contents
    assert "download.pytorch.org/whl/cu128" in contents


def test_pseudonymizer_dockerfile_installs_direct_dependencies_in_individual_layers():
    dockerfile = Path(__file__).parents[1] / "pseudonymizer/Dockerfile"
    lines = dockerfile.read_text(encoding="utf-8").lower().splitlines()
    pip_run_lines = [line.strip() for line in lines if line.strip().startswith("run python -m pip install")]

    assert len(pip_run_lines) >= 9
    expected_packages = {
        '"fastapi==0.115.12"',
        '"uvicorn[standard]==0.34.2"',
        '"pydantic==2.11.4"',
        '"transformers==4.51.3"',
        '"huggingface-hub==0.30.2"',
        '"tokenizers==0.21.1"',
        '"hf-xet==1.3.0"',
        '"pymorphy3==2.0.3"',
        '"peft==0.15.2"',
        '"gliner2[local]==2.0.0"',
    }
    for package in expected_packages:
        matching_lines = [line for line in pip_run_lines if package in line]
        assert len(matching_lines) == 1, f"Expected one individual install layer for {package}"
    assert not any(" -r " in line for line in lines if line.strip().startswith("run python -m pip install"))
    assert "python -m pseudonymizer.dependencies" not in "\n".join(lines)
    assert "base-requirements.txt" not in "\n".join(lines)


def test_runtime_installer_uses_constraints_and_skips_empty_requirements(monkeypatch, tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    constraints = tmp_path / "base-constraints.txt"
    requirements.write_text("\n# no optional additions yet\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(dependencies, "_pip", lambda *args: calls.append(args))
    dependencies.install_runtime_dependencies(requirements, constraints)
    assert calls == []

    requirements.write_text("rapidfuzz==3.12.2\n", encoding="utf-8")
    dependencies.install_runtime_dependencies(requirements, constraints)
    assert len(calls) == 1
    assert calls[0] == (
        "install", "--only-binary=:all:", "--index-url", dependencies.PYPI_INDEX,
        "-r", str(requirements), "-c", str(constraints),
    )


def test_bootstrap_validates_image_device_before_installing_delta(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("BASE_DEVICE", "cuda")
    monkeypatch.setattr(bootstrap, "REQUIREMENTS", tmp_path / "requirements.txt")
    installs = []
    monkeypatch.setattr(bootstrap, "install_runtime_dependencies", lambda *args: installs.append(args))
    with pytest.raises(RuntimeError, match="image contains the cuda"):
        bootstrap.main()
    assert installs == []


def test_bootstrap_checks_runtime_delta_and_starts_service(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("BASE_DEVICE", "cpu")
    requirements = tmp_path / "requirements.txt"
    monkeypatch.setattr(bootstrap, "REQUIREMENTS", requirements)
    constraints = tmp_path / "constraints.txt"
    monkeypatch.setattr(bootstrap, "BASE_CONSTRAINTS", constraints)
    installs = []
    monkeypatch.setattr(bootstrap, "install_runtime_dependencies", lambda *args: installs.append(args))
    checks = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: checks.append(args))

    class StopAfterStart(Exception):
        pass

    monkeypatch.setattr(bootstrap.os, "execvpe", lambda *args: (_ for _ in ()).throw(StopAfterStart()))
    with pytest.raises(StopAfterStart):
        bootstrap.main()
    assert installs == [(requirements, constraints)]
    assert checks == [([sys.executable, "-m", "pip", "check"],)]
