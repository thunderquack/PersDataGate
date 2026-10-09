import pytest


@pytest.fixture(autouse=True)
def isolated_pipeline_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_LOG_DIR", str(tmp_path / "pipeline-logs"))
