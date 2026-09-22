"""Fetching the large weights from object storage.

No test here touches the network: downloading is exercised through a stubbed
hf_hub_download, because a unit test that needs the Hub to be reachable is a
test that fails for reasons unrelated to this code.
"""

import pytest

from src import artifacts
from src.config import MODEL_KEYS, MODELS_DIR, REMOTE_ARTIFACTS
from src.inference.predictor import check_artifacts


def test_only_large_weights_are_remote():
    remote = artifacts.remote_files()
    assert all(name.endswith((".keras", ".safetensors", ".bin")) for name in remote)


def test_the_linear_model_is_not_remote():
    # Its pipeline is ~600 KB and stays in git, so a plain clone can still serve
    # predictions without fetching anything.
    assert REMOTE_ARTIFACTS["lr"] == ()
    assert artifacts.remote_files(["lr"]) == []


@pytest.mark.parametrize("model", [m for m in MODEL_KEYS if m != "lr"])
def test_each_neural_model_has_remote_weights(model):
    assert artifacts.remote_files([model])


def test_remote_paths_resolve_under_the_models_directory():
    for relative in artifacts.remote_files():
        assert MODELS_DIR in artifacts.local_path(relative).parents


def test_is_remote_distinguishes_fetched_from_git_tracked():
    assert artifacts.is_remote(MODELS_DIR / "lstm" / "model_final.keras") is True
    assert artifacts.is_remote(MODELS_DIR / "lstm" / "tokenizer.joblib") is False
    assert artifacts.is_remote(MODELS_DIR / "lr" / "pipeline.joblib") is False


def test_missing_lists_absent_files(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    assert artifacts.missing(["lstm"]) == list(REMOTE_ARTIFACTS["lstm"])


def test_missing_treats_a_pointer_stub_as_absent(monkeypatch, tmp_path):
    from tests.conftest import LFS_POINTER_BYTES

    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    for relative in REMOTE_ARTIFACTS["gru"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(LFS_POINTER_BYTES)

    assert artifacts.missing(["gru"]) == list(REMOTE_ARTIFACTS["gru"])


def test_fetch_downloads_only_what_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)

    requested = []

    def fake_download(repo_id, filename, revision, local_dir):
        requested.append(filename)
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download, raising=False)

    downloaded = artifacts.fetch(["gru"])
    assert requested == list(REMOTE_ARTIFACTS["gru"])
    assert len(downloaded) == len(requested)

    # Second call has nothing left to do.
    assert artifacts.fetch(["gru"]) == []


def test_fetch_force_redownloads(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    requested = []

    def fake_download(repo_id, filename, revision, local_dir):
        requested.append(filename)
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download, raising=False)

    artifacts.fetch(["roberta"])
    artifacts.fetch(["roberta"], force=True)
    assert requested == list(REMOTE_ARTIFACTS["roberta"]) * 2


def test_error_for_a_remote_artifact_points_at_the_fetch_command(monkeypatch, tmp_path):
    from src.inference import predictor as predictor_module

    # Relocate the models directory so the remote set resolves under tmp_path,
    # where the file genuinely does not exist.
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    absent = tmp_path / "lstm" / "model_final.keras"
    monkeypatch.setitem(predictor_module.REQUIRED_ARTIFACTS, "lstm", (absent,))

    reason = check_artifacts("lstm")
    assert "make fetch-weights" in reason
    assert "git lfs" not in reason


def test_error_for_a_git_tracked_artifact_still_mentions_lfs(monkeypatch, tmp_path):
    from src.inference import predictor as predictor_module

    monkeypatch.setitem(
        predictor_module.REQUIRED_ARTIFACTS, "lr", (tmp_path / "pipeline.joblib",)
    )
    reason = check_artifacts("lr")
    assert "git lfs" in reason
