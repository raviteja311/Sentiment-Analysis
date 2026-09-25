"""Fetching the large weights from object storage.

No test here touches the network: downloading is exercised through a stubbed
hf_hub_download, because a unit test that needs the Hub to be reachable is a
test that fails for reasons unrelated to this code.
"""

import hashlib
import json
import logging

import pytest

from src import artifacts, config
from src.config import DEFAULT_MODELS_REPO, MODEL_KEYS, MODELS_DIR, REMOTE_ARTIFACTS
from src.inference.predictor import check_artifacts

# The stub downloads below write these bytes, so this is what a manifest that
# agrees with them has to record.
WEIGHTS = b"weights"
WEIGHTS_SHA256 = hashlib.sha256(WEIGHTS).hexdigest()


def fake_download_into(root, requested=None):
    def fake_download(repo_id, filename, revision, local_dir):
        if requested is not None:
            requested.append(filename)
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(WEIGHTS)
        return str(path)

    return fake_download


def manifest_for(*relatives, repo=DEFAULT_MODELS_REPO, digest=WEIGHTS_SHA256):
    files = {rel: digest for rel in relatives}
    return {"repo": repo, "revision": "0" * 40, "files": files}


def test_only_large_weights_are_remote():
    remote = artifacts.remote_files()
    assert all(name.endswith((".keras", ".safetensors", ".bin")) for name in remote)


def test_training_only_files_are_not_fetched():
    # best.keras is the early-stopping checkpoint that model_final.keras
    # supersedes, and training_args.bin is a pickle of the Trainer's arguments.
    # Inference reads neither, so a fresh clone should not pay for them.
    remote = artifacts.remote_files()
    assert not any(name.endswith("best.keras") for name in remote)
    assert "roberta/training_args.bin" not in remote


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
    monkeypatch.setattr(artifacts, "load_remote_manifest", lambda: None)

    requested = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        fake_download_into(tmp_path, requested),
        raising=False,
    )

    downloaded = artifacts.fetch(["gru"])
    assert requested == list(REMOTE_ARTIFACTS["gru"])
    assert len(downloaded) == len(requested)

    # Second call has nothing left to do.
    assert artifacts.fetch(["gru"]) == []


def test_fetch_force_redownloads(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(artifacts, "load_remote_manifest", lambda: None)
    requested = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        fake_download_into(tmp_path, requested),
        raising=False,
    )

    artifacts.fetch(["roberta"])
    artifacts.fetch(["roberta"], force=True)
    assert requested == list(REMOTE_ARTIFACTS["roberta"]) * 2


# --- pinning and verification ----------------------------------------------
#
# The tokenizers live in git and the weights on the Hub. Without a pin, a
# newer publish silently paired new weights with an old tokenizer; without a
# digest, a corrupted or substituted download was loaded as if it were fine.


def test_a_download_matching_the_manifest_is_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        artifacts, "load_remote_manifest", lambda: manifest_for(*REMOTE_ARTIFACTS["gru"])
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_download_into(tmp_path), raising=False
    )

    downloaded = artifacts.fetch(["gru"])
    assert [p.read_bytes() for p in downloaded] == [WEIGHTS]


def test_a_download_with_the_wrong_digest_is_rejected_and_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    wrong = "f" * 64
    monkeypatch.setattr(
        artifacts,
        "load_remote_manifest",
        lambda: manifest_for(*REMOTE_ARTIFACTS["gru"], digest=wrong),
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_download_into(tmp_path), raising=False
    )

    with pytest.raises(artifacts.ArtifactVerificationError) as excinfo:
        artifacts.fetch(["gru"])

    # Both digests are named, and the bad file is gone rather than loadable.
    assert wrong in str(excinfo.value)
    assert WEIGHTS_SHA256 in str(excinfo.value)
    assert not (tmp_path / REMOTE_ARTIFACTS["gru"][0]).exists()


def test_a_file_the_manifest_does_not_list_is_kept_with_a_warning(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(artifacts, "load_remote_manifest", lambda: manifest_for())
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_download_into(tmp_path), raising=False
    )

    with caplog.at_level(logging.WARNING, logger="src.artifacts"):
        downloaded = artifacts.fetch(["gru"])
    assert len(downloaded) == 1
    assert "without verification" in caplog.text


def test_a_manifest_for_another_repo_does_not_reject_downloads(monkeypatch, tmp_path):
    # SENTIMENT_MODELS_REPO points somewhere the digests say nothing about.
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        artifacts,
        "load_remote_manifest",
        lambda: manifest_for(
            *REMOTE_ARTIFACTS["gru"], repo="someone/else", digest="f" * 64
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_download_into(tmp_path), raising=False
    )
    assert len(artifacts.fetch(["gru"], repo_id="my/fork")) == 1


def test_the_revision_comes_from_the_manifest():
    assert config.resolve_models_revision(None, {"revision": "abc123"}) == "abc123"


def test_the_environment_overrides_the_manifest_revision():
    assert config.resolve_models_revision("v2", {"revision": "abc123"}) == "v2"


def test_a_missing_manifest_falls_back_to_main_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="src.config"):
        assert config.resolve_models_revision(None, None) == "main"
    assert "main" in caplog.text


def test_an_unreadable_manifest_reads_as_absent(tmp_path):
    broken = tmp_path / "remote_manifest.json"
    broken.write_text("{not json", encoding="utf-8")
    assert config.load_remote_manifest(broken) is None
    assert config.load_remote_manifest(tmp_path / "nowhere.json") is None


def test_write_manifest_records_the_revision_and_every_digest_on_disk(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    for relative in artifacts.remote_files():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(WEIGHTS)

    target = tmp_path / "remote_manifest.json"
    written = artifacts.write_manifest("me/models", "a" * 40, path=target)

    assert written == json.loads(target.read_text(encoding="utf-8"))
    assert written["repo"] == "me/models"
    assert written["revision"] == "a" * 40
    assert written["files"] == {rel: WEIGHTS_SHA256 for rel in artifacts.remote_files()}


def test_write_manifest_keeps_digests_for_files_not_on_disk(monkeypatch, tmp_path):
    # Publishing one model must not unpin the others.
    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    target = tmp_path / "remote_manifest.json"
    target.write_text(
        json.dumps(
            manifest_for(*artifacts.remote_files(), repo="me/models", digest="b" * 64)
        ),
        encoding="utf-8",
    )
    only = tmp_path / REMOTE_ARTIFACTS["lstm"][0]
    only.parent.mkdir(parents=True, exist_ok=True)
    only.write_bytes(WEIGHTS)

    written = artifacts.write_manifest("me/models", "c" * 40, path=target)
    assert written["files"][REMOTE_ARTIFACTS["lstm"][0]] == WEIGHTS_SHA256
    assert written["files"][REMOTE_ARTIFACTS["gru"][0]] == "b" * 64


def test_the_committed_manifest_pins_every_remote_file():
    manifest = config.load_remote_manifest()
    assert manifest is not None, "models/remote_manifest.json is missing"
    assert manifest["repo"] == DEFAULT_MODELS_REPO
    # A commit SHA, not a branch name: branches move, pins must not.
    assert len(manifest["revision"]) == 40 and int(manifest["revision"], 16) >= 0
    assert set(manifest["files"]) == set(artifacts.remote_files())
    assert all(len(digest) == 64 for digest in manifest["files"].values())


@pytest.mark.parametrize("relative", artifacts.remote_files())
def test_the_local_weights_match_the_committed_manifest(relative):
    from src.inference.predictor import is_lfs_pointer
    from src.inference.versioning import file_digest

    path = artifacts.local_path(relative)
    if not path.is_file() or is_lfs_pointer(path):
        pytest.skip(f"{relative} not fetched")
    expected = config.load_remote_manifest()["files"][relative]
    assert file_digest(path) == expected, (
        f"{relative} on disk differs from the manifest; either re-fetch it or "
        "regenerate the manifest with `python -m src.artifacts --write-manifest`"
    )


def test_publishing_pins_the_uploaded_commit(monkeypatch, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "publish_weights", config.PROJECT_ROOT / "scripts" / "publish_weights.py"
    )
    publish = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publish)

    monkeypatch.setattr(artifacts, "MODELS_DIR", tmp_path)
    for relative in artifacts.remote_files():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(WEIGHTS)

    written = {}

    class FakeApi:
        def repo_exists(self, repo_id, repo_type):
            return True

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
            return type("Commit", (), {"oid": "d" * 40})()

    def fake_write_manifest(repo_id, revision):
        written.update(repo=repo_id, revision=revision)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(publish, "write_manifest", fake_write_manifest)

    assert publish.main(["--repo", "me/models"]) == 0
    assert written == {"repo": "me/models", "revision": "d" * 40}


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
