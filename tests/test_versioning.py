"""Content-addressed model versions.

The point of a version is that it changes when, and only when, the thing it
names changes. These tests pin that down against real files.
"""

import pytest

from src.config import MODEL_KEYS
from src.inference import versioning
from src.inference.predictor import model_status
from tests.conftest import requires_model


@pytest.fixture(autouse=True)
def _clear_digest_cache():
    versioning.clear_cache()
    yield
    versioning.clear_cache()


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    """A model whose artifacts live in a directory the test can edit."""
    weights = tmp_path / "model_final.keras"
    weights.write_bytes(b"weights v1")
    monkeypatch.setitem(versioning.REQUIRED_ARTIFACTS, "lstm", (weights,))
    monkeypatch.setitem(versioning.MODEL_DIRS, "lstm", tmp_path)
    return tmp_path, weights


# --- digests ---------------------------------------------------------------


def test_missing_file_has_no_digest(tmp_path):
    assert versioning.file_digest(tmp_path / "absent.bin") is None


def test_identical_content_gives_identical_digests(tmp_path):
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")
    assert versioning.file_digest(first) == versioning.file_digest(second)


def test_different_content_gives_different_digests(tmp_path):
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert versioning.file_digest(first) != versioning.file_digest(second)


def test_line_endings_do_not_change_a_text_digest(tmp_path):
    # git normalises these to LF in the repository, so a Windows checkout holds
    # CRLF and a Linux one holds LF. The same committed file must version the
    # same on both - this was caught by CI and a local run disagreeing.
    crlf, lf = tmp_path / "crlf.json", tmp_path / "lf.json"
    crlf.write_bytes(b'{\r\n  "temperature": 2.0\r\n}\r\n')
    lf.write_bytes(b'{\n  "temperature": 2.0\n}\n')
    assert versioning.file_digest(crlf) == versioning.file_digest(lf)


def test_line_endings_still_matter_for_binary_artifacts(tmp_path):
    # Normalising inside a .keras or .safetensors file would corrupt the digest
    # of weights that legitimately contain those bytes.
    first, second = tmp_path / "a.keras", tmp_path / "b.keras"
    first.write_bytes(b"\x00\r\n\x01")
    second.write_bytes(b"\x00\n\x01")
    assert versioning.file_digest(first) != versioning.file_digest(second)


def test_text_content_changes_still_change_the_digest(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_bytes(b'{"temperature": 1.0}')
    second.write_bytes(b'{"temperature": 2.0}')
    assert versioning.file_digest(first) != versioning.file_digest(second)


# --- versions --------------------------------------------------------------


def test_version_is_stable_across_calls(fake_model):
    assert versioning.model_version("lstm") == versioning.model_version("lstm")


def test_version_changes_when_the_weights_change(fake_model):
    _, weights = fake_model
    before = versioning.model_version("lstm")

    weights.write_bytes(b"weights v2 - retrained")
    versioning.clear_cache()

    assert versioning.model_version("lstm") != before


def test_version_changes_when_calibration_changes(fake_model):
    directory, _ = fake_model
    before = versioning.model_version("lstm")

    # Same weights, different confidence: callers see different numbers, so it
    # must be a different version.
    (directory / "calibration.json").write_text('{"temperature": 2.0}', encoding="utf-8")
    versioning.clear_cache()

    assert versioning.model_version("lstm") != before


def test_the_preprocessing_spec_is_part_of_the_version(fake_model):
    directory, _ = fake_model
    before = versioning.model_version("lstm")
    (directory / "preprocessing.json").write_text(
        '{"spec": "glove-v2"}', encoding="utf-8"
    )
    assert directory / "preprocessing.json" in versioning.versioned_files("lstm")
    # A retrained model shown different text is a different model, even if
    # the weights file happened to hash the same.
    assert versioning.model_version("lstm") != before


def test_calibration_is_part_of_the_version(fake_model):
    directory, _ = fake_model
    (directory / "calibration.json").write_text('{"temperature": 2.0}', encoding="utf-8")
    assert "calibration.json" in versioning.artifact_digests("lstm")


def test_version_is_none_when_nothing_is_on_disk(monkeypatch, tmp_path):
    monkeypatch.setitem(
        versioning.REQUIRED_ARTIFACTS, "gru", (tmp_path / "absent.keras",)
    )
    monkeypatch.setitem(versioning.MODEL_DIRS, "gru", tmp_path)
    assert versioning.model_version("gru") is None


def test_version_is_short_enough_to_read(fake_model):
    assert len(versioning.model_version("lstm")) == versioning.VERSION_LENGTH


# --- exposure --------------------------------------------------------------


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_status_reports_a_version_field_for_every_model(model):
    entry = model_status()[model]
    assert "version" in entry
    assert "artifacts" in entry


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_available_models_have_a_version(model):
    entry = model_status()[model]
    if entry["available"]:
        assert entry["version"]
        assert entry["artifacts"]


@requires_model("lr")
def test_predictions_carry_the_serving_version():
    from src.inference.predictor import load_predictor

    prediction = load_predictor("lr").predict("this is fantastic")
    assert prediction.version == model_status()["lr"]["version"]


@requires_model("lr")
def test_api_returns_the_version(api_client):
    body = api_client.post("/predict", json={"text": "great", "model": "lr"}).json()
    assert body["version"] == model_status()["lr"]["version"]
