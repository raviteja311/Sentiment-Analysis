"""Shared fixtures.

Tests are written to run on a machine that has only the inference stack, or even
less: anything needing TensorFlow, PyTorch or the trained weights skips itself
rather than failing. A clone without `git lfs pull` should still be able to run
the suite and learn something from it.
"""

import os
import sys
from pathlib import Path

import pytest

# Rate limiting is off for the suite: the limiter keys on client IP, every
# TestClient request shares one, and a few dozen API tests would otherwise
# start throttling each other. tests/test_rate_limit.py enables it explicitly.
os.environ.setdefault("RATE_LIMIT", "off")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predictor as predictor_module  # noqa: E402

LFS_POINTER_BYTES = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
    b"size 498615900\n"
)


def requires_model(model: str):
    """Skip a test when the model's artifacts are absent or are LFS stubs."""
    reason = predictor_module.check_artifacts(model)
    return pytest.mark.skipif(reason is not None, reason=f"{model} unavailable: {reason}")


@pytest.fixture(autouse=True)
def _clear_predictor_cache():
    """Keep cached predictors from leaking between tests."""
    predictor_module.clear_cache()
    yield
    predictor_module.clear_cache()


@pytest.fixture
def lfs_pointer(tmp_path) -> Path:
    path = tmp_path / "model_final.keras"
    path.write_bytes(LFS_POINTER_BYTES)
    return path


@pytest.fixture
def real_artifact(tmp_path) -> Path:
    path = tmp_path / "pipeline.joblib"
    path.write_bytes(b"\x80\x04\x95 not a pointer, just bytes")
    return path


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)
