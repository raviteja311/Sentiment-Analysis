"""Fetch the large model weights from object storage.

The transformer's weights are 498 MB and the four Keras files another 66 MB.
Kept in Git LFS they are paid for on every clone, fork and CI checkout, and a
clone without `git lfs pull` gets 133-byte pointer stubs instead of models. They
live on the Hugging Face Hub instead, and are downloaded on demand::

    make fetch-weights                    # everything
    python -m src.artifacts --models lr lstm

Everything small stays in git, so a plain clone can still serve predictions
without running this at all.

The repository is read from ``SENTIMENT_MODELS_REPO`` and defaults to the value
in :mod:`src.config`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import (
    MODEL_KEYS,
    MODELS_DIR,
    MODELS_REPO,
    MODELS_REPO_REVISION,
    REMOTE_ARTIFACTS,
    resolve_model,
)

LOGGER = logging.getLogger(__name__)


def remote_files(models: list[str] | None = None) -> list[str]:
    """Hub-relative paths of every remote artifact for the given models."""
    selected = models or list(MODEL_KEYS)
    files: list[str] = []
    for model in selected:
        files.extend(REMOTE_ARTIFACTS.get(resolve_model(model), ()))
    return files


def local_path(relative: str) -> Path:
    """Where a Hub-relative path lands on disk."""
    return MODELS_DIR / relative


def is_remote(path: Path) -> bool:
    """True if this artifact is fetched rather than stored in git."""
    candidates = {local_path(rel).resolve() for rel in remote_files()}
    try:
        return path.resolve() in candidates
    except OSError:
        return False


def missing(models: list[str] | None = None) -> list[str]:
    """Remote artifacts that are absent locally, or are LFS pointer stubs."""
    from src.inference.predictor import is_lfs_pointer

    absent = []
    for relative in remote_files(models):
        path = local_path(relative)
        if not path.exists() or is_lfs_pointer(path):
            absent.append(relative)
    return absent


def fetch(
    models: list[str] | None = None,
    force: bool = False,
    repo_id: str = MODELS_REPO,
    revision: str = MODELS_REPO_REVISION,
) -> list[Path]:
    """Download remote artifacts that are not already present.

    Returns the paths that were downloaded.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - dependency is pinned
        raise RuntimeError(
            "huggingface_hub is required to fetch weights. Install it with "
            "`pip install -r requirements/base.txt`."
        ) from error

    wanted = remote_files(models) if force else missing(models)
    if not wanted:
        LOGGER.info("All weights are already present.")
        return []

    downloaded = []
    for relative in wanted:
        LOGGER.info("Fetching %s from %s...", relative, repo_id)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=relative,
            revision=revision,
            local_dir=str(MODELS_DIR),
        )
        downloaded.append(Path(path))
        LOGGER.info("  -> %s", path)

    return downloaded


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS)
    parser.add_argument("--repo", default=MODELS_REPO)
    parser.add_argument("--revision", default=MODELS_REPO_REVISION)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if present"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    downloaded = fetch(
        models=args.models,
        force=args.force,
        repo_id=args.repo,
        revision=args.revision,
    )
    LOGGER.info("Downloaded %d file(s).", len(downloaded))
    return 0


if __name__ == "__main__":
    sys.exit(main())
