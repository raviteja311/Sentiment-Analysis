"""Fetch the large model weights from object storage.

The transformer's weights are 499 MB and the two Keras files another 75 MB.
Kept in Git LFS they are paid for on every clone, fork and CI checkout, and a
clone without `git lfs pull` gets 133-byte pointer stubs instead of models. They
live on the Hugging Face Hub instead, and are downloaded on demand::

    make fetch-weights                    # everything
    python -m src.artifacts --models lr lstm

Everything small stays in git, so a plain clone can still serve predictions
without running this at all.

Downloads are pinned and verified. ``models/remote_manifest.json`` records the
Hub commit the committed tokenizers and calibration files belong with, and the
sha256 of every file fetched from it; each download is hashed and compared,
and a mismatch is deleted rather than left on disk to be loaded later. The
manifest is rewritten by ``python -m src.artifacts --write-manifest`` and by
``scripts/publish_weights.py`` after an upload.

The repository is read from ``SENTIMENT_MODELS_REPO`` and the revision from
``SENTIMENT_MODELS_REVISION``; both default to the values in :mod:`src.config`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import (
    MODEL_ALIASES,
    MODEL_KEYS,
    MODELS_DIR,
    MODELS_REPO,
    MODELS_REPO_REVISION,
    REMOTE_ARTIFACTS,
    REMOTE_MANIFEST,
    load_remote_manifest,
    resolve_model,
)
from src.inference.versioning import file_digest

LOGGER = logging.getLogger(__name__)


class ArtifactVerificationError(RuntimeError):
    """A downloaded file's digest does not match the committed manifest."""


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


def verify_download(
    path: Path, relative: str, manifest: dict | None, repo_id: str
) -> None:
    """Compare a downloaded file with the manifest; delete it if it differs.

    A file the manifest does not cover is kept with a warning, so a manifest
    that is missing or stale degrades to the old unverified behaviour rather
    than making the weights unfetchable.
    """
    if manifest is None:
        LOGGER.warning(
            "%s not found; %s was downloaded without verification.",
            REMOTE_MANIFEST.name,
            relative,
        )
        return
    if manifest.get("repo") not in (None, repo_id):
        LOGGER.warning(
            "%s pins %s, not %s; %s was downloaded without verification.",
            REMOTE_MANIFEST.name,
            manifest.get("repo"),
            repo_id,
            relative,
        )
        return
    expected = (manifest.get("files") or {}).get(relative)
    if expected is None:
        LOGGER.warning(
            "%s is not listed in %s and was downloaded without verification.",
            relative,
            REMOTE_MANIFEST.name,
        )
        return

    actual = file_digest(path)
    if actual != expected:
        # Deleted, not left behind: a wrong file that stays on disk would be
        # reported as available and loaded on the next request.
        path.unlink(missing_ok=True)
        raise ArtifactVerificationError(
            f"{relative} does not match {REMOTE_MANIFEST.name}: expected sha256 "
            f"{expected}, downloaded {actual}. The file has been deleted. Either "
            "the Hub revision and the manifest have drifted apart, or the "
            "download was corrupted; re-run the fetch, and if it fails again "
            "regenerate the manifest with `python -m src.artifacts --write-manifest`."
        )
    LOGGER.info("  verified sha256 %s", actual[:16])


def fetch(
    models: list[str] | None = None,
    force: bool = False,
    repo_id: str = MODELS_REPO,
    revision: str = MODELS_REPO_REVISION,
) -> list[Path]:
    """Download remote artifacts that are not already present.

    Each download is verified against the committed manifest. Returns the paths
    that were downloaded.
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

    manifest = load_remote_manifest()
    downloaded = []
    for relative in wanted:
        LOGGER.info("Fetching %s from %s@%s...", relative, repo_id, revision)
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=relative,
                revision=revision,
                local_dir=str(MODELS_DIR),
            )
        )
        verify_download(path, relative, manifest, repo_id)
        downloaded.append(path)
        LOGGER.info("  -> %s", path)

    return downloaded


def write_manifest(repo_id: str, revision: str, path: Path = REMOTE_MANIFEST) -> dict:
    """Record the Hub commit and the sha256 of every remote artifact on disk.

    ``revision`` must be a commit SHA, not a branch name: a branch moves, and
    the whole point of the manifest is that the pin does not. Digests for
    files absent locally are carried over from the previous manifest when it
    pinned the same repository, so publishing one model does not unpin the
    others.
    """
    from src.inference.predictor import is_lfs_pointer

    previous = load_remote_manifest(path) or {}
    files: dict[str, str] = {}
    if previous.get("repo") == repo_id:
        files.update(previous.get("files") or {})

    for relative in remote_files():
        local = local_path(relative)
        if local.is_file() and not is_lfs_pointer(local):
            files[relative] = file_digest(local)
        elif relative not in files:
            LOGGER.warning(
                "%s is not on disk; the manifest has no digest for it.", relative
            )

    manifest = {
        "repo": repo_id,
        "revision": revision,
        # Only files still fetched; an entry for a dropped file would be noise.
        "files": {rel: files[rel] for rel in sorted(files) if rel in remote_files()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote %s pinned to %s@%s.", path, repo_id, revision)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        # Aliases are documented, so the CLI has to take them too.
        choices=[*MODEL_KEYS, *MODEL_ALIASES],
        metavar="MODEL",
    )
    parser.add_argument("--repo", default=MODELS_REPO)
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Hub revision to fetch "
            f"(default: the manifest's pin, {MODELS_REPO_REVISION})"
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if present"
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "hash the remote artifacts on disk and pin the Hub's current commit "
            f"in {REMOTE_MANIFEST.name}, instead of downloading"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.write_manifest:
        from huggingface_hub import HfApi

        # Resolved to a commit SHA even when a branch or tag was given, so the
        # manifest never pins something that can move.
        sha = HfApi().model_info(args.repo, revision=args.revision).sha
        write_manifest(args.repo, sha)
        return 0

    downloaded = fetch(
        models=args.models,
        force=args.force,
        repo_id=args.repo,
        revision=args.revision or MODELS_REPO_REVISION,
    )
    LOGGER.info("Downloaded %d file(s).", len(downloaded))
    return 0


if __name__ == "__main__":
    sys.exit(main())
