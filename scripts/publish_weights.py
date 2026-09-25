"""Upload the large model weights to the Hugging Face Hub.

Run this after training, to publish the artifacts that `src.artifacts` fetches::

    huggingface-cli login          # once, interactively
    python scripts/publish_weights.py --repo <user>/<repo>

After uploading it rewrites models/remote_manifest.json with the new commit's
SHA and the digests of the uploaded files, so that publishing and pinning are
one step. Commit the manifest together with the tokenizers and calibration
files from the same training run.

Kept out of src/ deliberately: this is the only code in the project that writes
to anything outside the working tree, and nothing at runtime should be able to
reach it by accident.

Authentication is left to `huggingface-cli login`, which stores a token in your
own keyring. The script never takes a token as an argument, so tokens cannot end
up in shell history, CI logs or a process list.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import local_path, remote_files, write_manifest  # noqa: E402
from src.config import MODEL_KEYS, MODELS_REPO  # noqa: E402

LOGGER = logging.getLogger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=MODELS_REPO)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be uploaded"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from huggingface_hub import HfApi

    files = remote_files(args.models)
    missing = [rel for rel in files if not local_path(rel).is_file()]
    if missing:
        LOGGER.error("Cannot upload, these are not on disk: %s", ", ".join(missing))
        return 1

    total = sum(local_path(rel).stat().st_size for rel in files)
    LOGGER.info("%d file(s), %.1f MB total:", len(files), total / 1e6)
    for relative in files:
        size_mb = local_path(relative).stat().st_size / 1e6
        LOGGER.info("  %-34s %8.1f MB", relative, size_mb)

    if args.dry_run:
        LOGGER.info("Dry run: nothing uploaded.")
        return 0

    api = HfApi()
    # A fine-grained token scoped to one repository can write to it without
    # being allowed to create repositories, so only create when it is absent.
    if api.repo_exists(repo_id=args.repo, repo_type="model"):
        LOGGER.info("Repository %s already exists.", args.repo)
    else:
        LOGGER.info("Creating repository %s...", args.repo)
        api.create_repo(repo_id=args.repo, repo_type="model", private=args.private)
    commit = None
    for relative in files:
        LOGGER.info("Uploading %s...", relative)
        commit = api.upload_file(
            path_or_fileobj=str(local_path(relative)),
            path_in_repo=relative,
            repo_id=args.repo,
            repo_type="model",
        )

    # Each upload is its own commit; the last one contains all of them, so its
    # SHA is the revision to pin.
    write_manifest(args.repo, commit.oid)

    LOGGER.info("Done: https://huggingface.co/%s/tree/%s", args.repo, commit.oid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
