"""Content-addressed versions for the served models.

A prediction is only traceable if you can tell which artifact produced it.
"latest" is not an answer: weights get retrained, re-fetched from the Hub, or
swapped by a volume mount, and none of those change a version string that a
human maintains by hand.

So a model's version is derived from its bytes - a short digest over every file
that determines its output, including the calibration temperature, because that
changes the confidence a caller sees even though the weights are identical.

Digests are cached against each file's size and modification time, so serving
does not re-hash half a gigabyte of safetensors on every request, but a swapped
artifact still produces a new version.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

from src.config import MODEL_DIRS, REQUIRED_ARTIFACTS

# Enough to be unambiguous in a log line without being unreadable.
VERSION_LENGTH = 12
DIGEST_LENGTH = 16

CHUNK_SIZE = 1024 * 1024


# Artifacts git treats as text, which therefore arrive with CRLF on Windows and
# LF elsewhere. Their line endings are normalised before hashing: without that,
# the same committed model gets a different version on a Windows checkout than
# on a Linux one, which is precisely the ambiguity a version is meant to remove.
TEXT_SUFFIXES = {".json", ".txt"}


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()

    if path.suffix.lower() in TEXT_SUFFIXES:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()

    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=64)
def _cached_digest(path_str: str, size: int, mtime_ns: int) -> str:
    """Digest keyed by identity *and* mtime, so a replaced file re-hashes."""
    return _digest_file(Path(path_str))


def file_digest(path: Path) -> str | None:
    """SHA-256 of a file, or None if it is not there."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return _cached_digest(str(path), stat.st_size, stat.st_mtime_ns)


def versioned_files(model: str) -> list[Path]:
    """Files whose contents determine what this model outputs."""
    files = list(REQUIRED_ARTIFACTS[model])
    calibration = MODEL_DIRS[model] / "calibration.json"
    if calibration.exists():
        files.append(calibration)
    return files


def artifact_digests(model: str) -> dict[str, str]:
    """Short digest per file, keyed by name."""
    digests = {}
    for path in versioned_files(model):
        digest = file_digest(path)
        if digest is not None:
            digests[path.name] = digest[:DIGEST_LENGTH]
    return digests


def model_version(model: str) -> str | None:
    """Short version string for a model, or None if nothing is on disk.

    Derived from the per-file digests rather than their concatenated contents,
    so it is stable regardless of the order the files are read in.
    """
    digests = artifact_digests(model)
    if not digests:
        return None

    combined = hashlib.sha256()
    for name in sorted(digests):
        combined.update(name.encode("utf-8"))
        combined.update(digests[name].encode("utf-8"))
    return combined.hexdigest()[:VERSION_LENGTH]


def clear_cache() -> None:
    """Forget cached digests (used by tests)."""
    _cached_digest.cache_clear()
