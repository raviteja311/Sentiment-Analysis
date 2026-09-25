"""The lockfiles agree with the requirements they were compiled from.

No network: this only reads the files. Freshness against PyPI is deliberately
not checked here, because a lock that is a week old is not wrong, only old.
"""

import re

import pytest

from src.config import PROJECT_ROOT

REQUIREMENTS = PROJECT_ROOT / "requirements"
ENTRY_POINTS = ("inference-cpu", "train", "dev")

PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")


def direct_pins(name: str) -> dict[str, str]:
    """Every `package==version` a .txt asks for, following its -r includes."""
    pins = {}
    for line in (REQUIREMENTS / f"{name}.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            pins.update(direct_pins(line[3:].strip().removesuffix(".txt")))
            continue
        match = PIN.match(line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def locked_pins(name: str) -> dict[str, set[str]]:
    """Every version a lock pins per package; a universal lock may list several."""
    pins: dict[str, set[str]] = {}
    for line in (REQUIREMENTS / f"{name}.lock").read_text(encoding="utf-8").splitlines():
        match = PIN.match(line.strip())
        if match:
            pins.setdefault(match.group(1).lower().replace("_", "-"), set()).add(
                match.group(2)
            )
    return pins


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_direct_requirement_is_locked_at_its_version(name):
    locked = locked_pins(name)
    for package, version in direct_pins(name).items():
        assert package in locked, f"{package} missing from {name}.lock"
        # torch is locked as 2.9.1+cpu off macOS and 2.9.1 on it; both satisfy the pin.
        assert any(
            v.split("+")[0] == version for v in locked[package]
        ), f"{package}: {name}.txt pins {version}, {name}.lock has {locked[package]}"


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_the_transitive_closure_is_locked(name):
    # starlette is FastAPI's dependency, not ours; the lock exists to pin it.
    locked = locked_pins(name)
    assert "starlette" in locked
    assert len(locked) > len(direct_pins(name))


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_the_lock_carries_the_cpu_index_and_no_cuda(name):
    text = (REQUIREMENTS / f"{name}.lock").read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in text
    assert "torch==2.9.1+cpu" in text
    assert not any(line.startswith("nvidia-") for line in text.splitlines())


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_the_lock_is_universal(name):
    # Both tensorflow branches must be present, each behind its marker.
    text = (REQUIREMENTS / f"{name}.lock").read_text(encoding="utf-8")
    assert "tensorflow-cpu==2.21.0 ;" in text
    assert "tensorflow==2.21.0 ;" in text


def test_installers_use_the_locks():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    ci = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "requirements/dev.lock" in makefile and "requirements/dev.lock" in ci
    assert "requirements/inference-cpu.lock" in dockerfile
