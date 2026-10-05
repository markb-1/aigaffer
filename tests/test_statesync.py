"""Tests for the state lock and the git sync the inbox records under.

The git half runs against scratch repositories: a bare remote in
``tmp_path`` and clones of it, so pull, commit and push are real and nothing
leaves the machine. The lock half holds the lock from a thread with its own
file handle — ``flock`` locks belong to an open file, so two opens in one
process contend exactly as two processes do.
"""

import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from aigaffer.statesync import PUSH_FAILED, GitStateSync, state_lock

# What the inbox writes, and so the only paths its commits may touch.
INBOX_PATHS = ("state/executed/",)


def recorded(box: Path, text: str = '{"gw": 2}\n') -> None:
    """A recording written into the clone, as the store writes it."""
    (box / "state" / "executed").mkdir(parents=True, exist_ok=True)
    (box / "state" / "executed" / "gw2.json").write_text(text)


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def clone_of(remote: Path, where: Path) -> Path:
    git("clone", "-q", str(remote), str(where), cwd=remote.parent)
    git("config", "user.name", "Test", cwd=where)
    git("config", "user.email", "test@example.invalid", cwd=where)
    return where


def scratch(tmp_path: Path) -> tuple[Path, Path]:
    """A bare remote on ``main`` holding a state file, and one clone of it."""
    remote = tmp_path / "remote.git"
    git("init", "-q", "--bare", str(remote), cwd=tmp_path)
    git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)
    seed = tmp_path / "seed"
    git("init", "-q", str(seed), cwd=tmp_path)
    git("config", "user.name", "Test", cwd=seed)
    git("config", "user.email", "test@example.invalid", cwd=seed)
    (seed / "state").mkdir()
    (seed / "state" / "aigaffer.db").write_text("v1")
    git("add", "state/", cwd=seed)
    git("commit", "-q", "-m", "seed", cwd=seed)
    git("push", "-q", str(remote), "HEAD:main", cwd=seed)
    return remote, clone_of(remote, tmp_path / "box")


def test_publish_commits_and_pushes_the_recording(tmp_path):
    remote, box = scratch(tmp_path)
    recorded(box)

    GitStateSync(box, INBOX_PATHS).publish("chore: record transfers made (inbox)")

    assert git("log", "-1", "--format=%s", "main", cwd=remote).strip() == (
        "chore: record transfers made (inbox)"
    )


def test_publish_commits_only_the_paths_it_is_given(tmp_path):
    # The database changed too (a run mid-flight, say): the inbox's commit
    # must not carry it, or it could collide with the other scheduler's
    # commit of the same binary file on the next rebase.
    remote, box = scratch(tmp_path)
    recorded(box)
    (box / "state" / "aigaffer.db").write_text("v2")
    git("add", "state/aigaffer.db", cwd=box)

    GitStateSync(box, INBOX_PATHS).publish("chore: record transfers made (inbox)")

    changed = git("show", "--name-only", "--format=", "main", cwd=remote).split()
    assert changed == ["state/executed/gw2.json"]
    assert "state/aigaffer.db" in git("status", "--porcelain", cwd=box)


def test_publish_with_nothing_changed_commits_nothing(tmp_path):
    remote, box = scratch(tmp_path)
    recorded(box)
    git("add", "state/", cwd=box)
    git("commit", "-qm", "already committed", cwd=box)
    git("push", "-q", "origin", "HEAD:main", cwd=box)

    GitStateSync(box, INBOX_PATHS).publish("chore: nothing")

    assert git("rev-list", "--count", "main", cwd=remote).strip() == "2"


def test_pull_brings_in_what_the_other_scheduler_pushed(tmp_path):
    remote, box = scratch(tmp_path)
    peer = clone_of(remote, tmp_path / "peer")
    (peer / "state" / "aigaffer.db").write_text("from the peer")
    git("commit", "-qam", "peer run", cwd=peer)
    git("push", "-q", "origin", "HEAD:main", cwd=peer)

    GitStateSync(box, INBOX_PATHS).pull()

    assert (box / "state" / "aigaffer.db").read_text() == "from the peer"


def test_a_push_that_fails_says_so_and_keeps_the_commit(tmp_path, capsys):
    # The recording is not lost: the commit stays local and the next hourly
    # tick pushes it (scripts/run-tick.sh, Task 14). The line names no URL.
    _, box = scratch(tmp_path)
    git("remote", "set-url", "origin", str(tmp_path / "nowhere.git"), cwd=box)
    recorded(box)

    GitStateSync(box, INBOX_PATHS).publish("chore: record transfers made (inbox)")

    out = capsys.readouterr().out
    assert out.startswith(PUSH_FAILED.split("(")[0])
    assert "nowhere" not in out
    assert git("log", "-1", "--format=%s", cwd=box).strip() == (
        "chore: record transfers made (inbox)"
    )
    assert not (box / ".git" / "rebase-merge").exists(), "no half-done rebase left"


def test_the_lock_waits_for_whoever_holds_it(tmp_path):
    lock = tmp_path / "inbox" / "state.lock"
    holding = threading.Event()

    def tick() -> None:
        with state_lock(lock):
            holding.set()
            time.sleep(0.5)

    holder = threading.Thread(target=tick)
    holder.start()
    holding.wait()
    started = time.monotonic()
    with state_lock(lock):
        waited = time.monotonic() - started
    holder.join()

    assert waited >= 0.4


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs util-linux flock(1)")
def test_the_lock_is_the_one_flock_1_takes(tmp_path):
    # run-tick.sh holds the lock with flock(1); the inbox must see it held.
    lock = tmp_path / "state.lock"
    lock.touch()
    shell = subprocess.Popen(["flock", str(lock), "sleep", "1"])
    try:
        time.sleep(0.3)
        started = time.monotonic()
        with state_lock(lock):
            waited = time.monotonic() - started
    finally:
        shell.wait()

    assert waited >= 0.4
