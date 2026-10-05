"""Tests for the box's two tick scripts and the inbox's systemd units.

``sh -n`` for both scripts; a scratch repository for the one behaviour of
``run-tick.sh`` this change adds — pushing a commit the inbox could not —
with a stub interpreter standing in for the gaffer, so the tick's git dance
runs for real and nothing else does; and, where util-linux ``flock(1)`` is
installed (the VM and the ubuntu CI runner, not a Mac), the lock the tick
and the inbox share.
"""

import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from aigaffer.statesync import state_lock

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DEPLOY = ROOT / "deploy"


@pytest.mark.parametrize("name", ["run-tick.sh", "inbox-tick.sh"])
def test_the_tick_scripts_parse(name):
    subprocess.run(["sh", "-n", str(SCRIPTS / name)], check=True)


def test_the_inbox_tick_is_executable():
    assert os.stat(SCRIPTS / "inbox-tick.sh").st_mode & stat.S_IXUSR


def test_the_inbox_timer_fires_every_minute_on_the_minute():
    timer = (DEPLOY / "aigaffer-inbox.timer").read_text()

    assert "OnCalendar=*-*-* *:*:00" in timer
    assert "AccuracySec=1s" in timer


def test_the_inbox_service_runs_the_inbox_tick_with_room_to_wait():
    service = (DEPLOY / "aigaffer-inbox.service").read_text()

    assert "ExecStart=/path/to/aigaffer/scripts/inbox-tick.sh" in service
    timeout = int(re.search(r"^TimeoutStartSec=(\d+)$", service, re.M).group(1))
    assert timeout >= 15 * 60


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def box(tmp_path):
    """A bare remote on main and a clone set up as the VM's: the real
    run-tick.sh, a .env, and a stub .venv/bin/python that runs nothing."""
    remote = tmp_path / "remote.git"
    git("init", "-q", "--bare", str(remote), cwd=tmp_path)
    git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)
    clone = tmp_path / "box"
    git("init", "-q", str(clone), cwd=tmp_path)
    git("config", "user.name", "Test", cwd=clone)
    git("config", "user.email", "test@example.invalid", cwd=clone)
    (clone / "scripts").mkdir()
    shutil.copy(SCRIPTS / "run-tick.sh", clone / "scripts" / "run-tick.sh")
    (clone / "state").mkdir()
    (clone / "state" / "aigaffer.db").write_text("v1")
    (clone / "README.md").write_text("# box\n")
    (clone / ".gitignore").write_text(".env\n.venv/\n")
    git("add", ".", cwd=clone)
    git("commit", "-q", "-m", "seed", cwd=clone)
    git("remote", "add", "origin", str(remote), cwd=clone)
    git("push", "-q", "origin", "HEAD:main", cwd=clone)
    git("fetch", "-q", "origin", cwd=clone)
    (clone / ".env").write_text("TELEGRAM_BOT_TOKEN=test\n")
    python = clone / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    return clone, remote


def tick(clone: Path, tmp_path: Path) -> None:
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "AIGAFFER_INBOX_DIR": str(tmp_path / "inbox"),
    }
    subprocess.run(
        ["sh", str(clone / "scripts" / "run-tick.sh")],
        cwd=clone, env=env, check=True, capture_output=True,
    )


def test_a_commit_the_inbox_could_not_push_goes_with_the_next_tick(box, tmp_path):
    clone, remote = box
    (clone / "state" / "aigaffer.db").write_text("recorded")
    git("commit", "-qam", "chore: record transfers made (inbox)", cwd=clone)

    tick(clone, tmp_path)

    assert git("log", "-1", "--format=%s", "main", cwd=remote).strip() == (
        "chore: record transfers made (inbox)"
    )


def test_a_recording_the_inbox_never_committed_goes_with_the_next_tick(box, tmp_path):
    # The inbox wrote the row and died before its commit: the tick's
    # ``git add state/`` covers state/executed/, so the row still reaches main.
    clone, remote = box
    (clone / "state" / "executed").mkdir()
    (clone / "state" / "executed" / "gw2.json").write_text('{"gw": 2}\n')

    tick(clone, tmp_path)

    files = git("ls-tree", "-r", "--name-only", "main", cwd=remote).split()
    assert "state/executed/gw2.json" in files


def test_a_tick_with_nothing_to_send_pushes_nothing(box, tmp_path):
    clone, remote = box

    tick(clone, tmp_path)

    assert git("rev-list", "--count", "main", cwd=remote).strip() == "1"


@pytest.mark.skipif(
    shutil.which("flock") is None,
    reason="util-linux flock(1) is the VM's; macOS has none",
)
def test_the_inbox_waits_while_a_tick_holds_the_state_lock(tmp_path):
    lock = tmp_path / "state.lock"
    holder = subprocess.Popen(["flock", str(lock), "sleep", "2"])
    time.sleep(0.5)  # let the shell's flock take it first
    started = time.monotonic()
    with state_lock(lock):
        waited = time.monotonic() - started
    holder.wait()

    assert waited >= 1.0
