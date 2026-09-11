"""The other scheduler's word, asked for once the expensive part is done.

Two schedulers tick the same store, and the store travels between them as a
git commit. Each tick reads the store before it starts and stands down if the
report exists — but a manager run is minutes long, and a tick that starts in
those minutes reads a store that does not know yet. On 10 Sep 2026 GitHub's
tick landed two minutes before the VM pushed the GW4 scout: two gaffers, two
texts saying different things, and a commit that could not rebase. So the
question is asked twice: once at the start from the store in hand, and once
more, right before the phone, from the store as it now stands on main.
"""

import os
import subprocess
from functools import partial
from pathlib import Path

import pytest

from aigaffer import __main__ as cli
from aigaffer import orchestrator
from aigaffer.orchestrator import run_pipeline
from aigaffer.peer import peer_has_run
from aigaffer.store import Store
from tests.test_manager_retry import phone_cfg
from tests.test_orchestrator import (
    PAST_THE_FLOOR,
    bootstrap_due_in,
    decided,
    make_client,
    pipeline_routes,
    serve,
    store,  # noqa: F401 — the CLI's configured environment, a fixture
    stub_gaffer,
)


@pytest.fixture
def phone(monkeypatch) -> list:
    sent: list = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args: sent.append(args))
    return sent


def scout_run(monkeypatch, tmp_path, **kwargs):
    """One scout with a manager and a phone; the report, the store, the manager."""
    gaffer = stub_gaffer(monkeypatch, decided)
    store_ = Store(tmp_path / "state" / "aigaffer.db")
    report = run_pipeline(
        phone_cfg(tmp_path), make_client(pipeline_routes()), store_, "scout",
        now=PAST_THE_FLOOR, **kwargs,
    )
    return report, store_, gaffer


# --- the pipeline, told the other scheduler got there first ----------------


def test_a_run_the_peer_overtook_sends_and_saves_nothing(
    monkeypatch, tmp_path, capsys, phone
):
    asked = []

    def overtaken(gw: int, mode: str) -> bool:
        asked.append((gw, mode))
        return True

    report, store_, gaffer = scout_run(monkeypatch, tmp_path, overtaken=overtaken)

    assert asked == [(2, "scout")]
    assert len(gaffer.consults) == 1, "the question comes after the manager, not instead"
    assert phone == []
    assert store_.last_runs() == []
    assert not (tmp_path / "GW2.md").exists()
    assert orchestrator.OVERTAKEN in capsys.readouterr().out
    assert report, "the report is still handed back, for a dry run to print"


def test_a_peer_who_has_not_run_changes_nothing(monkeypatch, tmp_path, phone):
    _, store_, _ = scout_run(monkeypatch, tmp_path, overtaken=lambda gw, mode: False)

    assert len(phone) == 1
    assert store_.has_run(2, "scout") is True


# --- the peer's store, read from git ---------------------------------------

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, env=GIT_ENV,
                   capture_output=True)


@pytest.fixture
def clones(tmp_path) -> tuple[Path, Path]:
    """Two clones of one hub: ``peer`` pushes, ``ours`` is behind."""
    hub = tmp_path / "hub.git"
    git("init", "-q", "--bare", "-b", "main", str(hub), cwd=tmp_path)
    peer = tmp_path / "peer"
    git("clone", "-q", str(hub), str(peer), cwd=tmp_path)
    (peer / "README.md").write_text("gaffer\n")
    git("add", "README.md", cwd=peer)
    git("commit", "-q", "-m", "init", cwd=peer)
    git("push", "-q", "origin", "HEAD:main", cwd=peer)
    ours = tmp_path / "ours"
    git("clone", "-q", str(hub), str(ours), cwd=tmp_path)
    return peer, ours


def test_a_run_the_peer_pushed_is_seen_from_a_clone_that_is_behind(clones):
    peer, ours = clones
    Store(peer / "state" / "aigaffer.db").save_run(2, "scout", "# report", {})
    git("add", "state", cwd=peer)
    git("commit", "-q", "-m", "chore: gaffer run", cwd=peer)
    git("push", "-q", "origin", "HEAD:main", cwd=peer)

    assert peer_has_run(ours, 2, "scout") is True
    assert peer_has_run(ours, 2, "deadline") is False
    assert peer_has_run(ours, 3, "scout") is False
    # Read, not merged: our own tree is exactly as it was.
    assert not (ours / "state").exists()


def test_a_hub_with_no_store_yet_is_no_run(clones):
    _, ours = clones

    assert peer_has_run(ours, 2, "scout") is False


def test_no_remote_to_ask_is_no_run(tmp_path, capsys):
    # Fail open: a check that cannot be made must not silence a report. A dry
    # run in a fresh checkout, a repo with no origin — the report goes out
    # and the line on stdout says the question went unanswered.
    lone = tmp_path / "lone"
    git("init", "-q", "-b", "main", str(lone), cwd=tmp_path)

    assert peer_has_run(lone, 2, "scout") is False
    assert "peer" in capsys.readouterr().out


# --- through the command line ----------------------------------------------


def cli_tick(monkeypatch, hours: float = 20) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(hours)))
    stub_gaffer(monkeypatch, decided)


def test_auto_asks_the_peer_before_the_phone(monkeypatch, tmp_path, capsys, store):
    asked = []

    def fake_peer(repo: Path, gw: int, mode: str) -> bool:
        asked.append((repo, gw, mode))
        return True

    monkeypatch.setattr(cli, "peer_has_run", fake_peer)
    cli_tick(monkeypatch)

    assert cli.main(["auto"]) == 0

    assert asked == [(tmp_path, 2, "deadline")], "the repo is the state dir's parent"
    assert store.has_run(2, "deadline") is False
    assert orchestrator.OVERTAKEN in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["deadline"], ["auto", "--force"]])
def test_a_named_or_forced_run_never_asks(monkeypatch, tmp_path, store, argv):
    monkeypatch.setattr(cli, "peer_has_run", lambda *a: pytest.fail("asked"))
    cli_tick(monkeypatch)

    assert cli.main(argv) == 0

    assert store.has_run(2, "deadline") is True
