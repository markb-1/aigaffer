"""The other scheduler's store, read from the branch the two of them share.

Two schedulers tick the same store — a VM at :35, GitHub's cron at :50 — and
the store travels between them as a commit on main. Each tick reads its copy
before it starts and stands down when the report exists; but a manager run is
minutes long, and a tick that starts inside those minutes reads a store that
does not know yet. So the question is asked a second time, right before the
phone: fetch main, read the store as it stands there, and answer from that.

The fetch is the only network the check needs, and the read is ``git show``
of one blob into a temporary file — nothing is merged, and the working tree
is exactly as it was. A check that cannot be made fails open: a report that
goes out twice is a nuisance, a report silenced by a flaky fetch is the
failure the whole arrangement exists to prevent.
"""

import sqlite3
import subprocess
import tempfile
from pathlib import Path

from aigaffer.store import DB_NAME, Store

REMOTE = "origin"
# The branch both schedulers commit to. The workflow's commit step pushes to
# whatever ref it ran on, so a dispatch from another branch would be asking
# about the wrong one — a dispatch never asks, and the schedule runs on main.
BRANCH = "main"
# Where the committed store lives in the repository: Config's default state
# directory, and the file name every host keeps it under.
STORE = f"state/{DB_NAME}"
TIMEOUT_SECONDS = 60

UNANSWERED = "the peer scheduler could not be asked ({reason}) — carrying on"


def peer_has_run(repo: Path, gw: int, mode: str) -> bool:
    """Whether ``gw``'s ``mode`` report is recorded in the store on the
    shared branch — the peer scheduler's word, as of right now."""
    try:
        _git(repo, "fetch", "-q", REMOTE, BRANCH)
    except (subprocess.SubprocessError, OSError) as error:
        print(UNANSWERED.format(reason=_reason(error)))
        return False
    try:
        blob = _git(repo, "show", f"FETCH_HEAD:{STORE}")
    except subprocess.SubprocessError:
        # No store on the branch yet: nothing has run, which is an answer.
        return False
    # Whatever is on the branch under the store's name — a truncated push, a
    # file that is not SQLite — the question must not take the run down with
    # it, not here, after the manager has been paid for.
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as copy:
            copy.write(blob)
            path = Path(copy.name)
        return Store(path).has_run(gw, mode)
    except (OSError, sqlite3.Error) as error:
        print(UNANSWERED.format(reason=_reason(error)))
        return False
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _git(repo: Path, *args: str) -> bytes:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        timeout=TIMEOUT_SECONDS,
    )
    return done.stdout


def _reason(error: Exception) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        return f"git {error.cmd[3]} exited {error.returncode}"
    return type(error).__name__
