"""The state branch, held still while the inbox writes to it.

On the VM two things write the store: the hourly tick
(``scripts/run-tick.sh``), which pulls, runs a report and pushes, and the
one-minute inbox, which records "Transfers made". They share one lock file,
``$AIGAFFER_INBOX_DIR/state.lock``: the tick holds it for its whole run (via
``flock(1)``), and the inbox takes it only around its own reads and writes —
pull, write the row, commit, push — so a minute tick never waits on the
network for anything but its own work. Both use ``flock(2)``, which is what
makes the shell's lock and Python's the same lock.

The git half mirrors ``run-tick.sh``: pull with rebase, commit under the
box's own identity (or a bot's on a fresh clone), pull again and push to
``main`` — but it commits only the paths it is given, which for the inbox
are the recorded weeks' text files (``state/executed/``). The database is
binary and the other scheduler commits it too; a rebase of two commits to
one binary file cannot be done, and would stop every later tick, while two
commits to different text files always rebase. It never raises. A failed pull records on the local copy;
a failed push leaves the commit local for the next hourly tick to push; and
either one first aborts any rebase it left half-done, because a clone stuck
mid-rebase stops every later tick. One line goes to the log, naming git's
exit status and never its output — a remote URL can carry credentials.
"""

import fcntl
import os
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

STATE_LOCK = "state.lock"
REMOTE = "origin"
BRANCH = "main"
TIMEOUT_SECONDS = 120
BOT_NAME = "aigaffer-bot"
BOT_EMAIL = "bot@aigaffer.local"

PULL_FAILED = "aigaffer inbox: state pull failed ({reason}) — recording on the local copy"
PUSH_FAILED = "aigaffer inbox: state push failed ({reason}) — the next hourly tick pushes it"


# What the owner is told when a what-if has to wait for the hourly tick: the
# tick holds this lock for a whole report, which can be most of an hour on a
# deadline day, and silence that long reads as a dead bot.
REPORT_RUNNING = "A report is being written — I'll start after it."


@contextmanager
def state_lock(
    path: Path, on_wait: Callable[[], None] | None = None
) -> Iterator[None]:
    """Hold the shared state lock for the body, waiting as long as it takes.

    Blocking on purpose: a tick holds it for a whole manager run, and an
    inbox that gave up would lose the owner's text; the inbox's own
    overlap guard (``flock -n`` in ``scripts/inbox-tick.sh``) is what stops
    minute ticks piling up behind it.

    ``on_wait``, when given, is called once — and only — when the lock is
    already held, before the wait begins: flock(2) has no "who holds it"
    query, so the only way to know a wait is coming is to try without
    blocking first. A what-if uses it to tell the owner why the numbers are
    late. Without it the lock is taken exactly as it always was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if on_wait is not None:
                on_wait()
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class StateSync(Protocol):
    """Bring the shared state in, and send it back out."""

    def pull(self) -> None: ...

    def publish(self, message: str) -> None: ...


class GitStateSync:
    """:class:`StateSync` over the clone at ``repo``, committing ``paths``
    (relative to it) and nothing else."""

    def __init__(self, repo: Path, paths: tuple[str, ...]) -> None:
        self.repo = repo
        self.paths = paths

    def pull(self) -> None:
        try:
            self._git("pull", "--rebase", "-q", REMOTE, BRANCH)
        except (subprocess.SubprocessError, OSError) as error:
            self._abort_rebase()
            print(PULL_FAILED.format(reason=_reason(error)))

    def publish(self, message: str) -> None:
        try:
            self._git("add", "--", *self.paths)
            staged = self._run("diff", "--cached", "--quiet", "--", *self.paths)
            if staged.returncode == 0:
                return
            name = self._config("user.name") or BOT_NAME
            email = self._config("user.email") or BOT_EMAIL
            # ``-- paths``: only these, whatever else happens to be staged.
            self._git(
                "-c", f"user.name={name}", "-c", f"user.email={email}",
                "commit", "-q", "-m", message, "--", *self.paths,
            )
            self._git("pull", "--rebase", "-q", "--autostash", REMOTE, BRANCH)
            self._git("push", "-q", REMOTE, f"HEAD:{BRANCH}")
        except (subprocess.SubprocessError, OSError) as error:
            self._abort_rebase()
            print(PUSH_FAILED.format(reason=_reason(error)))

    def _run(self, *args: str, **options) -> subprocess.CompletedProcess:
        """Every git call goes through here, so none can wait for a person.

        The state lock is held while git runs; a credential prompt on a
        terminal, or an ssh password prompt, would hold it until the
        timeout. No terminal prompt, no stdin, and ssh in batch mode (unless
        the box already chose its own ssh command).
        """
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=TIMEOUT_SECONDS,
            **options,
        )

    def _git(self, *args: str) -> None:
        self._run(*args, check=True, capture_output=True)

    def _config(self, key: str) -> str | None:
        done = self._run("config", key, capture_output=True, text=True)
        return done.stdout.strip() or None

    def _abort_rebase(self) -> None:
        """Leave the clone as it was before a rebase that failed half-way.
        Quiet when there is no rebase to abort, which is the usual case.
        Called from the failure handlers, so it must not raise itself: if git
        is missing or hangs there is nothing more to do but carry on."""
        try:
            self._run("rebase", "--abort", capture_output=True)
        except (subprocess.SubprocessError, OSError):
            pass


def _reason(error: Exception) -> str:
    """Git's exit status or the error's class — never its output."""
    if isinstance(error, subprocess.CalledProcessError):
        return f"git exited {error.returncode}"
    return type(error).__name__
