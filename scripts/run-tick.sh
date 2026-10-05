#!/bin/sh
# One tick of the gaffer on a box of your own: pull the latest code and
# state, ask auto, commit what changed, push. The store makes a duplicate
# tick (GitHub's :50 backup, a hand-run) harmless, and the late-open windows
# mean a missed tick costs lateness, never a report. It holds the state lock
# it shares with the inbox for the whole tick, and pushes any commit an inbox
# recording could not.
#
# Fired hourly at :35 by deploy/aigaffer.timer. Git auth is the box's own —
# a deploy key with write access is the usual answer — so nothing here names
# a host, a user or a key.
#
# The body is one function called on the last line: the pull below can
# rewrite this very file, and sh reads a script as it runs it, so the whole
# tick is parsed before the first git command touches the disk.

main() {
  set -e
  cd "$(dirname "$0")/.."

  # No secrets, no run: a tick without a Telegram token would mark reports
  # sent-and-done while the phone stays silent — the one failure worse
  # than not running.
  if [ ! -f .env ] || grep -q FILL_ME .env; then
    echo "run-tick: .env missing or still holds FILL_ME placeholders — standing down"
    exit 0
  fi
  set -a; . ./.env; set +a

  # The state lock, shared with the one-minute inbox (scripts/inbox-tick.sh):
  # held for the whole tick — pull, run, commit, push — so a "Transfers made"
  # recording never interleaves with a report. fd 9 stays open until the
  # script exits, which is what releases it. A box without util-linux's
  # flock runs the tick unlocked rather than not at all — stopping the
  # reports is the worse failure — and says so on stderr every tick. The
  # inbox refuses to run without flock, so there is no recording to collide
  # with; the lock is lost only against a hand-run one.
  if command -v flock >/dev/null 2>&1; then
    lockdir=${AIGAFFER_INBOX_DIR:-$HOME/.aigaffer}
    mkdir -p "$lockdir"
    exec 9>"$lockdir/state.lock"
    flock 9
  else
    echo "run-tick: flock(1) not found (util-linux) — running without the state lock" >&2
  fi

  # --autostash: an inbox that died between writing a recorded week and
  # committing it leaves a dirty state/executed/ file, and once the remote
  # moves on a plain rebase pull refuses — every tick after would die here.
  # Stashed across the pull, the row is committed below like any other.
  git pull --rebase -q --autostash origin main
  .venv/bin/python -m aigaffer auto

  # state/ includes state/executed/, the inbox's recorded weeks.
  git add state/
  if ls GW*.md >/dev/null 2>&1; then git add GW*.md; fi
  git add README.md
  if git diff --cached --quiet; then
    # Nothing new from this run — but an inbox recording whose push failed
    # left a commit behind, and this tick is the one that sends it.
    if [ -n "$(git rev-list origin/main..HEAD 2>/dev/null)" ]; then
      git push -q origin HEAD:main
    fi
    exit 0
  fi
  # A fresh clone has no identity to commit as; the box's own, if it has
  # one, wins.
  name=$(git config user.name || echo aigaffer-bot)
  email=$(git config user.email || echo bot@aigaffer.local)
  git -c user.name="$name" -c user.email="$email" commit -q \
    -m "chore: gaffer run $(date -u +%FT%H:%M) (${AIGAFFER_TICK_LABEL:-self-hosted})"
  git pull --rebase -q --autostash origin main
  git push -q origin HEAD:main
}

main "$@"
