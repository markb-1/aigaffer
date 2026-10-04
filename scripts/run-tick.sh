#!/bin/sh
# One tick of the gaffer on a box of your own: pull the latest code and
# state, ask auto, commit what changed, push. The store makes a duplicate
# tick (GitHub's :50 backup, a hand-run) harmless, and the late-open windows
# mean a missed tick costs lateness, never a report.
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

  git pull --rebase -q origin main
  .venv/bin/python -m aigaffer auto

  git add state/
  if ls GW*.md >/dev/null 2>&1; then git add GW*.md; fi
  git add README.md
  if git diff --cached --quiet; then
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
