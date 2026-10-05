#!/bin/sh
# One minute of the inbox on a box of your own: read what the owner has
# texted the bot and answer it (python -m aigaffer inbox). Fired every
# minute by deploy/aigaffer-inbox.timer; GitHub never runs this, and never
# reads Telegram.
#
# Two locks, both flock(2) files in the inbox directory, never in the repo.
# inbox.lock, taken here with -n, stops a minute's inbox starting while the
# last is still at work: it exits 0 instead, quietly. state.lock is the one
# this inbox shares with the hourly tick (scripts/run-tick.sh); Python takes
# it itself, and only around the state a recording reads and writes.
#
# No git pull here: the code is the hourly tick's to update, and a recording
# pulls the state it needs under the state lock. Like the tick, the body is
# one function called on the last line, so a pull that rewrites this file
# mid-run cannot change what runs.

main() {
  set -e
  cd "$(dirname "$0")/.."

  # No secrets, nothing to read: the inbox needs the token and the chat.
  if [ ! -f .env ] || grep -q FILL_ME .env; then
    exit 0
  fi
  set -a; . ./.env; set +a

  AIGAFFER_INBOX_DIR=${AIGAFFER_INBOX_DIR:-$HOME/.aigaffer}
  export AIGAFFER_INBOX_DIR
  mkdir -p "$AIGAFFER_INBOX_DIR"

  if command -v flock >/dev/null 2>&1; then
    exec flock -n -E 0 "$AIGAFFER_INBOX_DIR/inbox.lock" \
      .venv/bin/python -m aigaffer inbox
  fi
  exec .venv/bin/python -m aigaffer inbox
}

main "$@"
