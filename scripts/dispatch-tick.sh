#!/bin/sh
# A second scheduler's tick: ask GitHub to run the gaffer's `auto` mode.
#
# GitHub's own cron is best-effort — under load it drops scheduled runs, and
# it has dropped whole days of them — so this script exists to be fired from
# somewhere GitHub's load cannot touch: a launchd job on a Mac, a cron line
# on any box, a free ping service. Firing it is always safe: the dispatched
# run is `auto`, which asks the clock and the store exactly as a scheduled
# tick would, so a duplicate tick costs seconds and sends nothing twice.
#
# Auth comes from git's credential store for github.com (the same one a
# `git push` uses), so no token lives in this file. Failure is silent by
# design — a missed ping from the backup scheduler must never page anyone;
# the primary schedule and the late-open windows are still there.

REPO="markb-1/aigaffer"

# The same generous gate the workflow runs, run here instead: a manual
# dispatch stands the workflow's own gate aside, so an ungated hourly ping
# would bill a full checkout-and-install all year round. Ping only when the
# next deadline is within sixty-two hours — the superset of every report
# window — and fail open like the gate does: if the API or jq comes up
# empty, one wasted run is cheaper than one starved report.
next=$(curl -sf --max-time 30 https://fantasy.premierleague.com/api/bootstrap-static/ \
  | jq -r '[.events[].deadline_time // empty | fromdate? | select(. > now)] | min // empty') \
  || next=""
if [ -n "$next" ]; then
  secs=$(( next - $(date -u +%s) ))
  { [ "$secs" -lt 0 ] || [ "$secs" -gt 223200 ]; } && exit 0
fi

TOKEN=$(printf 'protocol=https\nhost=github.com\n\n' | git credential fill 2>/dev/null \
  | awk -F= '/^password=/{print $2}')
[ -n "$TOKEN" ] || exit 0

curl -s -o /dev/null --max-time 30 \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/actions/workflows/gaffer.yml/dispatches" \
  -d '{"ref":"main","inputs":{"mode":"auto"}}'
