# Operations

How to run your own: the secrets, the kill switches, the schedule, your own
box, the inbox, the cost and local development.

[Back to the README](../README.md)

The bot runs itself from `.github/workflows/gaffer.yml`. Fork the repo, then
set it up as below.

## Secrets

Add four under **Settings → Secrets and variables → Actions → New repository
secret**:

| Secret | Value |
| --- | --- |
| `FPL_TEAM_ID` | your FPL entry id (the number in your team's URL) |
| `TELEGRAM_BOT_TOKEN` | from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | the chat to send to: message the bot, then read `chat.id` from `https://api.telegram.org/bot<TOKEN>/getUpdates` |
| `ANTHROPIC_API_KEY` | from [the console](https://console.anthropic.com/settings/keys); this is what turns the gaffer on |

Set both Telegram secrets or neither. A token without a chat id delivers
nothing and says so in the log. `ANTHROPIC_API_KEY` is optional: leave it out
and every run is the solver's alone. The workflow needs `contents: write`
(already declared) to commit `state/`, `reports/`, the `GW{n}.md` verdicts and
the README's latest-verdict line back.

`Config.from_env()` also reads `AIGAFFER_STATE_DIR` (the state directory,
default `state`) and `AIGAFFER_MANAGER_MODEL` (the gaffer's model, default
`claude-opus-5-5`).

**The committed state is the bot's memory, and not all of it can be
regenerated.** The SQLite DB in `state/` holds the run history and the two
tables of the purchase ledger:

- `purchases`: what was paid for each player held, reconstructed by
  observation;
- `squads`: one snapshot per gameweek of the bank and fifteen the picks
  endpoint published, which the ledger's selling estimates are audited
  against.

Beside the DB are `state/news/ledger.json` (what the gaffer has learned about
each player), `state/executed/` (what you told it you entered),
`state/detailed/` (the per-run snapshots) and `state/reports/` (the diary).

Wipe `state/` mid-season and the next run reseeds every squad member at his
season-opening price. That's right for anyone held since GW1, and wrong for
every player bought since, until he is sold. A fork adopting the bot
mid-season starts in the same place. The reconciliation line in the report
shows how far adrift that leaves the estimates.

## Kill switches

Four repository variables, under **Settings → Secrets and variables → Actions
→ Variables**, stand parts of the system down without a commit and without
losing a secret:

| Variable | Value | Effect |
| --- | --- | --- |
| `AIGAFFER_MANAGER` | `0` | the solver alone, even with a key: byte-identical to the solver-only pipeline |
| `AIGAFFER_PLANNER` | `single` | the single-week solver alone: no window, so no chip calendar and no `Chips:` line |
| `AIGAFFER_CHIPS` | `off` | chips advisory only: priced in the panel, planned by nobody (the gaffer may still play an unspent chip) |
| `AIGAFFER_STRENGTH` | `off` | fixtures priced by FPL's strength columns, not the fitted team strengths |

Each switch takes exactly one literal value. Anything else (a typo, an empty
export, a `0` where `off` was meant) leaves the default standing, because a
misspelling should never turn a better engine off. The workflow sets all four
explicitly, so every scheduled run has *asked* for what it runs. A run that
asked for a gaffer and has no key to reach him prints one line saying so,
which makes a missing secret visible.

## The schedule, and running it by hand

**Actions → gaffer → Run workflow** runs one report by hand. The `mode` input
picks one by name instead of asking the clock, and `force` re-runs a report
already sent. A manual dispatch skips the schedule's gate.

There are two schedulers:

- **Primary: your own box**, running `python -m aigaffer auto` hourly at :35
  and pushing the state back. That's what `scripts/run-tick.sh` and the units
  in `deploy/` do (see [Your own box](#your-own-box)). FPL deadlines sit on the
  half hour, so a :35 tick lands each report about five minutes after its
  window opens.
- **Backup: GitHub Actions**, hourly at `50 * * * *`. Most of the week it does
  almost nothing: a curl-and-jq gate at the top checks the next deadline and
  stands the tick down in seconds unless it is within about 62 hours. Python
  stays the authority on the windows; the gate is a generous superset that
  only spares the pip install, and it fails open.

Each gameweek gets one report of each kind, because the SQLite store
remembers. A second tick inside a window does nothing, unless the gaffer was
asked and didn't decide. Then nothing was recorded, and the second tick is the
retry.

The two schedulers see each other only through the pushed state, and a gaffer
run takes minutes. So a scheduled tick reads the store twice: once from its
own checkout, and again from a fresh fetch of `main` after the gaffer has
spoken and before anything is sent. A tick the other scheduler overtook stands
down there. The remaining gaps are small:

- the few seconds between that second read and the push;
- a gaffer failing on both hosts in one window, which can cost a second
  withheld alert;
- a fetch that fails, which lets the tick carry on rather than silence it.

The early scout is the one report the gate never wakes for, because its
evening is days from the deadline. So only your own box sends it. If the box
is down that evening, the early scout is skipped and Thursday's scout is the
first word.

GitHub drops scheduled runs under load, sometimes for whole days. That's why
the windows stay open late: the first tick to get through sends whatever the
dropped ones owed. A reminder-hour tick with no full report on record runs the
full report first.

`scripts/dispatch-tick.sh` fires a tick from a second scheduler GitHub's load
can't touch: a `launchd` job, a cron line anywhere, or a ping service. It runs
the same gate locally, dispatches `auto`, and authorises itself from git's
stored GitHub credential. Duplicates send nothing twice, because the store
stands the loser down. A dispatched duplicate still pays for the checkout and
install it took to ask, which is why the script gates itself instead of firing
all year.

Whoever fires it, a failed run says so on stdout and sends one line to
Telegram. It never sends the exception's own text, which for an `httpx` error
contains the URL and so the bot token.

**GitHub disables scheduled workflows after 60 days without repository
activity.** That will happen over the off-season. Re-enable the workflow from
the Actions tab, or push any commit, before the new season's first deadline.

## Telling it what you entered

FPL's public API can't see a transfer until the deadline it was made for. On
its own, the bot would spend the rest of the gameweek recommending moves you
had already made.

So text **Transfers made** straight after you enter the latest recommendation
exactly, both its transfers and its chip. Every later report that gameweek then
works from the squad you entered:

- it says so near the top;
- it never sells a player you just bought, or buys back one you just sold;
- the T-3h reminder checks your armbands, shape, bench order and the fitness
  of your signings instead of re-recommending the moves.

If you changed anything, don't send it; the runs then behave as before.

A second "Transfers made" after a later report you also acted on adds that
report's moves on top. Sending the same one twice changes nothing. The reply
echoes the whole position (moves, armbands, squad, bank, free transfers), so a
mismatch shows at once. Prices are taken when you text, so text straight away;
an overnight price move between entering and texting is the one gap.

**Only your own box reads Telegram**, by running `python -m aigaffer inbox`
once a minute from `deploy/aigaffer-inbox.timer`. It obeys the configured chat
and nobody else. Recording takes the same state lock as the hourly tick, then
pulls, writes the row (`state/executed/gw{n}.json`), commits and pushes it. So
the GitHub workflow, which never reads Telegram, sees what you entered through
the store. If the push fails, the next hourly tick sends it.

## Asking about a chip

**Chip forecast** (or `chip forecast`) answers instantly and for free from the
newest saved report. It shows when the bot currently plans to play each chip
you hold, the week each is planned or saved for, what it's worth, when it
expires, the second set's chips and what you've played. It's only as fresh as
that report, and says which one.

**Wildcard?** or **Free hit?** (or `wildcard`, `wc`, `free hit`, `freehit`,
`fh`) works out what playing that chip at the coming deadline would do. An
acknowledgement comes first, then two messages:

1. **The numbers**, a few minutes later. You get a band: **Play**,
   **Marginal** (leaning one way) or **Hold**. It's based on the points the
   chip buys over the next six gameweeks against the best path that doesn't
   play it this week, net of hits and of what keeping it is worth. You also
   get the gain week by week and, when it's worth acting on, the moves or the
   whole free-hit team.
2. **The gaffer's view.** He reads the team news for the players it would
   bring in and says whether he agrees.

It's a simulation, and nothing is recorded; your reports are unchanged. If you
do play the chip, **don't** send "Transfers made", because that records the
latest verdict, not the what-if. The first report after the deadline will read
your real squad.

The limit is six what-ifs a day. The bot declines when:

- you don't hold the chip this gameweek;
- you've already entered another chip;
- for a free hit, you've already entered moves this week, or played a free hit
  last week.

If an hourly report is running, the what-if waits for it and tells you. Each
what-if is logged, numbers and verdict, to `whatifs.jsonl` in the inbox
directory on your box and never the repository. That way the bands can be
checked against what happened.

## Asking the gaffer

Any other text that reads as a question goes to the gaffer: it ends with "?",
or its first word is one of why, what, how, should, is, was, would, could,
can, and, who, which or when. Commands are matched first, so "transfer made"
is a typo that gets the help, not a question.

- He answers from the newest run's snapshots, rendered as that gameweek's
  detailed report. The run's own data is the whole of what he reads, and the
  plan is never changed.
- An answer is under 120 words, in plain text, from a conversation of at
  most six requests. A failed answer is retried once inside those six.
- The gameweek's thread keeps the last ten exchanges, so follow-ups work. If
  the plan has changed since your last question, the answer says so first.
- Before the week's first run there's no report, and he says so instead.
- Thirty answers a day. Only an answer counts; a question turned away is free.
- Without a gaffer configured (no key, or `AIGAFFER_MANAGER=0`), a question
  gets the help.

**The asking side's files live in the inbox directory**
(`AIGAFFER_INBOX_DIR`, default `~/.aigaffer/`), outside the checkout, and are
never committed: the day's count in `ask/cap.json` and each gameweek's thread
in `ask/gw{n}.jsonl`. The repo is public, and your questions are yours. The
same directory holds the inbox's place in Telegram (`offset`), its failure
marker, the what-if log and the two locks.

## Your own box

Any always-on Linux machine will do; a free-tier cloud VM is plenty. On it:

```sh
git clone git@github.com:YOU/aigaffer.git && cd aigaffer
python3 -m venv .venv && .venv/bin/pip install .
cp .env.example .env && chmod 600 .env   # then fill it in
```

- **Push access.** The tick commits state and verdicts back, so the clone
  needs write access. A deploy key with *Allow write access* (repo **Settings
  → Deploy keys**) is the narrowest way to give it.
- **The solver binary.** PuLP bundles CBC for x86-64. On ARM (Ampere cloud
  boxes, a Raspberry Pi) install it from the distribution with
  `sudo apt install coinor-cbc`, or every solve fails.
- **Check the key before you trust it.** A rejected Anthropic key doesn't fail
  the tick; it withholds the report and alerts you. So test it once by hand
  from the box, with the `.env` exported:
  `.venv/bin/python -c "import anthropic; print(anthropic.Anthropic().models.list().data[0].id)"`.
  Then do a dry run:
  `set -a; . ./.env; set +a; .venv/bin/python -m aigaffer scout --dry-run`.
- **The timer.** Copy `deploy/aigaffer.service` and `deploy/aigaffer.timer` to
  `/etc/systemd/system/`, fill in the user and path in the service, then run
  `sudo systemctl enable --now aigaffer.timer`. `systemctl list-timers
  aigaffer.timer` shows the next tick, and `journalctl -u aigaffer` shows what
  the last one said, including each window solve's time and proof line.
- **The inbox.** Copy `deploy/aigaffer-inbox.service` and
  `deploy/aigaffer-inbox.timer` beside them, fill in the same user and path,
  then run `sudo systemctl enable --now aigaffer-inbox.timer`. The tick
  (`scripts/run-tick.sh`) and the inbox (`scripts/inbox-tick.sh`) share one
  state lock through util-linux's `flock`; the inbox refuses to run without
  it. `journalctl -u aigaffer-inbox` shows what the inbox last said.

`run-tick.sh` stands down while `.env` is missing or still holds a `FILL_ME`.
It pulls before it runs, so a fix pushed from your laptop is live on the next
tick, and commits only when the run changed something. An out-of-window tick
costs one `git pull` and a few seconds of Python.

## What it costs

These are estimates, not measurements. They were first made at Opus 5 list
prices; the default model is now Opus 5.5, which is cheaper per token ($4 in,
$20 out per million, against $5 and $25). Either way the shape is the same.

**A decision run is typically well under $2**, and a search-heavy week can be
a few dollars. The driver isn't the searches themselves. It's that the whole
conversation is resent on every turn. A quiet week that searches twice and
decides in three turns costs cents. A bad one (a dozen searches, two
re-solves, twelve turns each re-reading everything before it) costs an order
of magnitude more.

There are three decision runs a gameweek: the early scout, the scout and the
deadline report. The reminder never wakes the gaffer. That's on the order of
$3–12 a week and $75–300 a season, weighted to the low end because most weeks
are quiet.

Smaller calls ride on top:

- the detailed report's reasons, at most two requests a decision run;
- the weekly review, at most two;
- each question you ask, at most six requests, with thirty a day;
- each chip what-if's opinion, at most six turns and five searches, with six a
  day.

The one way a week gets expensive is a gaffer who fails slowly and is asked
again. A withheld report is retried by every tick, and a refusal or a loop that
runs out of time is a full run each time. That's what the twelve-tick limit is
for: at worst a dozen runs, and the alert has told you by the first.

Three cache breakpoints, of the four the API allows, keep it down. Two never
move: the system prompt and the briefing, which can't change once the
conversation has started and are most of the input on turn one. The third
rides the end of the messages and moves forward every turn, so each turn's
searches and tool results are read from cache by the next turn instead of
being paid in full a dozen times.

## Local development

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

It needs Python 3.12 or newer. All tests run offline, with HTTP faked by
`httpx.MockTransport` and never the live API. So the same suite runs on every
push and pull request in `.github/workflows/tests.yml`, with no secrets in the
job at all.

The suite is nearly 1,900 tests and takes a few minutes. Almost all of that is
real MILP solves. The multi-week tests solve hand-built boards whose optimum is
worked out by hand in the comment above each test, and the pipeline tests
sweep real windows end to end. That's deliberate: a solver test that mocks the
solver tests nothing.

To run against the live API without touching state or Telegram:

```sh
FPL_TEAM_ID=1234567 python -m aigaffer scout --dry-run   # prints, saves nothing
python -m aigaffer backtest                              # grades the model, needs no team id
python -m aigaffer replay                                # replays real squads through each policy; slow
```

`--force` re-runs a report the store says has already gone. `backtest --gw 3`
grades one gameweek by name. `review` writes a finished gameweek's Looking
back by hand (`auto` runs it on its own once a week is final). `replay` takes
`--from` and `--to` for the range, defaulting to GW2 through the latest
finished gameweek. What the backtest and replay numbers mean, and what they
don't, is in [limitations.md](limitations.md).
