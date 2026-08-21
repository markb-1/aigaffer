# aigaffer

An autonomous AI manager for Fantasy Premier League. It pulls FPL data, projects
expected points, solves for the best transfers/lineup/captain, and sends a
recommendation report to Telegram on a schedule.

## Phase 1 — "The Analyst"

Phase 1 advises; it never acts. The pipeline is:

FPL public API (httpx + pydantic models) → xP model (expected minutes × season
rates per 90 × fixture factors over a 6-gameweek decayed horizon) → PuLP MILP
solver (candidate transfer plans) → markdown report → Telegram. State (SQLite DB + reports) lives
in `state/`, configurable via `AIGAFFER_STATE_DIR`.

No transfers are executed in Phase 1 — you make the final call.

### Configuration

`Config.from_env()` reads:

| Variable | Meaning |
| --- | --- |
| `FPL_TEAM_ID` | Your FPL entry id (required) |
| `TELEGRAM_BOT_TOKEN` | Bot token for report delivery (optional) |
| `TELEGRAM_CHAT_ID` | Chat to deliver reports to (optional) |
| `AIGAFFER_STATE_DIR` | State directory (default `state`) |

### Backtest sanity check

`backtest` grades the model against a gameweek that has already been played.
It rebuilds expected minutes from each player's history *before* that
gameweek, projects the single gameweek, and sets the projection against what
players actually scored. It needs no team id and writes nothing.

```sh
python -m aigaffer backtest            # the most recent finished gameweek
python -m aigaffer backtest --gw 3     # one by name, printing for example:
{'gw': 3, 'n': 143, 'spearman': 0.41, 'top20_hit_rate': 0.35}
```

`n` is the population: the 160-odd players on the pipeline's own shortlist
who had a fixture that gameweek — one API request each, so the run takes a
minute. `spearman` is the rank correlation between projected and actual
points over that population, and `top20_hit_rate` the share of the model's
top twenty who scored five or more.

**This is not a true backtest.** Three things here know how the gameweek
turned out, and all three flatter the model:

1. The season totals the rates are taken from, the prices and the team
   strengths come from the bootstrap as it stands today, which includes the gameweek being graded and every one since:
   the model is asked to rank a week it has already seen.
2. The shortlist admits only players available *today*, so anyone since injured
   or gone is never graded — the population is the survivors.
3. The shortlist then cuts by season-to-date total points, today's total,
   which includes everything scored *after* the graded gameweek: the field is
   skewed towards the players who went on to do well.

Only the minutes model is held honest. So the level of `spearman` is
optimistic: read it as a sanity check on ranking quality — is the order better
than chance, and is it holding up week to week? — and never as an estimate of
how the bot would have done.

## Operations

The bot runs itself from `.github/workflows/gaffer.yml`. To set it up on a
fork:

1. **Settings → Secrets and variables → Actions → New repository secret**, three
   of them:

   | Secret | Value |
   | --- | --- |
   | `FPL_TEAM_ID` | your FPL entry id (the number in your team's URL) |
   | `TELEGRAM_BOT_TOKEN` | from [@BotFather](https://t.me/BotFather) |
   | `TELEGRAM_CHAT_ID` | the chat to send to — message the bot, then read `chat.id` from `https://api.telegram.org/bot<TOKEN>/getUpdates` |

   Set all three or none: a token without a chat id delivers nothing and says
   so in the log. The workflow needs `contents: write` (already declared) to
   commit `state/` back.
2. **Actions → gaffer → Run workflow** to try it by hand; the `mode` input runs
   one report by name instead of asking the clock.

The schedule fires every three hours (`7 */3 * * *`) and does nothing at all
most of the time. Two windows before each deadline produce a report:

| Window | Report | What it is for |
| --- | --- | --- |
| 36–60 hours out | `scout` | transfer plans while there is still time to think |
| 0–3 hours out | `deadline` | the final call, after the press conferences |

Each gameweek gets one of each: the SQLite store remembers, so a second tick
inside a window is a no-op. A failed run says so on stdout and sends one line
to Telegram — never the exception's own text, which for an `httpx` error
contains the URL and so the bot token.

**GitHub disables scheduled workflows after 60 days without repository
activity**, and emails the owner first. That will happen over the off-season,
when nothing is being committed and the bot is standing down anyway; re-enable
it from the Actions tab (or push any commit) before the first deadline of the
new season.

## Known Phase 1 approximations

These are deliberate. Do not "fix" them without revisiting the design:

1. **Selling price = current price.** The public API does not expose your
   purchase prices, so profit-sharing on sales is ignored and the bank figure
   after a sale can be slightly off.
2. **Pre-deadline transfers by the user are invisible.** The public picks
   endpoint lags to the last deadline, so any transfer you make during the
   current gameweek is not reflected until the next deadline passes.

## Deferred to Phase 2

Three things the spec wants that Phase 1 deliberately does not ship:

1. **Weekly input-data snapshots.** The store keeps the report and the decision
   for each run, not the bootstrap and histories they were computed from, so a
   past recommendation can be read but not recomputed. Snapshotting the inputs
   is what would make a run reproducible after the API has moved on.
2. **A full-season vaastav backtest with a beat-the-average benchmark.** What
   ships is a single-gameweek ranking sanity check against the live API (above),
   which cannot say whether the bot would have beaten the average manager over a
   season — the question the spec actually asks.
3. **A fetch / project / solve seam in `run_pipeline`.** Phase 2 wants an LLM to
   override an input — a minutes estimate, an availability — and re-solve from
   there. Today the pipeline is one straight-through function, so there is
   nowhere to inject a changed projection and ask for the answer again.

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Requires Python 3.12 or newer. All tests run offline — HTTP is faked with
`httpx.MockTransport`, never the live API.
