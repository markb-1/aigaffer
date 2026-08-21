# aigaffer

An autonomous AI manager for Fantasy Premier League. It pulls FPL data, projects
expected points, solves for the best transfers/lineup/captain, and sends a
recommendation report to Telegram on a schedule.

## Phase 1 — "The Analyst"

Phase 1 advises; it never acts. The pipeline is:

FPL public API (httpx + pydantic models) → xP model (minutes × per-90 rates ×
fixture factors over a 6-gameweek decayed horizon) → PuLP MILP solver (candidate
transfer plans) → markdown report → Telegram. State (SQLite DB + reports) lives
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

**This is not a true backtest.** The per-90 rates, prices and team strengths
all come from the bootstrap as it stands today, which includes the gameweek
being graded and every one since: the model is being asked to rank a week it
has already seen. Availability leaks the other way — a player injured today
is projected at no minutes for a gameweek he was fit for. Only the minutes
model is held honest. So read the numbers as a sanity check on ranking
quality — is the order better than chance? — and never as an estimate of how
the bot would have done.

## Known Phase 1 approximations

These are deliberate. Do not "fix" them without revisiting the design:

1. **Selling price = current price.** The public API does not expose your
   purchase prices, so profit-sharing on sales is ignored and the bank figure
   after a sale can be slightly off.
2. **Pre-deadline transfers by the user are invisible.** The public picks
   endpoint lags to the last deadline, so any transfer you make during the
   current gameweek is not reflected until the next deadline passes.

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Requires Python 3.12 or newer. All tests run offline — HTTP is faked with
`httpx.MockTransport`, never the live API.
