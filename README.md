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
