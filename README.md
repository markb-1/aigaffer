# aigaffer

An autonomous AI manager for Fantasy Premier League. It pulls FPL data, projects
expected points, solves for the best transfers/lineup/captain, puts the result
to a Claude agent who reads the week's team news and can overrule it, and sends
a recommendation report to Telegram on a schedule.

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
| `ANTHROPIC_API_KEY` | Turns the Phase 2 manager on. No key, no manager |
| `AIGAFFER_MANAGER` | Set to `0` to run the solver alone even with a key |
| `AIGAFFER_MANAGER_MODEL` | Which model the manager is (default `claude-opus-5`) |
| `AIGAFFER_PLANNER` | Set to exactly `single` to solve one gameweek at a time instead of the window |

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

## Phase 2 — "The Gaffer"

The solver says what is optimal under the model. Phase 2 adds the half it
cannot do: what has happened since the numbers were computed. After the solve,
the week goes to a Claude agent (`aigaffer/manager/`) with four things it may
do and nothing else:

1. **Read the briefing** — the deadline, the date it was written, the squad,
   the candidate plans the solver reached, every player any of them would buy
   or sell with the expected minutes behind him, the chip EV panel and the
   chips already spent.
2. **Search the web** for what the model cannot know: injuries, suspensions,
   rotation, a press conference that made somebody a doubt. The tool is capped
   at eight searches **per request** — that is one assistant turn, not the run
   — so a long conversation can spend more than eight in total; see Deferred.
3. **Overrule the minutes and re-solve.** `adjust_players` sets a player's
   expected minutes for the coming gameweek absolutely — 0 for a player who is
   out, 20 for a substitute, 90 for a starter — and `resolve` re-projects and
   re-solves with those in force, coming back with fresh plans under fresh ids.
4. **Finalize** one plan, a captain and vice from that plan's eleven, a chip or
   none, and the rationale.

The guardrails are in code, not in the prompt. A plan is an id from a registry
minted by the loop, so no amount of text in the conversation — a briefing
quoting the FPL API, a web page quoting whoever wrote it — can pass for a plan
that was never solved. The armbands are checked against the eleven of the plan
he actually chose. A chip needs an argument, not a sentence, and is refused
outright if the season's history says it is gone — or if it is a wildcard or a
free hit, which suspend the transfer rules every plan on the board was solved
under. Those two go in the rationale for Mark to act on rather than into the
decision, because a chip-aware solve is not a thing Phase 2 has. Everything the solver already
enforced — budget, club quotas, the hit cap — it still enforces, because he
only ever picks from plans it produced.

What comes back is printed in the report as **The Gaffer's view**: the
rationale, the chip and its justification, the minute adjustments he settled on
and how many searches it took.

### The fallback doctrine

A deadline is never silently missed, and a manager who cannot be reached is not
a reason to miss one. **Every** failure degrades to the solver's own
recommendation — the exact Phase 1 answer — and the report still goes out on
time. There are two shapes of failure and they read differently:

**Asked and it went wrong.** A rate limit, an authentication error, a refusal, a
connection that dies, a turn the model never finishes, twelve turns that reach
no decision, a re-solve that finds nothing, a chip he tried to play twice, a
briefing that could not be built. The Gaffer's view section is still printed and
says whose pick this actually is: *"The gaffer was unavailable (…); this is the
solver's pick."* The reason is a class name or a phrase of ours — never an
exception's own words, which can carry a key or a token.

**Never asked at all.** No key, the kill switch, an initial squad draft
(fifteen players from nothing is not a week to read the news about), or a
dependency that will not import. Then there is no decision object to hang a
section on and the report is exactly Phase 1's. The first three are choices,
and they say nothing: that is the invariant below. The fourth is an accident,
so the report carries one line — *"Note: the manager is configured but was
unavailable this run; this is the solver's pick."* — because a fork whose
`anthropic` install broke would otherwise read a season of solver-only reports
and never be told.

The rule underneath: the Phase 1 path is the invariant. With
`AIGAFFER_MANAGER=0` the pipeline produces byte-identical output to Phase 1, and
a test holds it to that.

### What it costs

**Typically well under $2 per decision run** at Opus 5 rates, and a search-heavy
week can be a few dollars. The driver is not the searches themselves but that
the conversation is resent in full on every turn: a quiet week that searches
twice and decides in three turns is cents, and a pathological one — a dozen
searches, two re-solves, twelve turns each re-reading everything before them —
costs an order of magnitude more. Two decision runs a gameweek (scout and
deadline), so on the order of $2–8 a week and $50–200 for a season, weighted
towards the low end because most weeks are quiet.

Three cache breakpoints hold it down, of the four the API allows. Two never
move — the system prompt and the briefing, which are the whole of what cannot
change once the conversation has started, and most of the input on turn one.
The third rides the end of the messages and moves forward with every turn, so
each turn's searches and tool results are read from cache by the turn after it
instead of being paid for in full a dozen times.

### Turning it off

`AIGAFFER_MANAGER=0` in the environment, or delete the `ANTHROPIC_API_KEY`
secret. Either way the run is Phase 1's: a report goes out, a little worse, on
time. On the workflow the switch is a **repository variable** — Settings →
Secrets and variables → Actions → Variables → `AIGAFFER_MANAGER` = `0` — so the
gaffer can be stood down mid-season without editing the workflow or throwing
away the key.

`AIGAFFER_MANAGER` is an opt-out rather than an opt-in, and the workflow sets it
deliberately so that every scheduled run has *asked* for a manager: a run that
asked and has no key to reach him with prints one line saying so, which is what
a missing secret looks like from the outside instead of nothing at all.

## Phase 2.5 — "The Planner"

Phase 1's solver answers one question about one gameweek: given this squad,
this bank and these free transfers, what is the best fifteen reachable *this
week*? That is the question a manager asks and it is not the question a manager
answers. Money raised has to be spent the moment it is raised, a free transfer
rolled is a free transfer thrown away, and a plan that needs five moves and can
make three is infeasible rather than staged.

So the recommendation is now solved over the whole six-gameweek horizon
(`aigaffer/solver/multiweek.py`) as one mixed-integer program: a squad, an XI
and a captain for every gameweek in the window, with the bank and the
free-transfer count carried between them under the game's own rules — one free
transfer a week, never more than five in hand, four points for every move past
what is held and eight points a gameweek at the very most. Only the first
gameweek is ever entered, but it is chosen knowing what it is for. Three things
follow that the single-week solver cannot reach:

1. **A rolled transfer is a decision.** Banking this week's move to make two
   free ones next week is now a plan rather than an accident, and the model can
   see what the roll is *for*.
2. **A move can be made early for a gameweek that has not arrived.** The double
   gameweek is the case this was built for: five moves will not fit under one
   gameweek's ceiling, so the move that only raises the money goes a week ahead
   of the moves that spend it. The single-week solver never makes that move at
   all — selling a good player for a cheap one is a downgrade right up until the
   week it pays for something, and by then it is too late.
3. **The captain is inside the model.** The window prices every gameweek with
   the armband already on the best man in that gameweek's eleven, so a fifteen
   that unlocks a big captaincy is worth to the solver what it is worth to the
   season. Phase 1's objective has no captain in it at all — it maximises an
   eleven and a weighted bench, and the armband is chosen afterwards from
   whatever squad that produced. The armband finally *played* is still picked
   afterwards, by `pick_lineup` and then by the gaffer; what changed is that it
   is no longer invisible to the thing choosing the squad.

The gameweeks after this one are printed in the report under **The road ahead**,
and ride the end of every plan line in the manager's briefing. The report closes
that section the same way every week, and the line is the whole of the path's
status:

> Advisory — re-planned every run; only this week's moves are ever made.

Nothing in a path is a commitment. It is re-solved from scratch on the next run,
on projections that will have moved, and the one thing a reader must not do is
hold this week's move back because a path he read a fortnight ago says it
belongs in the next. The briefing makes the same point at more length, because a
model reading "GW4 +Pike -Byrne" beside a plan it is being asked to commit to
would otherwise reasonably read it as part of the commitment.

### The two engines, and which one answered

The window goes first, once per transfer count the shortlist considers, each
solve on a twenty-second leash. That is four to six solves, and on the boards it
has been run against they have come back in a second or two between them — the
leash is a ceiling, not a budget, and a solve that reaches it with a squad in
hand hands the squad over anyway. When the window has nothing at all to say —
an infeasible board, or a sweep where every count came back empty — the same
questions go to the single-week solver, because a gameweek with a deadline
needs a recommendation more than it needs the better model.

Which engine answered is written down nowhere, and does not need to be: a plan
off the window carries its path and a plan off the single-week solver does not.
So a report with a road-ahead section is a report off the window. A run that
asked for the window and got the other engine says so once, under the candidate
plans:

> Single-week engine (multi-week solve unavailable this run).

A run that asked for the single-week solver on purpose prints nothing — it got
what it asked for, and a report that apologised every week would be crying
wolf. The decision the store keeps carries the same fact as `engine`, `multi` or
`single`, read off what came back rather than off what was configured.

### Turning the window off

`AIGAFFER_PLANNER=single` in the environment, and the single-week solver answers
alone. The value has to be exactly `single`: a typo, an empty export, a `0` —
anything else at all leaves the window on, because turning the better engine off
is not something a misspelling should be able to do. It is the same convention
as `AIGAFFER_MANAGER`, where one literal value switches and everything else
leaves the default standing.

On the workflow it is the same arrangement too — Settings → Secrets and
variables → Actions → Variables → `AIGAFFER_PLANNER` = `single` — so a week the
window misbehaves is a week the single-week solver reports on time, without a
commit to the workflow and without a lost report.

## Operations

The bot runs itself from `.github/workflows/gaffer.yml`. To set it up on a
fork:

1. **Settings → Secrets and variables → Actions → New repository secret**, four
   of them:

   | Secret | Value |
   | --- | --- |
   | `FPL_TEAM_ID` | your FPL entry id (the number in your team's URL) |
   | `TELEGRAM_BOT_TOKEN` | from [@BotFather](https://t.me/BotFather) |
   | `TELEGRAM_CHAT_ID` | the chat to send to — message the bot, then read `chat.id` from `https://api.telegram.org/bot<TOKEN>/getUpdates` |
   | `ANTHROPIC_API_KEY` | from [the console](https://console.anthropic.com/settings/keys) — the Phase 2 manager. Already configured on this repo |

   Set all three Telegram ones or none: a token without a chat id delivers
   nothing and says so in the log. `ANTHROPIC_API_KEY` is independent of them
   and optional — leave it out and every run is Phase 1's. The workflow needs
   `contents: write` (already declared) to commit `state/` back.
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
3. **The opening weekend has no minutes to read, for a new face.** Expected
   minutes are a mean over the matches a player has actually played. The API
   adds a history row the moment a deadline goes — 0 minutes for a match that
   kicks off two days later — and those rows are dropped on the way in, fixture
   by fixture, because a zero for a match nobody has played is not a gameweek
   anybody sat out. That leaves GW1 with no history at all.

   For a player who was in the division last season this is now answered from
   the payload: element-summary carries his last season's minutes, and divided
   by 38 they are the floor under his first three gameweeks — 3230 minutes is
   85 a week, not the substitute's 20 the season's `starts` used to give him.
   A brand-new signing, a promoted club's player or a teenager has no Premier
   League season to read, so he still falls back on `starts`, which is zero for
   everybody in GW1, and he is still projected at a substitute's twenty
   minutes. The gaffer's own minute overrides remain the mitigation for those.

   Three edges of the prior are worth knowing. A January signing's half-season
   of minutes is spread over a whole one, so he is understated — the payload
   has no games-available column to do better with. A player whose role has
   genuinely changed is described by last season until this one reaches three
   gameweeks, which is when the history stands alone. And a prior can land
   *below* the guess it replaces: a returner coming off an injury-hit season
   floors at his own 11 minutes a week rather than at the 75 "he has started
   once" used to give him.

   That last one is understatement, and understatement is not free. For a
   player you do not own it costs a transfer you would have made. For a player
   already in your fifteen it can leave him out of the eleven or put him on the
   sell side — which is the case to watch in the first three gameweeks, and
   what the gaffer's own minute overrides are there to correct.

## Deferred

Still not shipped, deliberately. The fetch / project / solve seam that used to
head this list went in with Phase 2 — it is what `resolve` re-solves through —
and the multi-week planner above went in with Phase 2.5, which took the captain
into the model on the way past. Neither is a gap any more.

1. **Weekly input-data snapshots.** The store keeps the report and the decision
   for each run, not the bootstrap and histories they were computed from, so a
   past recommendation can be read but not recomputed. Snapshotting the inputs
   is what would make a run reproducible after the API has moved on. Phase 2
   sharpens this: the decision record now holds the manager's minute
   adjustments, but not the projections they were applied to, so his week can be
   read and not rebuilt.
2. **A full-season vaastav backtest with a beat-the-average benchmark.** What
   ships is a single-gameweek ranking sanity check against the live API (above),
   which cannot say whether the bot would have beaten the average manager over a
   season — the question the spec actually asks.
3. **Chip economics against the plan actually chosen.** The chip EV panel is
   priced once, against the plan that rolls the transfer, before the manager is
   asked anything. If he picks a different plan and plays a chip on it, the
   numbers he argued from describe a squad he did not enter. Re-pricing the
   chips per plan is a solve per chip per plan, which is why it is not done yet.
   The same missing piece is why a wildcard or a free hit cannot be finalized at
   all: planning one means solving the week with fifteen free transfers and no
   hits, and nothing here does that. The panel still prices both — the wildcard
   over the six-gameweek horizon, which is what the row says, and the other
   three over next gameweek — so the case can be made in the rationale.

   **This is now the next thing the solver should learn.** A chip is a gameweek
   whose transfer rules are different, which is one more quantity to carry from
   one gameweek to the next — exactly the shape of thing the multi-week window
   already carries the bank and the free transfers as. Chip-aware solving is not
   a solver of its own to write; it is a chip variable a gameweek in the one
   that is there, and the answer it would give is the question the report
   currently has to hand back to a human: *is this the week?*
4. **The mid-season chip reset.** `played_chips` counts a chip as gone the
   moment it appears in the season's history, without asking which half of the
   season it was played in. Modern FPL hands out a second set at the halfway
   point, so from GW20 the bot is too strict rather than too generous — it will
   decline to recommend a chip it actually holds. That is the safer of the two
   mistakes, and it is still a mistake.
5. **The bench order the design asks the manager for.** The spec has him
   returning a bench order with his decision; he is not asked for one. The
   order the report prints is the solver's — substitute keeper first, then by
   next gameweek's projection — and it is a good default and nobody's judgement
   about which of two fringe players is likelier to have a game at all.
6. **Price-change pressure in the briefing.** The spec lists it among the
   manager's inputs and the briefing does not carry it, so a player about to
   rise or fall reads to him exactly like one who is not, and "buy him this week
   rather than next" is an argument he cannot make. The bootstrap does publish
   the transfer counts it would be estimated from; the estimate itself is a
   model of an algorithm FPL does not document, which is why it is not in here
   pretending to be a fact.
7. **A search budget for the run.** `max_uses: 8` on the web search tool is a
   per-request cap — one assistant turn — not a budget for the conversation, so
   a twelve-turn run could in principle spend eight searches in each of them.
   Nothing counts them across turns or stops the loop when the total gets
   silly; what bounds a run is the turn cap and the twelve-minute clock, and
   the report prints the count afterwards. The cost note above is written for
   that worst case rather than against it.

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Requires Python 3.12 or newer. All tests run offline — HTTP is faked with
`httpx.MockTransport`, never the live API — which is why the same suite runs on
every push and pull request in `.github/workflows/tests.yml`, with no secrets
in the job at all.

It takes about half a minute rather than the second a mocked suite would, and
almost all of that is real MILP solves: the multi-week tests solve hand-built
universes whose optimum is worked out with a pencil in the comment above each
test, and the pipeline tests sweep real windows end to end. That is deliberate.
A solver test that mocks the solver tests nothing, and thirty seconds of CI is
a cheap way to find out that a formulation stopped being the game's rules.
