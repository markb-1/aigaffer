# aigaffer

An autonomous manager for Fantasy Premier League. On a schedule it pulls the
week's data from the FPL public API, projects expected points, solves for the
best transfers, lineup and captain over the coming six gameweeks, puts the
result to a Claude agent that reads the week's team news and may overrule it,
and messages the finished verdict to your phone. It advises and never acts:
no transfer is ever executed, and the final tap in the app is yours.

📋 **Latest verdict: [GW6](GW6.md)** <!-- latest-verdict -->

## What arrives, and when

Four messages per gameweek, three anchored to the deadline ahead and one
to the round just played:

| When | Report | What it is for |
| --- | --- | --- |
| the evening after the last match | `early` | the early scout: a first look at the plans while the results are fresh, from 18:00 UTC the day after the round's last kickoff |
| 24–60h before the deadline | `scout` | transfer plans while there is still time to think — aimed at ~60h out |
| 3–24h | `deadline` | the full verdict, aimed at T-24h so an evening is left to act on it |
| 0–3h | `reminder` | a short alert: the moves, and whether the plan survived the team news |

Each report is aimed at the top of its window and sent by the first
scheduled tick to land inside it; the windows stay open until the next
report's territory so that a run the scheduler dropped is made up late
rather than lost. Each is sent once — the store remembers. The early scout
is the scout under its own name, and its window closes when the scout's
opens: a round that ends close to the next deadline — a midweek one — gets
one scout, not two.

Telegram gets a digest, not the document: one line on where the season stands
(points, overall rank, team value, bank, free transfers), one `Chips:` line
under it (e.g. `Chips: TC GW15 · BB GW17 · FH GW12 · WC GW16 · expire GW19` —
`now` for the chip the gaffer plays this week, otherwise the planned or
saved-for week; not on the reminder), the ⏰ checklist, the
moves priced, the gaffer's opening paragraph and a pointer at the rest — a
phone message a person can act on in one screen. The reminder and the
withheld alert open with the same line. The full reports are kept as a diary in
`state/reports/`, and the polished verdict for each gameweek is committed by
CI to `GW{n}.md` at the repo root — the copy the GitHub homepage shows,
written first by the early scout, rewritten by the scout midweek and
overwritten by the deadline run.

The reminder is the solver alone — the manager is never woken for it — and
its ⚠️ means the *news* moved, not that the manager isn't a solver: it diffs
its fresh solve against the solver's own pre-manager plan, which the full
report records alongside the verdict, so a week where the gaffer overrode the
solver stays calm unless the team news actually changed the solver's answer.
Calm is one block: the gaffer's verdict, the operative plan. Moved news leads
with a ⚠️ section naming each change — sells and buys as separate lines, the
way the app takes them — and shows both plans, the gaffer's verdict first,
because the fresh solve is information and not an overruling. It goes to
Telegram and `state/reports/` only, never to the root `GW{n}.md`, which stays
the polished verdict — and unlike the full report it delivers before it
saves, so a failed buzz is retried by the next tick rather than marked done.

A full report reads like this (the players are invented; the shape is real):

```markdown
# AI Gaffer — GW12 deadline

Deadline: Sat 28 Nov 2026 13:30 UTC

## Do this

⏰ Make these by Sat 28 Nov 2026 13:30 UTC — GW12
SELL Byrne (MID AVL £6.4m) → BUY Reid (FWD CRV £9.5m)
CAPTAIN Reid · VICE Alvez
Set lineup (3-5-2): Alvez; Costa, Egan, Ferris; Hume, Innes, Jonker, Kerr, Salas; Reid, Moss

## Recommendation

1 transfer, no hit.

- Out: Byrne (MID, AVL, £6.4m)
- In: Reid (FWD, CRV, £9.5m)

## The Gaffer's view

WHAT I DID — Sold Byrne for Reid and gave Reid the armband. No chip.
WHAT I LEARNED — Quinn was ruled out in Friday's press conference; the
projection had him at 84 minutes, so I zeroed him and re-solved. …
WHY THIS PLAN — Plan 4 beats the roll by 3.8 over the window even with
Quinn out, and the alternative hit does not clear the noise. …

Minutes he overruled:

- Set Quinn to 0 mins — ruled out for GW12 at Friday's press conference

4 web searches. Decided by the gaffer.

## Candidate plans

xP and net are the whole window, decayed and with the armband in; transfers and hits shown are this week's unless the row says otherwise.

- 0 transfers | 0 hits | 251.2 xP | 251.2 net
- 1 transfer | 0 hits | 255.0 xP | 255.0 net  <- recommended
- 2 transfers | 1 hit now, 2 over the window | 257.1 xP | 249.1 net

## The road ahead

- GW13: out Fenn (£4.5m), in Dodd (£5.0m) — 1 FT, no hit
- GW15: Bench Boost

Advisory — re-planned every run; only this week's moves are ever made.

[… Starting XI and Chip EV follow, then …]

## Chip calendar

- Bench Boost (GW1–19): saved for GW17, worth +11.0 there; this week's bar 8.9
- Triple Captain (GW1–19): saved for GW15, worth +5.2 there; this week's bar 4.3
- Wildcard (GW2–19): this week's bar 35.0, falling to 0 as GW19 nears
- Free Hit (GW2–19): saved for GW12, worth +6.5 there; this week's bar 14.8

The first set expires after GW19.

[… and the Watchlist …]
```

## How it decides

The pipeline is: FPL public API (httpx + pydantic models) → expected-points
model → PuLP MILP solver (candidate transfer plans) → a Claude agent with the
right to overrule → markdown report → Telegram. State (SQLite DB + reports)
lives in `state/`, configurable via `AIGAFFER_STATE_DIR`.

It was built in phases — the analyst, the gaffer, the planner, the chip
planner — and the sections below describe each part as it now stands; the
commit history keeps the order they arrived in.

### What it reads — and what it cannot know

Everything live comes from the FPL public API: the bootstrap (players,
prices, season totals of goals, assists, expected goals and assists,
saves, bonus, defensive actions, minutes, transfer counts), the fixture
list with results, each shortlisted player's element-summary (his
gameweek-by-gameweek minutes this season and one row per past season),
and the entry's own picks, transfers and chip history. Beside it ride two
small vendored files, distilled once a year from the public
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
dataset: last season's per-player totals (`aigaffer/data/prior_season.csv`)
and two seasons of match results (`aigaffer/data/results.csv`) — the past
the API cannot serve. No paid data, no odds feeds, no scraped team news.

Three blind spots are structural rather than accidental. The API never says
what you paid for anyone — and purchase price is the number selling actually
turns on, since FPL pays a riser's owner only half the rise — so the bot
keeps a purchase ledger of its own, reconstructed by observation: seeded
from how far each price has moved since the season opened, maintained by
watching the picks for every arrival and departure, and audited once a
gameweek against the bank the game publishes, with one line in the report
when the two disagree. The public picks endpoint lags to the last deadline,
so a transfer you make mid-gameweek is invisible until the next deadline
passes — which is also the ledger's error bar, since a buy it first sights
then is recorded at that day's price, not your click's. And nothing in the
payload knows about this afternoon's press conference — which is the half of
the job the model cannot do, and the reason there is a manager at all.

### The model

Expected points are built fixture by fixture: the two teams' fitted
strengths set how likely a goal or a clean sheet is, the player's expected
minutes set how much of it he is around for, and his seasons — this one
and, discounted, the last — say what he does with the time. A gameweek is the sum over his club's fixtures in it, so a blank
scores nothing and a double scores twice without any special case.

**Expected minutes** are the mean of his last five *played* gameweeks —
phantom rows the API adds for matches not yet kicked off are dropped at the
fetch — scaled by his availability flag. Under three gameweeks of history the
mean is floored by a prior: his last Premier League season's minutes divided
by 38, a measurement rather than a guess, so a first-choice player rested on
the opening weekend is worth his eighty-five minutes and not a substitute's
twenty. A player with no Premier League season behind him falls back on the
old guess — 75 minutes if he has started this season, 20 if not.

**Per-90 rates** — goals, assists, saves, bonus, defensive contributions —
start as season totals divided by season minutes, with two lines of caution.
A rate is never taken over less than a full match, and no minutes is no rate
at all. A returning player's last season is pooled in first, as discounted
evidence — a third weight (a quarter for saves, and halved again for a
summer club move), joined on the player code that survives the id reset —
so in August an elite forward opens at an elite rate rather than at the
league average, and the new season's own minutes displace the old one's
around midwinter. Then the pooled rate is shrunk toward a coarse
league-average prior for the position, weighted as if the prior were six
nineties of the player's own play, which is what keeps one loud opening
gameweek from annualising a £4.5m defender into an elite striker — and
what a newcomer with no season behind him is priced on alone.

**Fixture factors** come from team strengths fitted on results — a
time-decayed Poisson with home advantage, refit on every run from the two
vendored seasons plus whatever the live one has finished, so the ratings
sharpen weekly for free. A player's attacking factor reads the opponent's
fitted defence and the venue (his own team's attack stays out — his per-90s
already embody it), clamped to 0.7–1.3 so no fixture ever triples anybody;
the goals his side concedes read *both* teams, which is what lets a strong
defence keep more clean sheets than a weak one against the same opponent.
FPL's editorial strength columns remain as the fallback: a failed fit says
so on the log and prices the week exactly as the model did before the fit
existed.

**The horizon** is six gameweeks, each step discounted by 0.85, because
points further out are worth less to a decision made today.

**The armband** is chosen on attacking EV — goals and assists, the
high-variance half that doubles into a haul — rather than on the total, so
the captaincy goes to a genuine goal threat and not to a cheap player whose
steady floor and kind fixture have padded his total past one.

### The solver

The single-week question — given this squad, this bank and these free
transfers, what is the best fifteen reachable *this week*? — is the question
a manager asks and it is not the question a manager answers. Money raised
has to be spent the moment it is raised, a free transfer rolled is a free
transfer thrown away, and a plan that needs five moves and can make three is
infeasible rather than staged.

So the recommendation is solved over the whole six-gameweek horizon
(`aigaffer/solver/multiweek.py`) as one mixed-integer program: a squad, an XI
and a captain for every gameweek in the window, with the bank and the
free-transfer count carried between them under the game's own rules — one
free transfer a week, never more than five in hand, four points for every
move past what is held and eight points a gameweek at the very most. Only
the first gameweek is ever entered, but it is chosen knowing what it is for.
Three things follow that a single-week solver cannot reach:

1. **A rolled transfer is a decision.** Banking this week's move to make two
   free ones next week is a plan rather than an accident, and the model can
   see what the roll is *for*.
2. **A move can be made early for a gameweek that has not arrived.** The
   double gameweek is the case this was built for: five moves will not fit
   under one gameweek's ceiling, so the move that only raises the money goes
   a week ahead of the moves that spend it. A single-week solver never makes
   that move at all — selling a good player for a cheap one is a downgrade
   right up until the week it pays for something, and by then it is too late.
3. **The captain is inside the model.** The window prices every gameweek
   with the armband already on the best man in that gameweek's eleven, so a
   fifteen that unlocks a big captaincy is worth to the solver what it is
   worth to the season. The armband finally *played* is still picked
   afterwards, by `pick_lineup` and then by the gaffer; what changed is that
   it is no longer invisible to the thing choosing the squad.

The gameweeks after this one are printed in the report under **The road
ahead**, and ride the end of every plan line in the manager's briefing. The
report closes that section the same way every week, and the line is the
whole of the path's status:

> Advisory — re-planned every run; only this week's moves are ever made.

Nothing in a path is a commitment. It is re-solved from scratch on the next
run, on projections that will have moved, and the one thing a reader must
not do is hold this week's move back because a path he read a fortnight ago
says it belongs in the next. The briefing makes the same point at more
length, because a model reading "GW4 +Pike -Byrne" beside a plan it is being
asked to commit to would otherwise reasonably read it as part of the
commitment.

**Two engines, and which one answered.** The window goes first, once per
transfer count the shortlist considers, each solve on a twenty-second leash.
That is four to six solves, and on the boards it has been run against they
have come back in a second or two between them — the leash is a ceiling, not
a budget, and a solve that reaches it with a squad in hand hands the squad
over anyway. When the window has nothing at all to say — an infeasible
board, or a sweep where every count came back empty — the same questions go
to the single-week solver, because a gameweek with a deadline needs a
recommendation more than it needs the better model. Which engine answered is
written down nowhere, and does not need to be: a plan off the window carries
its path and a plan off the single-week solver does not. A run that asked
for the window and got the other engine says so once, under the candidate
plans; a run that asked for the single-week solver on purpose prints nothing
— it got what it asked for, and a report that apologised every week would be
crying wolf.

**Chips are inside the model too.** A chip is a gameweek whose rules are
different — a bench boost scores all fifteen, a triple captain adds an
armband multiple, a wildcard makes that week's transfers free and uncapped,
a free hit fields a one-week squad that reverts — and each is a per-gameweek
binary the window may switch on, at most one chip a gameweek and each chip
at most once across the horizon. The free hit is priced by a small solve of
its own — the best legal one-week fifteen the pool holds inside the budget —
and a binary chooses the week to spend it, which keeps the revert
*structural*: the free-hit squad is never a variable of the window, so it
cannot leak into next week's team however the search wanders. When the
optimum plays a chip in the gameweek being decided the report leads with it
(**PLAY Free Hit**); chip weeks further out ride **The road ahead** exactly
as the transfer moves do.

**Chips come in two sets, and the first expires.** FPL hands out each chip
twice a season: one set playable in GW1–19 (the wildcard and free hit from
GW2) and a second in GW20–38. A first-set chip not played by the GW19 deadline
is lost, and the second set is not available before GW20. The bootstrap says
so (`aigaffer/chips.py` reads its `chips` list), and the bot holds each chip
*per half*: a chip played in the first half does not spend the second-half
one, and two of a kind can be held at once. The one cross-chip rule is one
chip a gameweek. `AIGAFFER_CHIPS=off` (and draft mode) holds no chips at all,
so the solver is the pre-chip model, byte for byte.

A chip is free to play, so a model with nothing pulling the other way would
burn every chip it holds in the best week of the window, however ordinary
that week is. What pulls the other way is the **chip calendar**
(`aigaffer/solver/calendar.py`), built once per run, which gives each held chip
a **bar for every week of the window**: the opportunity cost of not saving it.
A chip is played in a week only where its marginal points clear that week's bar.

- **Values beyond the window.** For each week from the window's last to the
  chip's expiry the calendar prices the chip by a small solve of its own:
  the triple captain is the best armband candidate who is available and
  expected to play 80 or more minutes (`TC_MIN_MINUTES`) — his projected points
  that week; the bench boost is the bench of the best fifteen bought with the
  bench weighted fully, at 0.9 of its worth; the free hit is the best one-week
  squad less the current squad's best one-week score. The projections are
  fixture-only and built from *base* minutes, so the gaffer's one-week overrides
  never reach GW17. Each value is then scaled by a proxy haircut
  (`PROXY_SCALE`: bench boost 0.7, triple captain 0.85, free hit 0.85), because
  a proxy prices a squad bought for the purpose.
- **Saved-for weeks.** Within each half, the triple captain, bench boost and
  free hit are assigned to distinct beyond-window weeks, or left unassigned,
  to maximise value discounted by `CHIP_DISCOUNT` ρ = 0.97 a week. With fewer
  weeks left than chips, the lowest-valued go unassigned and are played inside
  the window, which is the intended expiry behaviour.
- **Per-week bars.** A chip with a saved-for week w\* has bar v·ρ^(w\*−w) in
  window week w: play it now only if now is not clearly worse than the week it
  is saved for. A chip with no saved-for week has bar 0. The free hit adds an
  **option floor** of 10 points while any later week remains, since its real
  value is insurance against a blank or mass absences the projections cannot
  see. The wildcard is not valued week by week: its bar is 35
  (`WILDCARD_BAR`) ramping to 0 over its last eight weeks to expiry
  (`WILDCARD_RAMP_WEEKS`).
- **Expiry.** The bars reach 0 as GW19 nears, so by the end of the first half
  every unplayed chip is free points and the window plays it, stacked early
  if need be, rather than lose it.
- **If the calendar fails** the run does not: one line on stdout and the
  old flat bars (`FALLBACK_BARS`: bench boost 20, triple captain 18, free hit
  25, wildcard 35), except that a chip with no week left beyond the window has
  bar 0. The briefing and digest say "calendar unavailable".

There is no floor on the opening weeks. On a flat board — no doubles or blanks
yet announced — the triple captain and bench boost can therefore be played as
early as the next gameweek: with no doubles in sight there is nothing to wait
for, and the bars lean towards playing now unless a later week is clearly
better. The free hit is held by its option floor and the wildcard by its ramp.
Doubles and blanks that arrive later (cup clashes, rearrangements) show up in
the weekly re-run. The window still only *plans* six gameweeks, so the
calendar is as good as the fixtures it can see; the constants are on the
Deferred list below.

### The gaffer

The solver says what is optimal under the model. The manager
(`aigaffer/manager/`) adds the half it cannot do: what has happened since
the numbers were computed. After the solve, the week goes to a Claude agent
with four things it may do and nothing else:

1. **Read the briefing.** Built by `aigaffer/manager/briefing.py`, it is the
   whole of what the agent knows going in: the gameweek and its deadline,
   the date the briefing was written (so stale news can be recognised as
   stale), the bank, free-transfer count and squad value; the fifteen, a
   line each, with every player's id, price, availability status, the
   expected minutes the projection assumed and his projections for next
   gameweek and the horizon; the XI the solver's own choice would field; the
   candidate plans, numbered by the ids they will be finalized by, each with
   its window totals, hits and path; the chip EV panel with chips already
   spent marked; a ten-player watchlist of the best projections not held;
   and a research list — the thirty-odd players any plan holds, buys or
   sells — which is the whole of what it is asked to look up. When fewer
   than five gameweeks have finished, the briefing carries an early-season
   advisory: the rates are still stabilising, weight your own team-news
   findings over projection outliers.
2. **Search the web** for what the model cannot know: injuries, suspensions,
   rotation, a press conference that made somebody a doubt. The tool is
   capped at eight searches **per request** — that is one assistant turn,
   not the run — so a long conversation can spend more than eight in total;
   see Deferred.
3. **Overrule the minutes and re-solve.** `adjust_players` sets a player's
   expected minutes for the coming gameweek absolutely — 0 for a player who
   is out, 20 for a substitute, 90 for a starter — and `resolve` re-projects
   and re-solves with those in force, coming back with fresh plans under
   fresh ids. The coming gameweek only: every later week in the planner's
   window keeps the model's own minutes, so "out this week" does not sell a
   player for the month or price his later bench boost and armband at
   nothing. The model's minutes already carry the FPL injury flag across the
   whole window, so a flagged long-term absence stays at zero regardless;
   the cost is an absence the gaffer has read about and the API has not
   flagged yet, which reads as back after the coming week.
4. **Finalize** one plan, a captain and vice from that plan's eleven, a chip
   or none, and the rationale.

The guardrails are in code, not in the prompt. A plan is an id from a
registry minted by the loop, so no amount of text in the conversation — a
briefing quoting the FPL API, a web page quoting whoever wrote it — can pass
for a plan that was never solved. The armbands are checked against the
eleven of the plan he actually chose. A chip needs an argument, not a
sentence — a justification long enough to be a case and about the chip it is
playing — and all four are his to finalize, because the solver plans chip
weeks now: a plan that carries a chip was built for that chip. The one chip
refused is one not held for this gameweek — already spent in this half, or a
second-half chip before GW20 — and that refusal holds whatever he argues: the
pipeline checks his chip against the chip history before recording anything.
His system prompt carries a **chip playbook** (`aigaffer/manager/playbook.py`):
what makes a good week for each chip, and that in a chip's last two eligible
gameweeks it is free points — he finalizes the solver's chip unless team news
makes the week a dud, and says so in the rationale if he declines it in the
final week. His briefing has a `## Chip calendar` section after the Chip EV
panel: each held chip's window, saved-for week, this week's EV and bar.
Everything the solver already enforced — budget, club quotas, the hit cap — it
still enforces, because he only ever picks from plans it produced. The
conversation itself is bounded too: twelve assistant turns and twelve minutes,
whichever runs out first.

What comes back is printed in the report as **The Gaffer's view**: the
rationale, the chip and its justification, the minute adjustments he settled
on and how many searches it took.

### The fallback doctrine

A deadline is never silently missed, and a manager who cannot be reached is
not a reason to miss one. **Every** failure degrades to the solver's own
recommendation — the pipeline's answer with no manager in it — and that
recommendation always reaches the phone before the window closes. What
changes is *when*: see the withholding rule below. There are two shapes of
failure and they read differently:

**Asked and it went wrong.** A rate limit, an authentication error, a
refusal, a connection that dies, a turn the model never finishes, twelve
turns that reach no decision, a re-solve that finds nothing, a chip he tried
to play twice, a briefing that could not be built. The Gaffer's view section
is still printed and says whose pick this actually is: *"The gaffer was
unavailable (…); this is the solver's pick."* The reason is a class name or
a phrase of ours — never an exception's own words, which can carry a key or
a token.

**Never asked at all.** No key, the kill switch, an initial squad draft
(fifteen players from nothing is not a week to read the news about), or a
dependency that will not import. Then there is no decision object to hang a
section on and the report is exactly the solver's. The first three are
choices, and they say nothing: that is the invariant below. The fourth is an
accident, so the report carries one line — *"Note: the manager is configured
but was unavailable this run; this is the solver's pick."* — because a fork
whose `anthropic` install broke would otherwise read a season of solver-only
reports and never be told.

**Withheld, and asked again.** Either shape, while the report's window still
has more than three hours left, is not written down: no diary file, no run
row, no digest. The tick prints *"the report was withheld for the next
tick"*, sends the phone one short alert — the reason, when the retries stop,
and the solver's checklist so there is always a plan to act on — and stops.
The next tick, from either scheduler, runs the whole thing again; a manager
who has recovered (a key replaced, a rate limit lifted) produces the normal
report, and the alert is not repeated unless the failure changes. The
waiting ends three hours before the window closes, or after twelve withheld
ticks (`MANAGER_RETRY_LIMIT`), whichever comes first: the labelled solver
report goes out and counts, so the reminder still has a verdict to check the
news against and the week is never left without one. A dry run withholds
nothing and alerts nobody, and a run a person forced (`--force`, or the
workflow's button) takes whatever answer it got: forcing has always meant
overruling the store, and now it overrules the waiting too. The early scout
is the one exception to the ending: it is held for as long as the clock
would still choose it — until the Thursday scout's window opens — and past
the retry limit it is written down but not sent, because that scout is
hours away and will be asked properly; a labelled early scout would be a
second text saying less.

The rule underneath: the solver-only path is the invariant. With
`AIGAFFER_MANAGER=0` the pipeline produces byte-identical output to the
solver-only pipeline, and a test holds it to that.

## Run your own

The bot runs itself from `.github/workflows/gaffer.yml`. Fork the repo, then:

### Secrets

**Settings → Secrets and variables → Actions → New repository secret**, four
of them:

| Secret | Value |
| --- | --- |
| `FPL_TEAM_ID` | your FPL entry id (the number in your team's URL) |
| `TELEGRAM_BOT_TOKEN` | from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | the chat to send to — message the bot, then read `chat.id` from `https://api.telegram.org/bot<TOKEN>/getUpdates` |
| `ANTHROPIC_API_KEY` | from [the console](https://console.anthropic.com/settings/keys) — this is what turns the manager on |

Set both Telegram ones or neither: a token without a chat id delivers
nothing and says so in the log. `ANTHROPIC_API_KEY` is independent of them
and optional — leave it out and every run is the solver's alone. The
workflow needs `contents: write` (already declared) to commit `state/` and
the `GW{n}.md` verdicts back.

Beyond the secrets, `Config.from_env()` also reads `AIGAFFER_STATE_DIR`
(state directory, default `state`) and `AIGAFFER_MANAGER_MODEL` (which model
the manager is, default `claude-opus-5`).

The state the workflow commits back is the bot's memory, and not all of it
is regenerable. The SQLite DB in `state/` holds the run history beside two
tables the purchase ledger lives in: `purchases`, what was paid for each
player held — reconstructed by observation, because the API never says — and
`squads`, one snapshot per gameweek of the bank and fifteen the picks
endpoint published, which the ledger's selling estimates are audited
against. Wipe `state/` mid-season and the next run reseeds every squad
member at his season-opening price — right for anyone held since GW1, wrong
for every player bought since, and wrong until he is sold. A fork adopting
the bot mid-season starts in the same place: the seed assumes the squad it
finds was held from the start, so recent buys carry season-opening prices
until the reconciliation line in the report has had a gameweek or two to say
how far adrift that left the estimates.

### Kill switches

Four repository variables — **Settings → Secrets and variables → Actions →
Variables** — stand parts of the system down without a commit to the
workflow and without throwing away a secret:

| Variable | Value | Effect |
| --- | --- | --- |
| `AIGAFFER_MANAGER` | `0` | the solver alone, even with a key: byte-identical to the solver-only pipeline |
| `AIGAFFER_PLANNER` | `single` | the single-week solver answers alone, no window |
| `AIGAFFER_CHIPS` | `off` | chips advisory only: priced in the panel, held by no calendar and planned by nobody (the gaffer may still play an unspent chip) |
| `AIGAFFER_STRENGTH` | `off` | fixtures priced by FPL's strength columns, the fitted team-strength model asked nothing |

Each switch takes exactly one literal value, and everything else — a typo,
an empty export, a `0` where `off` was meant — leaves the default standing,
because turning a better engine off is not something a misspelling should be
able to do. The workflow sets all three deliberately rather than leaving
them unset, so that every scheduled run has *asked* for what it runs: a run
that asked for a manager and has no key to reach him with prints one line
saying so, which is what a missing secret looks like from the outside
instead of nothing at all.

### The schedule, and running it by hand

**Actions → gaffer → Run workflow** runs one report by hand; the `mode`
input runs one by name instead of asking the clock (a manual dispatch skips
the schedule's gate entirely).

The schedule fires hourly at `50 * * * *` as a **backup**: the primary
tick is best run from a machine of your own — any always-on box with the
repo cloned, running `python -m aigaffer auto` on an hourly timer at :35
and pushing the state back, which is what `scripts/run-tick.sh` and the
units in `deploy/` do (setup below; FPL deadlines sit on the half hour, so a :35
tick lands each report about five minutes after its window opens; the
store then stands GitHub's :50 tick down, and carries the report if your
box misses). The Actions workflow does nothing at all most of the week:
a curl-and-jq gate at its top checks the next deadline and stands the tick
down in
seconds unless it is within about sixty hours (Python stays the authority
on the windows; the gate is a generous superset that only exists to spare
the pip install). Each gameweek gets one report of each kind: the SQLite
store remembers, so a second tick inside a window is a no-op — unless the
manager was asked and did not decide, in which case nothing was recorded
and the second tick is the retry. The two schedulers see each other only
through the pushed state, and a manager run is minutes long — long enough
for the other scheduler's tick to start before this one has pushed. So a
scheduled tick reads the store twice: from its own checkout before it
starts, and again from a fresh fetch of `main` once the manager has spoken
and before anything is sent or saved; a tick the other scheduler overtook
stands down there, having spent a manager run and nothing else. What that
leaves is the few seconds between that second read and the push, which a
report that lands twice would still cross; a manager failing on both hosts
inside one window, which can cost a second withheld alert, since only a
recorded report stands the other tick down; and a fetch that fails leaves
the tick to carry on rather than silence it. The early scout is the one
report the gate never wakes for — its evening is days from the deadline —
so it is your own box's alone: if that box is down that evening the early
scout is skipped, and the scout on Thursday is the first word. GitHub drops
scheduled runs under load — occasionally for whole days at a stretch —
which is why the windows stay open late rather than closing at their
anchor: the first tick to survive sends whatever the dropped ones owed, and
a reminder-hour tick with no full report on record runs the full report
first.

Because GitHub's cron is best-effort, `scripts/dispatch-tick.sh` packages a
tick you can fire from a second scheduler GitHub's load cannot touch — a
`launchd` job on a Mac, a cron line anywhere, a free ping service. It runs
the same generous gate locally, dispatches the `auto` mode, and authorizes
itself from git's stored GitHub credential. Duplicate triggers send nothing
twice — the store stands the loser down — though a dispatched duplicate
bills the checkout and install it took to ask, which is why the script
gates itself instead of pinging year-round.

Whoever fires the tick, a failed run says so on stdout and sends one line
to Telegram — never the exception's own text, which for an `httpx` error
contains the URL and so the bot token.

**GitHub disables scheduled workflows after 60 days without repository
activity**, and emails the owner first. That will happen over the
off-season, when nothing is being committed and the bot is standing down
anyway; re-enable it from the Actions tab (or push any commit) before the
first deadline of the new season.

### Your own box

Any always-on Linux machine will do — a free-tier cloud VM is plenty. On
it:

```sh
git clone git@github.com:YOU/aigaffer.git && cd aigaffer
python3 -m venv .venv && .venv/bin/pip install .
cp .env.example .env && chmod 600 .env   # then fill it in
```

- **Push access.** The tick commits `state/` and the `GW{n}.md` verdicts
  back, so the clone needs write access: a deploy key with *Allow write
  access* (repo **Settings → Deploy keys**) is the narrowest way to give it.
- **The solver binary.** PuLP bundles CBC for x86-64; on ARM (Oracle's free
  Ampere boxes, a Raspberry Pi) install it from the distribution —
  `sudo apt install coinor-cbc` — or every solve fails.
- **Check the key before trusting it.** A rejected Anthropic key does not
  fail the tick, it withholds the report and alerts you, so prove it once by
  hand from the box itself:
  `.venv/bin/python -c "import anthropic; print(anthropic.Anthropic().models.list().data[0].id)"`
  with the `.env` exported. Then a dry run: `set -a; . ./.env; set +a;
  .venv/bin/python -m aigaffer scout --dry-run`.
- **The timer.** Copy `deploy/aigaffer.service` and `deploy/aigaffer.timer`
  to `/etc/systemd/system/`, fill in the user and path in the service, then
  `sudo systemctl enable --now aigaffer.timer`. `systemctl list-timers
  aigaffer.timer` shows the next tick and `journalctl -u aigaffer` what the
  last one said.

`run-tick.sh` stands down while `.env` is missing or still holds a
`FILL_ME`, pulls before it runs so a fix pushed from your laptop is live
on the next tick, and commits only when the run changed something — an
out-of-window tick costs one `git pull` and a few seconds of Python.

### What it costs

**Typically well under $2 per decision run** at Opus 5 rates, and a
search-heavy week can be a few dollars. The driver is not the searches
themselves but that the conversation is resent in full on every turn: a
quiet week that searches twice and decides in three turns is cents, and a
pathological one — a dozen searches, two re-solves, twelve turns each
re-reading everything before them — costs an order of magnitude more. Three
decision runs a gameweek (early scout, scout and deadline; the reminder
never wakes the manager), so on the order of $3–12 a week and $75–300 for a
season, weighted towards the low end because most weeks are quiet. The one way a week gets
expensive is a manager who fails slowly and is asked again: a withheld
report (see the fallback doctrine) is retried by every tick, and a refusal
or a loop that runs out of time is a full run each time, which is what the
twelve-tick limit is for — worst case a dozen runs, and the alert has told
you by the first of them.

Three cache breakpoints hold it down, of the four the API allows. Two never
move — the system prompt and the briefing, which are the whole of what
cannot change once the conversation has started, and most of the input on
turn one. The third rides the end of the messages and moves forward with
every turn, so each turn's searches and tool results are read from cache by
the turn after it instead of being paid for in full a dozen times.

### Local development

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Requires Python 3.12 or newer. All tests run offline — HTTP is faked with
`httpx.MockTransport`, never the live API — which is why the same suite runs
on every push and pull request in `.github/workflows/tests.yml`, with no
secrets in the job at all.

The suite takes about half a minute rather than the second a mocked one
would, and almost all of that is real MILP solves: the multi-week tests
solve hand-built universes whose optimum is worked out with a pencil in the
comment above each test, and the pipeline tests sweep real windows end to
end. That is deliberate. A solver test that mocks the solver tests nothing,
and thirty seconds of CI is a cheap way to find out that a formulation
stopped being the game's rules.

To run a report against the live API without touching state or Telegram:

```sh
FPL_TEAM_ID=1234567 python -m aigaffer scout --dry-run   # prints, saves nothing
python -m aigaffer backtest                              # grades the model, needs no team id
```

`--force` re-runs a report the store says has already gone; `backtest --gw 3`
grades one gameweek by name. What the backtest numbers mean — and what they
deliberately do not — is under Honest limitations below.

## Honest limitations

### Known approximations

These are deliberate. Do not "fix" them without revisiting the design:

1. **Selling prices are reconstructed, not read.** The public API still does
   not expose your purchase prices; the purchase ledger rebuilds them by
   observation, and the reconstruction has a known error bar. The seed —
   today's price minus its movement since the season opened — is exact only
   for a squad held since GW1. A player bought later is first sighted at his
   price on the run after the next deadline reveals him in the picks, which
   can trail what you actually clicked by however many £0.1m daily moves
   happened in between. The bank-reconciliation line in the report is the
   drift detector for exactly that: its first appearance after a week of
   manual transfers is the system working, not a bug, and the game's own
   published bank stays authoritative everywhere either way.
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
   minutes. The gaffer's own minute overrides remain the mitigation for those,
   for the coming gameweek: an override does not reach the weeks after it, so
   his eleven and armband are corrected but the window still undervalues him
   further out until his own history takes over.

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
   sell side — which is the case to watch in the first three gameweeks. The
   gaffer's own minute overrides correct the eleven he is left out of; they
   are for the coming gameweek only, so a sale the window recommends on his
   later weeks is the gaffer's to argue down, not the override's to fix.

### The backtest is a sanity check, not a backtest

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

1. The season totals the rates are taken from and the prices come from the
   bootstrap as it stands today, which includes the gameweek being graded
   and every one since: the model is asked to rank a week it has already
   seen. (The fitted team strengths are the exception — they are rebuilt
   from results before the graded round, dated at its deadline, and do not
   leak.)
2. The shortlist admits only players available *today*, so anyone since injured
   or gone is never graded — the population is the survivors.
3. The shortlist then cuts by season-to-date total points, today's total,
   which includes everything scored *after* the graded gameweek: the field is
   skewed towards the players who went on to do well.

Only the minutes model is held honest. So the level of `spearman` is
optimistic: read it as a sanity check on ranking quality — is the order better
than chance, and is it holding up week to week? — and never as an estimate of
how the bot would have done.

### Deferred

Still not shipped, deliberately:

1. **Weekly input-data snapshots.** The store keeps the report and the decision
   for each run, not the bootstrap and histories they were computed from, so a
   past recommendation can be read but not recomputed. Snapshotting the inputs
   is what would make a run reproducible after the API has moved on. The
   manager sharpens this: the decision record holds his minute adjustments,
   but not the projections they were applied to, so his week can be read and
   not rebuilt.
2. **A full-season vaastav backtest with a beat-the-average benchmark.** What
   ships is a single-gameweek ranking sanity check against the live API
   (above), which cannot say whether the bot would have beaten the average
   manager over a season — the question that actually matters.
3. **Chip economics against the plan actually chosen.** The Chip EV panel is
   priced once, against the plan that rolls the transfer, before the manager is
   asked anything. If he picks a different plan and plays a chip on it, the
   numbers he argued from describe a squad he did not enter. Re-pricing the
   chips per plan is a solve per chip per plan, which is why it is not done
   yet. Chip-aware *solving* itself is no longer here — the window puts a chip
   variable in every week and decides when each is played — but the advisory
   panel a human reads is still priced against the roll, not against whatever
   plan he finally chooses.
4. **Season-long chip timing beyond the fixtures.** The chip calendar now runs
   to expiry, but beyond the six-gameweek window it sees only fixture-only
   projections and proxies — a haircut bench, a bought-in captain, a free hit
   against today's squad — not the squad the bot will actually own, nor
   doubles and blanks FPL has not yet announced. It picks the week the board
   looks best *today* and re-plans every run. Pre-window bench building and
   ownership-aware timing are out of scope.
5. **The remaining chip constants are first guesses.** The bars are computed
   now, but what feeds them is tuned by hand to FPL norms, not measured against
   results: the discount ρ (0.97), the proxy scales, `TC_MIN_MINUTES` (80), the
   free hit's option floor (10) and the wildcard's bar and ramp (35 over eight
   weeks). Each lives in one place in `aigaffer/solver/calendar.py` so it can
   move under review, and refining them against real seasons is the dial most
   likely to change how early the planner reaches for a chip.
6. **A bench order from the manager.** He could return a bench order with
   his decision, but he is not asked for one. The
   order the report prints is the solver's — substitute keeper first, then by
   next gameweek's projection — and it is a good default and nobody's judgement
   about which of two fringe players is likelier to have a game at all.
7. **A rise-tonight alert between reports.** The briefing now carries a
   price watch — the market's movers among the relevant players, framed
   strictly as timing for a transfer already decided — but nothing pings
   between reports when a planned buy is surging, so a Tuesday-night rise
   ahead of a Friday deadline is only reported after it happened. FPL's
   change thresholds are undocumented, and the watch reports volumes rather
   than pretending to predict them.
8. **A search budget for the run.** `max_uses: 8` on the web search tool is a
   per-request cap — one assistant turn — not a budget for the conversation, so
   a twelve-turn run could in principle spend eight searches in each of them.
   Nothing counts them across turns or stops the loop when the total gets
   silly; what bounds a run is the turn cap and the twelve-minute clock, and
   the report prints the count afterwards. The cost note above is written for
   that worst case rather than against it.

## License

MIT — see [LICENSE](LICENSE), which also carries the notice for the
vendored vaastav data.
