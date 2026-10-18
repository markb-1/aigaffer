# How it decides

The pipeline runs FPL public API (httpx and pydantic models) → expected-points
model → PuLP mixed-integer solver → a Claude agent with the right to overrule
→ markdown reports → Telegram. State lives in `state/` (set
`AIGAFFER_STATE_DIR` to move it). It was built in phases: the analyst, the
gaffer, the planner, the chip planner. This page describes each part as it now
stands; the commit history keeps the order they arrived in.

[Back to the README](../README.md)

## What it reads, and what it cannot know

Everything live comes from the FPL public API:

- the bootstrap: players, prices, season totals (goals, assists, expected
  goals and assists, saves, bonus, defensive actions, minutes), transfer
  counts and the chip rules;
- the fixture list, with results;
- each shortlisted player's element-summary: his gameweek-by-gameweek minutes
  this season and one row per past season;
- the entry's own picks, transfers and chip history.

Two small vendored files sit beside the API. They are distilled once a year
from the public
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
dataset: last season's per-player totals (`aigaffer/data/prior_season.csv`) and
two seasons of match results (`aigaffer/data/results.csv`). There's no paid
data, no odds feed and no scraped team news.

Three blind spots are structural:

- **What you paid.** The API never says, and selling turns on it, because FPL
  pays a riser's owner only half the rise. So the bot keeps its own purchase
  ledger. It is seeded from how far each price has moved since the season
  opened, kept up by watching the picks for arrivals and departures, and
  audited each gameweek against the bank the game publishes. The report gets
  one line when the two disagree.
- **This week's transfers.** The public picks lag to the last deadline, so a
  transfer you make mid-week is invisible until the next deadline passes,
  unless you text "Transfers made", which also records what you paid. That lag
  is also the ledger's error bar: a buy first seen after the deadline is
  recorded at that day's price, not your click's.
- **This afternoon's press conference.** Nothing in the payload knows about
  it. That is the half of the job the model cannot do, and the reason there is
  a gaffer at all.

## The model

Expected points are built fixture by fixture. The two teams' fitted strengths
set how likely a goal or a clean sheet is. The player's expected minutes set
how much of that he is around for. His seasons, this one and last season at a
discount, say what he does with the time. A gameweek is the sum over his
club's fixtures in it, so a blank scores nothing and a double scores twice
with no special case.

**Expected minutes** are the mean of his last five *played* gameweeks, scaled
by his availability flag. The API adds a history row for a match the moment a
deadline passes, before it is played; those rows are dropped at the fetch.
Under three gameweeks of history, the mean is floored by a prior: his last
Premier League season's minutes divided by 38. So a first-choice player rested
on the opening weekend is worth his 85 minutes, not a substitute's 20. A player
with no Premier League season behind him falls back on a guess: 75 minutes if
he has started this season, 20 if not.

**Per-90 rates** (goals, assists, saves, bonus, defensive contributions) start
as season totals over season minutes. A rate is never taken over less than a
full match, and no minutes means no rate. A returning player's last season is
pooled in first at a third of its weight. For saves it's a quarter, and a
summer club move halves either again. The two seasons are joined on the player
code that survives the id reset. So in August an elite forward opens at an
elite rate, and this season's minutes take over around midwinter. The pooled
rate is then shrunk toward a coarse positional league average, weighted as six
nineties of the player's own play. That stops one loud opening gameweek
turning a £4.5m defender into an elite striker, and it is all a newcomer is
priced on.

**Fixture factors** come from team strengths fitted on results: a time-decayed
Poisson with home advantage, refit every run from the two vendored seasons plus
the live season's finished matches. A player's attacking factor reads the
opponent's defence and the venue; his own team's attack stays out, because his
per-90s already contain it. It is clamped to 0.7–1.3, so no fixture triples
anybody. Goals conceded read *both* teams, which lets a strong defence keep
more clean sheets than a weak one against the same opponent. If the fit fails,
the log says so and FPL's editorial strength columns price the week instead.

**The horizon** is six gameweeks, each step discounted by 0.85, because points
further out are worth less to a decision made today.

**The armband** goes to a midfielder or forward, ranked on attacking EV (goals
and assists, the half that doubles into a haul) rather than on total points.
So a goal threat gets it, not a cheap player whose steady floor and kind
fixture have padded his total.

## The solver

A single-week solver answers the wrong question. Money raised has to be spent
at once, a rolled free transfer is thrown away, and a plan that needs five
moves when three are allowed is infeasible rather than staged.

So the plan is solved over the whole six-gameweek window as one mixed-integer
program (`aigaffer/solver/multiweek.py`). It picks a squad, an eleven and a
captain for every gameweek. The bank and free transfers carry between weeks
under the game's rules:

- one free transfer a week, and none in the week after a wildcard or free hit;
- never more than five in hand;
- four points for each move beyond those, and at most two hits (eight points)
  in a gameweek.

Only the first gameweek is ever entered, but it is chosen knowing what it is
for. Three things follow:

1. **A rolled transfer is a decision.** Banking this week's move to make two
   next week is a plan, and the model can see what the roll is *for*.
2. **A move can be made early.** Five moves don't fit in one gameweek's
   ceiling, so the move that raises money goes a week ahead of the moves that
   spend it. That's the double-gameweek case. A single-week solver would never
   make that move, because selling a good player for a cheap one is a
   downgrade until the week it pays for something.
3. **The captain is inside the model.** Every gameweek is priced with the
   armband on the best man in its eleven, so a squad that unlocks a big
   captaincy is worth what it is worth. The armband actually played is still
   picked afterwards, by `pick_lineup` and then by the gaffer.

The later gameweeks are printed under **The road ahead**, which always closes
with the same line:

> Advisory — re-planned every run; only this week's moves are ever made.

Nothing in a path is a commitment. It is re-solved from scratch next run, on
projections that will have moved. So never hold this week's move back because
an old path put it next week. The briefing tells the gaffer the same thing at
more length.

### The live settings

The window values three things a single-week solver can't see. They come as a
`WindowSettings` in `aigaffer/solver/optimizer.py`, and the live run uses
`DEFAULT_SETTINGS = WINDOW`:

- **The bench by order.** A backup keeper is worth 0.03 of his points. The
  outfield subs are worth 0.25, 0.08 and 0.02, first to third, because the
  first sub comes on far more often than the third. This is written as a floor
  and two bonus slots, which are exact without binaries. In a Bench Boost week
  all fifteen count in full.
- **Free transfers kept past the window.** They are credited at diminishing
  values relative to holding one: the second is worth 2.0, then 1.6, 1.3 and
  1.1 for the third to fifth.
- **A small cost per transfer.** Each costs 0.2, so a marginal move waits.
  Wildcard and free-hit weeks are exempt.

`LEGACY`, the old flat model with every bench man at a tenth, no credit and no
cost, is one line away in the same file.

**The bench-buy charge.** Even by order, the bench is worth little, which
would make a £6m backup keeper bought for one Bench Boost week look nearly
free. So a signing above his position's enabler price costs 0.5 for each
window week he sits on the bench. The enabler price is £4.5m for a keeper or
defender and £5.0m for a midfielder or forward. Bench Boost and free-hit weeks
charge nothing, and a player already owned when the window opens is never
charged; his bench is sunk.

### Two engines, and the sweep

The window is solved once per opening transfer count: zero to three, or up to
the free transfers banked (at most five). When a wildcard could be played this
week, there's one more solve with the count left free. **Each sweep solve has
a 20-second limit.** A solve that hits the limit with a squad in hand still
returns it. Each one logs a line to stderr with its time and whether CBC proved
it best, for example `solver: window 1: 3.2s, proven`. That log is how you
tell whether 20 seconds is enough on your machine.

When the window has nothing to say (an infeasible board, or every solve came
back empty), the same questions go to the single-week solver. A gameweek with
a deadline needs a recommendation more than it needs the better model. You can
tell which engine answered because a window plan carries a path. A run that
asked for the window and got the other engine says so once, under the
candidate plans. A run that asked for the single-week solver on purpose says
nothing.

### Chips

A chip is a gameweek with different rules. A Bench Boost scores all fifteen.
A Triple Captain adds an armband multiple. A Wildcard makes that week's
transfers free and uncapped. A Free Hit fields a one-week squad that reverts.
Each is a per-gameweek binary the window may switch on: one chip a gameweek,
and each chip once. The free hit is priced by a small solve of its own (the
best legal one-week fifteen within the budget), and a binary picks its week.
That keeps the revert *structural*, because the free-hit squad is never a
variable of the window and can't leak into next week's team. A chip played
this week leads the report (**PLAY Free Hit**). Later chip weeks show under
**The road ahead**.

**Why chip solves now prove.** Each chip carries one aggregate row: four
boosted bench men in a boosted week, and one tripled armband in a tripled week.
Every integer solution satisfies these anyway. Without them, the LP relaxation
could spread a fractional chip over every week and sit tens of points above any
real plan, so CBC couldn't close the gap within its limit.

**Two sets, and the first expires.** FPL gives each chip twice a season: one
set for GW1–19 (the wildcard and free hit from GW2) and another for GW20–38. A
first-set chip not played by the GW19 deadline is lost. The bot reads the
rules from the bootstrap's `chips` list (`aigaffer/chips.py`) and holds each
chip *per half*, so two of a kind can be held at once. A chip the rules don't
list isn't planned. If the rules leave out one of the four, the run says so on
stdout (`chip rules: bench_boost not offered (unknown chips: …)`). If FPL
renamed a chip, the fix is one line in `CHIP_API_NAMES`. `AIGAFFER_CHIPS=off`
and a draft hold no chips at all, so the solver is the pre-chip model, byte for
byte.

### The chip calendar

A chip is free to play, so without a brake the model would burn every chip in
the best week of the window, however ordinary. The brake is the chip calendar
(`aigaffer/solver/calendar.py`). It is built once per run and gives each held
chip a **bar for every week of the window**: the cost of not saving it. A chip
is played only where its gain clears that week's bar.

- **Values beyond the window.** Each week from the window's end to the chip's
  expiry is priced by a small solve:
  - Triple Captain: the projected points of the best attacking candidate who
    is available and expected to play at least 80 minutes (`TC_MIN_MINUTES`).
  - Bench Boost: the best fifteen bought for the week, lined up, and its bench
    scored in full less what the live settings already credit it by order.
  - Free Hit: the best one-week squad less the current squad's best one-week
    score.

  These use fixture-only projections on *base* minutes, so the gaffer's
  one-week overrides never reach them. Each is cut by a proxy haircut
  (`PROXY_SCALE`: Bench Boost 0.7, Triple Captain 0.85, Free Hit 0.85),
  because a proxy prices a squad bought for the purpose.
- **Saved-for weeks.** Within each half, the Triple Captain, Bench Boost and
  Free Hit are each given a different week beyond the window, or none. The
  assignment maximises value discounted by `CHIP_DISCOUNT` ρ = 0.97 a week.
  With fewer weeks left than chips, the lowest-valued go unassigned and are
  played inside the window.
- **Per-week bars.** A chip saved for week w\* has bar v·ρ^(w\*−w) in window
  week w: play it now only if now isn't clearly worse. A chip with no
  saved-for week has bar 0. The free hit adds an **option floor** of 10 points
  while any later week remains, as insurance against a blank or mass absences.
  The floor tapers over the chip's last five weeks (`FLOOR_TAPER_WEEKS`) to 0
  at expiry. The wildcard isn't valued week by week. Its bar is 35
  (`WILDCARD_BAR`), scaled down once fewer than eight of its eligible weeks
  (`WILDCARD_RAMP_WEEKS`) lie beyond the window.
- **Expiry.** The bars reach 0 as GW19 nears, so every unplayed chip becomes
  free points and the window plays it, stacked early if need be.
- **If the calendar fails**, the run doesn't. Stdout gets one line, and the old
  flat bars apply (`FALLBACK_BARS`: Bench Boost 20, Triple Captain 18, Free Hit
  25, Wildcard 35). A chip with no week left beyond the window still gets bar
  0. The briefing and digest say "calendar unavailable".

There's no floor on the opening weeks. On a flat board, with no doubles or
blanks announced, the Triple Captain and Bench Boost can go as early as the
next gameweek. The free hit is held by its option floor and the wildcard by its
bar. Doubles and blanks announced later show up in the weekly re-run, and the
calendar is only as good as the fixtures it can see.

### The replay

`python -m aigaffer replay` checks a change to the window settings against
real managers. It samples squads from the overall league: a top group (ranks
to about 10k) and a deep group (about 100k to 1M). Each squad starts as it
stood before the first gameweek of a finished range. Each policy (`LEGACY`,
`WINDOW` and the alternative `ALT`) then plays that range as a season of its
own: its own transfers, bank and free transfers, one solve a gameweek, and
each week scored on what the players actually did, automatic substitutions
included.

The gate compares each candidate with `LEGACY`, squad by squad. A cell is one
group at one end credit, where leftover free transfers are worth 0, 1.5 or 2
apiece. It holds the mean paired season difference and a standard error
clustered by gameweek. The verdict:

- **no**, if any cell falls more than one standard error below zero;
- **ship**, if every cell is at or above zero;
- **the owner's call** otherwise.

The gate asks "not worse", not "better". A handful of gameweeks can't separate
a point or two a week. The rows name real managers, so they go to the
git-ignored `state/replay/` and never to stdout.

The first replay ran GW2–6 over 151 squads. It found `WINDOW` level with
`LEGACY` within its noise: ahead in the top group, a fraction behind deeper
down. The gate left the call to the owner, who turned `WINDOW` on, because it
fixes what `LEGACY` can't see: a backup keeper bought to sit, and a transfer
spent for nothing. The replay's leaks are in
[limitations.md](limitations.md#the-backtest-and-the-replay-are-sanity-checks).

## The gaffer

The solver says what is optimal under the model. The gaffer
(`aigaffer/manager/`) adds what it can't know: what has happened since the
numbers were computed. After the solve, the week goes to a Claude agent that
can do four things and nothing else.

1. **Read the briefing** (`aigaffer/manager/briefing.py`). It's all he knows
   going in:
   - the gameweek, its deadline and today's date, so stale news looks stale;
   - the bank, free transfers and squad value;
   - the fifteen, with each player's id, price, status, assumed minutes and
     projections;
   - the solver's eleven;
   - the candidate plans, numbered by the ids he will finalize by;
   - the Chip EV panel and the chip calendar;
   - a ten-player watchlist and a price watch;
   - the research list: every player any plan holds, buys or sells;
   - **What we already know**: the news ledger (`state/news/ledger.json`),
     with what earlier runs learned about each player and whether it is still
     fresh or needs a re-check;
   - the search budget;
   - last week's lessons from the review.

   Before five gameweeks have finished, it also carries an early-season
   caution.
2. **Search the web** for injuries, suspensions, rotation, and press
   conferences. The tool allows eight searches per request (one assistant
   turn). On top of that, the briefing states a run budget of 15 searches.
   It's soft: a run that goes over says so on its log line, and the run is
   never stopped.
3. **Overrule the minutes and re-solve.** `adjust_players` sets a player's
   expected minutes for the coming gameweek (0 if out, 20 for a sub, 90 for a
   starter). `resolve` re-projects and re-solves with those in force and
   returns fresh plans under fresh ids. The change covers the coming gameweek
   only, so "out this week" doesn't sell a player for the month. A long-term
   absence is already zeroed across the window by the API's injury flag. The
   cost is an absence the gaffer has read about but the API hasn't flagged,
   which reads as "back after this week".
4. **Finalize** one plan, a captain and vice from its eleven, a chip or none,
   and the rationale.

The guardrails are in code, not in the prompt:

- A plan is an id from a registry the loop mints, so no text in the
  conversation can pass for a plan that was never solved.
- The armbands are checked against the chosen plan's eleven.
- A chip needs a real justification about that chip. All four are his to
  play, because a plan that carries a chip was built for it. A chip not held
  this gameweek (spent in this half, or a second-half chip before GW20) is
  refused whatever he argues: the pipeline checks it against the chip history
  before recording anything.
- He only ever picks from plans the solver produced, so budget, club quotas
  and the hit cap still hold.
- The conversation is capped at twelve assistant turns and twelve minutes,
  whichever comes first.

His system prompt carries a **chip playbook** (`aigaffer/manager/playbook.py`).
It says what makes a good week for each chip, and that in a chip's last two
eligible gameweeks it is free points. There he finalizes the solver's chip
unless team news makes the week a dud, and says why if he declines it.

The report prints what he decided as **The Gaffer's view**: his rationale, the
chip and its justification, the minutes he changed and the number of searches.
After the phone message has gone, he also writes short reasons into the
detailed report: for each transfer, the captain, the chip, the plan over the
others, and each player passed over. If that fails,
those slots say "reason unavailable" and nothing else is lost.

## The fallback doctrine

A deadline is never silently missed, and a gaffer who can't be reached is no
reason to miss one. **Every** failure degrades to the solver's own
recommendation, and that always reaches the phone before the window closes.
What changes is *when*. Failures come in two shapes.

**Asked, and it went wrong:** a rate limit, an auth error, a refusal, a dropped
connection, an unfinished turn, twelve turns with no decision, a re-solve that
finds nothing, a chip played twice, or a briefing that couldn't be built. The
Gaffer's view still prints and says *"The gaffer was unavailable (…); this is
the solver's pick."* The reason is a class name or a phrase of ours, never an
exception's own words, which can carry a key or a token.

**Never asked at all:** no key, the kill switch, a first-squad draft, or a
dependency that won't import. Then the report is exactly the solver's. The
first three are choices and say nothing. The fourth is an accident, so the
report carries one line: *"Note: the manager is configured but was unavailable
this run; this is the solver's pick."* Without it, a fork whose `anthropic`
install broke would read a season of solver-only reports and never be told.

**Withheld, and asked again.** Either way, if the report's window has more
than three hours left, nothing is written: no diary file, no run row, no
digest. The tick prints *"the report was withheld for the next tick"*, and the
phone gets one short alert with the reason, when the retries stop, and the
solver's checklist so there is always a plan. The next tick, from
either scheduler, runs the whole thing again, and the alert isn't repeated
unless the failure changes. The waiting ends three hours before the window
closes, or after twelve withheld ticks (`MANAGER_RETRY_LIMIT`), whichever comes
first. Then the labelled solver report goes out and counts, so the reminder
still has a verdict to check the news against.

A dry run withholds nothing and alerts nobody. A forced run (`--force`, or the
workflow's button) takes whatever answer it got. The early scout is the one
exception: it is held until the scout's window opens, and past the retry limit
it is written down but not sent, because the scout is hours away.

**The reminder** never wakes the gaffer. It re-solves and diffs the fresh solve
against the solver's own pre-gaffer plan, so a week where the gaffer overrode
the solver stays calm unless the news actually moved the solver's answer.
Calm is one block: the operative plan. Moved news leads with a ⚠️ section
naming each change (sells and buys on separate lines, the way the app takes
them) and shows both plans, the gaffer's verdict first. The reminder goes to
Telegram and `state/reports/` only, never to `GW{n}.md`. Unlike the full
report it sends before it saves, so a failed send is retried next tick rather
than marked done.

The rule underneath all of this: with `AIGAFFER_MANAGER=0` the pipeline's
output is byte-identical to the solver-only pipeline, and a test holds it to
that.
