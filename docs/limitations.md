# Honest limitations

What the bot approximates on purpose, what its look-backs can and cannot
tell you, and what isn't built yet.

[Back to the README](../README.md)

## Known approximations

These are deliberate. Don't "fix" them without revisiting the design.

1. **Selling prices are reconstructed, not read.** The public API doesn't
   expose your purchase prices, so the purchase ledger rebuilds them by
   observation, within a known error bar. The seed (today's price less its
   movement since the season opened) is exact only for a squad held since GW1.
   A player bought later is first seen at his price on the run after the next
   deadline reveals him. That can trail what you actually paid by however many
   £0.1m daily moves happened in between. The bank-reconciliation line in the
   report detects that drift. If it appears after a week of manual transfers,
   that's the system working, not a bug. The game's published bank stays
   authoritative everywhere. A signing recorded with "Transfers made" is
   ledgered at the price when you texted.
2. **Your own transfers are invisible until the deadline.** The public picks
   endpoint lags to the last deadline. Texting "Transfers made" closes the gap
   for a week you entered exactly as recommended. A week you deviated from
   stays invisible until the deadline.
3. **A new face has no minutes to read on the opening weekend.** Expected
   minutes are a mean over matches actually played. The API adds a 0-minute
   history row the moment a deadline passes, for matches not yet kicked off,
   and those rows are dropped on the way in. So GW1 has no history at all.

   For a player who was in the division last season, the payload answers
   this. His last season's minutes, divided by 38, floor his first three
   gameweeks: 3,230 minutes is 85 a week, not the substitute's 20. A brand-new
   signing, a promoted club's player or a teenager has no Premier League season
   to read. He falls back on `starts`, which is zero for everybody in GW1, so
   he is projected at a substitute's 20 minutes. The gaffer's minute overrides
   are the mitigation, for the coming gameweek only. His eleven and armband
   are corrected, but the window still undervalues him further out until his
   own history takes over.

   Three edges of the prior are worth knowing:
   - A January signing's half-season of minutes is spread over a whole one, so
     he is understated. The payload has no games-available column to do
     better.
   - A player whose role has changed is described by last season until this
     one reaches three gameweeks.
   - A prior can land *below* the guess it replaces. A returner from an
     injury-hit season floors at, say, his own 11 minutes a week, not the 75
     that "he has started once" would give him.

   That understatement isn't free. For a player you don't own, it costs a
   transfer you would have made. For one in your fifteen, it can leave him out
   of the eleven or put him on the sell side, which is the case to watch in
   the first three gameweeks. The gaffer's overrides correct the eleven. A sale
   the window recommends on his later weeks is the gaffer's to argue down.
4. **Prices don't move inside the window.** The six-week plan holds every
   price at today's. The squad you start with sells at the ledger's figures,
   and a player bought inside the window sells at what he cost.

## The backtest and the replay are sanity checks

### The backtest

`backtest` grades the model against a gameweek already played. It rebuilds
expected minutes from each player's history *before* that gameweek, projects
that one gameweek, and compares the projection with what players actually
scored. It needs no team id and writes nothing.

```sh
python -m aigaffer backtest            # the most recent finished gameweek
python -m aigaffer backtest --gw 3     # one by name, printing for example:
{'gw': 3, 'n': 143, 'spearman': 0.41, 'top20_hit_rate': 0.35}
```

- `n` is the population: the players on the pipeline's own shortlist (the
  squad and the top 40 a position, so about 160) who had a fixture that
  gameweek. It's one API request each, so the run takes a minute.
- `spearman` is the rank correlation between projected and actual points over
  that population.
- `top20_hit_rate` is the share of the model's top twenty who scored five or
  more.

**This isn't a true backtest.** Three things here know how the gameweek turned
out, and all three flatter the model:

1. The season totals behind the rates, and the prices, come from today's
   bootstrap. That includes the graded gameweek and every one since, so the
   model is asked to rank a week it has already seen. The fitted team
   strengths are the exception: they are rebuilt from results before the
   graded round and don't leak.
2. The shortlist admits only players available *today*, so anyone since
   injured or gone is never graded. The population is the survivors.
3. The shortlist then cuts by today's season total, which includes everything
   scored *after* the graded gameweek. The field leans toward players who went
   on to do well.

Only the minutes model is held honest. So `spearman` is optimistic. Read it as
a sanity check on ranking (is the order better than chance, and is it holding
up week to week?), never as an estimate of how the bot would have done.

### The replay

`replay` plays real managers' squads through each way of valuing the bench and
free transfers (see [how-it-decides.md](how-it-decides.md#the-replay)). Its
gate is a sanity check too, with its own stated leaks:

- **Rates and prices.** These come from today's bootstrap, except where a
  player's history gives that gameweek's own value.
- **The shortlist.** It is today's players, each offered only once his history
  shows him registered, and every one taken as available.
- **The minutes fallback.** It counts today's `starts`, and the replay hands
  in no last-season prior.
- **Chips are off.** The replay measures the bench and free-transfer terms and
  nothing of chip timing.
- **Selling.** It sells at that gameweek's value, without the half-the-rise
  rule.
- **The fixture list is today's,** including matches since moved.

Two more things limit what it can say:

- After week one, each policy holds a squad no real manager held.
- A haul or a blank lands on every policy at once, so the error is clustered
  by gameweek, and a handful of gameweeks makes it rough.

## Not yet

Still not built:

1. **Re-running a past week.** Each run now keeps a snapshot in
   `state/detailed/`: the projected players' fixtures, xP and minutes (after
   the gaffer's overrides), every plan he could have chosen and the decision.
   So his week can be read in full. It holds no bootstrap or histories,
   though, and only the projected set, so a past recommendation still can't be
   recomputed once the API has moved on. The reminder isn't snapshotted.
2. **A full-season backtest against the average manager.** The backtest ranks
   one gameweek. The replay compares policies with each other over a range,
   not with the average manager over a season, which is the question that
   matters.
3. **A longer replay.** The first replay covered GW2–6, too few gameweeks to
   separate a point or two a week. It should be re-run once more gameweeks
   have finished.
4. **Chips in the replay.** Every policy replays chip-free, so nothing yet
   measures chip timing against real results.
5. **Chip economics against the plan actually chosen.** The Chip EV panel is
   priced once, against the plan that rolls the transfer, before the gaffer is
   asked. If he picks another plan and plays a chip on it, the numbers he
   argued from describe a squad he didn't enter. The window itself plans chips
   inside every plan; it's only the advisory panel that is priced against the
   roll.
6. **Season-long chip timing beyond the fixtures.** Beyond the six-week window
   the chip calendar sees only fixture-only projections and proxies: a
   haircut bench, a bought-in captain, a free hit against today's squad. It
   doesn't see the squad the bot will actually own, or doubles and blanks FPL
   hasn't announced. It picks the week that looks best *today* and re-plans
   every run. Building a bench ahead of a boost, and ownership-aware timing,
   are out of scope.
7. **The constants are first guesses.** The chip calendar's inputs are set by
   hand to FPL norms, not measured against results:
   - the discount ρ (0.97);
   - the proxy scales;
   - `TC_MIN_MINUTES` (80);
   - the free hit's option floor (10, tapering over five weeks);
   - the wildcard's bar and ramp (35 over eight weeks).

   Each lives in one place in `aigaffer/solver/calendar.py`. The `WINDOW`
   bench weights and free-transfer values are an FPL expert's set, checked only
   by the first replay.
8. **A bench order from the gaffer.** The solver now values the bench by order,
   so it buys knowing the first sub matters more than the third. But the order
   the report prints is still `pick_lineup`'s: substitute keeper first, then by
   next gameweek's projection. The gaffer isn't asked for one. It's a good
   default, but it isn't anyone's judgement about which of two fringe players
   is likelier to get a game.
9. **"What if I…" questions that re-solve the plan.** Asking the gaffer
   explains the plan from the week's report and never changes it. A question
   like "what if I keep Saka?" can't yet put a move to the solver and come
   back with a new plan. The chip what-ifs are the only re-solving questions.
10. **A faster solver.** Each sweep solve has 20 seconds on CBC, and the log
    says whether it was proven. If the box's solve log shows unproven solves
    at 20 seconds, the next step is a faster solver such as HiGHS.
11. **Planning around price rises and the half-profit rule.** The window holds
    prices still, so it can't buy ahead of a rise, sell ahead of a fall, or
    weigh that a seller keeps only half of a rise.
12. **A rise-tonight alert between reports.** The briefing carries a price
    watch (the market's movers among the relevant players, framed as timing
    for a transfer already decided). But nothing pings between reports when a
    planned buy is surging, so a Tuesday-night rise ahead of a Friday deadline
    is only reported after it happened. FPL's thresholds are undocumented, so
    the watch reports volumes rather than pretending to predict them.
