# Transfer card: why this player, in numbers

Date: 2026-10-30 · Status: principal approved with changes (applied); awaiting Mark

## Why

On GW9 the bot said "Calafiori out, Gabriel in", and Mark's first reaction was
that Gabriel's fixtures looked tough (liv, HUL, new, MCI, bre, tot). What he
does with a transfer suggestion is open the FPL app and check **both players'
fixtures**. What settled it for him was not the alternatives but the
**projected points**: Gabriel's projection holds up even against those
fixtures, and he is still the best defender we can afford.

Today none of that reaches the phone. The digest and the full report give the
move as two bare lines under `## Recommendation`:

```
- Out: Calafiori (DEF, ARS, £5.9m)
- In: Gabriel (DEF, ARS, £7.9m)
```

The Week plan doesn't describe the incoming player either. The only "why" is
the gaffer's paragraph, which is mostly about rolling versus moving and names
risks ("a tough away game at Liverpool") without any numbers.

## What success looks like

Mark reads one block per transfer and doesn't need to open the app to check
fixtures. Each of the window's fixtures, for the player leaving and the player
arriving, sits next to its projected points, so a tough-looking run comes with
the number that answers it.

## The card (Mark approved this shape)

One block for each of **this week's** transfers:

```
Calafiori → Gabriel · DEF · £7.9m (bank £1.5m left)
Out  liv 0.8 · HUL 3.6 · new 3.3 · MCI 3.0 · bre 3.2 · tot 3.5 = 17.3
In   liv 4.8 · HUL 5.8 · new 5.3 · MCI 5.0 · bre 5.2 · tot 5.6 = 31.6
+4.0 this week, +14.3 over 6 weeks
Next best: Tarkowski 28.9 (£6.2m) · Konsa 28.6 (£4.7m)
```

- **Fixtures** use the Week plan strip's bracket-free format (`_strip`,
  `aigaffer/weekplan.py`): home UPPER, away lower, `—` for a blank, `+`
  joining a double. *Not* `games_text`, which prints `LEE(h)`. The venue rule
  already exists in three copies (`games_text`, `_strip`, detailed `_cell`).
  The builder extracts one bracket-free phone helper that both `_strip` and
  the card call, rather than adding a fourth copy. A double gameweek shows
  **one** number, the week's total.
- **Numbers** are the run's per-gameweek projections (the snapshot's `x`,
  stored to 2dp), printed to one decimal place, across the solver's window.
  That's six weeks today, but it follows the window and is never hard-coded.
- **Totals and gains are plain (undecayed) sums.** *My call, which the
  principal agreed with:* the mock-up I showed Mark used the decayed window
  total (21.8 / 11.3), which the row's numbers don't sum to. A total is the
  sum of the stored 2dp values, so the printed 1dp cells can differ from it
  by a few tenths. Tests assert on the stored sums, never on adding up the
  printed cells.
- **"This week"** is GW-now in for GW-now out. **"Over N weeks"** is the
  difference of the two sums.
- **Next best**: the two best others at the same position, ranked by the same
  plain sum. Eligibility:
  - Not in the squad after this week's moves, and not one of this week's
    **outgoing** players either. On a downgrade-to-fund week, the player
    being sold would otherwise top the list.
  - Affordable in the incoming player's place: `price(alt) <= pr(in) +
    bank_left`, where `bank_left = bank + Σ sell(out) − Σ pr(in)` over all
    this week's moves. `sell` is the snapshot's `sp`, falling back to `pr`
    when there is no ledger price. `bank` is the snapshot's `bank`, which is
    the bank *before* moves.
  - Doesn't break the three-per-club rule against the rest of the new squad.
  - Taken from the run's projected set (`set(inputs.histories) |
    candidate_pool(costed) | squad`, the snapshot's `players`), because
    anyone else has no projection. Ruled-out players already project near
    zero (`aigaffer/model/minutes.py`), so no availability rule is needed.
    A test pins it: an injured defender with a stale high projection must
    not appear.

  If an alternative projects **higher** than the pick, it is still shown in
  rank order, with no extra wording. That will happen more often than "the
  solver picks squads" suggests: on plain sums, decay alone can invert the
  order between a front-loaded run and a back-loaded one. Hiding a higher
  number would be dishonest, and Mark can ask the gaffer why.
- **Pairing** (*my call*): the plan keeps `out` and `in` as two sorted id
  lists, not pairs. Cards pair them like for like by position, as the
  detailed report's `_pairs` does (`aigaffer/detailed/render.py`). Within a
  position they pair in ascending id order, not strongest first. Inside one
  position the pairing is presentational, and id order needs no projections,
  so the checklist, which has none, can share the rule. `_pairs` moves to it
  too. The "Do this" checklist
  (`_moves` in `aigaffer/report/render.py`) and `entered_moves` currently
  zip the two sorted lists, which can print a defender out against a
  midfielder in on a two-position week. They move to the same pairing in this
  work, so the card and the SELL → BUY line above it always agree. The
  pairing helper moves somewhere both renderers can import.
- **Bank** (*my call*): "(bank £X left)" is the bank after **all** of this
  week's moves, the same figure on every card. That is the number the owner
  sees once he has made them, and the figure "Next best" affordability uses.
- **Price** on the header line is the buy price. The sell price already
  shows in "Do this" (`sells £5.7m`) and is not repeated.

### Club line (my call; principal agreed; Mark can veto)

When the move takes a club to the three-player limit, add one line:
`3 from Arsenal now (Gabriel, Rice, Ødegaard)` (an illustration; GW9's
actual move leaves two from Arsenal, so it would not fire). It's cheap and states a fact, and
it covers the double-up risk raised in conversation. Only the current week counts. Moves planned for later
weeks (Konsa in GW10) are not counted.

## Where it goes

| Place | Placement |
|---|---|
| Phone digest (scout and deadline) | Replaces the bare `- Out:` / `- In:` bullets inside `## Recommendation`. The "1 transfer, no hit." line stays. |
| Full report (diary / GitHub) | The same, since both renderers call `_recommendation`. |
| Week plan | Opens text 3 ("Why this plan"), above the gaffer's reasoning. |
| Detailed report | **No change** (*my call*). Its `## Transfers` tables (`_transfers`, `aigaffer/detailed/render.py`) already give each pair every window fixture with xP to two decimals, plain and decayed totals, and a reason. A card beside them would repeat the same numbers at a different precision. |
| 3-hour reminder | **No change.** It is meant to be short. |

## Scope

**In:**
- This week's transfers in the recommended plan only.
- One card per transfer, paired and ordered as under **Pairing** above.
- The full report's **watchlist** switches to the plain window sum (both how
  it ranks and what it prints). Otherwise the same document would give
  Tarkowski one six-week number on the card and another a few sections down.
  The weighted figure then appears only in the detailed report, under its
  "Weighted" caption.
- The Week plan's "Passed over" gap stays decayed: its threshold is defined
  on that basis. Its label gains "weighted" so it can't be read against the
  card's plain "+14.3".

**Out (Mark's call):**
- The road ahead (GW10+ moves) stays as one-line moves, because it's
  re-planned every run.

**Out (my calls):**
- **Wildcard and Free Hit weeks get no cards.** Up to fifteen cards is a wall
  on a phone. The existing sections stay, plus one line pointing at the
  detailed report. On a Free Hit week, `choice` already carries no transfers
  (the FH team lives on `path.week1_freehit_xi`), so "Roll the transfer."
  already prints. The builder checks what the pointer line attaches to
  there. Wildcard is the real case.
- **Draft weeks get no cards.**
- No new numbers are computed. The card only arranges projections the run
  already made.
- No LLM involvement. The card is deterministic, so it can't get a fixture
  wrong. That's the GW9 lesson.

## Anti-goals

- A card that disagrees with the run's own numbers. It must read the same
  projections the solver used, after the gaffer's minute changes, i.e. the
  re-solved projections the recommended plan was chosen on.
- A digest that fails to send because the card failed. The card is rendered
  in the same pure, no-I/O way as the rest of the digest, and missing data
  (a player with no projection) is a bug to raise in tests, not something to
  paper over.

## Constraints

- Telegram: `MAX_CHARS` 4000, HTML parse mode, whole-line chunking. A card
  is about 5 lines and about 300 characters, so three transfers add about
  1k. The digest already chunks.
- Readers stay pure readers. The Week plan renders only from the snapshot.
  The digest and full report render from the live run. One renderer
  function serves both, fed **one row dict built once by one builder**:
  - Lift the per-player row construction out of `build_snapshot`
    (`aigaffer/detailed/snapshot.py`, the `players[pid] = {...}` loop) into
    a public row builder.
  - Call it once in the orchestrator's report path, after `costed` is set.
  - Pass the same dict to `render_report`, `render_digest` and
    `build_snapshot`, which takes it as an argument in place of its inline
    loop.
  - That way the 2dp rounding happens once, and the digest and Week plan
    cards can't come out 0.1 apart.
  - The squad-without-projection `ValueError` stays inside `build_snapshot`.
    The row builder is total, so it can't kill the digest.
- **Projections are `costed`** (post-gaffer, the set the recommended plan was
  chosen on), not `projections`. Both are in scope where the digest is
  rendered, and `projections` is used deliberately nearby for the
  reminder's yardstick, so naming the wrong one is an easy slip.
- The digest must **not** depend on the snapshot write succeeding: that
  write is fail-soft and the digest is not. The principal confirmed the
  `save` and `send` branches are already independent.
- `_recommendation` gains the week's chip and the bank. `played_chip` is
  already computed in both callers. That means signature churn across the
  `render_report` / `render_digest` call sites and their tests.

## Left to the builder

- The module name and home for the renderer, the row builder and the shared
  pairing helper.
- Exact whitespace and alignment within the approved shape.
- Test layout. Expected cases: a double, a blank, an affordability boundary,
  the three-per-club exclusion, a higher-projected alternative, a two-move
  week, and the FH/WC/draft no-card cases.
