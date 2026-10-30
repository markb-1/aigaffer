# Transfer Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each of this week's transfers gets a card: both players' fixtures, each with its projected points, the totals and gains, and the next best alternatives. It appears in the phone digest, the full report and Week plan text 3.

**Architecture:**
- A new pure module, `aigaffer/report/transfer_card.py`, renders cards from snapshot-shaped player rows. It also owns the one out→in pairing rule.
- The rows are built once by a `player_rows` function lifted out of `build_snapshot`. The orchestrator calls it right after `costed` is set and hands the same dict to `render_report`, `render_digest` and `build_snapshot`.
- The Week plan renders cards from the snapshot's own `players`.

**Tech Stack:** Python 3.12, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-30-transfer-card-design.md`

## Global Constraints

- **Totals are plain sums of the stored 2dp `x` values.** Cells print at 1dp. Totals and gains print at 1dp, signed (`+4.0`, `-1.2`). They are never `PlayerProjection.total`, which is decayed.
- **Projections are `costed`** (post-gaffer), never `projections`, in the orchestrator.
- **Cells use plain `f"{x:.1f}"`.** No Decimal and no half-up rounding, so the card agrees with every other 1dp number in the house. Test fixtures avoid exact `.x5` values (`f"{5.85:.1f}"` is `"5.8"`).
- **Fixtures are bracket-free:** home UPPER, away lower, `+` joins a double, `—` is a blank. A double shows one number, the week's total. A blank shows `—` alone.
- **Pairing:** like for like by position. Within a position, pair in ascending id order. Any leftovers are zipped in id order, so nothing is dropped when the two lists are the same length. With unequal lengths, zip drops the surplus, which keeps the detailed report's existing note and its test. The same function serves the card, the "Do this" checklist, `entered_moves` and the detailed report.
  - **Deviation from spec, to record:** the spec said "strongest first within a position". Id order is used instead, because within one position the pairing is presentational and id order needs no projections. That lets the checklist, which has none, use the same rule.
- **No cards on Wildcard, Free Hit or draft weeks.** A Wildcard week prints one pointer line instead.
- Code style: match the surrounding modules. Docstrings explain why, in the house voice. No new comments that restate code.
- Run tests with `../../.venv/bin/python -m pytest -q` from the worktree, without xdist.

## Review Focus

1. **Downgrade-to-fund week** (sell an expensive player, buy a cheaper one): the outgoing player must not appear in Next best, and the bank arithmetic must use `sp`, falling back to `pr` when `sp` is None. Tests are in Task 4.
2. **Two moves in two positions** (DEF→DEF plus MID→MID, with ids interleaved): the checklist and the cards must pair DEF with DEF. Tests are in Tasks 2 and 5.
3. **A player in the projected set with status injured and a high stale `t` but near-zero `x`** must not lead Next best. Ranking is on plain `x` sums, which handles it. Test in Task 4.
4. **The card reads only the row keys every snapshot already has** (`n p c cf pr sp f x`). Test in Task 6.
5. **The digest still sends when `cards` is None** (draft, no squad, or any caller that doesn't pass it): it prints the old `- Out:` / `- In:` bullets. Test in Task 5.

---

### Task 1: One bracket-free phone fixture helper

**Files:**
- Modify: `aigaffer/data/schedule.py` (add next to `games_text`)
- Modify: `aigaffer/weekplan.py` (`_strip`)
- Test: `tests/test_schedule.py`, `tests/test_weekplan.py` (existing strip tests must still pass unchanged)

**Interfaces:**
- Produces: `phone_games(games: Sequence[Sequence[str]]) -> str`. It takes `[(opp, "h"|"a"), ...]` (tuples or the JSON lists a snapshot holds) and returns `"BHA"`, `"lee"`, `"BHA+ars"`, or `"—"` for an empty list.

- [ ] **Step 1: Write failing tests**
  - `test_phone_games_home_upper_away_lower`: `phone_games([("BHA","h")]) == "BHA"` and `phone_games([["LEE","a"]]) == "lee"`.
  - `test_phone_games_double_and_blank`: `phone_games([("BHA","h"),("ARS","a")]) == "BHA+ars"` and `phone_games([]) == "—"`.
- [ ] **Step 2:** Run `../../.venv/bin/python -m pytest tests/test_schedule.py -q`. Expect FAIL (import error).
- [ ] **Step 3:** Implement `phone_games`. Make `_strip` call it instead of its inline join.
- [ ] **Step 4:** Run `../../.venv/bin/python -m pytest tests/test_schedule.py tests/test_weekplan.py -q`. Expect all to pass.
- [ ] **Step 5: Commit** with `refactor(schedule): one bracket-free phone fixture helper`.

### Task 2: One pairing rule everywhere

**Files:**
- Create: `aigaffer/report/transfer_card.py` (pairing only for now; the module docstring introduces the card)
- Modify: `aigaffer/report/render.py` (`_moves`, `entered_moves`)
- Modify: `aigaffer/detailed/render.py` (`_pairs` delegates to the shared rule)
- Modify: `aigaffer/manager/detailed_writer.py` (its import of `_pairs`)
- Test: `tests/test_transfer_card.py` (new), `tests/test_render.py`

**Interfaces:**
- Produces: `pair_moves(out_ids: Sequence[int], in_ids: Sequence[int], position: Callable[[int], object]) -> list[tuple[int, int]]`.
  - It groups by `position(pid)` and zips within each group in ascending id order.
  - Matched groups come first, in order of their lowest out id, then the leftovers.
  - Leftovers on either side are zipped together afterwards, in id order.
  - Every id appears exactly once whenever the two lists are the same length.
- `transfer_card.py` must not import `aigaffer.report.render`, because `render` will import it.

- [ ] **Step 1: Write failing tests** in `tests/test_transfer_card.py`:
  - `test_pairs_like_for_like_across_interleaved_ids`: out `[10, 30]` with positions `{10:"DEF", 30:"MID"}`, and in `[20, 25]` with positions `{20:"MID", 25:"DEF"}`, gives `[(10, 25), (30, 20)]`.
  - `test_pairs_within_a_position_by_id`: out `[5, 3]` and in `[9, 7]`, all DEF, gives `[(3, 7), (5, 9)]`.
  - `test_unmatched_positions_are_still_paired_not_dropped`: out `[1]` DEF and in `[2]` MID gives `[(1, 2)]`.
  - In `tests/test_render.py`, `test_checklist_pairs_sales_and_buys_by_position`: a plan where `transfers_out=[a_def, b_mid]` and `transfers_in=[c_mid, d_def]`, with ids chosen so a plain zip would cross positions. The "Do this" lines must read `SELL <def> → BUY <def>` and `SELL <mid> → BUY <mid>`.
- [ ] **Step 2:** Run the new tests. Expect FAIL.
- [ ] **Step 3:** Implement `pair_moves`. The position callable must tolerate a missing player, as `_who` does: `players[pid].element_type if pid in players else None`.
  - `_moves` uses `position=lambda pid: players[pid].element_type`, and `entered_moves` does the same.
  - `detailed.render._pairs(snapshot)` becomes `pair_moves(plan["out"], plan["in"], lambda pid: players[pid]["p"] if pid in players else "?")`. Keep the name `_pairs` so the detailed writer's import still works.
  - Update `_pairs`' and `entered_moves`' docstrings to match.
- [ ] **Step 4:** Run `../../.venv/bin/python -m pytest -q`. Expect everything to pass. Tests that pinned the old t-ordered pairing within one position may need their expected order updated; say so in the commit body.
- [ ] **Step 5: Commit** with `fix(render): the checklist pairs sales and buys like for like`.

### Task 3: Rows built once

**Files:**
- Modify: `aigaffer/detailed/snapshot.py`
- Modify: `aigaffer/orchestrator.py` (`_run_snapshot` and the report path after `costed`, about line 600)
- Test: `tests/test_detailed_snapshot.py`, `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `player_rows(*, bootstrap: Bootstrap, fixtures: list[Fixture], projections: dict[int, PlayerProjection], xmins: dict[int, float], ids: Iterable[int], selling_prices: dict[int, int] | None) -> dict[int, dict]`.
  - It returns exactly today's row (`n p c cf pr sp st nw xm f x t`, int keys, `x` at 2dp) for each id that has both a bootstrap element and a projection.
  - It is total: it never raises for a missing player.
- `build_snapshot`'s final signature drops `bootstrap`, `fixtures`, `selling_prices`, `xmins` and `projected_ids`, all of which are dead after the lift, and adds `players: dict[int, dict]`. It keeps `projections`, which it still reads for `events`.
  - The test helper `snapshot(**overrides)` in `tests/test_detailed_snapshot.py` builds its rows via `player_rows`, so callers like `snapshot(fixtures=postponed, projected_ids={1,2,4})` keep working through the helper.
  - It keeps the squad-without-projection `ValueError`: any `squad_ids` not in `players` is reported.
  - It must not mutate the dict it is given. The digest holds the same object.
- Produces, for Task 5: in `orchestrator`, a local `rows` built once after `costed`, on non-draft weeks with a squad, from
  - `ids = set(inputs.histories) | set(candidate_pool(effective.players, costed, squad_ids, limit=multiweek.CANDIDATES_PER_POSITION)) | set(squad_ids)`
  - `xmins = _costed_minutes(xmins, gaffer)`
  - `selling_prices = prices`.

  It is passed to `_run_snapshot` (new parameter `rows`), which passes it on. On draft weeks, or when there is no squad, `rows = None`.

- [ ] **Step 1: Write failing tests**
  - `test_player_rows_skips_unprojected_and_rounds`: an id with no projection is absent, and `x` values are rounded to 2dp.
  - `test_build_snapshot_names_a_squad_player_missing_from_rows`: `ValueError` naming the id.
  - `test_build_snapshot_leaves_the_rows_it_is_given_untouched`: a deepcopy taken before the call equals the dict after it.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Lift the loop into `player_rows`. Rewire `build_snapshot` and `_run_snapshot`. Build `rows` in the orchestrator as above.
  - Use `squad_ids = list(effective.squad.player_ids)` under the guard `not solved.draft_mode and effective.squad is not None`, mirroring the snapshot's own guard.
  - This moves `candidate_pool` and row building out of the fail-soft snapshot `try` and onto the path before `render_report`. That is intended (the spec accepts it), and `player_rows` is total.
  - Rewrite the comment above that `try` ("never at the week's expense") and the `_run_snapshot` docstring's "projected set" bullet so both say where the rows now come from.
- [ ] **Step 4:** Run `../../.venv/bin/python -m pytest -q`. Expect everything to pass. Existing snapshot tests must produce byte-identical snapshots.
- [ ] **Step 5: Commit** with `refactor(snapshot): player rows are built once and shared`.

### Task 4: The card renderer

**Files:**
- Modify: `aigaffer/report/transfer_card.py`
- Test: `tests/test_transfer_card.py`

**Interfaces:**
- Produces:
  - `class CardInputs(NamedTuple)` with fields:
    - `rows: Mapping[int, dict]`
    - `events: list[int]`
    - `squad: list[int]`, the squad **before** this week's moves
    - `bank: int`, in tenths, **before** moves
  - `transfer_cards(cards: CardInputs, out_ids: Sequence[int], in_ids: Sequence[int], chip: str) -> list[str]`, returning one string per card (lines joined by `\n`), in `pair_moves` order. It returns:
    - `[]` when there are no moves, or when `chip == FREE_HIT`;
    - `[WILDCARD_POINTER.format(count=len(in_ids))]` when `chip == WILDCARD`.
  - Constant `WILDCARD_POINTER = "Wildcard: {count} transfers, too many for cards here. The detailed report lists each with its fixtures and points."`.
  - `chip` values are the strings in `aigaffer.chips`. Import `FREE_HIT` and `WILDCARD` from there.
- Card layout. Spaces matter: two after `Out`, three after `In`.
  ```
  {out_n} → {in_n} · {in_p} · {£in_pr} (bank {£bank_left} left)
  Out  {cell} · {cell} … = {out_sum:.1f}
  In   {cell} · {cell} … = {in_sum:.1f}
  {this_week:+.1f} this week, {gain:+.1f} over {len(events)} weeks
  Next best: {name} {sum:.1f} ({£pr}) · {name} {sum:.1f} ({£pr})
  {3} from {in_cf} now ({names})
  ```
  - A cell is `phone_games(f[gw])`, then a space and `x[gw]` at 1dp. A blank is `—` alone.
  - Money is `f"£{tenths / 10:.1f}m"`, a local helper (the same as `render.price`, which can't be imported here).
  - `bank_left = bank + Σ (sp if sp is not None else pr)(out) − Σ pr(in)` over **all** this week's moves. It is the same on every card.
  - `this_week` is `x[events[0]]` in minus out. `gain` is in_sum minus out_sum.
  - **Next best** shows up to two, ranked by plain sum descending, ties by name. Eligible players:
    - are in `rows`, at the same `p` as the in player;
    - are not in `new_squad = squad − outs + ins`;
    - are not among `out_ids`;
    - have `pr <= pr(in) + bank_left`;
    - have fewer than 3 of their club (`c`) in `new_squad` minus this in player.

    The line is omitted when nobody is eligible.
  - **Club line:** only when `new_squad` holds exactly 3 of the in player's club. `names` are those three players' `n`, sorted, joined by `", "`.

- [ ] **Step 1: Write failing tests.** Fixture: Calafiori id 1 (DEF ARS, cf Arsenal, pr 59, sp 57, f `{9:[("LIV","a")],10:[("HUL","h")]}`, x `{9:0.81,10:3.62}`) out, Gabriel id 2 (DEF ARS, pr 79, x `{9:4.82,10:5.86}`) in. The squad holds 1 plus a Rice (MID ARS). Bank is 59 and events are `[9,10]`.
  - `test_card_matches_the_approved_shape` asserts these lines exactly:
    ```
    Calafiori → Gabriel · DEF · £7.9m (bank £3.7m left)
    Out  liv 0.8 · HUL 3.6 = 4.4
    In   liv 4.8 · HUL 5.9 = 10.7
    +4.0 this week, +6.2 over 2 weeks
    ```
  - `test_double_shows_the_weeks_number_once` gives `BHA+ars 7.1`. Nothing is summed: `x[gw]` is already the week's number. `test_blank_cell_is_a_dash` gives `—`.
  - `test_next_best_excludes_the_outgoing_player_on_a_downgrade`
  - `test_next_best_affordability_boundary`: a player priced exactly `pr(in) + bank_left` is in, and one tenth more is out.
  - `test_next_best_uses_listed_price_when_no_selling_price`: `sp` None.
  - `test_next_best_respects_three_per_club`
  - `test_higher_projected_alternative_is_still_listed_in_rank_order`
  - `test_injured_player_with_stale_total_does_not_lead`: high `t`, `x` all 0.0.
  - `test_club_line_only_at_three`
  - `test_two_moves_share_one_bank_figure`
  - `test_wildcard_gives_the_pointer_and_free_hit_gives_nothing`
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `../../.venv/bin/python -m pytest tests/test_transfer_card.py -q`. Expect everything to pass.
- [ ] **Step 5: Commit** with `feat(card): each transfer shows both players' fixtures and points`.

### Task 5: Cards in the digest and the full report; plain-sum watchlist

**Files:**
- Modify: `aigaffer/report/render.py` (`_recommendation`, `render_report`, `render_digest`, `_watchlist`)
- Modify: `aigaffer/orchestrator.py` (pass `cards` to both renderers)
- Test: `tests/test_render.py`, `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `CardInputs` and `transfer_cards` from Task 4, and `rows` from Task 3.
- `render_report(..., cards: CardInputs | None = None)` and `render_digest(..., cards: CardInputs | None = None)` are both keyword-only and default to None.
- `_recommendation(choice, players, clubs, *, cards: CardInputs | None = None, chip: str = NO_CHIP)`:
  - Keep the `"{n} transfer(s), {cost}."` line.
  - `_recommendation` owns the chip branch:
    - On a Wildcard week it keeps the `- Out:` / `- In:` bullets exactly as today and appends the pointer line.
    - Otherwise, when `cards` is not None and `transfer_cards(...)` returns cards, they follow the count line, separated by blank lines, in place of the bullets.
  - On a Free Hit week `choice` carries no transfers, so "Roll the transfer." prints first and there is no pointer. That is by design.
  - Otherwise print the bullets exactly as today.
  - Both renderers already hold `chip` (`played_chip`).
- The orchestrator passes `cards=CardInputs(rows, sorted(projected_events(costed)), squad_ids, effective.squad.bank)` when `rows` is not None, and None otherwise.
- `_watchlist` gets its own sort key: the plain window sum, `sum(p.per_gw.values())` where `p = projections.get(pid)`, or 0.0 when there is no projection (as `projected_points` does), ties by name. It ranks and prints by that. It still shows 1dp and the label `xP`. The team sheet's `_ranked` is unchanged.

- [ ] **Step 1: Write failing tests**
  - `test_digest_carries_the_card_under_recommendation`
  - `test_full_report_carries_the_same_card_text_as_the_digest`: the same `cards` produce the identical block in both.
  - `test_without_cards_the_bullets_are_unchanged`: `cards=None` gives today's output byte for byte.
  - `test_wildcard_keeps_the_bullets_and_adds_the_pointer`
  - `test_watchlist_ranks_and_prints_plain_window_sums`: a back-loaded player outranks a front-loaded one whose decayed total is higher.
  - In `tests/test_orchestrator.py`, `test_run_passes_costed_rows_to_the_digest`: on a run where the gaffer re-solved with different minutes, the digest card's numbers are `costed`'s, not `projections`'.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `../../.venv/bin/python -m pytest -q`. Expect everything to pass.
- [ ] **Step 5: Commit** with `feat(report): the recommendation shows each transfer's card`.

### Task 6: Cards open Week plan text 3

**Files:**
- Modify: `aigaffer/weekplan.py` (`_why`, the passed-over line, and the module docstring's text 3 description)
- Test: `tests/test_weekplan.py`

**Interfaces:**
- Consumes: `CardInputs` and `transfer_cards` from Task 4.
- `_why(cur)` puts the cards straight after the title line and before the gaffer's block:
  - `transfer_cards(CardInputs(cur["players"], cur["events"], cur["squad"], cur["bank"]), plan["out"], plan["in"], chip(cur))`, using `chip` from `aigaffer.detailed.render`, which `weekplan` already imports
  - where `plan = cur["plans"][cur["chosen"]]`.
- The passed-over gap reads `({gap:+.1f} weighted)`.
- **Bank fix:** text 1's `_summary` prints `cur["bank"]`, which is the bank *before* moves, next to "1 FT used". It reads as the bank afterwards: GW9 says £3.7m when the real figure after Calafiori → Gabriel is £1.5m. Make it print the bank after this week's moves.
  - Make the card module's `bank_left(cards: CardInputs, out_ids: Sequence[int], in_ids: Sequence[int]) -> int` public and use it here, so text 1 and the card can never disagree.
  - Wildcard and Free Hit weeks keep their wording, with the same after-moves figure.

- [ ] **Step 1: Write failing tests**
  - `test_text_three_opens_with_the_transfer_card`
  - `test_text_three_has_no_card_on_a_roll`
  - `test_passed_over_gap_says_weighted`
  - `test_summary_bank_is_after_the_weeks_moves`: GW9 shape, bank 37, sell 57, buy 79, gives `bank £1.5m`.
  - `test_card_reads_only_the_row_keys_old_snapshots_have`: rows holding only `n p c cf pr sp f x`.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `../../.venv/bin/python -m pytest -q`. Expect everything to pass. Check that text 3 still fits the existing chunking tests.
- [ ] **Step 5: Commit** with `feat(weekplan): text 3 opens with each transfer's card`.

### Task 7: Live check (senior engineer, not the implementer)

- [ ] Render the Week plan from `state/detailed/gw9-scout.json` (main's copy). Text 3 must open with the Calafiori → Gabriel card:
  - Out 17.3, In 31.6, `+4.0 this week, +14.3 over 6 weeks`.
  - GW10 shows `HUL 5.8`.
  - `(bank £1.5m left)`.
  - Next best: Tarkowski 28.9, Konsa **28.6**. The spec mock-up's 28.7 and £3.7m are wrong, so don't fix the code toward them.
  - No club line (Arsenal stays at 2).
  - Text 1 shows `bank £1.5m`.
- [ ] Dry-run the scout digest on live data (`python -m aigaffer scout --force --dry-run`). The card must be under Recommendation, with numbers that match the snapshot.
