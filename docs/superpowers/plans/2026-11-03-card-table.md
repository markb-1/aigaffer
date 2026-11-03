# Transfer Card Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lay out each transfer card as a per-gameweek fenced table that reads on the phone and on GitHub, and teach the Telegram sender to deliver a fenced block as one `<pre>`.

**Architecture:** `transfer_card.py` keeps its public shape (`transfer_cards` returns one string per card) but builds a header line, a fenced table and `- ` bullets instead of two long fixture lines. The bank moves to the report's count line through a shared `covers` predicate. `telegram.py` gains a unit layer: a closed bare fence becomes one `(plain, html)` unit, which the chunker never cuts.

**Tech Stack:** Python 3.12, pytest, httpx (`MockTransport` in tests).

**Spec:** `docs/superpowers/specs/2026-11-03-card-table-design.md`

## Global Constraints

- The table is fenced with lines that are exactly ```` ``` ````. No language tag.
- Columns: GW (width 2), two-space gap, Out cell, two-space gap, In cell. In the header row, `Out` is padded to the Out cell's width, then the gap, then `In`, as in `GW  Out      In`. The totals row is labelled `=`.
- Cell: the fixture (`phone_games`) is left-padded to the column's widest fixture, then one space, then the 1dp points right-aligned to the column's widest number *including the column's sum*. A blank shows `—` with an empty points slot. Every row is `rstrip`ped.
- Sums are plain `_window` sums (stored 2dp values), never decayed.
- Width: a double on both sides with points under 10 is exactly 30 characters. The measured phone ceiling is 32. Nothing truncates.
- Card header: `{out} → {in} · {POS} · {price}`, with no bank.
- Bullets: `- {+this} this week, {+gain} over {n} weeks`, `- Next best: A 27.6 (£5.0m), B 26.9 (£5.0m)`, `- 3 from {Club} now: A, B, C`.
- Count line, when `covers` holds: `{n transfers}, {cost}, {£x.xm} left in the bank.` Otherwise it stays as today.
- Telegram: a fenced unit's content goes through `_escape` only, never heading detection. "Too long" means the unit's HTML exceeds `MAX_CHARS` in UTF-16 units.
- Week plan text 3 is unchanged in code. It drops the bank because the card does.
- Match the surrounding code: docstrings in the house prose style, and test comments that work the expected figures.

## Review Focus

1. **Model text carrying a tagged fence** (```` ```python ````) must arrive as ordinary escaped lines, never as `<pre>` and never paired with a card's fence. Pinned in Task 3.
2. **An unclosed bare fence after the card** (a model's stray ```` ``` ````) must leave the card's `<pre>` intact, and the stray line must be sent as itself. Pinned in Task 3.
3. **A card falling at a chunk boundary** must be moved whole to the next message, never split. Pinned in Task 3.
4. **Telegram's 400 plain fallback for a chunk with a table** must read as the table's lines with no backticks. Pinned in Task 3.
5. **A blank gameweek in the Out column** (the week plan's Justin case) must keep its alignment, with `—` in the fixture slot and no stray number. Pinned in Task 1.

---

### Task 1: The card builds the table

**Files:**
- Modify: `aigaffer/report/transfer_card.py` (module docstring; `_cell` and `_line` replaced; `transfer_cards`; new `covers`)
- Test: `tests/test_transfer_card.py`, `tests/test_render.py:1195-1210` (the `CARD` constant only), `tests/test_weekplan.py:550-610` (card expectations only)

**Interfaces:**
- Produces: `covers(cards: CardInputs, out_ids: Sequence[int], in_ids: Sequence[int]) -> bool`, which is true when `cards.rows` holds every id in `out_ids` and `in_ids`. `transfer_cards` uses it in place of its inline `all(...)` (transfer_card.py:192). Task 2 uses it for the count line.
- Produces: `FENCE = "```"`, a module constant that `_table` writes on both sides.
- `transfer_cards` keeps its signature and its `[]`/pointer behaviour.

- [ ] **Step 1: Rewrite the card-shape tests to the new layout**

In `tests/test_transfer_card.py`:

```python
def test_card_matches_the_approved_shape():
    [card] = cards_for(board(), [1], [2], [1, 3], 59)

    # Out points slot is 3 wide (4.4 is its widest); In is 4 (10.7), hence the
    # two spaces in "liv  4.8".
    assert card.splitlines() == [
        "Calafiori → Gabriel · DEF · £7.9m",
        "```",
        "GW  Out      In",
        "9   liv 0.8  liv  4.8",
        "10  HUL 3.6  HUL  5.9",
        "=       4.4      10.7",
        "```",
        "- +4.0 this week, +6.2 over 2 weeks",
    ]
```

- `test_double_shows_the_weeks_number_once`: the table rows are `"9   liv 0.8  BHA+ars  7.1"`, `"10  HUL 3.6  HUL      5.9"` and `"=       4.4          13.0"`. Assert the slice `card.splitlines()[3:6]`.
- `test_blank_cell_is_a_dash`: the rows are `"9   liv 0.8  liv 4.8"`, `"10  HUL 3.6  —"` and `"=       4.4      4.8"`.
- `next_best` helper and `test_no_next_best_line_when_nobody_is_eligible`: match the prefix `"- Next best:"`. Every existing next-best assertion changes its separator from `" · "` to `", "` and gains the `"- "` prefix.
- `test_club_line_only_at_three`: `"- 3 from Arsenal now: Gabriel, Rice, Zed"`.
- `test_two_moves_share_one_bank_figure`: rename it `test_two_moves_head_their_own_cards`, and assert `["DefOut → DefIn · DEF · £7.9m", "MidOut → MidIn · MID · £8.0m"]`.
- `test_a_missing_row_gives_no_cards_at_all`: also assert `covers(...)` is False for `[2, 99]` and True for `[2, 4]`.

- [ ] **Step 2: Add the width test**

```python
def test_a_double_on_both_sides_fits_thirty_characters():
    # Six double weeks of 9.9 each side: sums 59.4, so each points slot is 4
    # wide and each cell is 7 + 1 + 4 = 12. A row is 2 + 2 + 12 + 2 + 12 = 30.
    events = list(range(10, 16))
    doubles = lambda a, b: {gw: [(a, "h"), (b, "a")] for gw in events}
    rows = {
        1: row("Out", "DEF", "AAA", 50, doubles("BHA", "ARS"), {gw: 9.9 for gw in events}),
        2: row("In", "DEF", "BBB", 50, doubles("MCI", "TOT"), {gw: 9.9 for gw in events}),
    }

    [card] = cards_for(rows, [1], [2], [1], 0, events=events)
    table = card.splitlines()[2:10]

    assert table[1] == "10  BHA+ars  9.9  MCI+tot  9.9"
    assert max(len(line) for line in table) == 30
```

- [ ] **Step 3: Update the card text other suites assert**

- `tests/test_render.py` `CARD`: `["Gale → Reid · FWD · £9.5m", "```", "GW  Out      In", "2   ASH 1.0  ASH 4.0", "3   brw 2.0  brw 5.0", "=       3.0      9.0", "```", "- +3.0 this week, +6.0 over 2 weeks"]`. In `test_digest_carries_the_card_under_recommendation`, change the slice to `[:9]`. Its count line stays `"1 transfer, no hit."` until Task 2.
- `tests/test_weekplan.py` `test_text_three_opens_with_the_transfer_card`: `third.splitlines()[:10]` is `["3/5 · 🤔 Why this plan", "", "Justin → Cucurella · DEF · £6.1m", "```", "GW  Out      In", "8   LEE 3.0  wol 2.0", "9   —        LEE 4.0", "=       3.0      6.0", "```", "- -1.0 this week, +3.0 over 2 weeks"]`. That last row is Review Focus 5.
- `test_text_three_has_no_card_on_a_roll`: unchanged; `"this week,"` is still absent.
- `test_card_reads_only_the_row_keys_old_snapshots_have`: the header becomes `"Justin → Cucurella · DEF · £6.1m"`.

- [ ] **Step 4: Run the tests and see them fail**

Run: `.venv/bin/pytest tests/test_transfer_card.py tests/test_render.py tests/test_weekplan.py -q`
Expected: `tests/test_transfer_card.py` fails at collection with an ImportError, because it imports `covers` at the top. That's expected and not a broken plan. In the other two files, the render-card and week-plan card tests FAIL.

- [ ] **Step 5: Implement the table in `transfer_card.py`**

- `covers(cards, out_ids, in_ids) -> bool`, as in the Interfaces block. `transfer_cards` calls it.
- `_table(out_row: dict, in_row: dict, events: Sequence[int]) -> list[str]`: the fence, header, one row per event, the `=` row and the fence, per the Global Constraints. `_window` is unchanged. A blank's points slot is empty. Rows are rstripped.
- `transfer_cards` assembles the header line, `*_table(...)`, the `- ` gain bullet, the `- Next best: ` bullet joined with `", "` (only when there are alternatives), and the `- {CLUB_LIMIT} from {cf} now: {names}` bullet (only at the limit).
- Update the module docstring and the docstrings of `transfer_cards` and `bank_left`. `bank_left`'s docstring should say the count line prints the figure, not the card.

- [ ] **Step 6: Run the tests and see them pass**

Run: `.venv/bin/pytest tests/test_transfer_card.py tests/test_render.py tests/test_weekplan.py -q`
Expected: PASS, including the frozen-snapshot test `test_the_committed_snapshots_render_five_texts_under_telegrams_limit`.

- [ ] **Step 7: Commit**

```bash
git add aigaffer/report/transfer_card.py tests/test_transfer_card.py tests/test_render.py tests/test_weekplan.py
git commit -m "feat(card): the transfer card is a per-gameweek table"
```

### Task 2: The count line carries the bank

**Files:**
- Modify: `aigaffer/report/render.py:1436-1482` (`_recommendation` and its docstring)
- Test: `tests/test_render.py` (around lines 1213-1270)

**Interfaces:**
- Consumes: `covers` and `bank_left` from `aigaffer.report.transfer_card` (Task 1), and render.py's existing `price(tenths) -> str` (`£x.xm`, used at render.py:1191).

- [ ] **Step 1: Write the failing tests**

- `test_digest_carries_the_card_under_recommendation`: the slice is `["1 transfer, no hit, £2.5m left in the bank.", *CARD.splitlines()]`. The comment above `CARD_ROWS` already works the arithmetic: 80 + 40 − 95 = 25.
- `test_without_cards_the_bullets_are_unchanged` and `test_a_card_with_a_missing_row_falls_back_to_the_bullets`: unchanged. Their `"1 transfer, no hit."` lines are the proof that a fallback gains no bank.
- `test_wildcard_keeps_the_bullets_and_adds_the_pointer`: the third line becomes `"1 transfer, no hit, £2.5m left in the bank."`.
- New test, `test_hits_and_bank_share_the_count_line`: `dataclasses.replace(ONE, hits=1)` (`Plan` is a dataclass, solver/optimizer.py:145). The line is `"1 transfer, -4 pts in hits, £2.5m left in the bank."`.

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/pytest tests/test_render.py -q -k "recommendation or card or wildcard or count_line"`
Expected: FAIL on the three changed assertions.

- [ ] **Step 3: Implement it in `_recommendation`**

When `cards is not None and covers(cards, choice.transfers_out, choice.transfers_in)`, the count line is `f"{moves}, {cost}, {price(bank_left(...))} left in the bank."`. Otherwise it is today's `f"{moves}, {cost}."`. The Free Hit/roll early return is untouched. Update the docstring to say where the bank lives now.

- [ ] **Step 4: Run the tests and see them pass**

Run: `.venv/bin/pytest tests/test_render.py tests/test_orchestrator.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aigaffer/report/render.py tests/test_render.py
git commit -m "feat(render): the count line carries the bank once a week"
```

### Task 3: Telegram sends a fenced block as one `<pre>`

**Files:**
- Modify: `aigaffer/report/telegram.py` (module docstring; `_chunks`; new `_units`)
- Test: `tests/test_telegram.py` (module docstring; new tests at the end of the chunking group, after `test_an_emoji_heavy_line_is_budgeted_in_utf16_units`)

**Interfaces:**
- Produces: `_units(text: str) -> Iterator[tuple[str, str]]`, which yields `(plain, html)` pairs in order. `_chunks` iterates over these instead of `_lines` plus `_as_html`. It keeps packing by `_utf16_len(html)` and keeps joining both renderings with `\n`. `_lines` and `_as_html` are unchanged and still serve ordinary lines.

This supersedes the spec's wording, in which `_lines` yields the unit and `_as_html` renders it (spec, Delivery, first two bullets). The outcome is the same. Write the docstrings to the code.

`_units` algorithm (the tests don't pin the scan, so here it is):

```
lines = text.split("\n"); i = 0
while i < len(lines):
    if lines[i].strip() == "```":
        j = next index > i with lines[j].strip() == "```", else None
        if j is not None:
            body = lines[i+1:j]
            html = "<pre>" + _escape("\n".join(body)) + "</pre>"
            if not body: pass                                   # empty block: nothing
            elif _utf16_len(html) <= MAX_CHARS: yield ("\n".join(body), html)
            else: yield from ((l, _as_html(l)) for b in body for l in _lines(b))  # fence lines dropped
            i = j + 1; continue
    for piece in _lines(lines[i]): yield (piece, _as_html(piece))
    i += 1
```

In the oversize fallback, a body line starting `# ` would be bolded by `_as_html`. That is acceptable, because those are ordinary lines by then. Only the `<pre>` path must never call heading detection.

- [ ] **Step 1: Write the failing tests**

```python
# Not aligned to the card rules: these tests check escaping and units, not layout.
TABLE = "Gale → Reid · FWD · £9.5m\n```\nGW  Out  In\n2   A&B 1.0  <x> 4.0\n```\n- +3.0 this week"

def test_a_fenced_block_arrives_as_one_pre_with_the_fences_dropped():
    assert texts(send(TABLE)) == [
        "Gale → Reid · FWD · £9.5m\n<pre>GW  Out  In\n2   A&amp;B 1.0  &lt;x&gt; 4.0</pre>\n- +3.0 this week"
    ]

def test_a_heading_inside_a_fence_is_not_bolded():
    assert texts(send("```\n# not a heading\n```")) == ["<pre># not a heading</pre>"]

def test_a_tagged_fence_is_ordinary_lines():          # Review Focus 1
    text = "```python\nx = 1\n```"
    assert texts(send(text)) == ["```python\nx = 1\n```"]

def test_an_unclosed_fence_after_a_card_leaves_the_card_whole():   # Review Focus 2
    text = TABLE + "\n```\nthe gaffer's stray line"
    [html] = texts(send(text))
    assert html.count("<pre>") == 1 and html.endswith("\n```\nthe gaffer's stray line")

def test_two_fenced_blocks_are_two_pres():
    assert texts(send("```\na\n```\n```\nb\n```")) == ["<pre>a</pre>\n<pre>b</pre>"]

def test_an_empty_fenced_block_sends_nothing_for_it():
    assert texts(send("before\n```\n```\nafter")) == ["before\nafter"]

def test_a_fenced_block_at_a_chunk_boundary_moves_whole():          # Review Focus 3
    # Fill the first message to within 20 units of the limit, then a 3-line
    # block: it cannot fit, so it opens the second message intact.
    filler = "x" * (LIMIT - 20)
    first, second = texts(send(filler + "\n```\naaaaaaaaaa\nbbbbbbbbbb\n```"))
    assert first == filler
    assert second == "<pre>aaaaaaaaaa\nbbbbbbbbbb</pre>"

def test_a_rejected_chunk_with_a_table_falls_back_without_backticks():   # Review Focus 4
    requests = send_html_rejected(TABLE)
    assert texts(requests)[1] == "Gale → Reid · FWD · £9.5m\nGW  Out  In\n2   A&B 1.0  <x> 4.0\n- +3.0 this week"

def test_an_oversized_fenced_block_degrades_to_ordinary_lines():
    body = [f"{i:03d} " + "x" * 95 for i in range(45)]          # ~4500 units
    sent = texts(send("```\n" + "\n".join(body) + "\n```"))
    assert len(sent) == 2 and "<pre>" not in "".join(sent) and "```" not in "".join(sent)
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/pytest tests/test_telegram.py -q`
Expected: the new fence tests FAIL (fences arrive literally) and the existing tests PASS.

- [ ] **Step 3: Implement `_units` and switch `_chunks` to it, as in the Interfaces block**

Update `_chunks`'s own docstring: it packs units now, not whole lines. Rewrite the module docstring paragraph about "every tag opens and closes on its own line" so it covers units. In the stray-fence paragraph, record why a bare fence in model prose can't precede a card: render.py:347-350 and 583-584, and weekplan.py:415-427. Update the test module's docstring in the same way.

- [ ] **Step 4: Run the tests and see them pass**

Run: `.venv/bin/pytest -q`
Expected: the whole suite PASSES.

- [ ] **Step 5: Commit**

```bash
git add aigaffer/report/telegram.py tests/test_telegram.py
git commit -m "feat(telegram): a fenced block goes as one pre, never cut"
```

## Final verification (after Task 3)

- `.venv/bin/pytest -q`: everything passes.
- Feed `render_week_plan(load_snapshots(GW9_WEDNESDAY, 9), NOW, None)[2]` (as in tests/test_weekplan.py:748) through `telegram._chunks`. Print the HTML and check by eye that the `<pre>` is whole, with the bullets after it.
- After merge, the next live digest is the real check on the phone. Ask Mark for a screenshot of the first one.
