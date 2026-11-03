# Transfer card as a table: one row per gameweek

Date: 2026-11-03 · Status: principal approved with changes (applied); awaiting Mark

## Why

The transfer card (#27, `aigaffer/report/transfer_card.py`) has the right
numbers but is hard to read on the phone. Mark's screenshot from 3 Nov, digest
`## Recommendation`:

```
Thomas → Konsa · DEF · £4.7m (bank
£0.8m left)
Out  eve 1.7 · CRY 1.9 · lee 1.7 · IPS 1.9 ·
mun 1.6 · AVL 1.7 = 10.4
In   HUL 5.6 · new 5.0 · MCI 4.8 · bre 4.9 ·
tot 5.3 · BOU 5.3 = 30.9
+3.9 this week, +20.6 over 6 weeks
Next best: Van Hecke 27.6 (£5.0m) · Silva
26.9 (£5.0m)
3 from Arsenal now (Gabriel, Konsa, Rice)
```

Three faults:

1. **Six cells don't fit on a phone line**, so the `Out` and `In` rows each
   wrap, at different points. The comparison the card is for (this week, him
   against him) can't be read across.
2. **The alignment padding is lost.** `"Out  "` / `"In   "` assume equal-width
   characters. Telegram's font is proportional, so the padding collapses.
3. **The header and "Next best" lines carry too much** and break
   mid-phrase: `(bank` / `£0.8m left)`.

The same card also lands in the committed `GW*.md` reports, where GitHub
markdown joins its consecutive lines into one paragraph. That's a fourth
fault, seen by fewer people.

## What success looks like

Mark reads one gameweek per row, the outgoing player's fixture and points next
to the incoming player's, with the decimals lined up, and nothing wraps on an
iPhone at the default text size. The same block reads as a table on GitHub.

## The card (Mark approved this shape in chat)

````
Thomas → Konsa · DEF · £4.7m
```
GW  Out       In
10  eve  1.7  HUL  5.6
11  CRY  1.9  new  5.0
12  lee  1.7  MCI  4.8
13  IPS  1.9  bre  4.9
14  mun  1.6  tot  5.3
15  AVL  1.7  BOU  5.3
=       10.4      30.9
```
- +3.9 this week, +20.6 over 6 weeks
- Next best: Van Hecke 27.6 (£5.0m), Silva 26.9 (£5.0m)
- 3 from Arsenal now: Gabriel, Konsa, Rice
````

(The GW numbers above are illustrative.)

### The header line

`{out} → {in} · {POS} · {price}`, plain text and short. The bank moves off
the card (see "Bank" below).

### The table

- **A fenced code block** (```` ``` ```` lines above and below). GitHub renders
  it as monospace. The Telegram sender turns it into `<pre>` (see "Delivery").
- **Rows:** a header row `GW  Out  In`, one row per gameweek in the window
  (`cards.events`, never hard-coded to six), and a totals row.
- **Columns are `Out` and `In`, not the players' names.** The header line
  already names them, and long names (`Alexander-Arnold`) would break the
  width budget.
- **A cell is two slots, a fixture slot and a points slot, with one space
  between them.**
  - The fixture comes from `aigaffer.data.schedule.phone_games`, unchanged:
    home UPPER, away lower, `+` joining a double, `—` for a blank.
  - The fixture slot is left-aligned and padded to the column's widest
    fixture (`—` counts, at width 1).
  - The points slot is 1dp, right-aligned and padded to the column's widest
    number, *including that column's sum*. A sum of 10 or more therefore
    makes the slot 4 wide, which is why the mock-up shows `eve  1.7` with
    two spaces. The decimal points line up down the column.
  - A blank gameweek prints `—` in the fixture slot and leaves the points
    slot as spaces. It adds 0 to the sum, as `_window` already does.
- **The last row is labelled `=`** and carries each column's plain window
  sum (as today: `_window`, the sum of the stored 2dp values, never the
  decayed total), in that column's points slot. A one-character label keeps
  the GW column 2 wide. A 3-character `Sum` would cost the doubles case its
  fit.
- **Column gaps are two spaces**, so a row is `GW(2) + 2 + Out + 2 + In`.
- **Width budget: 30 characters, asserted.**
  - Two single fixtures with sums of 10 or more: 2 + 2 + 8 + 2 + 8 = 22.
  - A double on both sides (`BHA+ars`, 7 characters), with points under 10
    and sums of 10 or more: 2 + 2 + 12 + 2 + 12 = **30**.
  - A test builds exactly that doubles case and asserts every row is at
    most 30 characters.
  - A double-gameweek week of 10+ points, or a triple gameweek, can go to
    31 or more. That's accepted, not designed for: nothing truncates.
  - **Measured on 3 Nov 2026.** The principal worked back from the first
    screenshot that 30 characters might be at the edge, so a throwaway
    `<pre>` ruler was sent to Mark's chat. On his iPhone (Telegram iOS,
    default text size), lines of 22, 28, 30 and **32** characters all stayed
    on one line, with room to spare. The sample 30-character double row was
    tested and also fit. So the 30 budget holds with a margin, and **32 is
    the measured ceiling** for the next card change to work to. Telegram
    draws a `</>` badge at the top right of the block, beside the first
    line. The table's first line is the short header row, so the two don't
    collide.

### The lines under the table

Markdown bullets (`- `), house style for the report's lists, so GitHub shows
a list and Telegram shows `- ` lines that wrap as a group:

- `- {+this week} this week, {+window} over {n} weeks` (unchanged numbers).
- `- Next best: A 27.6 (£5.0m), B 26.9 (£5.0m)`: the same alternatives and
  rule as today. The separator changes from ` · ` to `, ` because the bullet
  wraps more cleanly on a comma.
- `- 3 from {Club} now: A, B, C` when the club limit is reached (the same rule
  as today, with the brackets replaced by a colon).

### Bank

The bank after this week's moves is the same on every card of the week, so
it is printed once per week rather than once per card:

- **Digest/report:** the count line becomes
  `1 transfer, no hit, £0.8m left in the bank.` (the figure is `bank_left`,
  as today). It carries the bank whenever `bank_left` can be computed: when
  `cards` is given and the run holds a row for every man moved. That
  includes the Wildcard path. The "a row for every man moved" check becomes
  one small predicate in `transfer_card.py`, shared by `transfer_cards` and
  the count line, not two copies. When it can't be computed (no `cards`, or a
  missing row), the line stays as it is today, without the bank. Today's
  fallback bullets never printed the bank either, so that path loses
  nothing. A Free Hit week still prints "Roll the transfer.".
- **Week plan:** text 1's `_summary` already prints `bank £0.8m` from the
  same function, so text 3's cards simply drop it.

My call: the screenshot's worst break was inside the bank bracket, and
printing it per card adds nothing on a two-transfer week.

## Delivery: fences in Telegram

`aigaffer/report/telegram.py` builds its HTML one line at a time, with every
tag opening and closing on its own line, so that no chunk cut can leave a tag
half-sent. A fenced block changes that rule from "per line" to "per unit":

- **`_lines` yields a fenced block as one unit**: the fence lines plus
  everything between them, joined with `\n`. `_chunks` treats a unit like a
  line, so it never cuts inside one. A card's table is about 10 lines and 300
  characters, far below the 4000 budget.
- **`_as_html` renders a fenced unit as `<pre>…</pre>`**: the fence lines are
  dropped and the content goes through `_escape` only, never through heading
  detection. A `<b>` inside `<pre>` is invalid Telegram HTML and would send
  the whole chunk to the plain fallback. There is no language tag, because
  Telegram would then show a "copy" header.
- **Plain fallback** (the 400 resend): the fence lines are dropped and the
  content is kept, so a fallback reads as the old line-by-line text rather
  than showing literal backticks.
- **A fence line is exactly ```` ``` ```` after `strip()`.** A tagged
  (```` ```python ````) or indented-then-tagged line is an ordinary line, so a
  model's code fence never pairs with the card's.
- **Pairing is a left-to-right scan.** When a fence line opens, the scan
  looks ahead for the next fence line. If there is one, everything from the
  opener to that closer is one unit. If there isn't, the opener is an
  ordinary line and the scan carries on from the line after it. Nothing is
  swallowed, and no `<pre>` is left open.
- **The stray-fence case is accepted.** A bare ```` ``` ```` line in model
  text placed before a card would pair with the card's opener, putting the
  wrong lines in `<pre>`. The result looks wrong but still sends, because
  every unit closes. No model text produces a bare fence line today, and
  guarding against it isn't worth a second syntax. More to the point, in
  every sent text the model's prose comes *after* the card:
  - full report: `_recommendation` before `_gaffer` (render.py:347-350);
  - digest: `_recommendation` before `_gaffer_digest` (render.py:583-584);
  - week plan text 3: the cards before WHY THIS PLAN (weekplan.py:415-427).

  So the case cannot arise today. A reorder of those sections must revisit
  this paragraph.
- **A fenced unit too long for one message** (not something a card can
  produce), meaning its HTML, tags included, exceeds `MAX_CHARS` in UTF-16
  units, the same measure `_chunks` packs by: it falls back to ordinary lines, each escaped, with the fence
  lines dropped. Correctness over looks; no `<pre>` is split.

The week plan's texts go through the same `send_report` path (`_say` →
`send_message`), so text 3 gets the `<pre>` table without its own code. Its
"five texts under Telegram's limit" test over the frozen snapshots must still
pass with the fences added.

## Where the change lands

- `aigaffer/report/transfer_card.py`: `transfer_cards` returns the new block.
  It still returns `list[str]`, one string per card, and the
  Wildcard/Free Hit/missing-row behaviour is unchanged. `_line` and `_cell`
  are replaced by the table builder.
- `aigaffer/report/render.py`: the count line carries the bank. The cards
  are joined as today.
- `aigaffer/weekplan.py` `_why`: unchanged apart from receiving the new
  block. It already splits each card on `\n`.
- `aigaffer/report/telegram.py`: fenced units, as above.
- Tests: the card's expected text in `tests/test_transfer_card.py`,
  `tests/test_render.py` and `tests/test_weekplan.py`; new telegram tests
  for a fenced unit (HTML `<pre>`, the plain fallback, an unclosed fence, a
  unit kept whole across a chunk boundary); a width test with a double
  gameweek on both sides.
- Docstrings: the card module's, `transfer_cards`', `_recommendation`'s and
  telegram.py's module docstring (its "every tag opens and closes on its own
  line" paragraph) are updated to match.

## Non-goals

- No change to what the card says: the same numbers, alternatives, club
  rule, pairing (`pair_moves`) and window.
- No change to the venue convention (UPPER home / lower away). Mark was
  offered an explicit h/a marker and didn't take it up. It can be added
  later without changing the layout.
- No markdown rendering in Telegram beyond headings and these fences.
  Bold, italics and links stay literal, as today.
- The detailed report's own per-player tables are untouched.

## Anti-goals

- A table that wraps on the phone. That is the whole bug.
- A `<pre>` split across two messages, or Telegram rejecting a chunk because
  of a stray fence.
- Totals that don't match the numbers printed in their column, beyond the
  2dp-to-1dp rounding the original card spec already accepts.
