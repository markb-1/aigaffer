# Week plan button — design

Status: draft for the owner's approval · 2026-10-29

## Intent

The phone digest is built to be acted on. It says what to do before the deadline and little else. The detailed report carries the reasoning, the team news, the plan for later weeks and the fixtures, but it is a long markdown file in the repo that nobody reads on a phone.

**Week plan** is an optional button that sends the detailed plan to Telegram as a short series of texts. It answers "what are we doing this week, why, what's the news, and where is this heading?" without opening GitHub. It's opt-in, so it can carry a fair amount, but each text must stand on its own and lead with its bottom line.

## Goals

- One tap on the Telegram keyboard sends 4–5 texts that cover:
  - the actions
  - team news
  - why this plan
  - the road ahead and chips
  - upcoming fixtures
- Answer within seconds, at no LLM cost, as **Chip forecast** does.
- Say the same thing as the report it summarises: built from the same snapshot, reusing the report's own text where it exists.
- Always say how fresh the plan is ("as of Wednesday's scout run, 23:35").

## Non-goals (v1)

- No new analysis. Nothing is solved, and no LLM is called. Everything shown is already in the latest detailed snapshot.
- No inline buttons or callbacks. The bot uses a persistent reply keyboard, and a tap sends its label as text (`telegram.py` `KEYBOARD`), so the new button needs no new update type.
- No per-section drill-down ("send me only the fixtures"). That could come later as a text command.
- No scheduled or automatic sending. It's sent only when the owner asks.

## Anti-goals (a failure even if it works)

- **Contradicting the report**, e.g. a captain or transfer that differs from the digest. The action card is the snapshot's `do_this`, printed verbatim.
- **Passing off a stale plan as current.** The first line always names the run and its time. If the owner has since recorded "Transfers made", or the gameweek's deadline has passed, the texts say so.
- **A wall of text.** Each text has a target length of about 1,200 characters, comfortably under Telegram's 4,000 limit, and each one leads with its bottom line.
- **Noise.** No 15-player six-week table, no list of every alternative plan, no team value, and no minute overrides for players outside the eleven unless they changed a decision. (FPL expert's list of things to cut.)

## The texts

The order and contents come from the FPL expert's recommendation, trimmed to what the snapshot holds today. Each section below is one Telegram message.

### 1 · This week: what to do

- Header: `📋 GW9 week plan · as of Wed scout run 23:35 · deadline Sat 11:00 (Irish time)`.
- **What changed since the last run**, only when something did, as the detailed report's `what_changed` bullets (capped at about 6, then "+N more").
- **The action card:** the snapshot's `do_this`, verbatim. That's the chip banner, the "Do this" checklist with selling prices, captain and vice, and the lineup.
- A one-line summary: `1 FT used, 0 banked next week · bank £3.7m · no hit`.

### 2 · Team news

Only for the players who matter: the XI and bench first, then the planned targets (this week's signings and the road ahead's), then the captain's rivals.

- **What the gaffer learned:** the `WHAT I LEARNED` block of his rationale, verbatim. These are his researched notes with sources, such as "Arteta said…".
- **Flags:** players in that set with an FPL news flag or projected minutes under 65, one line each: `Thomas (DEF) 35 min · 75% · Forced off vs Fulham; Lampard 'hoping'…`. The reason is the gaffer's override reason when he set one, otherwise the FPL news string.
- **Not checked:** when his rationale says some clubs weren't searched, that line is shown as a caution.
- The certainty tags the expert suggested (confirmed / likely / rumour) aren't in the data today. That needs the gaffer to tag his findings (see Phase 2).

### 3 · Why this plan

- `WHY THIS PLAN`, the gaffer's bullets, verbatim.
- **Runner-up:** the best non-chosen plan and its margin, plus rolling the transfer and its margin, from the options table: `Next best: +Konsa for a hit, 1.6 behind · Rolling: 1.9 behind`.
- **Captain:** captain, vice and the two rivals with their window and this-week xP, from "Captain and vice".
- **Skipped:** the two or three highest-gain transfers that were passed over, each with the reason given in the report.

### 4 · The road ahead and chips

- The road ahead lines, as the detailed report prints them (`GW10: out Thomas (£4.0m), in Konsa (£4.7m) — 1 FT, no hit`), headed **"Current plan, re-planned every run"**.
- Chips:
  - Played this week or not.
  - Each chip's calendar line: the week it's saved for, its value there, and this week's bar.
  - This week's chip values, on one line.

### 5 · Upcoming gameweeks

- **The squad's week-by-week projected total over the window:** `GW9 54.3 · GW10 64.9 · …`, with a note that GW+4 to GW+6 are softer.
- **One line per XI player and planned target:** opponents and home/away across the window, then his window total. `Haaland  BHA nfo FUL ars LEE bre — 32.9`. Home is upper case and away lower case, as in the report.
- **Blanks and doubles** marked: `—` for a blank, `+` joining a double.
- **One "swing" line**, worked out from the data: the club whose fixtures turn most, best to worst or the reverse, between the first and last three weeks, among the squad's clubs. Leave it out when nothing turns sharply.

Price watch isn't in the snapshot today, so it moves to Phase 2.

## How it works

- **Trigger:**
  - Add a **"Week plan"** button to `KEYBOARD`. New layout:
    - row 1: `Transfers made` / `Week plan`
    - row 2: `Wildcard?` / `Free hit?`
    - row 3: `Chip forecast` / `Help`
  - Add `week plan` (and `plan`) to the inbox's `dispatch`, ahead of the question check. Without that, "plan?" would go to ask.
  - Add a line to `HELP`.
- **Source:** `latest_snapshot(state_dir)` for the next gameweek, plus that week's earlier snapshots from `load_snapshots` for "what changed". It's read under `state_lock`, as `ask` does, which says "report running" if the hourly tick holds the lock. If there's no snapshot for the coming gameweek, it falls back to the newest one and says so.
- **Rendering:**
  - New module `aigaffer/weekplan.py`: a pure `render_week_plan(snapshots, now, executed) -> list[str]`, one string per text, plus the inbox handler `week_plan(cfg, sent_at, reply)`.
  - It reuses the detailed renderer's helpers rather than parsing the markdown: `what_changed`, `_road_ahead` / `_planned`, `captain_rivals`, `settled` / `noted`, `chip_label`, `price`, and the cell format. The private ones become public.
- **Sending:** each text goes through the inbox's `reply`, which is `send_message` and handles HTML escaping, splitting and the plain-text fallback. The handler returns None, as `chip_whatif` does. The keyboard rides on the last text.
- **Freshness checks:**
  - If an executed record ("Transfers made") is newer than the snapshot, text 1 says: "You've entered moves since this run; the next report works from your squad."
  - If the snapshot's gameweek deadline has passed, it says the plan is for a gameweek now underway.
- **Failure:** the inbox's existing failed-marker retry and apology apply. Rendering is pure, so tests can feed it the committed GW8 and GW9 snapshots.

## Phase 2 (not in v1, listed so v1 leaves room)

1. **The gaffer writes "what would change my mind" and decision triggers** ("If Rice is ruled out on Friday: Rice → Ødegaard now"). This needs two new fields in the manager's finalize tool and the decision record. It's the expert's highest-value addition.
2. **Certainty tags on team news** (confirmed / likely / rumour, with the quote's date). The gaffer tags each finding in a structured list rather than prose.
3. **Price watch in the snapshot**, for owned players and planned targets: likely rise or fall tonight, and a "transfer before about 01:30" line.
4. **Captain ownership / effective ownership and the template gap.** This needs ownership data beyond `selected_by_percent`, and is worth it only if the owner plays for rank.
5. **Confidence per road-ahead move**, as in FPL Review's sensitivity: how often a move survives nearby alternatives.

## Testing

- `render_week_plan` on hand-built snapshot dicts, in the style of `test_detailed_render.py`:
  - each text's contents and order
  - "changes" left out when nothing changed
  - old snapshots without `do_this` or `hits`
  - a solver-only run (no rationale blocks)
  - a free-hit week
  - the stale-snapshot and entered-moves notices
  - every text under 4,000 characters for the committed GW9 snapshot
- Inbox: `week plan` and `plan` dispatch to the handler; `plan?` goes to week plan, not ask; the keyboard has the button.
- Rendering the committed GW8 and GW9 snapshots runs without error, and the output is checked by eye.

## Left to the builder

- Exact wording and emoji. Keep to the house voice of the existing digest.
- The thresholds: 65-minute flag, 6 change bullets, 3 skipped transfers, and the cut-off for the swing line.
- Whether "Week plan" also accepts `wp`.

## Example (GW9, from the committed scout snapshot, abridged)

> **1/5** 📋 GW9 week plan · as of Wed scout run 23:35 · deadline Sat 31 Oct 11:00
> Changed since Monday: chip now Triple Captain · Calafiori 25→15 min · Rice 86→70 min · …
> ⏰ Make these by Sat 31 Oct 11:00 UTC — GW9 / SELL Calafiori … → BUY Gabriel … / CAPTAIN Haaland · VICE Mbeumo / Set lineup (3-5-2): …
>
> **2/5** Team news — Calafiori (hip) out of Fleetwood tie, "genuine concern" for Anfield → sold. Rice trained Monday, 70 min. Thomas forced off, 35 min, benched. João Pedro out "a few weeks". ⚠️ Not checked: Man Utd, Hull, Everton, Bournemouth, Leeds.
>
> **3/5** Why — Calafiori injured and not starting; Gabriel nailed. Move worth 1.9 over rolling; Konsa as well for a hit only 0.3 ahead. Captain Haaland 32.9 over window (Tavernier 30.6, Calvert-Lewin 28.6).
>
> **4/5** Current plan (re-planned every run) — GW10 Thomas → Konsa · GW11 Rice → Ødegaard · GW12 Vuskovic → Tarkowski. Chips: TC saved for GW15 (+6.0, bar 7.0) · FH saved for GW17 · WC held.
>
> **5/5** Next 6 — squad xP GW9 54.3 · GW10 64.9 · GW11 58.0 · … / Haaland BHA nfo FUL ars LEE bre 32.9 / Gabriel liv HUL new MCI bre tot 31.6 / …
