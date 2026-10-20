# aigaffer

An autonomous manager for Fantasy Premier League. On a schedule it pulls the
week's data from the FPL public API, projects expected points, plans the best
transfers, lineup, captain and chips over the coming six gameweeks, and puts
that plan to a Claude agent that reads the week's team news and may overrule
it. The verdict goes to your phone. It advises and never acts: no transfer is
ever made for you, and the final tap in the app is yours.

📋 **Latest verdict: [GW8](GW8.md)** <!-- latest-verdict -->

## What you get, and when

Four reports a gameweek. Three are timed against the deadline ahead, and one
against the round just played:

| When | Report | What it is for |
| --- | --- | --- |
| from 18:00 UTC the day after the round's last kickoff | `early` | a first look at the plans while the results are fresh |
| 18–60h before the deadline | `scout` | transfer plans while there is still time to think |
| 3–18h before, aimed at T-18h | `deadline` | the full verdict, after Friday's press conferences and the flags that follow |
| the last 3h | `reminder` | a short alert: the moves, and whether the plan survived the team news |

Each report is sent once, by the first scheduled tick inside its window. A
window stays open until the next report's begins, so a tick the scheduler
dropped is made up late rather than lost. When a round ends close to the next
deadline, as a midweek one does, there is one scout and no early scout.

**On Telegram** you get a digest, not the document. It opens with one line on
where the season stands (points, overall rank, team value, bank, free
transfers). Under it is one `Chips:` line, such as `Chips: TC GW15 · BB GW17 ·
FH GW12 · WC — · expire GW19`, where `now` marks a chip played this week. Then
come the ⏰ checklist of moves, priced; the eleven and the bench; the gaffer's
opening paragraph; and a pointer to the full report. It is one screen you can
act on. The reminder and the withheld-report alert open with the same season
line.

**Where things live:**

- `GW{n}.md`, at the repo root: the verdict for each gameweek. The early scout
  writes it first, the scout rewrites it and the deadline run overwrites it.
  Once the week's points are final it gains a **Looking back** review: what
  went well, what went wrong, and the lessons the next week's gaffer reads.
  See [GW7.md](GW7.md) for a real verdict, and [GW6.md](GW6.md) for one with
  its review.
- `reports/GW{n} Detailed Report.md`: the long read. It has every transfer
  beside its fixtures and xP, the squad, the armbands, the transfers it
  skipped and why each lost, and what changed between the week's runs.
- `state/reports/`: the run-by-run history, including every reminder.
- `state/detailed/`: the per-run data the detailed report is built from.

## Talking to it on Telegram

The chat keeps five buttons under the message box. Tapping one sends its
text, or you can type:

- **Transfers made**: send it straight after you enter the latest
  recommendation exactly, transfers and chip. Later reports that gameweek then
  work from the squad you entered. Don't send it if you changed anything.
- **Wildcard?** / **Free hit?**: what playing that chip at the coming deadline
  would do. The numbers come in a few minutes and the gaffer's view after
  them. Nothing is recorded. Six a day.
- **Chip forecast**: when the bot currently plans to play each chip you hold,
  read instantly from the latest report.
- **help**: the list of commands.
- **Ask the gaffer a question.** Any text that ends with "?", or starts with
  why, what, how, should, is, was, would, could, can, and, who, which or when,
  goes to the gaffer, for example "Why Saka?". He answers from this week's
  detailed report in a short reply and remembers the last ten exchanges, so
  "and Saka?" follows on. He explains the plan and never changes it. Thirty
  answers a day, when a gaffer is configured.

The buttons win over questions, so a mistyped command gets the help, not an
answer. Only your own box reads Telegram. The details are in
[docs/operations.md](docs/operations.md#telling-it-what-you-entered), down to
[asking the gaffer](docs/operations.md#asking-the-gaffer).

## How it decides

An expected-points model projects every relevant player fixture by fixture.
It uses fitted team strengths, recent minutes and a shrunk per-90 rate. A
mixed-integer solver then plans the next six gameweeks as one problem: squad,
eleven, captain, bank, free transfers and chips in every week. It values the
bench by order and charges for a signing who would only sit. The candidate
plans then go to a Claude agent, the gaffer. He reads the team news on the web,
may overrule players' minutes and re-solve, and picks the plan to enter. If he
can't be reached or doesn't decide, the solver's own pick goes out instead,
labelled as such, and a deadline is never missed for want of him. The full
story is in [docs/how-it-decides.md](docs/how-it-decides.md).

## Run your own

The bot runs itself from `.github/workflows/gaffer.yml`, with an always-on box
of your own as the primary scheduler if you want one. Fork the repo and add
these repository secrets:

| Secret | Value |
| --- | --- |
| `FPL_TEAM_ID` | your FPL entry id (the number in your team's URL) |
| `TELEGRAM_BOT_TOKEN` | from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | the chat to send to |
| `ANTHROPIC_API_KEY` | optional; without it every run is the solver's alone |

Run one report by hand from **Actions → gaffer → Run workflow**, or locally
without touching state or Telegram:

```sh
FPL_TEAM_ID=1234567 python -m aigaffer scout --dry-run
```

Four repository variables stand parts of it down without a commit:
`AIGAFFER_MANAGER=0` (no gaffer), `AIGAFFER_PLANNER=single` (no six-week
window), `AIGAFFER_CHIPS=off` (chips advisory only) and
`AIGAFFER_STRENGTH=off` (FPL's strength columns instead of the fitted model).

Secrets, the schedule, your own box, the inbox and local development are all
in [docs/operations.md](docs/operations.md).

## What it costs

The FPL data is free; the Claude calls are the cost. Three decision runs a
gameweek come to roughly $3–12 a week, mostly at the low end because most
weeks are quiet, and the smaller calls (reasons, review, questions, chip
what-ifs) add a little on top. The breakdown is in
[docs/operations.md](docs/operations.md#what-it-costs).

## Honest limits

The bot can't see your purchase prices, so it rebuilds them by watching your
squad, within a known error bar. It also can't see a transfer you made
mid-week until the deadline passes, unless you text "Transfers made". A new
signing with no Premier League season behind him is underrated in the opening
weeks. The backtest and the replay are sanity checks, not measurements of how
the bot would have done. The details, and what isn't built yet, are in
[docs/limitations.md](docs/limitations.md).

## License

MIT — see [LICENSE](LICENSE), which also carries the notice for the vendored
vaastav data.
