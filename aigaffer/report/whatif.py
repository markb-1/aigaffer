"""Message 1 of a chip what-if: the numbers, read on a phone in a minute.

The order is the FPL expert's: the call and the deadline first; one sentence of
why; the gain week by week; whose minutes these are; then — only when he might
act on it — what to enter; the doubts in it; and a footer saying nothing was
recorded and nothing can be undone. A Hold gets the week to hold for in place of
a squad nobody will enter.

Plain text, one line per line: :func:`aigaffer.report.telegram.send_message`
escapes each line into Telegram HTML exactly as it does the digest, so markup
made here would be escaped twice. Every sentence is a constant, because the
inbox handler's tests read them back.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from aigaffer.chips import BENCH_BOOST, FREE_HIT, TRIPLE_CAPTAIN, WILDCARD, HeldChip, held_for, held_in_week
from aigaffer.data.free_transfers import MAX_FREE_TRANSFERS
from aigaffer.orchestrator import PreparedWeek
from aigaffer.report.render import STATUS_WORDS, plural, price
from aigaffer.whatif import BREAK_DAYS, HOLD, MARGINAL, PLAY, WhatIf

LONDON = ZoneInfo("Europe/London")
UK_FORMAT = "%a %d %b %H:%M"
CHIP_NAMES = {WILDCARD: "Wildcard", FREE_HIT: "Free hit"}
CHIP_WORDS = {WILDCARD: "wildcard", FREE_HIT: "free hit"}
CHIP_TAGS = {WILDCARD: "WC", FREE_HIT: "FH", BENCH_BOOST: "BB", TRIPLE_CAPTAIN: "TC"}
POSITION_TAGS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
BAND_WORDS = {PLAY: "PLAY", MARGINAL: "MARGINAL", HOLD: "HOLD"}
# Expected minutes under this, in the eleven, is a doubt worth a flag.
FLAG_MINUTES = 80.0
# A free-hit knock-on bigger than this either way is said in words.
KNOCK_ON_WORDS = 3.0

HEADER = "🃏 {chip} GW{gw} — {band} ({net} net; noise ±{margin:.0f})"
LEAN = "{band}, lean {lean}"
DEADLINE_LINE = "Deadline {when} UK ({countdown})"
COUNTDOWN_DAYS = "in {days}d {hours}h"
COUNTDOWN_HOURS = "in {hours}h {minutes:02d}m"
PASSED = "passed"
WHY_WILDCARD = "Now buys {gain} over GW{first}–{last} vs your best path without it; {bars}."
KEEPING = "keeping it is worth ~{bars:.0f}"
CHIP_TIMING = "chip timing {bars}"
WHY_FREE_HIT = (
    "FH team vs your GW{gw} XI {one_week} · knock-on GW{next}–{last} {knock}"
    " · keeping it {keeping} = net {net}"
)
KNOCK_ON_DOWN = "Your planned transfers wait a week — that costs {knock:.1f} over the weeks after."
KNOCK_ON_UP = "The weeks after gain {knock:.1f} — unusual for a free hit; treat the numbers as rough."
OFF_LINE = "(off: {moves} · {chip}: {where})"
OFF_ROLL = "roll, {ft} FTs into GW{next}"
OFF_MOVES = "{count} ({pairs}), {ft} FTs into GW{next}"
NOT_IN_WINDOW = "not in the window"
CHIPS_LINE = "Chips — on: {on} · off: {off}"
NO_CHIPS = "none"
GAIN_LINE = "Gain by week: {gains}"
MINUTES_FROM = "Minutes: {source} ({calls})."
MINUTES_NO_CHANGES = "no changes"
MINUTES_MODEL = "Minutes: the model's own (no report yet this gameweek)."
PLAY_WILDCARD = "If you play it — {moves}:"
PLAY_FREE_HIT = "If you play it — your free-hit team:"
BANK_LINE = "Bank {bank} · C {captain} · V {vice} · Bench: {bench}"
BENCH_LINE = "Bench: {bench}"
ARMBANDS = "C {captain} · V {vice}"
ANYWAY_WILDCARD = "(Playing it anyway would mean {moves}.)"
ANYWAY_FREE_HIT = "(Playing it anyway would mean a free-hit team of {changes} new players.)"
HOLD_OFF_PATH = "Hold → GW{gw}{provisional}: the path without it plays it then, worth {value:.0f} more than now."
HOLD_CALENDAR = "Hold → GW{gw}{provisional}: the calendar saves it for then, worth ~{value:.0f}."
HOLD_PAST_WINDOW = (
    "Hold: nothing in GW{first}–{last} beats keeping it. It's yours until the GW{stop}"
    " deadline ({until} UK). Keeping it is priced at ~{value:.0f} now on a generic curve,"
    " not a specific week. It becomes a Play when 4+ starters need changing, a fixture"
    " swing lines up, or a bench boost is 1–3 weeks ahead."
)
HOLD_PAST_WINDOW_FREE_HIT = (
    "Hold: nothing in GW{first}–{last} beats keeping it. It's yours until the GW{stop}"
    " deadline ({until} UK)."
)
PROVISIONAL = " (provisional)"
ASK_AGAIN_NEXT = "Ask again from GW{next}."
ASK_AGAIN_BREAK = "Ask again after the international break."
FLAGS = "⚠️ {flags}"
UNSURE = "Solver unsure: {which} stopped on its time limit — treat the numbers as rough."
UNSURE_SIDES = {
    (False, True): "the solve with it",
    (True, False): "the solve without it",
    (False, False): "both solves",
}
DIFFERS = "This differs from your latest report's plan ({plan}) — a quicker solve."
ROLL = "roll"
FOOTER = (
    'Not recorded. If you play it, don\'t send "Transfers made" — the first report'
    " after the deadline will see your squad. A chip can't be undone once confirmed."
)
GAFFER_FOLLOWS = "The gaffer's view follows in a few minutes."
NUMBERS_ONLY = "Numbers only — the gaffer is switched off."


def render_numbers(
    whatif: WhatIf,
    week: PreparedWeek,
    *,
    verdict: dict | None,
    now: datetime,
    numbers_only: bool,
) -> str:
    """Message 1, ready for :func:`~aigaffer.report.telegram.send_message`.

    ``verdict`` is the latest decision record for the gameweek (the one whose
    minutes the numbers inherited), or None; ``now`` is timezone-aware.
    """
    players = week.effective.players
    blocks = [
        [_header(whatif), _deadline(week, now)],
        _why(whatif, week, verdict),
    ]
    acting = whatif.band in (PLAY, MARGINAL)
    holding = HOLD in (whatif.band, whatif.lean)
    squad_block: list[str] = []
    if acting:
        squad_block = _squad_block(whatif, week)
        flags = _flags(whatif, week)
        if flags:
            squad_block.append(flags)
    else:
        squad_block = [_anyway(whatif, week)]
    if holding:
        squad_block.extend(line for line in (_hold(whatif, week), _ask_again(week)) if line)
    blocks.append(squad_block)
    extras = [line for line in (render_unsure_line(whatif), _differs(whatif, verdict, players)) if line]
    if extras:
        blocks.append(extras)
    blocks.append([f"{FOOTER} {NUMBERS_ONLY if numbers_only else GAFFER_FOLLOWS}"])
    return "\n\n".join("\n".join(block) for block in blocks if block)


def render_unsure_line(whatif: WhatIf) -> str | None:
    """Which solve stopped on its time limit, or None when both were proven."""
    sides = (whatif.on.path.proven, whatif.off.path.proven)
    if all(sides):
        return None
    return UNSURE.format(which=UNSURE_SIDES[sides])


def _header(whatif: WhatIf) -> str:
    band = BAND_WORDS[whatif.band]
    if whatif.band == MARGINAL and whatif.lean is not None:
        band = LEAN.format(band=band, lean=whatif.lean)
    return HEADER.format(
        chip=CHIP_NAMES[whatif.kind], gw=whatif.event, band=band,
        net=_signed(whatif.net, 0), margin=whatif.margin,
    )


def _deadline(week: PreparedWeek, now: datetime) -> str:
    deadline = week.effective.event.deadline_time
    left = deadline - now
    if left <= timedelta(0):
        countdown = PASSED
    elif left.days >= 1:
        countdown = COUNTDOWN_DAYS.format(days=left.days, hours=left.seconds // 3600)
    else:
        countdown = COUNTDOWN_HOURS.format(hours=left.seconds // 3600, minutes=(left.seconds % 3600) // 60)
    return DEADLINE_LINE.format(when=_uk(deadline), countdown=countdown)


def _why(whatif: WhatIf, week: PreparedWeek, verdict: dict | None) -> list[str]:
    first, last = whatif.window[0], whatif.window[-1]
    lines = []
    if whatif.kind == FREE_HIT and whatif.fh_terms is not None:
        terms = whatif.fh_terms
        lines.append(WHY_FREE_HIT.format(
            gw=first, one_week=_signed(terms.one_week, 1), next=first + 1, last=last,
            knock=_signed(terms.knock_on, 1), keeping=_signed(-terms.keeping, 1),
            net=_signed(whatif.net, 1),
        ))
        if terms.knock_on < -KNOCK_ON_WORDS:
            lines.append(KNOCK_ON_DOWN.format(knock=-terms.knock_on))
        elif terms.knock_on > KNOCK_ON_WORDS:
            lines.append(KNOCK_ON_UP.format(knock=terms.knock_on))
    else:
        bars = (
            KEEPING.format(bars=whatif.bars_diff)
            if _only_keeping(whatif)
            else CHIP_TIMING.format(bars=_signed(-whatif.bars_diff, 0))
        )
        lines.append(WHY_WILDCARD.format(gain=_signed(whatif.gain, 0), first=first, last=last, bars=bars))
    lines.append(_off_line(whatif, week))
    lines.append(CHIPS_LINE.format(on=_chips(whatif.on.chips), off=_chips(whatif.off.chips)))
    lines.append(GAIN_LINE.format(gains=" ".join(_signed(whatif.weekly_gain[gw], 0) for gw in whatif.window)))
    lines.append(_minutes(whatif, verdict, week.effective.players))
    return lines


def _only_keeping(whatif: WhatIf) -> bool:
    """Is the bars difference only "what keeping it is worth"? Only when the
    off path plays the chip nowhere in the window and both paths play every
    other chip in the same weeks — otherwise it is timing, not keeping."""
    if whatif.kind in whatif.off.chips.values():
        return False
    others_on = {gw: chip for gw, chip in whatif.on.chips.items() if gw != whatif.event}
    return others_on == whatif.off.chips


def _off_line(whatif: WhatIf, week: PreparedWeek) -> str:
    plan = whatif.off.plan
    ft = week.effective.free_transfers or 0
    used = len(plan.transfers_in) - plan.hits
    next_ft = min(MAX_FREE_TRANSFERS, max(0, ft - used) + 1)
    if plan.transfers_in:
        moves = OFF_MOVES.format(
            count=plural(len(plan.transfers_in), "move"),
            pairs=_pairs(plan.transfers_out, plan.transfers_in, week.effective.players),
            ft=next_ft, next=whatif.event + 1,
        )
    else:
        moves = OFF_ROLL.format(ft=next_ft, next=whatif.event + 1)
    later = sorted(gw for gw, chip in whatif.off.chips.items() if chip == whatif.kind)
    where = f"GW{later[0]}" if later else NOT_IN_WINDOW
    return OFF_LINE.format(moves=moves, chip=CHIP_WORDS[whatif.kind], where=where)


def _chips(played: dict[int, str]) -> str:
    if not played:
        return NO_CHIPS
    return ", ".join(f"{CHIP_TAGS.get(chip, chip)} GW{gw}" for gw, chip in sorted(played.items()))


def _minutes(whatif: WhatIf, verdict: dict | None, players: dict) -> str:
    if verdict is None or whatif.minutes_source is None:
        return MINUTES_MODEL
    calls = [
        f"{_name(int(entry['player_id']), players)} {float(entry['expected_minutes']):.0f}"
        for entry in verdict.get("adjustments") or []
    ]
    return MINUTES_FROM.format(source=whatif.minutes_source, calls=", ".join(calls) or MINUTES_NO_CHANGES)


def _squad_block(whatif: WhatIf, week: PreparedWeek) -> list[str]:
    players = week.effective.players
    if whatif.kind == FREE_HIT:
        lineup = whatif.on.lineup
        lines = [PLAY_FREE_HIT]
        for position, tag in POSITION_TAGS.items():
            names = [_name(pid, players) for pid in lineup.xi if players[pid].element_type == position]
            lines.append(f"{tag:<3} {', '.join(names)}")
        lines.append(BENCH_LINE.format(bench=", ".join(_name(pid, players) for pid in lineup.bench)))
        lines.append(ARMBANDS.format(captain=_name(lineup.captain, players), vice=_name(lineup.vice, players)))
        return lines
    plan = whatif.on.plan
    lines = [PLAY_WILDCARD.format(moves=plural(len(plan.transfers_in), "move"))]
    for position, tag in POSITION_TAGS.items():
        outs = [pid for pid in plan.transfers_out if players[pid].element_type == position]
        ins = [pid for pid in plan.transfers_in if players[pid].element_type == position]
        if outs:
            lines.append(f"{tag:<3} {_pairs(outs, ins, players)}")
    lineup = whatif.on.lineup
    lines.append(BANK_LINE.format(
        bank=price(_bank_after(plan, week)),
        captain=_name(lineup.captain, players),
        vice=_name(lineup.vice, players),
        bench=", ".join(_name(pid, players) for pid in lineup.bench),
    ))
    return lines


def _bank_after(plan, week: PreparedWeek) -> int:
    """The bank once the wildcard's moves are made: sales at what they raise
    (the ledger's prices), buys at today's price."""
    players = week.effective.players
    prices = week.prices or {}
    raised = sum(prices.get(pid, players[pid].now_cost) for pid in plan.transfers_out)
    spent = sum(players[pid].now_cost for pid in plan.transfers_in)
    return week.effective.squad.bank + raised - spent


def _flags(whatif: WhatIf, week: PreparedWeek) -> str | None:
    """The doubts in what he would be fielding new: a wildcard's signings, or
    a free hit's whole fifteen (every one of them plays for him this week)."""
    players = week.effective.players
    if whatif.kind == FREE_HIT:
        newcomers = list(whatif.on.path.week1_freehit_squad or [])
        eleven = set(whatif.on.path.week1_freehit_xi or [])
    else:
        newcomers = list(whatif.on.plan.transfers_in)
        eleven = set(whatif.on.lineup.xi)
    flags = []
    for pid in newcomers:
        player = players[pid]
        chance = player.chance_of_playing_next_round
        if chance is not None and chance < 100:
            flags.append(f"{player.web_name} {chance}%")
        elif player.status != "a":
            flags.append(f"{player.web_name} {STATUS_WORDS.get(player.status, player.status)}")
        elif pid in eleven and week.xmins.get(pid, 90.0) < FLAG_MINUTES:
            flags.append(f"{player.web_name} {week.xmins[pid]:.0f} mins")
    return FLAGS.format(flags=" · ".join(flags)) if flags else None


def _anyway(whatif: WhatIf, week: PreparedWeek) -> str:
    if whatif.kind == FREE_HIT:
        squad = set(whatif.on.path.week1_freehit_squad or [])
        return ANYWAY_FREE_HIT.format(changes=len(squad - set(week.effective.squad.player_ids)))
    return ANYWAY_WILDCARD.format(moves=plural(len(whatif.on.plan.transfers_in), "move"))


def _hold(whatif: WhatIf, week: PreparedWeek) -> str | None:
    hold = whatif.hold_week
    if hold is None:
        return None
    provisional = PROVISIONAL if hold.provisional else ""
    if hold.source == "off_path":
        return HOLD_OFF_PATH.format(gw=hold.event, provisional=provisional, value=hold.value or 0.0)
    if hold.source == "calendar":
        return HOLD_CALENDAR.format(gw=hold.event, provisional=provisional, value=hold.value or 0.0)
    chip = _held_chip(whatif, week)
    stop = chip.stop_event if chip is not None else None
    until = _deadline_of(week, stop)
    template = HOLD_PAST_WINDOW if whatif.kind == WILDCARD and hold.value is not None else HOLD_PAST_WINDOW_FREE_HIT
    return template.format(
        first=whatif.window[0], last=whatif.window[-1], stop=stop,
        until=_uk(until) if until is not None else "?", value=hold.value or 0.0,
    )


def _held_chip(whatif: WhatIf, week: PreparedWeek) -> HeldChip | None:
    effective = week.effective
    held = held_in_week(effective.bootstrap, effective.chips_used, effective.event.id, effective.executed)
    return held_for(held, whatif.kind, whatif.event)


def _deadline_of(week: PreparedWeek, gw: int | None) -> datetime | None:
    if gw is None:
        return None
    return next((event.deadline_time for event in week.effective.bootstrap.events if event.id == gw), None)


def _ask_again(week: PreparedWeek) -> str:
    this = week.effective.event
    following = next(
        (event for event in sorted(week.effective.bootstrap.events, key=lambda e: e.id) if event.id > this.id),
        None,
    )
    if following is not None and following.deadline_time - this.deadline_time > timedelta(days=BREAK_DAYS):
        return ASK_AGAIN_BREAK
    return ASK_AGAIN_NEXT.format(next=this.id + 1)


def _differs(whatif: WhatIf, verdict: dict | None, players: dict) -> str | None:
    """Said when the off path's moves this week are not the verdict's — so the
    "off" line never reads as the bot contradicting the report he just read."""
    if verdict is None or verdict.get("event", whatif.event) != whatif.event:
        return None
    off = whatif.off.plan
    if set(off.transfers_in) == set(verdict.get("transfers_in") or []) and set(off.transfers_out) == set(
        verdict.get("transfers_out") or []
    ):
        return None
    ins, outs = verdict.get("transfers_in") or [], verdict.get("transfers_out") or []
    plan = _pairs(outs, ins, players) if ins else ROLL
    return DIFFERS.format(plan=plan)


def _pairs(outs: list[int], ins: list[int], players: dict) -> str:
    """``A → B, C → D``: sales and signings paired off in order, by position
    where the caller grouped them."""
    return ", ".join(f"{_name(out, players)} → {_name(inn, players)}" for out, inn in zip(outs, ins, strict=False))


def _name(pid: int, players: dict) -> str:
    player = players.get(pid)
    return player.web_name if player is not None else f"player {pid}"


def _signed(value: float, places: int) -> str:
    """``+3`` / ``−10`` with a true minus sign, and never ``−0``."""
    rounded = round(value, places)
    if rounded == 0:
        rounded = 0.0
    text = f"{rounded:+.{places}f}"
    return text.replace("-", "−")


def _uk(stamp: datetime) -> str:
    return stamp.astimezone(LONDON).strftime(UK_FORMAT)
