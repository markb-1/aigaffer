"""Message 1 of a chip what-if: the call, the points, the team.

Read on a phone in a few seconds, so only what the owner acts on: the judgement
(PLAY, MAYBE, HOLD, with the week to hold for when there is one), the expected
points change, the deadline, and the lineup and bench the chip would field —
always, even on a Hold, because he may still decide to play it. The why, the
chip calendar, the minutes and the doubts in the numbers belong to the gaffer's
briefing, which already carries them, and his synopsis follows as message 2.

Plain text, one line per line: :func:`aigaffer.report.telegram.send_message`
escapes each line into Telegram HTML exactly as it does the digest, so markup
made here would be escaped twice. Every sentence is a constant, because the
inbox handler's tests read them back.

The owner is in Ireland, so times carry the label ``Irish``. Europe/Dublin has
the same offsets as Europe/London all year, so only the label differs.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.orchestrator import PreparedWeek
from aigaffer.report.render import STATUS_WORDS, phone_lineup, price
from aigaffer.whatif import HOLD, MARGINAL, PLAY, WhatIf

IRISH = ZoneInfo("Europe/Dublin")
IRISH_FORMAT = "%a %d %b %H:%M"
CHIP_NAMES = {WILDCARD: "Wildcard", FREE_HIT: "Free hit"}
# The phone's words. The code's band constant stays "marginal"; the owner
# reads MAYBE.
BAND_WORDS = {PLAY: "PLAY", MARGINAL: "MAYBE", HOLD: "HOLD"}
# Expected minutes under this, in the eleven, is a doubt worth a note.
FLAG_MINUTES = 80.0

HEADER = "🃏 {chip} GW{gw} — {band}"
BAND_WITH_HINT = "{band} ({hint})"
LEANS = "leans {lean}"
GW_LOOKS_BETTER = "GW{gw} looks better"
PROVISIONAL = "provisional"
NOTHING_BEATS_IT = "nothing in GW{first}–{last} beats keeping it"
POINTS_WILDCARD = "{gain} xP over GW{first}–{last} · {net} net of keeping it"
POINTS_FREE_HIT = "{one_week} xP this week vs your XI · {net} net over GW{first}–{last}"
DEADLINE_LINE = "Deadline {when} Irish ({countdown})"
COUNTDOWN_DAYS = "in {days}d {hours}h"
COUNTDOWN_HOURS = "in {hours}h {minutes:02d}m"
PASSED = "passed"
PLAY_WILDCARD = "If you played it — C {captain} · V {vice} · bank {bank}:"
PLAY_FREE_HIT = "If you played it — your free-hit team · C {captain} · V {vice}:"
UNSURE = "Solver unsure: {which} stopped on its time limit — treat the numbers as rough."
UNSURE_SIDES = {
    (False, True): "the solve with it",
    (True, False): "the solve without it",
    (False, False): "both solves",
}
FOOTER = 'Not recorded — if you play it, don\'t send "Transfers made". {last}'
GAFFER_FOLLOWS = "Gaffer's view next."
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

    Four blocks: the call, the points and the deadline (and the unsure line
    when a solve was not proven); the lineup the chip would field; the footer.
    ``now`` is timezone-aware. ``verdict`` (the latest decision record) is
    accepted because the inbox handler passes it, but nothing here reads it
    any more: the minutes it carried are in the gaffer's briefing.
    """
    head = [_header(whatif), _points(whatif), _deadline(week, now)]
    unsure = render_unsure_line(whatif)
    if unsure is not None:
        head.append(unsure)
    footer = FOOTER.format(last=NUMBERS_ONLY if numbers_only else GAFFER_FOLLOWS)
    return "\n\n".join(["\n".join(head), _team(whatif, week), footer])


def render_unsure_line(whatif: WhatIf) -> str | None:
    """Which solve stopped on its time limit, or None when both were proven."""
    sides = (whatif.on.path.proven, whatif.off.path.proven)
    if all(sides):
        return None
    return UNSURE.format(which=UNSURE_SIDES[sides])


def _header(whatif: WhatIf) -> str:
    """``🃏 Wildcard GW7 — HOLD (GW9 looks better)``.

    A Marginal reads ``MAYBE (leans hold)``; when it leans hold, or the band is
    Hold, the week to hold for joins the parenthesis. Play has none.
    """
    band = BAND_WORDS[whatif.band]
    hints = []
    if whatif.band == MARGINAL and whatif.lean is not None:
        hints.append(LEANS.format(lean=whatif.lean))
    if HOLD in (whatif.band, whatif.lean):
        hints.extend(_hold_hint(whatif))
    if hints:
        band = BAND_WITH_HINT.format(band=band, hint=", ".join(hints))
    return HEADER.format(chip=CHIP_NAMES[whatif.kind], gw=whatif.event, band=band)


def _hold_hint(whatif: WhatIf) -> list[str]:
    """What to put in the parenthesis after a Hold: the better week (marked
    provisional when it is far out), or that nothing in the window beats
    keeping the chip. Nothing at all when the numbers found no week."""
    hold = whatif.hold_week
    if hold is None:
        return []
    if hold.event is None:
        return [NOTHING_BEATS_IT.format(first=whatif.window[0], last=whatif.window[-1])]
    hints = [GW_LOOKS_BETTER.format(gw=hold.event)]
    if hold.provisional:
        hints.append(PROVISIONAL)
    return hints


def _points(whatif: WhatIf) -> str:
    """The expected points change, in whole points: ``xP`` after the first
    number only. A free hit leads with its one-week gain against the XI he has
    now; a wildcard with what it buys over the window."""
    first, last = whatif.window[0], whatif.window[-1]
    net = _signed(whatif.net, 0)
    if whatif.kind == FREE_HIT and whatif.fh_terms is not None:
        return POINTS_FREE_HIT.format(
            one_week=_signed(whatif.fh_terms.one_week, 0), net=net, first=first, last=last,
        )
    return POINTS_WILDCARD.format(gain=_signed(whatif.gain, 0), net=net, first=first, last=last)


def _deadline(week: PreparedWeek, now: datetime) -> str:
    deadline = week.effective.event.deadline_time
    left = deadline - now
    if left <= timedelta(0):
        countdown = PASSED
    elif left.days >= 1:
        countdown = COUNTDOWN_DAYS.format(days=left.days, hours=left.seconds // 3600)
    else:
        countdown = COUNTDOWN_HOURS.format(hours=left.seconds // 3600, minutes=(left.seconds % 3600) // 60)
    return DEADLINE_LINE.format(when=_irish(deadline), countdown=countdown)


def _team(whatif: WhatIf, week: PreparedWeek) -> str:
    """The lineup the chip would field, under its own heading.

    :func:`~aigaffer.report.render.phone_lineup` draws the eleven and the
    bench; its first line (the formation heading every other text carries) is
    swapped for the ``If you played it`` line, which names the armbands and,
    for a wildcard, the bank — the formation can be read off the rows. On a
    free hit ``whatif.on.lineup`` is already the free-hit team.
    """
    players = week.effective.players
    lineup = whatif.on.lineup
    captain, vice = _name(lineup.captain, players), _name(lineup.vice, players)
    if whatif.kind == FREE_HIT:
        heading = PLAY_FREE_HIT.format(captain=captain, vice=vice)
    else:
        heading = PLAY_WILDCARD.format(
            captain=captain, vice=vice, bank=price(_bank_after(whatif.on.plan, week)),
        )
    body = phone_lineup(lineup, players, notes=_notes(whatif, week)).split("\n", 1)[1]
    return f"{heading}\n{body}"


def _bank_after(plan, week: PreparedWeek) -> int:
    """The bank once the wildcard's moves are made: sales at what they raise
    (the ledger's prices), buys at today's price."""
    players = week.effective.players
    prices = week.prices or {}
    raised = sum(prices.get(pid, players[pid].now_cost) for pid in plan.transfers_out)
    spent = sum(players[pid].now_cost for pid in plan.transfers_in)
    return week.effective.squad.bank + raised - spent


def _notes(whatif: WhatIf, week: PreparedWeek) -> dict[int, str]:
    """The doubts in what he would be fielding new, one short note a player:
    a wildcard's signings, or a free hit's whole fifteen (every one of them
    plays for him this week). A chance under 100 reads ``75%``; a player not
    available reads his status word; a player in the eleven with under
    :data:`FLAG_MINUTES` expected minutes reads ``60 mins``."""
    players = week.effective.players
    if whatif.kind == FREE_HIT:
        newcomers = list(whatif.on.path.week1_freehit_squad or [])
        eleven = set(whatif.on.path.week1_freehit_xi or [])
    else:
        newcomers = list(whatif.on.plan.transfers_in)
        eleven = set(whatif.on.lineup.xi)
    notes: dict[int, str] = {}
    for pid in newcomers:
        player = players[pid]
        chance = player.chance_of_playing_next_round
        if chance is not None and chance < 100:
            notes[pid] = f"{chance}%"
        elif player.status != "a":
            notes[pid] = STATUS_WORDS.get(player.status, player.status)
        elif pid in eleven and week.xmins.get(pid, 90.0) < FLAG_MINUTES:
            notes[pid] = f"{week.xmins[pid]:.0f} mins"
    return notes


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


def _irish(stamp: datetime) -> str:
    return stamp.astimezone(IRISH).strftime(IRISH_FORMAT)
