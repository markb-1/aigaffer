"""The week's decision, written down.

The same string is committed to the repo as a managerial diary and sent to
Telegram, where nothing is rendered at all — so the markdown stays at the
plain end of the language. Headings and list items, because a run of bare
lines collapses into one paragraph when something does render it, and a
bulleted line reads the same either way. No pipe tables: the alignment row
they need is noise to a reader who only ever sees the source.

Every number is spelled out where it is used rather than left to a column
header — ``1 transfer | 0 hits | 255.5 xP | 255.5 net`` — because a row that
has to be read against something else three lines up is a row that gets
misread on a phone. Prices are turned into pounds here and nowhere else:
they travel as tenths of a million right up to the moment they are printed.

The renderer decides nothing. It is handed the plans, the choice among them,
the eleven and the chip numbers, and it says what they are.

:func:`deadline`, :func:`price` and :func:`plural` are public because they are
the house vocabulary rather than this module's private business: the manager's
briefing (:mod:`aigaffer.manager.briefing`) says the same things to a different
reader and must say them the same way. The deadline especially — reading a
naive timestamp as the runner's local clock is a mistake worth making once.
"""

from collections import defaultdict
from datetime import UTC
from typing import TYPE_CHECKING

from aigaffer.data.models import Bootstrap, Event, Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.lineup import ChipEvs, Lineup
from aigaffer.solver.optimizer import (
    DEFENDER,
    FORWARD,
    GOALKEEPER,
    HIT_POINTS,
    MIDFIELDER,
    Plan,
    projected_points,
)

if TYPE_CHECKING:  # the manager imports this module, so never the reverse
    from aigaffer.manager.agent import ManagerDecision

POSITIONS = {GOALKEEPER: "GKP", DEFENDER: "DEF", MIDFIELDER: "MID", FORWARD: "FWD"}
OUTFIELD = (DEFENDER, MIDFIELDER, FORWARD)

WATCHLIST_SIZE = 5
DEADLINE_FORMAT = "%a %d %b %Y %H:%M UTC"

# The manager's own vocabulary, repeated here rather than imported: the manager
# package imports this module for the house formatting, so the dependency runs
# one way only and a report that reached back into it would close the circle.
# ``ManagerDecision.source`` is either this word or ``solver-fallback: reason``,
# and ``ManagerDecision.chip`` is either this word or a chip.
DECIDED = "manager"
NO_CHIP = "none"

# The two headings the minute lists carry. An adjustment only reaches the
# projections through a re-solve, so the second list is what he wrote down and
# never spent — a note about the week, not a part of the decision under it.
OVERRULED = "Minutes he overruled:"
NOT_APPLIED = "Noted but not applied (no re-solve followed):"

# What the chip panel is measuring, said once above it. The wildcard is the odd
# one out and carries its own units on its own line.
CHIP_UNITS = (
    "Points each chip would add this gameweek — except the wildcard, which is"
    " priced over the whole horizon, because that is what a wildcard buys."
)

# What the shortlist says when the manager's own plan is not on it: he adjusted
# somebody's minutes and solved again, and what came back was a fifteen this
# list never reached.
RESOLVED_ELSEWHERE = (
    "The gaffer re-solved after his adjustments;"
    " his pick above is not on this list."
)


def render_report(
    mode: str,
    event: Event,
    plans: list[Plan],
    choice: Plan,
    lineup: Lineup,
    chips: ChipEvs,
    bootstrap: Bootstrap,
    projections: dict[int, PlayerProjection],
    gaffer: "ManagerDecision | None" = None,
) -> str:
    """The whole report as one markdown string. Pure; no I/O.

    ``choice`` is the plan being recommended and is expected to be one of
    ``plans`` — it is flagged there by identity. Every player id in the plans
    and the lineup must be a ``bootstrap`` element, which it is: they came
    from there. The string ends in a newline, because it is written out as a
    file as well as sent as a message.

    ``gaffer`` is the manager's decision, when there was a manager: the plan
    and the eleven above are his by then, and the section this adds is the
    half of a decision that is words rather than numbers — why, what he
    overruled, and whose pick this actually is. Omitted, the report is byte
    for byte the one this wrote before there was a manager at all.
    """
    players = {player.id: player for player in bootstrap.elements}
    clubs = {team.id: team.short_name for team in bootstrap.teams}

    sections = [
        _header(mode, event),
        _recommendation(choice, players, clubs),
        *([] if gaffer is None else [_gaffer(gaffer, players)]),
        _team_sheet(lineup, players, projections),
        _candidates(plans, choice, decided=gaffer is not None),
        _chip_panel(chips, horizon_of(projections)),
        _watchlist(choice.squad, players, clubs, projections),
    ]
    return "\n\n".join(sections) + "\n"


def _header(mode: str, event: Event) -> str:
    return f"# AI Gaffer — GW{event.id} {mode}\n\nDeadline: {deadline(event)}"


def deadline(event: Event) -> str:
    """The deadline in UTC, whatever clock it arrived on.

    A naive datetime is read as UTC rather than handed to ``astimezone``,
    which would take it for the runner's local time and quietly move the
    deadline by however many hours that machine happens to be out.
    """
    stamp = event.deadline_time
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC).strftime(DEADLINE_FORMAT)


def _recommendation(
    choice: Plan, players: dict[int, Player], clubs: dict[int, str]
) -> str:
    """What to do, and what it costs.

    An opening draft buys fifteen and sells nobody, and it is the only plan
    that does: every other week swaps like for like. So the sales bullet is
    omitted when there are none rather than printed with nothing after it,
    which is how a report that has lost half its own list reads.
    """
    lines = ["## Recommendation", ""]
    if not choice.transfers_in and not choice.transfers_out:
        lines.append("Roll the transfer.")
        return "\n".join(lines)

    moves = plural(len(choice.transfers_in), "transfer")
    cost = f"-{choice.hits * HIT_POINTS} pts in hits" if choice.hits else "no hit"
    lines += [f"{moves}, {cost}.", ""]
    if choice.transfers_out:
        lines.append("- Out: " + _listed(choice.transfers_out, players, clubs))
    lines.append("- In: " + _listed(choice.transfers_in, players, clubs))
    return "\n".join(lines)


def _gaffer(gaffer: "ManagerDecision", players: dict[int, Player]) -> str:
    """The manager's week in his own words, and whose week it actually is.

    Five things, in the order they are worth reading: what he decided and why,
    the chip he is spending if he is spending one, the minutes he overruled to
    get there, the minutes he only wrote down, and — last, because it qualifies
    everything above it — how much research it cost and whether the manager
    reached a decision at all. A failed conversation prints this section too: a
    report that quietly reverts to the solver is a report that has told the
    reader something untrue about where its recommendation came from.

    The two minute lists are not one list. An adjustment reaches the numbers
    only through a re-solve, so anything he said after the last one — or never
    re-solved on at all — is a note beside this decision rather than a part of
    it, and printing the two together would credit the recommendation with a
    minutes model it was never costed on.
    """
    lines = ["## The Gaffer's view", "", _prose(gaffer.rationale)]

    if gaffer.chip != NO_CHIP:
        spoken = gaffer.chip.replace("_", " ")
        lines += ["", f"Playing the {spoken}. {_prose(gaffer.chip_justification)}"]

    applied = _settled(gaffer.adjustments)
    if applied:
        lines += ["", OVERRULED, ""]
        lines += [
            f"- Set {_who(record['player_id'], players)}"
            f" to {record['expected_minutes']:.0f} mins"
            f" — {_one_line(record['reason'])}"
            for record in applied
        ]

    noted = _noted(gaffer.unapplied, applied)
    if noted:
        lines += ["", NOT_APPLIED, ""]
        lines += [
            f"- {_who(record['player_id'], players)}"
            f" at {record['expected_minutes']:.0f} mins"
            f" — {_one_line(record['reason'])}"
            for record in noted
        ]

    lines += ["", f"{_searches(gaffer.searches)}. {_source(gaffer.source)}"]
    return "\n".join(lines)


def _searches(count: int) -> str:
    """``1 web search``, ``3 web searches``. The house :func:`plural` adds an
    s and would make it "web searchs"; a noun that pluralizes differently is
    spelled out where it is used rather than taught to the vocabulary."""
    return f"{count} web search" if count == 1 else f"{count} web searches"


def _noted(unapplied: list[dict], applied: list[dict]) -> list[dict]:
    """The minutes that changed nothing, one line per player.

    A player who was adjusted, re-solved on, and then adjusted again is not a
    player the solver ignored: his last word is in ``applied`` and the earlier
    one is a superseded line of the record, so he is listed once, above. What
    is left here is the players whose latest number never reached a re-solve at
    all, which is the only thing the heading claims.
    """
    settled = {record["player_id"] for record in applied}
    return [
        record
        for record in _settled(unapplied)
        if record["player_id"] not in settled
    ]


def _settled(adjustments: list[dict]) -> list[dict]:
    """One entry per player: the last thing he said about each of them.

    The manager's record keeps every adjustment he made, in the order he made
    them, superseded ones included — it is the conversation's own history and
    the store keeps it whole. What the projection actually used, though, is
    the last number set for each player, and that is what a report claiming
    "he set Gale to 30 minutes" has to say. Each player keeps the place he
    first appears in, because that is the order he thought about them in.
    """
    settled: dict[int, dict] = {}
    for record in adjustments:
        settled[record["player_id"]] = record
    return list(settled.values())


def _who(pid: int, players: dict[int, Player]) -> str:
    """His name, or his id if this board has never heard of him. Nothing here
    is worth losing a written report over."""
    player = players.get(pid)
    return player.web_name if player is not None else f"player {pid}"


def _source(source: str) -> str:
    """Whose pick this is, said plainly.

    ``source`` is ``manager`` or ``solver-fallback: <reason>``, and the reason
    is a class name or a phrase of ours — never an exception's own words,
    which can carry a key or a token.
    """
    if source == DECIDED:
        return "Decided by the gaffer."
    reason = source.partition(": ")[2] or source
    return f"The gaffer was unavailable ({reason}); this is the solver's pick."


def _prose(text: str) -> str:
    """A paragraph of his, kept out of the report's own structure.

    The rationale and the argument for a chip are written by a model that has
    just spent the afternoon reading whatever the web served it, and they land
    in a document whose sections are carried by lines beginning with ``##`` —
    a document that is committed to the repo and read on a phone. A forged
    heading here would fool no parser, because nothing parses this; it would
    fool a reader, which is the worse of the two.

    So a line that opens a section is pushed off the margin and reads as the
    sentence it is. The line breaks are kept: this is prose, and a paragraph
    flattened into one line is a paragraph nobody finishes.
    """
    return "\n".join(
        f" {line}" if line.lstrip().startswith("#") else line
        for line in str(text).splitlines()
    )


def _one_line(text: str) -> str:
    """Somebody else's sentence, flattened onto the line it was given.

    The reason is the model's own prose and it goes onto a list item. A
    newline in it ends the list; a newline and two hashes start a section that
    the solver never produced, in a document that is committed to the repo and
    sent to a phone.
    """
    return " ".join(str(text).split())


def _team_sheet(
    lineup: Lineup,
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
) -> str:
    """The eleven a row to a position, then the bench in the order it is read.

    Within a row the best projected total goes first — the same number the
    watchlist ranks on, and near enough to the next gameweek alone that the
    armbands were chosen on. It is a way of laying out an eleven that is
    already picked, not a second opinion about it: the bench keeps the order
    it was given, because that order is a substitution list.
    """
    rows: dict[int, list[str]] = defaultdict(list)
    for pid in _ranked(lineup.xi, players, projections):
        rows[players[pid].element_type].append(_armband(pid, lineup, players))

    formation = "-".join(str(len(rows[position])) for position in OUTFIELD)
    lines = [f"## Starting XI ({formation})", ""]
    lines += [
        f"- {label}: " + ", ".join(rows[position])
        for position, label in POSITIONS.items()
    ]
    lines.append(
        "- Bench: "
        + ", ".join(
            f"{order}. {players[pid].web_name} ({POSITIONS[players[pid].element_type]})"
            for order, pid in enumerate(lineup.bench, start=1)
        )
    )
    return "\n".join(lines)


def _candidates(plans: list[Plan], choice: Plan, decided: bool = False) -> str:
    """Every plan the solver came back with, in the order it ranked them.

    Exactly one row carries the recommendation, and that has to hold however
    the recommendation was arrived at. A manager who re-solved on his own
    minutes finalizes a plan object the solver minted after this list was
    drawn up, so the row is found by the squad it leaves behind and not by
    identity — the same fifteen is the same plan, whatever the numbers beside
    it were recomputed to.

    ``decided`` says a manager chose. When his choice is genuinely not on this
    list — a squad the original solve never reached — no row is flagged and
    the section says why, because a list with no recommendation on it and a
    recommendation above it with no list behind it is a report that has
    stopped explaining itself.
    """
    pick = _pick(plans, choice)
    lines = ["## Candidate plans", ""]
    for index, plan in enumerate(plans):
        recommended = "  <- recommended" if index == pick else ""
        lines.append(
            f"- {plural(len(plan.transfers_in), 'transfer')}"
            f" | {plural(plan.hits, 'hit')}"
            f" | {plan.xp_total:.1f} xP"
            f" | {plan.objective:.1f} net{recommended}"
        )
    if decided and pick is None:
        lines += ["", RESOLVED_ELSEWHERE]
    return "\n".join(lines)


def _pick(plans: list[Plan], choice: Plan) -> int | None:
    """Which row is the recommendation, if any of them is.

    Identity first, because in a run with no manager the choice is one of
    these objects and two plans could in principle field the same fifteen by
    different routes. Then the squad, which is what makes a re-solved plan the
    same plan as the one this list already has.
    """
    for index, plan in enumerate(plans):
        if plan is choice:
            return index
    squad = sorted(choice.squad)
    for index, plan in enumerate(plans):
        if sorted(plan.squad) == squad:
            return index
    return None


def _chip_panel(chips: ChipEvs, horizon: int) -> str:
    """The chip numbers, signed: a chip can be worth less than nothing.

    Three of the four are next gameweek's: the bench that would have scored,
    the captain counted once more, the eleven a free hit would field instead.
    The wildcard is not, and never was — it is priced as the difference between
    two decayed horizon totals, because a wildcard is bought for the run of
    fixtures rather than for Saturday — so its row says which number it is.
    Four figures under one heading, one of them measuring something else, is
    how a chip gets played on a comparison nobody made.
    """
    panel = {
        "Bench boost": chips.bench_boost,
        "Triple captain": chips.triple_captain,
        "Free hit": chips.free_hit,
    }
    lines = ["## Chip EV", "", CHIP_UNITS, ""]
    lines += [f"- {label}: {points:+.1f}" for label, points in panel.items()]
    lines.append(f"- Wildcard: {wildcard_ev(chips.wildcard, horizon)}")
    return "\n".join(lines)


def wildcard_ev(points: float, horizon: int) -> str:
    """``+12.0 xP over 6 GWs (horizon)`` — the wildcard row, in its own units.

    Public because the manager's briefing prints the same panel to a different
    reader, and a number that means one thing in the report and another in the
    briefing is worse than a number nobody prints.
    """
    return f"{points:+.1f} xP over {horizon} GWs (horizon)"


def horizon_of(projections: dict[int, PlayerProjection]) -> int:
    """How many gameweeks the projections cover, for the labels that say so."""
    return max((len(p.per_gw) for p in projections.values()), default=0)


def _watchlist(
    squad: list[int],
    players: dict[int, Player],
    clubs: dict[int, str],
    projections: dict[int, PlayerProjection],
) -> str:
    """The best projections we do not own — the shortlist for next week."""
    owned = set(squad)
    field = [pid for pid in players if pid not in owned]
    lines = ["## Watchlist", ""]
    lines += [
        f"- {_described(pid, players, clubs)}"
        f" — {projected_points(projections, pid):.1f} xP"
        for pid in _ranked(field, players, projections)[:WATCHLIST_SIZE]
    ]
    return "\n".join(lines)


def _ranked(
    pids: list[int],
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
) -> list[int]:
    """Best projected first, ties broken by name so the order never wobbles."""
    return sorted(
        pids,
        key=lambda pid: (-projected_points(projections, pid), players[pid].web_name),
    )


def _armband(pid: int, lineup: Lineup, players: dict[int, Player]) -> str:
    marker = {lineup.captain: " (C)", lineup.vice: " (V)"}.get(pid, "")
    return f"{players[pid].web_name}{marker}"


def _listed(pids: list[int], players: dict[int, Player], clubs: dict[int, str]) -> str:
    return ", ".join(_described(pid, players, clubs) for pid in pids)


def _described(pid: int, players: dict[int, Player], clubs: dict[int, str]) -> str:
    """``Reid (FWD, CRV, £9.5m)`` — who he is, in one parenthesis."""
    player = players[pid]
    position = POSITIONS[player.element_type]
    club = clubs[player.team]
    return f"{player.web_name} ({position}, {club}, {price(player.now_cost)})"


def price(now_cost: int) -> str:
    """Tenths of a million as a manager reads them: 55 is £5.5m."""
    return f"£{now_cost / 10:.1f}m"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
