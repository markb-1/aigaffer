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
        _candidates(plans, choice),
        _chip_panel(chips),
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
    """What to do, and what it costs."""
    lines = ["## Recommendation", ""]
    if not choice.transfers_in and not choice.transfers_out:
        lines.append("Roll the transfer.")
        return "\n".join(lines)

    moves = plural(len(choice.transfers_in), "transfer")
    cost = f"-{choice.hits * HIT_POINTS} pts in hits" if choice.hits else "no hit"
    lines += [
        f"{moves}, {cost}.",
        "",
        "- Out: " + _listed(choice.transfers_out, players, clubs),
        "- In: " + _listed(choice.transfers_in, players, clubs),
    ]
    return "\n".join(lines)


def _gaffer(gaffer: "ManagerDecision", players: dict[int, Player]) -> str:
    """The manager's week in his own words, and whose week it actually is.

    Four things, in the order they are worth reading: what he decided and why,
    the chip he is spending if he is spending one, the minutes he overruled to
    get there, and — last, because it qualifies everything above it — how much
    research it cost and whether the manager reached a decision at all. A
    failed conversation prints this section too: a report that quietly reverts
    to the solver is a report that has told the reader something untrue about
    where its recommendation came from.
    """
    lines = ["## The Gaffer's view", "", _prose(gaffer.rationale)]

    if gaffer.chip != NO_CHIP:
        spoken = gaffer.chip.replace("_", " ")
        lines += ["", f"Playing the {spoken}. {_prose(gaffer.chip_justification)}"]

    adjustments = _settled(gaffer.adjustments)
    if adjustments:
        lines += ["", "Minutes he overruled:", ""]
        lines += [
            f"- Set {_who(record['player_id'], players)}"
            f" to {record['expected_minutes']:.0f} mins"
            f" — {_one_line(record['reason'])}"
            for record in adjustments
        ]

    lines += ["", f"{_searches(gaffer.searches)}. {_source(gaffer.source)}"]
    return "\n".join(lines)


def _searches(count: int) -> str:
    """``1 web search``, ``3 web searches``. The house :func:`plural` adds an
    s and would make it "web searchs"; a noun that pluralizes differently is
    spelled out where it is used rather than taught to the vocabulary."""
    return f"{count} web search" if count == 1 else f"{count} web searches"


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


def _candidates(plans: list[Plan], choice: Plan) -> str:
    """Every plan the solver came back with, in the order it ranked them."""
    lines = ["## Candidate plans", ""]
    for plan in plans:
        recommended = "  <- recommended" if plan is choice else ""
        lines.append(
            f"- {plural(len(plan.transfers_in), 'transfer')}"
            f" | {plural(plan.hits, 'hit')}"
            f" | {plan.xp_total:.1f} xP"
            f" | {plan.objective:.1f} net{recommended}"
        )
    return "\n".join(lines)


def _chip_panel(chips: ChipEvs) -> str:
    """The chip numbers, signed: a chip can be worth less than nothing."""
    panel = {
        "Bench boost": chips.bench_boost,
        "Triple captain": chips.triple_captain,
        "Free hit": chips.free_hit,
        "Wildcard": chips.wildcard,
    }
    lines = ["## Chip EV", "", "Points each chip would add this gameweek.", ""]
    lines += [f"- {label}: {points:+.1f}" for label, points in panel.items()]
    return "\n".join(lines)


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
