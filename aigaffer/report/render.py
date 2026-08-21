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
"""

from collections import defaultdict
from datetime import UTC

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

POSITIONS = {GOALKEEPER: "GKP", DEFENDER: "DEF", MIDFIELDER: "MID", FORWARD: "FWD"}
OUTFIELD = (DEFENDER, MIDFIELDER, FORWARD)

WATCHLIST_SIZE = 5
DEADLINE_FORMAT = "%a %d %b %Y %H:%M UTC"


def render_report(
    mode: str,
    event: Event,
    plans: list[Plan],
    choice: Plan,
    lineup: Lineup,
    chips: ChipEvs,
    bootstrap: Bootstrap,
    projections: dict[int, PlayerProjection],
) -> str:
    """The whole report as one markdown string. Pure; no I/O.

    ``choice`` is the plan being recommended and is expected to be one of
    ``plans`` — it is flagged there by identity. Every player id in the plans
    and the lineup must be a ``bootstrap`` element, which it is: they came
    from there. The string ends in a newline, because it is written out as a
    file as well as sent as a message.
    """
    players = {player.id: player for player in bootstrap.elements}
    clubs = {team.id: team.short_name for team in bootstrap.teams}

    sections = [
        _header(mode, event),
        _recommendation(choice, players, clubs),
        _team_sheet(lineup, players, projections),
        _candidates(plans, choice),
        _chip_panel(chips),
        _watchlist(choice.squad, players, clubs, projections),
    ]
    return "\n\n".join(sections) + "\n"


def _header(mode: str, event: Event) -> str:
    return f"# AI Gaffer — GW{event.id} {mode}\n\nDeadline: {_deadline(event)}"


def _deadline(event: Event) -> str:
    """The deadline in UTC, whatever clock it arrived on.

    A naive datetime is read as UTC rather than handed to ``astimezone``,
    which would take it for the runner's local time and quietly move the
    deadline by however many hours that machine happens to be out.
    """
    deadline = event.deadline_time
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return deadline.astimezone(UTC).strftime(DEADLINE_FORMAT)


def _recommendation(
    choice: Plan, players: dict[int, Player], clubs: dict[int, str]
) -> str:
    """What to do, and what it costs."""
    lines = ["## Recommendation", ""]
    if not choice.transfers_in and not choice.transfers_out:
        lines.append("Roll the transfer.")
        return "\n".join(lines)

    moves = _plural(len(choice.transfers_in), "transfer")
    cost = f"-{choice.hits * HIT_POINTS} pts in hits" if choice.hits else "no hit"
    lines += [
        f"{moves}, {cost}.",
        "",
        "- Out: " + _listed(choice.transfers_out, players, clubs),
        "- In: " + _listed(choice.transfers_in, players, clubs),
    ]
    return "\n".join(lines)


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
            f"- {_plural(len(plan.transfers_in), 'transfer')}"
            f" | {_plural(plan.hits, 'hit')}"
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
    price = _price(player.now_cost)
    return f"{player.web_name} ({position}, {clubs[player.team]}, {price})"


def _price(now_cost: int) -> str:
    """Tenths of a million as a manager reads them: 55 is £5.5m."""
    return f"£{now_cost / 10:.1f}m"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
