"""The week's position, written for the manager rather than for Mark.

The report (:mod:`aigaffer.report.render`) is read on a phone by someone who
already knows his own squad. This is read by a model that knows nothing at
all and is about to spend web searches on it, so the two documents pull in
opposite directions: the report leaves out what a manager can supply from
memory, and the briefing spells out everything and repeats itself where
repeating is cheaper than a wrong inference.

Three things follow from that.

* **Every player carries his id.** The agent's only way to name a player back
  to us is ``adjust_players``, which takes an integer, so a line that names a
  player without his id is a line he cannot act on.
* **Every plan carries the id it will be finalized by, and the caller sets
  it.** :func:`format_plans` is public and takes ``(id, plan)`` pairs because
  a re-solve mid-conversation produces a fresh shortlist that must be
  presented with *continuing* ids — plan 3 said once must never mean a
  different plan later. Deriving the id from a list position is exactly the
  bug that would cause.
* **Numbers are labelled where they are used.** Two projections appear on
  most lines — next gameweek alone, and the decayed horizon total the solver
  actually maximises — and a model that confuses them will argue for the
  wrong transfer with great confidence.
* **The one number he can change is shown to him.** ``adjust_players`` sets a
  player's expected minutes absolutely, so a briefing that hides what the
  model currently assumes is asking him to overwrite a number he cannot see.
  Pass ``xmins`` and every player he may adjust carries it.

Nothing here decides anything, and nothing here talks to an API. The house
formatting vocabulary — the price, the deadline, the plural — is imported
from the report renderer rather than written twice, so the two documents can
never come to disagree about what £4.0m or a naive deadline means.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from aigaffer.data.models import Player
from aigaffer.manager.tools import played_chips
from aigaffer.model.xp import PlayerProjection
from aigaffer.report.render import (
    POSITIONS,
    deadline,
    horizon_of,
    plural,
    price,
    wildcard_ev,
)
from aigaffer.solver.lineup import ChipEvs, Lineup
from aigaffer.solver.optimizer import AVAILABLE, Plan, projected_points

if TYPE_CHECKING:  # the orchestrator imports the manager, so never the reverse
    from aigaffer.orchestrator import PipelineInputs, SolveResult

# What the API's one-letter status codes mean. Everything but "a" is shouted,
# because a squad list is fifteen lines long and the one that matters is the
# one the model must not skim past.
STATUS_FLAGS = {
    "a": "fit",
    "d": "DOUBTFUL",
    "i": "INJURED",
    "s": "SUSPENDED",
    "u": "UNAVAILABLE",
    "n": "INELIGIBLE",
}
FULLY_FIT = 100

# Twice the report's five: the report is a recommendation and this is research
# material, and the field is what tells the manager whether the solver's
# shortlist was drawn from a sane pool.
WATCHLIST_SIZE = 10

# The research list is thirty names of nothing but names, so it is wrapped
# rather than given thirty lines it would not earn.
RELEVANT_PER_LINE = 6

DATE_FORMAT = "%a %d %b %Y"

PICK_MARKER = "  <- solver pick"

# What a chip already spent this season is marked with, and the line that says
# what the mark means. A chip is played once, so the panel has to distinguish
# "worth nothing" from "not yours to play".
PLAYED_MARK = " (already played)"
PLAYED_GUARD = (
    "A chip marked (already played) has been used this season and is gone,"
    " whatever it is priced at above. Finalize with chip 'none' instead."
)

# What the panel is measuring, and which of it he may actually play. The
# wildcard is priced over the horizon and the other three over one gameweek, so
# the four numbers are not comparable and the line above them says so; and two
# of the four are refused by finalize_decision, so the panel says that too
# rather than leaving him to meet the rule as an error.
CHIP_UNITS = (
    "Points each chip would add next gameweek — except the wildcard, which is a"
    " horizon number: the decayed total of the best squad fifteen free"
    " transfers could reach, against the plan in hand. It is not comparable"
    " with the three above it."
)
PLANNED_GUARD = (
    "Only bench_boost and triple_captain can be finalized: they are played on"
    " the team a plan already fields. A wildcard or a free hit is a different"
    " squad, and no plan on this board was solved for one, so"
    " finalize_decision refuses them. If you think this is the week for one,"
    " finalize with chip 'none' and argue for it in your rationale."
)

# The longest a name, club or status out of the payload may be before it is
# cut short. Real ones are a dozen characters; the cap is not about tidiness
# but about a single field being unable to swamp the document it sits in.
MAX_FIELD = 60


@dataclass(frozen=True)
class _Board:
    """The lookups every line needs, carried together rather than threaded.

    ``horizon`` is how many gameweeks the projections cover; it is read off
    them rather than passed in, because the number's only job is to label the
    column it came from. ``xmins`` is None when the caller did not supply the
    minutes, and then the minutes column does not exist at all — not a column
    of blanks, which would be a column of questions nobody asked.
    """

    players: dict[int, Player]
    clubs: dict[int, str]
    projections: dict[int, PlayerProjection]
    horizon: int
    xmins: dict[int, float] | None = None


def build_briefing(
    inputs: "PipelineInputs",
    solve: "SolveResult",
    projections: dict[int, PlayerProjection],
    free_transfers: int | None,
    today: date | None = None,
    xmins: dict[int, float] | None = None,
) -> str:
    """The whole briefing as one plain-text string. Pure; no I/O.

    ``free_transfers`` is an argument rather than read off ``inputs`` so that
    a caller who has already spent a move can say so. ``today`` is an argument
    for the same reason the deadline is printed at all: the manager's job
    includes deciding whether a piece of team news is stale, and he cannot do
    it without knowing what day it is. It defaults to today, which is what
    every real caller means.

    ``xmins`` is the expected minutes the projections were built on — the
    first half of what :func:`aigaffer.orchestrator.build_projections` returns.
    Supplied, it appears against every player the manager may adjust, because
    ``adjust_players`` sets that number absolutely and he cannot sensibly
    overwrite what he was never shown. Omitted, the briefing is byte for byte
    what it was without it.

    A draft — a run with no squad — is a different question and says so in
    its title: fifteen players from nothing, no bank to spend, no free
    transfers to weigh and no chips to price.
    """
    board = _Board(
        players=inputs.players,
        clubs={team.id: team.short_name for team in inputs.bootstrap.teams},
        projections=projections,
        horizon=horizon_of(projections),
        xmins=xmins,
    )
    # A draft has no squad to list, so it lists the fifteen it just drafted.
    held = solve.choice.squad if inputs.squad is None else inputs.squad.player_ids
    numbered = initial_plan_ids(solve)
    pick = next((pid for pid, plan in numbered if plan is solve.choice), None)
    event = inputs.event.id

    sections = [
        _situation(
            inputs, board, held, free_transfers, today or date.today(), solve.draft_mode
        ),
        _squad(held, board, event, solve.draft_mode),
        _team_sheet(solve.lineup, board, event, pick),
        _candidates(numbered, board, pick),
        _chip_panel(
            solve.chips,
            solve.draft_mode,
            played_chips(inputs.chips_used),
            board.horizon,
        ),
        _watchlist(held, board, event),
        _relevant(solve, board),
    ]
    return "\n\n".join(sections)


def format_plans(
    plans: list[tuple[int, Plan]],
    players: dict[int, Player],
    clubs: dict[int, str],
    projections: dict[int, PlayerProjection],
    recommended: int | None = None,
) -> str:
    """The candidate plans, one line each, numbered by the ids given.

    ``plans`` is ``(id, plan)`` pairs and the ids come from the caller: the
    agent finalizes on one of them, and the manager loop keeps a registry that
    goes on counting across re-solves. Nothing here may infer an id from a
    position in the list. The lines come back in the order given, which is the
    order the solver ranked them.

    ``clubs`` maps team id to short name — ``{t.id: t.short_name for t in
    bootstrap.teams}``. ``recommended`` is the id the solver itself would pick,
    flagged on its line; None flags nothing, which is what a re-solve wants
    when it is presenting alternatives rather than a recommendation.

    Each plan is measured against the one in the same set that moves fewest
    players — the roll, in any ordinary shortlist — because the number that
    decides a transfer is what it gains over doing nothing, and two totals
    that agree to within a point of each other are exactly the subtraction a
    model gets wrong once in a while. The baseline is named on every line
    rather than assumed to be plan 0, so a shortlist that reaches no roll at
    all still reads honestly.
    """
    if not plans:
        return ""

    board = _Board(players, clubs, projections, horizon_of(projections))
    baseline_id, baseline = min(
        plans, key=lambda pair: (len(pair[1].transfers_in), pair[0])
    )

    lines = []
    for plan_id, plan in plans:
        against = (
            "baseline"
            if plan_id == baseline_id
            else f"{plan.objective - baseline.objective:+.1f} net vs plan {baseline_id}"
        )
        lines.append(
            f"plan {plan_id}: {_moves(plan, board)}"
            f" | {plan.xp_total:.1f} xP"
            f" | {plural(plan.hits, 'hit')}"
            f" | {plan.objective:.1f} net"
            f" | {against}"
            f"{PICK_MARKER if plan_id == recommended else ''}"
        )
    return "\n".join(lines)


def initial_plan_ids(solve: "SolveResult") -> list[tuple[int, Plan]]:
    """The solver's shortlist, numbered from zero in the order it ranked them.

    The one place the opening ids are minted. :func:`build_briefing` presents
    these, the manager loop seeds its registry from them, and a re-solve goes
    on counting from ``len(...)`` — so if this convention lives in two places
    they will eventually disagree, and plan 1 will mean one thing in the
    briefing and another in the registry that validates ``finalize_decision``.
    It is a one-line function precisely so that nobody writes the line twice.
    """
    return list(enumerate(solve.plans))


def relevant_players(solve: "SolveResult") -> list[int]:
    """Everyone any plan holds, buys or sells — the research list, sorted.

    It is the whole of what the manager is asked to look up: about thirty
    players, against a board of six hundred. A player nobody could end up
    owning is a web search spent on nothing, and a transfer candidate from a
    plan the solver ranked third is still a player who might have been ruled
    out an hour ago.
    """
    pool: set[int] = set()
    for plan in solve.plans:
        pool.update(plan.squad, plan.transfers_in, plan.transfers_out)
    return sorted(pool)


def _situation(
    inputs: "PipelineInputs",
    board: _Board,
    held: list[int],
    free_transfers: int | None,
    today: date,
    drafting: bool,
) -> str:
    """Which gameweek, which deadline, what money, and how to read the columns."""
    event = inputs.event
    draft_label = " (initial squad draft)" if drafting else ""
    value = price(sum(board.players[pid].now_cost for pid in held))

    if inputs.squad is None:
        money = (
            "No squad yet: this is fifteen players from nothing,"
            f" not a transfer decision. Drafted value: {value}"
        )
    else:
        moves = "unknown" if free_transfers is None else free_transfers
        money = (
            f"Bank: {price(inputs.squad.bank)}"
            f" | Free transfers: {moves}"
            f" | Squad value: {value}"
        )

    # The minutes clause exists only when the minutes do: a briefing built
    # without them must read exactly as it read before they were an option.
    minutes = (
        ""
        if board.xmins is None
        else (
            ' "xMins" is the expected minutes the projection was built on, and'
            " the one number adjust_players overwrites;"
        )
    )

    return "\n".join(
        [
            f"# AI Gaffer — manager briefing: GW{event.id}{draft_label}",
            "",
            f"Today: {today.strftime(DATE_FORMAT)}",
            f"Deadline: {deadline(event)}",
            money,
            f"Numbers:{minutes}"
            f' "xP GW{event.id}" is next gameweek alone;'
            f' "xP{board.horizon}" is the decayed {board.horizon}-gameweek'
            " total the solver maximises.",
        ]
    )


def _squad(held: list[int], board: _Board, event: int, drafting: bool) -> str:
    """The fifteen, a line each, by position and then by next gameweek.

    Every player carries a status, "fit" included: a blank would read as "not
    looked up" to a model whose whole job is deciding what still needs looking
    up. The transfer lines in the plans below do the opposite and stay silent
    when there is nothing to report, because there a status is news.
    """
    heading = "## Drafted fifteen" if drafting else "## Current squad"
    lines = [heading, ""]
    lines += [
        f"- {_player_line(pid, board, event)}"
        for pid in _by_position(held, board, event)
    ]
    return "\n".join(lines)


def _team_sheet(lineup: Lineup, board: _Board, event: int, pick: int | None) -> str:
    """The eleven the solver's own choice would field, and its bench.

    The manager names a captain and a vice, and they have to come from the XI
    of the plan he finalizes — but this is one plan's XI, not every plan's, so
    the section says which plan it belongs to and what happens if he picks a
    different one. Told flatly that the captain must come from the eleven
    below, he would read it as a rule about all of them and name a captain
    the plan he chose does not own.

    It is ranked on next gameweek alone — the week the armband is worn — and
    the bench keeps the order it was given, because that order is a
    substitution list.
    """
    whose = "the recommended plan" if pick is None else f"plan {pick}"
    rows: dict[int, list[str]] = defaultdict(list)
    for pid in _ranked(lineup.xi, board, event):
        rows[board.players[pid].element_type].append(_named(pid, board))

    lines = [
        "## Solver XI",
        "",
        f"If you finalize {whose}, the XI below is the one that applies for"
        f" GW{event}. For any other plan the lineup is re-picked from that"
        " plan's squad, and your captain and vice must belong to it.",
        "",
    ]
    lines += [
        f"- {label}: " + ", ".join(rows[position])
        for position, label in POSITIONS.items()
    ]
    lines.append(
        f"- Captain: {_named(lineup.captain, board)}"
        f" | vice: {_named(lineup.vice, board)}"
    )
    lines.append(
        "- Bench: "
        + ", ".join(
            f"{order}. {_named(pid, board)}"
            for order, pid in enumerate(lineup.bench, start=1)
        )
    )
    return "\n".join(lines)


def _candidates(plans: list[tuple[int, Plan]], board: _Board, pick: int | None) -> str:
    """The shortlist, numbered, with the boundary the manager may not cross."""
    return "\n".join(
        [
            "## Candidate plans",
            "",
            "Finalize on one of these ids and no other. Each has already been"
            " checked for budget, club quotas and the hit cap; a squad that is"
            " not on this list is not reachable.",
            "",
            format_plans(plans, board.players, board.clubs, board.projections, pick),
        ]
    )


def _chip_panel(
    chips: ChipEvs, drafting: bool, played: set[str], horizon: int
) -> str:
    """The chip numbers, signed: a chip can be worth less than nothing.

    Three of the four are next gameweek's and the wildcard is a horizon total,
    so it is labelled rather than left to be read as the other three are — the
    manager is asked to argue from these numbers, and an argument from the
    wrong units is one the code cannot catch.

    Two of the four are also not his to finalize, and the panel says so where
    he looks them up: the validator refuses a wildcard or a free hit, and a
    rule he meets first as an error is a turn spent learning it.

    A chip already spent is priced anyway and then marked. The number is worth
    reading — it says what this week would have been worth with it — but the
    chip is not on the table, and the manager is told so in the one place he
    looks the chips up. The guardrail line is printed only when something has
    actually been played, because a warning about nothing is a line of noise
    in a document that is already long.
    """
    if drafting:
        return (
            "## Chip EV\n\nChips are not priced for a draft:"
            " a chip is played against a squad, and there is not one yet."
        )

    panel = {
        "bench_boost": ("Bench boost", f"{chips.bench_boost:+.1f}"),
        "triple_captain": ("Triple captain", f"{chips.triple_captain:+.1f}"),
        "free_hit": ("Free hit", f"{chips.free_hit:+.1f}"),
        "wildcard": ("Wildcard", wildcard_ev(chips.wildcard, horizon)),
    }
    lines = ["## Chip EV", "", CHIP_UNITS, ""]
    lines += [
        f"- {label}: {points}{PLAYED_MARK if chip in played else ''}"
        for chip, (label, points) in panel.items()
    ]
    lines += ["", PLANNED_GUARD]
    if played:
        lines += ["", PLAYED_GUARD]
    return "\n".join(lines)


def _watchlist(held: list[int], board: _Board, event: int) -> str:
    """The best projections we do not hold.

    Not a shopping list — the manager may only finalize a plan the solver
    reached — but the answer to "is the shortlist above drawn from a sane
    field", which is a question worth him being able to ask.
    """
    owned = set(held)
    field = [pid for pid in board.players if pid not in owned]
    ranked = sorted(
        field,
        key=lambda pid: (
            -projected_points(board.projections, pid),
            board.players[pid].web_name,
            pid,
        ),
    )
    # The count is what was listed, not what was asked for: a briefing that
    # announces ten and prints six has told the reader something untrue about
    # the size of the field, which is the one thing this section is for.
    shown = ranked[:WATCHLIST_SIZE]
    lines = [
        "## Watchlist",
        "",
        f"The {plural(len(shown), 'best projection')} we do not hold."
        " Context only: a player here is buyable only if a plan above buys him.",
        "",
    ]
    lines += [f"- {_player_line(pid, board, event)}" for pid in shown]
    return "\n".join(lines)


def _relevant(solve: "SolveResult", board: _Board) -> str:
    """The research list, wrapped: names and ids and nothing else."""
    pids = relevant_players(solve)
    entries = []
    for pid in pids:
        minutes = _minutes(pid, board)
        entries.append(_named(pid, board, extra=f", {minutes}" if minutes else ""))
    lines = [
        "## Relevant players",
        "",
        f"Research these {len(pids)} and nobody else — they are the squad and"
        " every player any plan above would buy or sell. adjust_players takes"
        " the id.",
        "",
    ]
    lines += [
        ", ".join(entries[start : start + RELEVANT_PER_LINE])
        for start in range(0, len(entries), RELEVANT_PER_LINE)
    ]
    return "\n".join(lines)


def _player_line(pid: int, board: _Board, event: int) -> str:
    """``Alvez (id 1, GKP, ASH, £5.5m) | xMins 89 | 4.0 xP GW2 | 20.0 xP6 | fit``.

    The minutes come between who he is and what he is worth, because they are
    the assumption the worth was computed from: read left to right, the line
    says who, for how long, to what end, and whether any of it is in doubt.
    """
    minutes = _minutes(pid, board)
    return (
        f"{_described(pid, board)}"
        f"{f' | {minutes}' if minutes else ''}"
        f" | {_next_gw(pid, board, event):.1f} xP GW{event}"
        f" | {projected_points(board.projections, pid):.1f} xP{board.horizon}"
        f" | {_flag(board.players[pid])}"
    )


def _moves(plan: Plan, board: _Board) -> str:
    """What the plan does, in the fewest words that name every player involved.

    A plan that sells nobody and buys a full squad is a draft rather than a
    transfer — there is no fifteen to move from — and printing fifteen
    signings on one line would say nothing the drafted-fifteen section above
    has not already said better.
    """
    if not plan.transfers_in and not plan.transfers_out:
        return "roll — no transfers"
    if not plan.transfers_out:
        return f"draft — buys all {len(plan.transfers_in)} places"
    return (
        "out " + _listed(plan.transfers_out, board)
        + "; in " + _listed(plan.transfers_in, board)
    )


def _listed(pids: list[int], board: _Board) -> str:
    return ", ".join(_priced(pid, board) for pid in pids)


def _priced(pid: int, board: _Board) -> str:
    """``Reid (id 18, FWD, CRV, £9.5m, 39.0 xP6)`` — the descriptor and what he
    is worth, plus a flag if the API doubts him. The flag is silent when there
    is nothing to say: on a transfer line, unlike a squad line, a status is
    news or it is noise."""
    player = board.players[pid]
    worth = f"{projected_points(board.projections, pid):.1f} xP{board.horizon}"
    flag = _flag(player)
    concern = "" if flag == STATUS_FLAGS[AVAILABLE] else f", {flag}"
    return _described(pid, board, extra=f", {worth}{concern}")


def _described(pid: int, board: _Board, extra: str = "") -> str:
    """``Alvez (id 1, GKP, ASH, £5.5m)`` — who he is, in one parenthesis.

    The report's own descriptor with the id in front of it, because the id is
    how the manager names him back to us. ``extra`` adds fields inside the
    same parenthesis rather than a second one.
    """
    player = board.players[pid]
    position = POSITIONS[player.element_type]
    club = _safe(board.clubs[player.team])
    cost = price(player.now_cost)
    name = _safe(player.web_name)
    return f"{name} (id {pid}, {position}, {club}, {cost}{extra})"


def _named(pid: int, board: _Board, extra: str = "") -> str:
    """``Alvez (id 1)`` — where the line has already said the rest."""
    return f"{_safe(board.players[pid].web_name)} (id {pid}{extra})"


def _safe(text: str) -> str:
    """One line's worth of somebody else's string.

    Names, club abbreviations and status codes arrive from a public API and go
    straight into a document whose structure is carried by its line breaks:
    ``##`` opens a section, ``plan 3:`` opens a plan. A ``web_name`` holding a
    newline and a heading is therefore a name that can forge a section the
    solver never produced, or a plan worth nine hundred points that cannot be
    finalized because it does not exist — and the reader is a model that has
    been told to trust this document about what is reachable.

    So every such field is flattened to a single line before it is
    interpolated — ``str.split`` with no argument breaks on every kind of
    whitespace Python knows, the line and paragraph separators included — and
    capped, because a field cannot be allowed to bury the rest of the page
    either.

    The pipe goes with them. It is the column separator on every line in this
    document, so a name holding one forges columns rather than sections — a
    weaker lie, still a lie, and no player is legitimately called it. A slash
    stands in, which reads as somebody's punctuation rather than as ours.

    Structure comes from this module and from nowhere else.
    """
    return " ".join(text.replace("|", "/").split())[:MAX_FIELD]


def _minutes(pid: int, board: _Board) -> str:
    """``xMins 89`` — what the model expects of him, or nothing to say at all.

    Rounded to the minute: the column exists to be argued with, and nobody
    argues in tenths. A player the fetch never asked for a history has no
    entry and reads as zero, which is not a gap in the briefing but exactly
    what the projection did with him — and therefore the number most worth a
    search.
    """
    if board.xmins is None:
        return ""
    return f"xMins {round(board.xmins.get(pid, 0.0))}"


def _flag(player: Player) -> str:
    """What the API says about his availability, in a word.

    A chance of playing is printed whenever it is short of certain, whatever
    the status letter says: the two fields disagree often enough on a
    Friday afternoon that reporting both is the only honest thing to do.
    """
    label = STATUS_FLAGS.get(player.status, _safe(player.status).upper())
    chance = player.chance_of_playing_next_round
    if chance is None or chance >= FULLY_FIT:
        return label
    return f"{label} ({chance}% chance)"


def _by_position(pids: list[int], board: _Board, event: int) -> list[int]:
    """Keepers, defenders, midfielders, forwards; best next gameweek first."""
    return sorted(
        pids,
        key=lambda pid: (
            board.players[pid].element_type,
            -_next_gw(pid, board, event),
            board.players[pid].web_name,
            pid,
        ),
    )


def _ranked(pids: list[int], board: _Board, event: int) -> list[int]:
    """Best next gameweek first, ties broken so the order never wobbles."""
    return sorted(
        pids,
        key=lambda pid: (
            -_next_gw(pid, board, event),
            board.players[pid].web_name,
            pid,
        ),
    )


def _next_gw(pid: int, board: _Board, event: int) -> float:
    """His projection for the gameweek in hand; a player without one is worth
    nothing, not a guess."""
    projection = board.projections.get(pid)
    return projection.per_gw.get(event, 0.0) if projection else 0.0
