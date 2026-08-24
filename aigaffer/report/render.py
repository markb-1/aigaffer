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

One section is not about this week at all. A recommendation off the multi-week
planner carries the rest of the window it was chosen for, and "the road ahead"
prints it — under a standing caveat, because only the gameweek above it is ever
entered and the rest is re-planned from scratch on the next run. A
recommendation off the single-week solver has no such thing to print, and when
the window was configured on and could not answer, the shortlist says so: which
engine drew a list up is part of what the list means.

:func:`deadline`, :func:`price`, :func:`plural`, :func:`hits_taken` and
:func:`carries_a_path` are public because they are the house vocabulary rather
than this module's private business: the manager's briefing
(:mod:`aigaffer.manager.briefing`) says the same things to a different reader
and must say them the same way. The deadline especially — reading a naive
timestamp as the runner's local clock is a mistake worth making once — and the
hits, where the report and the briefing are printing this gameweek's number
beside a total for several.
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
    from aigaffer.solver.multiweek import PlannedMove

POSITIONS = {GOALKEEPER: "GKP", DEFENDER: "DEF", MIDFIELDER: "MID", FORWARD: "FWD"}
OUTFIELD = (DEFENDER, MIDFIELDER, FORWARD)

# The chip whose fields on a free-hit week hold the temporary team. Named so the
# renderer and the orchestrator agree on which chip fields an eleven of its own.
FREE_HIT = "free_hit"

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

# What the shortlist says when the window was asked and had nothing to say. It
# is printed only when the better engine was configured on and did not answer:
# a run that asked for the single-week solver got what it asked for, and a
# report that apologised for it every week would be crying wolf.
SINGLE_WEEK = "Single-week engine (multi-week solve unavailable this run)."

# What a row on a window's board is measured in, said once above the list.
# Every number on it is honest and two of them are counted over different
# stretches of time: the transfers are this Saturday's, and the two totals are
# the whole window's with every gameweek's hits already taken off. Printed only
# when the board came off the window — a single-week row has one stretch of
# time in it and nothing to explain.
WINDOW_UNITS = (
    "xP and net are the whole window, decayed and with the armband in;"
    " transfers and hits shown are this week's unless the row says otherwise."
)

# What the chip-EV panel says once the solver actually plans one this week: the
# number above is not only a price now, it is a move the plan makes. Named so
# the panel can distinguish a chip priced from a chip planned.
PLANNED_THIS_WEEK = (
    "Planned this gameweek: {chip}. The solver's plan plays it now, not only"
    " prices it — see the Do this block."
)

# The label a free-hit week hangs on the eleven it fields. The team on the sheet
# is the temporary one the chip buys for a week, not the standing squad, and it
# reverts — so the section and the checklist both say so, in the same words.
FREE_HIT_XI = "Free Hit XI (this week only)"

# The line under the path, every week. The gameweeks after this one are solved
# on a projection of a projection and re-planned from scratch on the next run;
# printed without this they would read as a commitment, and the one thing a
# reader must not do is hold a move back this week because a plan he read a
# fortnight ago says it belongs in the next.
ADVISORY = "Advisory — re-planned every run; only this week's moves are ever made."

# The reminder's four fixed sentences. The calm one closes the ordinary
# reminder; the loud one opens the rare one, and it is deliberately the only
# line in either document that shouts — a warning that appears every week is a
# warning nobody reads by October. Loud means the *news* moved: the shout is
# earned by the solver disagreeing with its own day-old answer, never by the
# gaffer having disagreed with the solver, which was settled at T-24h.
REMINDER_UNCHANGED = (
    "The news has not moved since the full report — the plan above stands."
)
NEWS_MOVED = "⚠️ THE NEWS HAS MOVED since the full report"
NO_FULL_REPORT = (
    "There was no full report to compare against — the day-before run never"
    " happened. This is the solver's fresh answer, unreviewed."
)
# What a changed reminder must say about authority, in so many words: the
# verdict was the gaffer's — or the solver's wearing his label, a day ago with
# the manager in the loop — and the fresh block is a solver that has read
# nothing. Neither overrules the other; the person holding the phone does.
HUMAN_JUDGES = (
    "The verdict is the gaffer's and the fresh solve is the solver's;"
    " neither overrules the other. Read both, then enter one."
)

# The two blocks a changed reminder shows side by side, in this order: the
# decision that was actually made — the operative plan — then the news that
# questions it.
GAFFER_VERDICT = "## The gaffer's verdict (the operative plan)"
FRESH_SOLVE = "## The solver's fresh answer"


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
    *,
    engine_expected: bool = False,
    free_transfers: int | None = None,
) -> str:
    """The whole report as one markdown string. Pure; no I/O.

    ``choice`` is the plan being recommended and is expected to be one of
    ``plans`` — it is flagged there by identity. Every player id in the plans
    and the lineup must be a ``bootstrap`` element, which it is: they came
    from there. The string ends in a newline, because it is written out as a
    file as well as sent as a message.

    ``free_transfers`` is the bank, and it is read in one place: the action
    block at the top says how many the week is banking when it rolls. It is not
    needed to draft — a draft has no action block — so it defaults to None, and
    a None on a week that rolls simply drops the count rather than guessing one.

    ``gaffer`` is the manager's decision, when there was a manager: the plan
    and the eleven above are his by then, and the section this adds is the
    half of a decision that is words rather than numbers — why, what he
    overruled, and whose pick this actually is. Omitted, the report is byte
    for byte the one this wrote before there was a manager at all.

    ``engine_expected`` says the multi-week planner was configured on. Whether
    it *answered* is written on the recommendation — a plan off the window
    carries a path and a plan off the single-week solver does not — and the
    two together are the only way to tell a fallback from a run that asked for
    the single-week solver on purpose. The renderer decides nothing: the caller
    knows what it configured, and this knows what came back.
    """
    players = {player.id: player for player in bootstrap.elements}
    clubs = {team.id: team.short_name for team in bootstrap.teams}
    road = choice.path.moves if choice.path is not None else []
    # The chip this week plays, decided in one place. It moves the checklist, the
    # team sheet's label and the panel's note together, so none of them can say
    # a different thing about the same decision.
    chip = played_chip(choice, gaffer)
    free_hit = _free_hitting(choice, chip)

    sections = [
        _header(mode, event),
        *(
            []
            if _drafting(choice)
            else [
                _do_this(
                    event, choice, lineup, players, clubs, chip, free_hit,
                    free_transfers,
                )
            ]
        ),
        _recommendation(choice, players, clubs),
        *([] if gaffer is None else [_gaffer(gaffer, players)]),
        _team_sheet(lineup, players, projections, free_hit),
        _candidates(
            plans,
            choice,
            decided=gaffer is not None,
            fell_back=engine_expected and choice.path is None,
        ),
        *([_road_ahead(road, players)] if road else []),
        _chip_panel(chips, horizon_of(projections), chip),
        _watchlist(choice.squad, players, clubs, projections),
    ]
    return "\n\n".join(sections) + "\n"


def render_reminder(
    event: Event,
    fresh: dict,
    stored: dict | None,
    changes: dict,
    bootstrap: Bootstrap,
) -> str:
    """The short alert, three hours out. Pure; no I/O.

    Not a report: the reminder exists so that the phone buzzes once with the
    operative plan and whether the news has moved under it. It is built from
    **actions dicts** — ``transfers`` as ``[out, in]`` pairs, ``captain``,
    ``vice``, ``chip``, ``formation``: ``stored`` is the gaffer's verdict
    read back from the deadline record, ``fresh`` is this run's solver-only
    solve — and from ``changes``, the diff the orchestrator made between the
    *solver's* stored pre-manager plan and that fresh solve
    (:func:`aigaffer.orchestrator.diff_actions`). The renderer decides
    nothing about whether the news moved; it only says so — which is why
    ``changes`` is never derived from the two dicts on show: the verdict and
    the fresh solve may differ simply because the gaffer overrode the solver
    a day ago, and that is a settled decision, not news.

    Three shapes. ``stored`` is None when the full report never ran, and the
    fresh block goes out with one line admitting there was nothing to check
    it against. An empty ``changes`` over a stored plan is the ordinary
    reminder: the gaffer's verdict — the operative plan, the one a person
    enters, however far it sits from what a solver alone would do — and a
    calm sentence. A non-empty ``changes`` leads with the one loud line
    either document is allowed, lists what the news moved, and then shows
    both weeks labelled — the gaffer's verdict first, because it is the
    decision that was actually made, and the fresh solve second, because it
    is information and not an overruling. The reader judges.

    The stored plan is read back from the diary and rendered against today's
    bootstrap, so a player the API has since dropped or renumbered prints as
    ``player {id}`` rather than costing the alert.
    """
    players = {player.id: player for player in bootstrap.elements}
    clubs = {team.id: team.short_name for team in bootstrap.teams}

    sections = [_header("reminder", event)]
    if stored is None:
        sections += [_actions_block("## Do this", event, fresh, players, clubs)]
        sections += [NO_FULL_REPORT]
    elif not changes:
        sections += [_actions_block("## Do this", event, stored, players, clubs)]
        sections += [REMINDER_UNCHANGED]
    else:
        sections += [_changed(changes, players, clubs)]
        sections += [_actions_block(GAFFER_VERDICT, event, stored, players, clubs)]
        sections += [_actions_block(FRESH_SOLVE, event, fresh, players, clubs)]
        sections += [HUMAN_JUDGES]
    return "\n\n".join(sections) + "\n"


def _changed(changes: dict, players: dict[int, Player], clubs: dict[int, str]) -> str:
    """What the news moved between yesterday's solve and today's, a line each.

    The order is the order the moves are entered in: transfers, then the
    chip, then the armbands, then the shape. Sells and buys are named on
    separate lines — two lists is how the FPL app takes them, and a paired
    line here would claim to know which sale funds which signing, which
    nothing does. ``Now`` is what the fresh solve wants and yesterday's did
    not; ``No longer`` the other way about.
    """
    lines = [NEWS_MOVED, ""]
    for key, label in (
        ("sells_added", "Now selling"),
        ("sells_dropped", "No longer selling"),
        ("buys_added", "Now buying"),
        ("buys_dropped", "No longer buying"),
    ):
        for pid in changes.get(key, []):
            lines.append(f"- {label}: {_tagged(pid, players, clubs)}")
    if "chip" in changes:
        before, after = changes["chip"]
        lines.append(
            f"- Chip changed from {_spoken(before)} to {_spoken(after)}"
        )
    for band in ("captain", "vice"):
        if band in changes:
            before, after = changes[band]
            lines.append(
                f"- {band.title()} moved from {_who(before, players)}"
                f" to {_who(after, players)}"
            )
    if "formation" in changes:
        before, after = changes["formation"]
        lines.append(f"- Formation changed from {before} to {after}")
    return "\n".join(lines)


def _actions_block(
    heading: str,
    event: Event,
    actions: dict,
    players: dict[int, Player],
    clubs: dict[int, str],
) -> str:
    """One plan as the ⏰ checklist, under ``heading``.

    The same shape as the full report's "Do this" block, drawn from an
    actions dict instead of a plan: the deadline, a swap to a line, the chip
    if one plays, the armbands, and — always, where the report only nudges
    when a signing starts — the formation, because a reminder with no team
    sheet under it has nowhere else to say the shape. A stored plan from
    before formations were kept simply drops the line.
    """
    lines = [heading, "", f"⏰ Make these by {deadline(event)} — GW{event.id}"]
    if not actions["transfers"]:
        lines.append("No transfers — roll.")
    lines += [
        _swap(out, bought, players, clubs) for out, bought in actions["transfers"]
    ]
    if actions["chip"] != NO_CHIP:
        lines.append(f"PLAY {chip_label(actions['chip'])}")
    lines.append(
        f"CAPTAIN {_who(actions['captain'], players)}"
        f" · VICE {_who(actions['vice'], players)}"
    )
    if actions.get("formation"):
        lines.append(f"Formation: {actions['formation']}")
    return "\n".join(lines)


def _swap(
    out: int, bought: int, players: dict[int, Player], clubs: dict[int, str]
) -> str:
    """``SELL Gale (DEF CRV £4.0m) → BUY Reid (FWD CRV £9.5m)`` — one move,
    sell first, the way the app takes it and the way the report says it."""
    return (
        f"SELL {_tagged(out, players, clubs)} → BUY {_tagged(bought, players, clubs)}"
    )


def _tagged(pid: int, players: dict[int, Player], clubs: dict[int, str]) -> str:
    """:func:`_named`, or the id when this board has never heard of him.

    The report never needs the net — every id it prints came off this run's
    bootstrap — but the reminder prints a plan read back from the diary, and
    an id the API has since retired is not worth losing the alert over.
    """
    if pid in players:
        return _named(pid, players, clubs)
    return f"player {pid}"


def _spoken(chip: str) -> str:
    """A chip as a sentence says it: ``bench boost``, or ``none``."""
    return chip.replace("_", " ")


def _free_hitting(choice: Plan, chip: str) -> bool:
    """Is this a free-hit week with a temporary team to field?

    Only then does the report field an eleven the standing squad is not: a free
    hit the solver actually planned carries the fifteen it priced. Any other
    chip, or a free hit nobody put a squad behind, leaves the standing team on
    the sheet.
    """
    return (
        chip == FREE_HIT
        and choice.path is not None
        and bool(choice.path.week1_freehit_xi)
    )


def chip_label(chip: str) -> str:
    """``bench_boost`` as a person reads it: ``Bench Boost``.

    Public because the manager's briefing names the same chips on the same
    paths, and a chip that reads one way in the report and another in the
    briefing is a chip nobody can match between the two documents.
    """
    return chip.replace("_", " ").title()


def played_chip(choice: Plan, gaffer: "ManagerDecision | None") -> str:
    """The chip this week actually plays, or ``"none"``.

    One place decides it, so the checklist, the panel and the record cannot come
    to disagree. With a manager it is his: he is the decision, and he may play a
    chip the solver planned, a different one, or none. Without a manager it is
    the solver's own week-1 chip, which the window sets on the recommended
    plan's path — a plan off the single-week solver has no path and plays
    nothing, which is the pre-chip behaviour to the byte.
    """
    if gaffer is not None:
        return gaffer.chip
    if choice.path is not None:
        return choice.path.week1_chip
    return NO_CHIP


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


def _drafting(choice: Plan) -> bool:
    """A squad bought from nothing: players in, and none sold.

    The one week a plan buys without selling. Every other week the squad is
    already fifteen and every move is a swap, so a plan that sells nobody is a
    plan drafting from an empty squad — which is the one report with no action
    block, because the whole squad is the action and the recommendation below is
    already the fifteen to buy.
    """
    return bool(choice.transfers_in) and not choice.transfers_out


def _do_this(
    event: Event,
    choice: Plan,
    lineup: Lineup,
    players: dict[int, Player],
    clubs: dict[int, str],
    chip: str,
    free_hit: bool,
    free_transfers: int | None,
) -> str:
    """The week's moves as a checklist, first and imperative.

    The bot advises and never executes: the moves are made by hand, on a phone,
    against a deadline. So the top of the report is a short list of exactly what
    to do, in the order the FPL app takes it — the deadline, the transfers, the
    chip if there is one, the armbands, and a nudge to fix the lineup when a
    signing has to start. Everything below it explains this; this is the part he
    acts on, and it draws from the same decision without adding a fact to it.

    ``chip`` is the chip this week plays, ``"none"`` for most weeks. A free-hit
    week is its own checklist: the chip fields a whole temporary team that
    reverts, so there is no swap on the standing squad to make — the block names
    the eleven to build and says it is for one week, and ``lineup`` is already
    that temporary team rather than the standing one.

    A signing that starts is the one lineup change worth flagging here: the app
    drops a transferred-in player onto the bench, so an eleven that needs him in
    it needs the bench reordered by hand. The shape is named and the full eleven
    is left to the team sheet below — a checklist, not a second copy of it.
    """
    lines = ["## Do this", "", f"⏰ Make these by {deadline(event)} — GW{event.id}"]
    if free_hit:
        lines.append(f"PLAY {chip_label(chip)}")
        lines.append(
            f"{FREE_HIT_XI}: {_eleven(lineup, players)}"
            " — a temporary team; it reverts next week"
        )
        lines.append(
            f"CAPTAIN {_who(lineup.captain, players)}"
            f" · VICE {_who(lineup.vice, players)}"
        )
        return "\n".join(lines)

    lines += _moves(choice, players, clubs, free_transfers)
    if chip != NO_CHIP:
        lines.append(f"PLAY {chip_label(chip)}")
    lines.append(
        f"CAPTAIN {_who(lineup.captain, players)} · VICE {_who(lineup.vice, players)}"
    )
    if set(choice.transfers_in) & set(lineup.xi):
        lines.append(f"Set lineup: {formation(lineup, players)}")
    return "\n".join(lines)


def _eleven(lineup: Lineup, players: dict[int, Player]) -> str:
    """The eleven by name, for a checklist that has to name a whole team at once
    — the free-hit build, where the standing squad is not the one that plays."""
    return ", ".join(players[pid].web_name for pid in lineup.xi)


def _moves(
    choice: Plan,
    players: dict[int, Player],
    clubs: dict[int, str],
    free_transfers: int | None,
) -> list[str]:
    """The transfers a swap to a line, or the one line that makes none.

    Sell then buy, the way the app takes a transfer, so a line is a move he can
    make without holding two of them in his head. A week that rolls says how
    many free transfers it is banking, because that count is the whole of the
    non-move and is the one number he checks it against; a run that was handed
    no count drops it rather than inventing one.
    """
    if not choice.transfers_in and not choice.transfers_out:
        if free_transfers is None:
            return ["No transfers — roll."]
        return [f"No transfers — roll (bank {plural(free_transfers, 'free transfer')})."]
    return [
        f"SELL {_named(out, players, clubs)} → BUY {_named(bought, players, clubs)}"
        for out, bought in zip(choice.transfers_out, choice.transfers_in)
    ]


def _named(pid: int, players: dict[int, Player], clubs: dict[int, str]) -> str:
    """``Gale (DEF CRV £4.0m)`` — the checklist's own parenthesis.

    Spaces, not the commas :func:`_described` uses: on a line read at arm's
    length the position, club and price want to run together as one tag on the
    name rather than read as a list of three things.
    """
    player = players[pid]
    return (
        f"{player.web_name} ({POSITIONS[player.element_type]}"
        f" {clubs[player.team]} {price(player.now_cost)})"
    )


def formation(lineup: Lineup, players: dict[int, Player]) -> str:
    """``3-4-3`` — the shape of the eleven, defenders through forwards.

    Public because the decision record keeps it now: the reminder three hours
    out diffs the fresh solve against the full report's plan, and a shape that
    moved is one of the things it has to be able to say. The orchestrator
    computes it once, off the same lineup the record's armbands come off.
    """
    counts: dict[int, int] = defaultdict(int)
    for pid in lineup.xi:
        counts[players[pid].element_type] += 1
    return "-".join(str(counts[position]) for position in OUTFIELD)


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

    So the hash that would open a section is escaped, and the line reads as the
    sentence it is. Indenting it instead — which is what this did first — is
    not enough: CommonMark allows three spaces before an ATX heading, so a
    pushed-off line is still a heading on GitHub, where the diary is read. A
    backslash is a heading nowhere; on the phone, where nothing is rendered at
    all, it costs one visible character in a line that was trying to lie.

    The line breaks are kept: this is prose, and a paragraph flattened into one
    line is a paragraph nobody finishes.
    """
    return "\n".join(_unheaded(line) for line in str(text).splitlines())


def _unheaded(line: str) -> str:
    """One line of his, made unable to open a section of ours."""
    body = line.lstrip()
    if not body.startswith("#"):
        return line
    return f"{line[: len(line) - len(body)]}\\{body}"


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
    free_hit: bool = False,
) -> str:
    """The eleven a row to a position, then the bench in the order it is read.

    Within a row the best projected total goes first — the same number the
    watchlist ranks on, and near enough to the next gameweek alone that the
    armbands were chosen on. It is a way of laying out an eleven that is
    already picked, not a second opinion about it: the bench keeps the order
    it was given, because that order is a substitution list.

    ``free_hit`` says this eleven is the temporary team a free hit fields, not
    the standing squad, so the heading says which — the standing squad reverts
    and is not the team taken to the deadline this week.
    """
    rows: dict[int, list[str]] = defaultdict(list)
    for pid in _ranked(lineup.xi, players, projections):
        rows[players[pid].element_type].append(_armband(pid, lineup, players))

    label = f" — {FREE_HIT_XI}" if free_hit else ""
    lines = [f"## Starting XI ({formation(lineup, players)}){label}", ""]
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


def _candidates(
    plans: list[Plan], choice: Plan, decided: bool = False, fell_back: bool = False
) -> str:
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

    A board off the window is captioned, because its rows mix two stretches of
    time — see :data:`WINDOW_UNITS`, and :func:`hits_taken` for the one cell
    that carries both. The caption is read off the rows rather than off the
    recommendation, since it is the rows it is explaining: a single-week board
    prints the section exactly as it printed before there was a window at all.

    ``fell_back`` says the window was asked and had nothing to say about the
    plan being recommended, which in the ordinary run is a fact about this
    whole list: the caller reads it off that plan, and that plan is one of
    these. After a manager re-solve it need not be one of these, and then the
    line is still true of the recommendation above and merely unproven of the
    rows — which is the right way round, since the recommendation is what a
    reader is being asked to trust. It goes under the list, above the note
    about the recommendation.
    """
    pick = _pick(plans, choice)
    lines = ["## Candidate plans", ""]
    if any(carries_a_path(plan) for plan in plans):
        lines += [WINDOW_UNITS, ""]
    for index, plan in enumerate(plans):
        recommended = "  <- recommended" if index == pick else ""
        lines.append(
            f"- {plural(len(plan.transfers_in), 'transfer')}"
            f" | {hits_taken(plan)}"
            f" | {plan.xp_total:.1f} xP"
            f" | {plan.objective:.1f} net{recommended}"
        )
    if fell_back:
        lines += ["", SINGLE_WEEK]
    if decided and pick is None:
        lines += ["", RESOLVED_ELSEWHERE]
    return "\n".join(lines)


def hits_taken(plan: Plan) -> str:
    """``0 hits``, or ``0 hits now, 1 over the window``.

    The hit cell on a plan line, and the one place the two stretches of time a
    window's row is measured over are reconciled. ``xp_total`` and ``objective``
    are the whole window's and the objective has every gameweek's hits taken
    off it; ``hits`` is this gameweek's alone. Printed as one number beside the
    other two, a plan that pays four points in three weeks' time is a row whose
    net is four short of its total for no reason the reader can see — and the
    reader is being asked to make this week's transfer on the strength of it.

    So a path that pays hits puts both numbers on the line and says which week
    each of them belongs to. A path that pays none does not: the two would be
    the same number, and a row that prints it twice reads as a row with
    something to explain. A plan with no path at all is a single gameweek from
    end to end and prints exactly what it always printed.

    Public because the briefing prints the same cell to a different reader, and
    a number that means one thing in the report and another in the briefing is
    worse than a number nobody prints.
    """
    now = plural(plan.hits, "hit")
    if plan.path is None:
        return now
    later = sum(move.hits for move in plan.path.moves)
    return now if not later else f"{now} now, {plan.hits + later} over the window"


def carries_a_path(plan: Plan) -> bool:
    """Whether this plan has gameweeks after this one to speak for it.

    A path with no moves in it is a window that means to do nothing further,
    and everything that reads a path treats it as no path at all: there is no
    road to print, no notation to explain and nothing on the line to caption.
    Public, and one line, so that the two documents cannot come to disagree
    about what counts as a plan off the window.
    """
    return plan.path is not None and bool(plan.path.moves)


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


def _road_ahead(moves: list["PlannedMove"], players: dict[int, Player]) -> str:
    """The gameweeks after this one, as the window means to play them.

    A gameweek to a line, and only the gameweeks it means to move in: a plan
    that leaves a week alone has no move in it, which is what makes a path a
    list of intentions rather than a calendar. Each line says who goes, who
    comes, and what the move spends — the free transfers it uses and the hit
    it takes, because a plan that pays four points in three weeks' time is
    making an argument about this week's move that the reader is entitled to
    weigh.

    The section exists only when there is a path with moves on it, and it
    closes with :data:`ADVISORY` every time. Nothing here is ever entered: the
    week above it is the decision, and this is what the decision is for.
    """
    lines = ["## The road ahead", ""]
    lines += [f"- GW{move.event}: {_planned(move, players)}" for move in moves]
    lines += ["", ADVISORY]
    return "\n".join(lines)


def _planned(move: "PlannedMove", players: dict[int, Player]) -> str:
    """``out Gale (£4.0m), in Dodd (£4.5m) — 1 FT, no hit``, and the chip.

    The sales bullet is omitted when there are none, as the recommendation's
    is: a week the window only buys in is not a week to print an empty list
    for. In practice a squad is fifteen every gameweek and every move is a
    swap, so this is a defence against a solver that changed rather than a
    case anybody has seen.

    A chip the window means to play that gameweek rides on the end — and a
    gameweek that only plays a chip, moving nobody, is the chip alone: a bench
    boost or a wildcard the plan schedules three weeks out is a large part of
    the argument the road ahead is making for the opening move.
    """
    parts = []
    if move.transfers_out:
        parts.append("out " + _priced(move.transfers_out, players))
    if move.transfers_in:
        parts.append("in " + _priced(move.transfers_in, players))
    moved = ", ".join(parts) + f" — {_spent(move)}" if parts else ""
    chip = chip_label(move.chip) if move.chip != NO_CHIP else ""
    if moved and chip:
        return f"{moved} — {chip}"
    return moved or chip


def _spent(move: "PlannedMove") -> str:
    """``1 FT, no hit`` — what a future gameweek's moves cost it.

    The free transfers are the moves the hits did not pay for: the path model
    charges four points for every move beyond the bank, so what is left is
    what the bank covered. Both halves are printed because a reader deciding
    whether to trust a path wants to know whether it is spending points or
    only patience.
    """
    free = max(0, len(move.transfers_in) - move.hits)
    cost = f"-{move.hits * HIT_POINTS} pts in hits" if move.hits else "no hit"
    return f"{plural(free, 'FT')}, {cost}"


def _chip_panel(chips: ChipEvs, horizon: int, chip: str = NO_CHIP) -> str:
    """The chip numbers, signed: a chip can be worth less than nothing.

    Three of the four are next gameweek's: the bench that would have scored,
    the captain counted once more, the eleven a free hit would field instead.
    The wildcard is not, and never was — it is priced as the difference between
    two decayed horizon totals, because a wildcard is bought for the run of
    fixtures rather than for Saturday — so its row says which number it is.
    Four figures under one heading, one of them measuring something else, is
    how a chip gets played on a comparison nobody made.

    ``chip`` is the one the plan actually plays this week, if any. Its row is a
    price like the others until the plan plays it, so the panel says below the
    numbers that this one is planned — the reader is being asked to act on it,
    not only to weigh it.
    """
    panel = {
        "Bench boost": chips.bench_boost,
        "Triple captain": chips.triple_captain,
        "Free hit": chips.free_hit,
    }
    lines = ["## Chip EV", "", CHIP_UNITS, ""]
    lines += [f"- {label}: {points:+.1f}" for label, points in panel.items()]
    lines.append(f"- Wildcard: {wildcard_ev(chips.wildcard, horizon)}")
    if chip != NO_CHIP:
        lines += ["", PLANNED_THIS_WEEK.format(chip=chip_label(chip))]
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


def _priced(pids: list[int], players: dict[int, Player]) -> str:
    """``Gale (£4.0m), Fenn (£4.5m)`` — a name and a price each.

    What a move three gameweeks out turns on, and no more than that: the
    position and the club are for the week being decided, where the reader is
    weighing one player against another. On a path they would be detail about
    a move that will be planned again before anybody makes it.
    """
    return ", ".join(
        f"{players[pid].web_name} ({price(players[pid].now_cost)})" for pid in pids
    )


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
