"""Which chips are in hand, and when each may be played.

FPL hands out chips in two sets: one to play by the GW19 deadline and a fresh
one from GW20 (the bootstrap's ``chips`` list says exactly which gameweeks each
chip may go in). A first-set chip not played by its stop_event is gone. So a
chip is not a name but a name *and a window*: this module turns the rules and
the season's chip history into the chips actually held, each with the
gameweeks it may still be played in.

It sits below the solver, the orchestrator, the manager and the report — all
four ask it the same questions — and depends on nothing but the data models,
so the dependency always runs towards it.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from aigaffer.data.models import Bootstrap, ChipRule

if TYPE_CHECKING:  # read for its fields only; the import never runs
    from aigaffer.executed import Executed
    from aigaffer.orchestrator import PipelineInputs

BENCH_BOOST = "bench_boost"
TRIPLE_CAPTAIN = "triple_captain"
WILDCARD = "wildcard"
FREE_HIT = "free_hit"
# The order chips are listed, compared and tie-broken in, everywhere.
CHIP_ORDER = (BENCH_BOOST, TRIPLE_CAPTAIN, WILDCARD, FREE_HIT)

# What the FPL API calls each chip, against what we call it.
CHIP_API_NAMES = {
    BENCH_BOOST: "bboost",
    TRIPLE_CAPTAIN: "3xc",
    FREE_HIT: "freehit",
    WILDCARD: "wildcard",
}
_FROM_API = {api: chip for chip, api in CHIP_API_NAMES.items()}

FIRST_GW, LAST_GW = 1, 38


@dataclass(frozen=True)
class ChipWindow:
    """One chip the rules hand out, and the gameweeks it may be played in."""

    chip: str
    start_event: int
    stop_event: int


@dataclass(frozen=True)
class HeldChip:
    """A chip in hand: its kind and the window it must be played inside.

    Two of a kind can be held at once — the first set's bench boost and the
    second's — so anything keyed downstream is keyed by :attr:`id`, never by
    the kind.
    """

    chip: str
    start_event: int
    stop_event: int

    @property
    def id(self) -> str:
        return f"{self.chip}@{self.stop_event}"

    def allows(self, event: int) -> bool:
        return self.start_event <= event <= self.stop_event


def chip_windows(rules: list[ChipRule]) -> tuple[ChipWindow, ...]:
    """The bootstrap's chip rules in our names, unknown chips dropped.

    No rules at all — a payload recorded before the field was read, a board a
    test built by hand — is one window per chip over the whole season: the
    model the bot had before it knew about halves.
    """
    windows = tuple(
        ChipWindow(_FROM_API[rule.name], rule.start_event, rule.stop_event)
        for rule in rules
        if rule.name in _FROM_API
    )
    if windows:
        return windows
    return tuple(ChipWindow(chip, FIRST_GW, LAST_GW) for chip in CHIP_ORDER)


def rule_gaps(rules: list[ChipRule]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Our chips the rules leave out, and API chip names we do not know.

    :func:`chip_windows` trusts the rules: a chip they do not list is a chip
    the game does not offer, and it is never planned. That is right when FPL
    drops a chip and silently wrong when FPL renames one — ``bboost`` becoming
    something else would read exactly like a season with no bench boost. The
    payload cannot tell the two apart, so this only names the evidence for
    the run to say out loud: which of ours went missing, and which names
    arrived that the map does not know (a rename shows up as one of each).

    No rules at all is :func:`chip_windows`' fallback, not a gap, and a new
    chip beside all four of ours costs the plan nothing — both are silent.
    """
    if not rules:
        return (), ()
    listed = {rule.name for rule in rules}
    missing = tuple(chip for chip in CHIP_ORDER if CHIP_API_NAMES[chip] not in listed)
    if not missing:
        return (), ()
    return missing, tuple(sorted(listed - set(_FROM_API)))


def held_chips(
    windows: tuple[ChipWindow, ...], chips_used: list[dict], event: int
) -> tuple[HeldChip, ...]:
    """Every chip still in hand at ``event``, soonest to expire first.

    A window is spent when the history has that chip played on a gameweek
    inside it, and closed once ``event`` is past its stop_event. History we
    cannot place — a chip we do not plan, an entry with no gameweek — marks
    nothing: it is somebody else's payload, and guessing would cost a chip.

    The rules also bar a free hit the gameweek after a free hit (the official
    2026/27 wording: use the first in GW19 and you cannot play the second in
    GW20). The two sets meet only at that boundary, but it is stated for any
    gameweek: a still-held free hit whose window opens the week after one the
    history shows played is held with its window opening a week later. Its id,
    keyed by stop_event, does not change, so everything downstream — the MILP's
    eligible weeks, the calendar, ``held_for`` and so the gaffer's belt and the
    briefing panel — sees the later start and nothing else needs to know.
    """
    played = [
        (_FROM_API.get(str(entry.get("name", "")).strip().lower()), entry.get("event"))
        for entry in chips_used
    ]
    free_hit_events = {
        played_at
        for chip, played_at in played
        if chip == FREE_HIT and isinstance(played_at, int)
    }
    held = [
        HeldChip(
            window.chip,
            window.start_event + 1
            if window.chip == FREE_HIT and window.start_event - 1 in free_hit_events
            else window.start_event,
            window.stop_event,
        )
        for window in windows
        if window.stop_event >= event
        and not any(
            chip == window.chip
            and isinstance(played_at, int)
            and window.start_event <= played_at <= window.stop_event
            for chip, played_at in played
        )
    ]
    return tuple(
        sorted(held, key=lambda chip: (chip.stop_event, CHIP_ORDER.index(chip.chip)))
    )


def held_for(held: tuple[HeldChip, ...], chip: str, event: int) -> HeldChip | None:
    """The held chip of that kind playable at ``event``, if any — at most one
    can be, since a kind's windows do not overlap."""
    return next(
        (candidate for candidate in held if candidate.chip == chip and candidate.allows(event)),
        None,
    )


def held_in_week(
    bootstrap: Bootstrap,
    chips_used: list[dict],
    event_id: int,
    executed: "Executed | None" = None,
) -> tuple[HeldChip, ...]:
    """The chips in hand at ``event_id``, by the bootstrap's rules and the
    chip history alone — whatever the chip switch says.

    The body of :func:`held_by_rules`, taking the four facts it reads rather
    than the run's inputs, for the one caller that has no inputs to hand: the
    chip what-if's gate, which answers "is this chip yours this week?" from
    the bootstrap, the picks and the chip history — three requests — before
    the full fetch's two hundred are worth making for a chip he cannot play.

    ``executed`` is the owner's recorded week, if any; ``chips_used`` must
    already include its chip (the effective inputs' history does —
    :func:`aigaffer.executed.apply_executed` adds it — and the gate adds it
    itself). A gameweek with a chip recorded by "Transfers made" plays no
    other (see the comment below).
    """
    held = held_chips(chip_windows(bootstrap.chips), chips_used, event_id)
    # One chip a gameweek. When the owner has recorded a chip for this one —
    # it is in the chip history above already, so that chip is spent — no
    # other may be played in it either: every held chip opens next week at
    # the earliest, and one whose window ends this week is gone. So the solver
    # plans none in week 1, the briefing marks all four, and the belt refuses
    # whatever the gaffer finalizes but "none" (the recorded chip stands).
    if executed is None or executed.chip not in CHIP_ORDER:
        return held
    return tuple(
        replace(chip, start_event=max(chip.start_event, event_id + 1))
        for chip in held
        if chip.stop_event > event_id
    )


def held_by_rules(inputs: "PipelineInputs") -> tuple[HeldChip, ...]:
    """The chips in hand at the run's gameweek, by the rules and the history
    alone — whatever the chip switch says.

    The switch decides whether the solver plans chips, not whether a chip may
    be played: switched off, chips are advisory, priced in the panel and
    planned by nobody, and the gaffer may still play one we hold. So the belt
    that refuses a chip and the briefing panel that marks one both ask this,
    while the solver and the calendar ask the orchestrator's switch-aware
    ``_held``. ``inputs`` is read for its fields only (bootstrap rules, chip
    history, gameweek, recorded week), which keeps this module below everyone
    who asks; :func:`held_in_week` is the same answer from those fields.
    """
    return held_in_week(
        inputs.bootstrap,
        inputs.chips_used,
        inputs.event.id,
        getattr(inputs, "executed", None),
    )


def whole_season(*chips: str) -> tuple[HeldChip, ...]:
    """Held chips with one GW1–38 window each — the no-rules fallback, and a
    convenience for tests that predate halves."""
    return tuple(
        HeldChip(chip, FIRST_GW, LAST_GW)
        for chip in CHIP_ORDER
        if chip in chips
    )


def playable_in(held: tuple[HeldChip, ...], events: list[int]) -> tuple[HeldChip, ...]:
    """The held chips at least one of ``events`` may play."""
    return tuple(chip for chip in held if any(chip.allows(event) for event in events))
