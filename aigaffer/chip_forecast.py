"""The Chip forecast: when the bot currently plans to play each chip.

The owner taps **Chip forecast** (or texts the words) and gets, within
seconds, the chip thinking of the newest report on file: which held chip the
plan puts in which week, what the calendar is saving it for, and when it
expires. Nothing is solved and no gaffer is asked, so it costs nothing and
answers at once; the price is that it is only as fresh as the last report,
which the first line says (``as of Wednesday's scout (GW7)``). The plan is
re-made every run, so the footer reminds him it is the current thinking and
not a commitment.

It reads three things and makes two requests: the saved decision record
(:meth:`aigaffer.store.Store.latest_verdict`), the bootstrap (the next
gameweek and the chip rules) and the chip history. The record already carries
the plan's path and the chip calendar (``decision["path"]``,
``decision["chip_calendar"]``), so the report's own logic is mirrored here
from the record alone — :func:`aigaffer.report.render.chip_line` works from
live objects, this from their JSON — and the wording is kept to hand.

:func:`render_forecast` is pure; :func:`chip_forecast` is the inbox's handler,
the same shape as :func:`aigaffer.recording.transfers_made`. It takes no
state lock: it reads a few rows and writes nothing.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from aigaffer.chips import (
    CHIP_API_NAMES,
    CHIP_ORDER,
    WILDCARD,
    HeldChip,
    held_in_week,
)
from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.recording import MODE_NAMES, VERDICT_MODES
from aigaffer.report.render import chip_label
from aigaffer.store import DB_NAME, Store

NO_GAMEWEEK = "No gameweek ahead — nothing to forecast."
NO_PLAN = "No plan yet — the first report for GW{gw} hasn't run."
HEADER = "🗓 Chip forecast — as of {day}'s {mode} (GW{gw})"
THIS_WEEK = "GW{gw} (this week)"
WORTH = " · worth ~+{value:.0f}"
EXPIRES = " · expires GW{stop}"
NO_WEEK = "no week yet"
WILDCARD_KEPT = "no week yet: nothing in GW{first}–{last} beats keeping it"
CALENDAR_UNAVAILABLE = "calendar unavailable"
SECOND_SET = "Second set (GW{start}–{stop}): {kinds}"
PLAYED = "Played: {chips}"
NO_CURRENT_CHIPS = "No chip in hand to play before this half closes."
FOOTER = "Re-planned every run — this is the current thinking, not a commitment."

# The owner's clock: a verdict stamped late on a Tuesday UTC night is
# Wednesday's to him.
LOCAL_ZONE = ZoneInfo("Europe/Dublin")

_FROM_API = {api: chip for chip, api in CHIP_API_NAMES.items()}


def _name(chip: str) -> str:
    """``bench_boost`` as lower-case words: ``bench boost``."""
    return chip_label(chip).lower()


def _earliest_path_week(record: dict, held: HeldChip, gw: int) -> int | None:
    """The first week the plan's path plays ``held``'s kind inside its window.

    Weeks before ``gw`` are skipped: a record from last gameweek still lists
    its own first week, which has gone, and a forecast names only what is to
    come.
    """
    weeks = [
        move["event"]
        for move in record.get("path") or []
        if move.get("chip") == held.chip
        and isinstance(move.get("event"), int)
        and move["event"] >= gw
        and held.allows(move["event"])
    ]
    return min(weeks) if weeks else None


def _window(record: dict, record_gw: int, entry: dict | None) -> tuple[int, int] | None:
    """The weeks the plan looked across: the record's gameweek to the last
    week of its path, else to the last bar week of the wildcard's calendar
    entry; None when it names neither."""
    path_weeks = [
        move["event"]
        for move in record.get("path") or []
        if isinstance(move.get("event"), int)
    ]
    if path_weeks:
        return record_gw, max(path_weeks)
    bars = [int(week) for week in (entry or {}).get("bars") or {}]
    if bars:
        return record_gw, max(bars)
    return None


def _when(record: dict, record_gw: int, gw: int, held: HeldChip) -> str:
    """What a held chip's line says about its week — rules 1 to 5 of the brief,
    in the order they are tried: this week's chip, the path's week, the
    calendar's saved-for week (with its worth), a wildcard kept, nothing."""
    if record_gw == gw and record.get("chip") == held.chip and held.allows(gw):
        return THIS_WEEK.format(gw=gw)
    week = _earliest_path_week(record, held, gw)
    if week is not None:
        return f"GW{week}"
    calendar = record.get("chip_calendar")
    if calendar is None or calendar.get("fell_back"):
        return CALENDAR_UNAVAILABLE
    entry = next(
        (item for item in calendar.get("entries") or [] if item.get("chip") == held.id),
        None,
    )
    # A wildcard is never earmarked for a week, as in the report's own line.
    # A saved-for week already gone (a record from last gameweek) is no
    # forecast, as a past path week is not.
    saved = entry.get("saved_for") if entry else None
    if saved is not None and saved >= gw and held.chip != WILDCARD:
        text = f"GW{saved}"
        if entry.get("value") is not None:
            text += WORTH.format(value=entry["value"])
        return text
    if held.chip == WILDCARD:
        window = _window(record, record_gw, entry)
        if window is not None:
            return WILDCARD_KEPT.format(first=window[0], last=window[1])
    return NO_WEEK


def render_forecast(
    record: dict,
    verdict_mode: str,
    verdict_ts: str,
    record_gw: int,
    gw: int,
    held: tuple[HeldChip, ...],
    chips_used: list[dict],
) -> str:
    """The forecast as plain text: one line per chip in the current set, one
    for the second set, one for what has been played.

    A held chip is in the *current set* when its window has opened
    (``start_event <= gw``) and in the *second set* when it has not. The
    first line names the record the plan comes from — its weekday in Dublin,
    the report's own name, its gameweek — because a forecast from last
    gameweek's deadline verdict reads differently from this week's scout.
    """
    day = datetime.fromisoformat(verdict_ts).astimezone(LOCAL_ZONE).strftime("%A")
    mode = MODE_NAMES.get(verdict_mode, verdict_mode)
    lines = [HEADER.format(day=day, mode=mode, gw=record_gw)]

    order = lambda chip: (chip.stop_event, CHIP_ORDER.index(chip.chip))  # noqa: E731
    current = sorted((chip for chip in held if chip.start_event <= gw), key=order)
    later = sorted((chip for chip in held if chip.start_event > gw), key=order)

    if not current:
        lines.append(NO_CURRENT_CHIPS)
    for chip in current:
        when = _when(record, record_gw, gw, chip)
        lines.append(
            f"{_name(chip.chip).capitalize()} — {when}"
            + EXPIRES.format(stop=chip.stop_event)
        )
    if later:
        lines.append(
            SECOND_SET.format(
                start=min(chip.start_event for chip in later),
                stop=max(chip.stop_event for chip in later),
                kinds=", ".join(_name(chip.chip) for chip in later),
            )
        )

    played = sorted(
        (
            (entry["event"], _FROM_API[str(entry.get("name", "")).strip().lower()])
            for entry in chips_used
            if str(entry.get("name", "")).strip().lower() in _FROM_API
            and isinstance(entry.get("event"), int)
        ),
        key=lambda item: item[0],
    )
    if played:
        lines.append(
            PLAYED.format(
                chips=", ".join(f"{_name(chip)} GW{event}" for event, chip in played)
            )
        )
    lines.append(FOOTER)
    return "\n".join(lines)


def chip_forecast(cfg: Config, client: FplClient, sent_at: datetime) -> str:
    """The inbox's handler for "Chip forecast": fetch, read the store, render.

    Two requests — the bootstrap and the chip history — and no solve. The plan
    is the newest early, scout or deadline record for the next gameweek, or
    failing that for the one before it (the first report of a gameweek has
    not always run). ``sent_at`` is accepted to keep the handler's shape
    beside ``transfers_made``; the forecast is the newest record whenever it
    was texted, so it is not used.

    A chip the owner has recorded for this gameweek ("Transfers made") is
    added to the history, so it is spent here exactly as the report treats
    it. The "one chip a week" shift :func:`held_in_week` can apply for a
    recorded week is deliberately not asked for: it would push every other
    current-set chip's window to next week and so list the whole current set
    as the second set. A failure fetching the chip history is allowed to
    raise: the inbox's apology covers it.
    """
    bootstrap = client.bootstrap()
    event = bootstrap.next_event()
    if event is None:
        return NO_GAMEWEEK
    gw = event.id
    # Opening a Store creates the database; one momentarily absent while the
    # hourly tick pulls must not be conjured empty under its rebase.
    if not (cfg.state_dir / DB_NAME).exists():
        return NO_PLAN.format(gw=gw)
    store = Store(cfg.state_dir / DB_NAME)
    verdict = store.latest_verdict(gw, VERDICT_MODES)
    record_gw = gw
    if verdict is None:
        record_gw = gw - 1
        verdict = store.latest_verdict(record_gw, VERDICT_MODES)
    if verdict is None:
        return NO_PLAN.format(gw=gw)

    chips_used = list(client.chips_used(cfg.team_id))
    recorded = store.executed(gw)
    if recorded is not None and recorded.chip in CHIP_API_NAMES:
        # Unless the API already lists it (after the deadline it does): once.
        api = CHIP_API_NAMES[recorded.chip]
        if not any(
            entry.get("name") == api and entry.get("event") == gw for entry in chips_used
        ):
            chips_used.append({"name": api, "event": gw})
    held = held_in_week(bootstrap, chips_used, gw)
    return render_forecast(
        verdict.decision, verdict.mode, verdict.ts, record_gw, gw, held, chips_used
    )
