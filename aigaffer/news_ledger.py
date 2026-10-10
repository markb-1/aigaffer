"""The news ledger: what the gaffer learned about a player, kept between runs.

Every run used to research team news from scratch and forget it at the end:
it ran out of searches, called the same news two ways on two days, and the
on-demand what-if started knowing nothing. The ledger keeps the latest thing
known about each player — what kind of news (``category``), how good the
source (``tier``), and the date of the *quote*, not of the article that
recycled it — in ``state/news/ledger.json``, beside the rest of the state the
runs commit. It is public football news and nothing else.

An entry is not trusted on a timer. Football news arrives at known points —
the club's press conference, FPL's flag update after it, a match — so an
entry stays fresh until the earlier of its category's time limit and the next
of those points (spec: news-ledger design §5, from the FPL expert). One pure
function, :func:`evaluate_ledger`, judges the whole file once per run with one
clock; the briefing, the record and (Phase B) the minutes all read that one
judgement.

Phase A applies no minutes from here: a fresh entry is shown to the gaffer,
who carries it into ``adjust_players`` as it stands.
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from aigaffer.config import DEADLINE_ANCHOR_HOURS
from aigaffer.data.models import Event, Fixture, Player

LEDGER_DIR = "news"
LEDGER_FILE = "ledger.json"

NAILED, ROTATION, DOUBT, TEST_ON_DAY = "nailed", "rotation", "doubt", "test_on_day"
INJURED, SUSPENDED, ILL = "injured", "suspended", "ill"
CATEGORIES = (NAILED, ROTATION, DOUBT, TEST_ON_DAY, INJURED, SUSPENDED, ILL)
# 1: FPL's flag, the manager's own words, an official club update. 2: club
# reporters, The Athletic, BBC, Sky, Premier Injuries. 3: predicted line-ups.
# Social media is not a tier: it may prompt a search and never sets anything.
TIERS = (1, 2, 3)

# Searches a run should need when the ledger is doing its job (the expert's
# estimates). Passing one is a defect signal on the log, never a stop.
SEARCH_BUDGET = {"early": 6, "scout": 6, "deadline": 10}

# How long an entry stays fresh after it was checked, by category. A test on
# the day is never fresh on read (one look per run); a nailed starter is also
# capped at the gameweek after his entry's.
LIMITS = {
    NAILED: timedelta(days=7),
    ROTATION: timedelta(hours=72),
    DOUBT: timedelta(hours=24),
    ILL: timedelta(hours=24),
    INJURED: timedelta(days=14),
    SUSPENDED: timedelta(days=14),
}
# The kinds of news a new gameweek, or a press conference, overtakes.
SHORT_LIVED = (DOUBT, ROTATION, ILL, TEST_ON_DAY)
PRESSER_CATEGORIES = (ROTATION, DOUBT, ILL)
# Pre-match press conferences are mandatory and fall 18–72 hours before the
# club's kickoff: done for certain by K − 18h, and a quote dated before the
# day of K − 72h was given before the window opened.
PRESSER_LAST = timedelta(hours=18)
PRESSER_FIRST = timedelta(hours=72)

FPL_FIELDS = ("status", "chance_of_playing_next_round", "news", "news_added")

FPL_MOVED = "FPL's flag or news has changed"
CLUB_PLAYED = "his club has played since"
NEW_GAMEWEEK = "a new gameweek"
ON_THE_DAY = "a late fitness test — one look per run"
RETURN_DUE = "his return gameweek has come"
TIME_LIMIT = "older than its time limit"
PRESSER_SINCE = "his club's press conference came after the quote"

UNREADABLE = "aigaffer news ledger: unreadable ({reason}) — starting empty"
WRITE_FAILED = "aigaffer news ledger: not written ({reason})"


@dataclass(frozen=True)
class Entry:
    """The latest thing known about one player (spec §4)."""

    player_id: int
    gw: int
    category: str
    expected_minutes: float
    tier: int
    quote_date: str
    source: str | None
    note: str
    return_gw: int | None
    checked_at: str
    run: str
    fpl: dict

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict) -> "Entry":
        """Raises KeyError or TypeError on a malformed entry; the reader skips it."""
        return cls(**{field.name: raw[field.name] for field in fields(cls)})


@dataclass(frozen=True)
class Judged:
    """An entry, and whether it is fresh — and if not, the rule that said so."""

    entry: Entry
    fresh: bool
    reason: str | None


@dataclass(frozen=True)
class LedgerView:
    """The whole ledger judged once, with one clock, for one run."""

    entries: dict[int, Judged]


def ledger_path(state_dir: Path) -> Path:
    """Where the ledger lives: beside the database, in the state the runs commit."""
    return state_dir / LEDGER_DIR / LEDGER_FILE


def read_ledger(path: Path) -> dict:
    """The raw ledger, or ``{}``. A missing file is a ledger nobody has written
    yet and is said nowhere; an unreadable one is one line on stdout — the
    exception's class, never its words — and the run carries on without it."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as error:
        print(UNREADABLE.format(reason=type(error).__name__))
        return {}
    try:
        raw = json.loads(text)
    except ValueError as error:
        print(UNREADABLE.format(reason=type(error).__name__))
        return {}
    if not isinstance(raw, dict):
        print(UNREADABLE.format(reason=type(raw).__name__))
        return {}
    return raw


def evaluate_ledger(
    raw: dict, bootstrap, fixtures: list[Fixture], event: Event, now: datetime
) -> LedgerView:
    """Every readable entry for a player still in the game, judged at ``now``.

    ``bootstrap`` is read for its ``elements`` alone. A malformed entry, or one
    for a player the bootstrap no longer has, is dropped without a word: the
    next write leaves it out of the file too.
    """
    players = {player.id: player for player in bootstrap.elements}
    judged: dict[int, Judged] = {}
    for value in raw.values():
        try:
            entry = Entry.from_json(value)
            checked = _instant(entry.checked_at)
            date.fromisoformat(entry.quote_date)
            float(entry.expected_minutes)
            if entry.category not in CATEGORIES or entry.tier not in TIERS:
                continue
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        player = players.get(entry.player_id)
        if player is None:
            continue
        reason = _stale(entry, checked, player, fixtures, event, _aware(now))
        judged[entry.player_id] = Judged(entry, reason is None, reason)
    return LedgerView(judged)


def write_ledger(
    path: Path,
    view: LedgerView,
    records: list[dict],
    *,
    gw: int,
    run: str,
    now: datetime,
    players: dict[int, Player],
) -> None:
    """Rewrite the file from ``view`` plus this run's ``records``, atomically.

    ``records`` are the gaffer's ``adjust_players`` calls in the order he made
    them, applied or not, so a later call for a player wins. A record for a
    player not on the board is skipped. Each new entry is stamped with this
    run's clock, mode and gameweek and with the player's FPL fields now — the
    snapshot a later run compares against. Written to a temp file and renamed
    over the old one, with sorted keys, so a crash leaves the last good file
    and a change is a readable diff in the state commit.
    """
    entries = {pid: judged.entry.to_json() for pid, judged in view.entries.items()}
    for record in records:
        player = players.get(record["player_id"])
        if player is None:
            continue
        entries[player.id] = Entry(
            player_id=player.id,
            gw=gw,
            category=record["category"],
            expected_minutes=record["expected_minutes"],
            tier=record["tier"],
            quote_date=record["quote_date"],
            source=record.get("source"),
            note=record["reason"],
            return_gw=record.get("return_gw"),
            checked_at=_aware(now).isoformat(),
            run=run,
            fpl=_snapshot(player),
        ).to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps({str(pid): entries[pid] for pid in sorted(entries)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stale(
    entry: Entry, checked: datetime, player: Player, fixtures: list[Fixture], event: Event, now: datetime
) -> str | None:
    """The first rule that makes ``entry`` a re-check, or None when it is fresh."""
    if _snapshot(player) != entry.fpl:
        return FPL_MOVED
    if _club_played_since(player.team, fixtures, checked):
        return CLUB_PLAYED
    if event.id > entry.gw and entry.category in SHORT_LIVED:
        return NEW_GAMEWEEK
    if entry.category == TEST_ON_DAY:
        return ON_THE_DAY
    if entry.category in (INJURED, SUSPENDED) and entry.return_gw is not None and event.id >= entry.return_gw:
        return RETURN_DUE
    if entry.category == NAILED and event.id > entry.gw + 1:
        return TIME_LIMIT
    if now - checked > LIMITS[entry.category]:
        return TIME_LIMIT
    if entry.category in PRESSER_CATEGORIES and _presser_since(entry, player.team, fixtures, event, now):
        return PRESSER_SINCE
    return None


def _presser_since(entry: Entry, team: int, fixtures: list[Fixture], event: Event, now: datetime) -> bool:
    """Spec §5 rule 3: the club's press conference is done and the quote is older
    than its window. Done by ``K − 18h`` for certain, or at the deadline run for
    every club — one that has not spoken by then speaks after the FPL deadline,
    so nothing better can arrive. No fixture this gameweek: no rule."""
    kickoffs = [
        _instant(fixture.kickoff_time)
        for fixture in fixtures
        if fixture.event == event.id and team in (fixture.team_h, fixture.team_a) and fixture.kickoff_time
    ]
    if not kickoffs:
        return False
    kickoff = min(kickoffs)
    deadline_run = _aware(event.deadline_time) - timedelta(hours=DEADLINE_ANCHOR_HOURS)
    done = now >= kickoff - PRESSER_LAST or now >= deadline_run
    return done and date.fromisoformat(entry.quote_date) < (kickoff - PRESSER_FIRST).date()


def _club_played_since(team: int, fixtures: list[Fixture], checked: datetime) -> bool:
    """Has ``team`` played a match — full time is enough — that kicked off after ``checked``?"""
    return any(
        fixture.played
        and team in (fixture.team_h, fixture.team_a)
        and fixture.kickoff_time
        and _instant(fixture.kickoff_time) > checked
        for fixture in fixtures
    )


def _snapshot(player: Player) -> dict:
    return {name: getattr(player, name) for name in FPL_FIELDS}


def _instant(text: str) -> datetime:
    """An ISO timestamp, with or without the API's ``Z``, as an aware datetime."""
    return _aware(datetime.fromisoformat(text.replace("Z", "+00:00")))


def _aware(moment: datetime) -> datetime:
    """A naive clock is UTC, as a naive deadline is elsewhere in the pipeline."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
