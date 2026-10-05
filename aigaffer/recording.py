"""Recording "Transfers made": which verdict, the position it leaves, the echo.

The owner texts "Transfers made" after entering the latest recommendation
exactly — its transfers and its chip. This module turns that into the
gameweek's :class:`~aigaffer.executed.Executed` row and the reply that echoes
it back, so a mismatch is spotted on the phone and not at the deadline.

The row is **cumulative**. He may act on Tuesday's scout and then on a
further move from Thursday's deadline verdict, which was solved from the
squad Tuesday left; a second text composes Thursday's verdict onto the row
rather than replacing it, and a re-send of the same verdict changes nothing.
Which squad a verdict was solved from is on the verdict itself
(``squad_before``): built on the row, it composes; built on the API's squad
— the other scheduler raced the row — it *is* the position and replaces the
row; built on neither, it is refused rather than applied to a squad it never
saw. Every cumulative fact is recomputed against the API's squad, so a
reversal nets out by construction.

Prices are the live bootstrap's at recording — the nearest the bot gets to
the moment he clicked — and sales are the ledger's selling prices then. A
nightly price move between entering and texting is the one gap, which is why
the help text tells him to text straight away.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.data.models import Bootstrap, Player, Squad
from aigaffer.executed import NO_CHIP, Executed, StaleVerdict, Verdict
from aigaffer.ledger import observe, selling_price
from aigaffer.report.render import POSITIONS, entered_moves, plural, price
from aigaffer.statesync import StateSync, state_lock
from aigaffer.store import DB_NAME, Store

# The reports a person can enter. The reminder is a check on one of these,
# never a verdict of its own.
VERDICT_MODES = ("early", "scout", "deadline")

# The report as the echo names it; the early scout is the scout and says so.
MODE_NAMES = {"early": "early scout", "scout": "scout", "deadline": "deadline"}

RECORDED = "Recorded for GW{gw} (from {day}'s {mode} verdict):"
LATER_RUNS = "Later runs this gameweek work from this squad."

# A wildcard or a free hit leaves the free transfers where they were: FPL
# banks them through either chip.
BANKING_CHIPS = (WILDCARD, FREE_HIT)


def compose(
    row: Executed | None,
    verdict: Verdict,
    real_squad: list[int],
    real_bank: int,
    players: dict[int, Player],
    selling_prices: dict[int, int],
    now: datetime,
) -> Executed:
    """The gameweek's position once ``verdict`` is entered on top of ``row``.

    ``real_squad``/``real_bank`` are the API's picks — what every cumulative
    fact is measured against. ``players`` is the live bootstrap, read for
    the price each signing cost and his status flag now. ``selling_prices``
    is the ledger's reading of the real squad's sales.

    Returns ``row`` itself when the verdict is already in it — the re-send
    that must change nothing. Raises :class:`StaleVerdict` for a verdict
    solved from a squad that is neither the row's nor the API's.
    """
    pair = [verdict.mode, verdict.ts]
    if row is not None and pair in row.verdicts:
        return row

    decision = verdict.decision
    base = _base(row, decision.get("squad_before"), real_squad)
    base_squad = real_squad if base is None else base.squad_after
    base_bank = real_bank if base is None else base.bank_after
    bought_before = {} if base is None else base.buy_prices
    sold_before = {} if base is None else base.sell_prices

    if (
        base is not None
        and (decision.get("chip") or NO_CHIP) == FREE_HIT
        and (base.transfers_in or base.transfers_out)
    ):
        return _free_hit_over_moves(base, verdict, real_squad, real_bank, players, now)

    ins = list(decision.get("transfers_in") or [])
    outs = list(decision.get("transfers_out") or [])

    bank = base_bank
    for pid in outs:
        listed = players[pid].now_cost
        if pid in bought_before:
            # A signing from earlier this week sold again. The week-1 lock
            # means no verdict does this; if one ever did, the game would
            # pay the selling rule on the price he paid.
            bank += selling_price(bought_before[pid], listed)
        else:
            bank += selling_prices.get(pid, listed)
    for pid in ins:
        bank -= players[pid].now_cost

    squad_after = sorted((set(base_squad) - set(outs)) | set(ins))
    real = set(real_squad)
    transfers_in = sorted(set(squad_after) - real)
    transfers_out = sorted(real - set(squad_after))
    buy_prices = {
        pid: bought_before.get(pid, players[pid].now_cost) for pid in transfers_in
    }
    sell_prices = {
        pid: sold_before.get(pid, selling_prices.get(pid, players[pid].now_cost))
        for pid in transfers_out
    }

    chip = decision.get("chip") or NO_CHIP
    if chip == NO_CHIP and base is not None:
        chip = base.chip
    free = decision.get("free_transfers") or 0
    # The week's free transfers before any recorded move: the verdict's own
    # when it was solved from the API's squad, else carried on the row —
    # recording never fetches the transfer history to work it out again.
    ft_before = free if base is None else base.ft_before
    # A wildcard or free hit played after earlier moves absorbs them: their
    # hits are cancelled and the free-transfer bank stands where it did
    # before the week's moves. An ordinary week spends one a move.
    ft_after = ft_before if chip in BANKING_CHIPS else max(0, free - len(ins))

    freehit_squad = freehit_xi = None
    if chip == FREE_HIT:
        freehit_squad = decision.get("freehit_squad") or (
            None if base is None else base.freehit_squad
        )
        freehit_xi = decision.get("freehit_xi") or (
            None if base is None else base.freehit_xi
        )

    arrivals = set(transfers_in) | (set(freehit_squad or []) - set(squad_after))
    known = {} if base is None else base.arrival_status
    arrival_status = {
        pid: known.get(pid, players[pid].status)
        for pid in sorted(arrivals)
        if pid in known or pid in players
    }

    return Executed(
        gw=int(decision["event"]),
        mode=verdict.mode,
        recorded_at=now.isoformat(),
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        squad_after=squad_after,
        chip=chip,
        captain=int(decision["captain"]),
        vice=int(decision["vice"]),
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        bank_after=bank,
        ft_after=ft_after,
        ft_before=ft_before,
        verdicts=([] if base is None else [list(p) for p in base.verdicts]) + [pair],
        freehit_squad=freehit_squad,
        freehit_xi=freehit_xi,
        arrival_status=arrival_status,
    )


def _free_hit_over_moves(
    base: Executed,
    verdict: Verdict,
    real_squad: list[int],
    real_bank: int,
    players: dict[int, Player],
    now: datetime,
) -> Executed:
    """A free hit played after earlier moves this gameweek.

    FPL reverts a free-hit week to the *previous* deadline's squad, and that
    undoes the week's earlier transfers too: hits refunded, bank restored, the
    free-transfer bank left at its pre-week value. So the position is the
    API's squad and bank with nothing locked (a hold week), and the earlier
    signings are *dropped* — buy/sell prices and arrival flags with them — so
    that after the deadline the ledger never seeds a purchase for a player
    the owner did not keep. The verdicts stay, so a re-send is still a no-op
    and the echo can still say which verdict this was.
    """
    decision = verdict.decision
    freehit_squad = decision.get("freehit_squad") or None
    freehit_xi = decision.get("freehit_xi") or None
    arrivals = set(freehit_squad or []) - set(real_squad)
    return Executed(
        gw=int(decision["event"]),
        mode=verdict.mode,
        recorded_at=now.isoformat(),
        transfers_in=[],
        transfers_out=[],
        squad_after=sorted(real_squad),
        chip=FREE_HIT,
        captain=int(decision["captain"]),
        vice=int(decision["vice"]),
        buy_prices={},
        sell_prices={},
        bank_after=real_bank,
        ft_after=base.ft_before,
        ft_before=base.ft_before,
        verdicts=[list(p) for p in base.verdicts] + [[verdict.mode, verdict.ts]],
        freehit_squad=freehit_squad,
        freehit_xi=freehit_xi,
        arrival_status={
            pid: players[pid].status for pid in sorted(arrivals) if pid in players
        },
    )


def _base(
    row: Executed | None, squad_before: list[int] | None, real_squad: list[int]
) -> Executed | None:
    """The recorded position a verdict builds on — the row, or None for the
    API's squad — or :class:`StaleVerdict` when it builds on neither.

    A record from before ``squad_before`` was kept cannot say, and the only
    such records in a live store predate the inbox entirely; the row is the
    honest guess when there is one.
    """
    if squad_before is None:
        return row
    if row is not None and set(squad_before) == set(row.squad_after):
        return row
    if set(squad_before) == set(real_squad):
        return None
    raise StaleVerdict("the verdict was solved from a squad nobody recorded")


def echo(row: Executed, verdict: Verdict, players: dict[int, Player]) -> str:
    """The reply: the whole new position, so a mismatch shows on the phone.

    Which verdict it matches (by weekday and report), the moves and the
    armbands, the fifteen by position — and the free-hit team beside it on
    a free-hit week — then the money and the free transfers left.
    """
    day = datetime.fromisoformat(verdict.ts).astimezone(UTC).strftime("%A")
    mode = MODE_NAMES.get(verdict.mode, verdict.mode)
    moves = entered_moves(row, players)
    lines = [
        RECORDED.format(gw=row.gw, day=day, mode=mode),
        f"{moves[0].upper()}{moves[1:]}."
        f" {_name(row.captain, players)} (C), {_name(row.vice, players)} (V).",
        f"Squad: {_by_position(row.squad_after, players)}",
    ]
    if row.chip == FREE_HIT and row.freehit_squad:
        lines.append(f"Free Hit team: {_by_position(row.freehit_squad, players)}")
    undone = _undone_moves(row, verdict, players)
    if undone:
        earlier = datetime.fromisoformat(row.verdicts[-2][1]).astimezone(UTC)
        lines.append(
            f"Free Hit played: {earlier.strftime('%A')}'s moves ({undone})"
            " are undone for this gameweek."
        )
    lines.append(
        f"Bank {price(row.bank_after)}"
        f" · {plural(row.ft_after, 'free transfer')} left"
    )
    lines.append(LATER_RUNS)
    return "\n".join(lines)


def _undone_moves(
    row: Executed, verdict: Verdict, players: dict[int, Player]
) -> str:
    """The earlier moves a free hit has undone, or "" when it undid none.

    The row has dropped them, but the verdict says which squad it was solved
    from: against the row's (real) squad, the difference is what was undone.
    """
    before = verdict.decision.get("squad_before")
    if row.chip != FREE_HIT or len(row.verdicts) < 2 or not before:
        return ""
    gone = sorted(set(before) - set(row.squad_after))  # signed earlier
    back = sorted(set(row.squad_after) - set(before))  # sold earlier
    return ", ".join(
        f"{_name(out, players)} → {_name(bought, players)}"
        for out, bought in zip(back, gone)
    )


def _name(pid: int, players: dict[int, Player]) -> str:
    player = players.get(pid)
    return player.web_name if player is not None else f"player {pid}"


def _by_position(pids: list[int], players: dict[int, Player]) -> str:
    """``GKP Alvez, Byrne · DEF …`` — keeper to forwards, names sorted within
    a position; an id the board has never heard of goes on the end as an id."""
    groups = []
    for position, label in POSITIONS.items():
        names = sorted(
            players[pid].web_name
            for pid in pids
            if pid in players and players[pid].element_type == position
        )
        if names:
            groups.append(f"{label} {', '.join(names)}")
    unknown = [f"player {pid}" for pid in pids if pid not in players]
    if unknown:
        groups.append(", ".join(unknown))
    return " · ".join(groups)


NO_GAMEWEEK = "Nothing to record: the API has no gameweek ahead."
NOTHING_YET = "Nothing to record yet: no recommendation for GW{gw} has gone out."
CLOSED = "GW{gw} has closed; the API now shows your squad."
NO_SQUAD = "Nothing to record: the API shows no squad to work from."
STALE = (
    "Not recorded: GW{gw}'s latest verdict was worked out from a squad that is"
    " neither the API's nor the one on record. Wait for the next report, enter"
    " it, and text again."
)
PUBLISH_MESSAGE = "chore: record transfers made (inbox)"


@dataclass(frozen=True)
class Outcome:
    """The reply to send, and whether the store changed (so state is pushed)."""

    reply: str
    changed: bool


def record_transfers_made(
    store: Store,
    bootstrap: Bootstrap,
    squad: Squad | None,
    chips_used: list[dict],
    sent_at: datetime,
    now: datetime,
) -> Outcome:
    """Record the verdict the owner says he entered, and say what was recorded.

    The verdict is the newest early, scout or deadline record for the next
    gameweek saved at or before ``sent_at`` — the text's own time, because
    the inbox may have waited minutes on the hourly tick's lock and a verdict
    that landed meanwhile is one he never saw. None for the next gameweek but
    one for the gameweek before is a text about a gameweek that has closed.

    The sales are priced by the ledger read in memory (``persist=False``):
    recording writes the row and nothing else.
    """
    event = bootstrap.next_event()
    if event is None:
        return Outcome(NO_GAMEWEEK, False)
    verdict = store.latest_verdict(event.id, VERDICT_MODES, at_or_before=sent_at)
    if verdict is None:
        if store.latest_verdict(event.id - 1, VERDICT_MODES, at_or_before=sent_at):
            return Outcome(CLOSED.format(gw=event.id - 1), False)
        return Outcome(NOTHING_YET.format(gw=event.id), False)
    if squad is None:
        return Outcome(NO_SQUAD, False)

    players = {player.id: player for player in bootstrap.elements}
    row = store.executed(event.id)
    ledger = observe(store, squad, players, chips_used, persist=False)
    try:
        new = compose(
            row, verdict, squad.player_ids, squad.bank, players,
            ledger.selling_prices, now,
        )
    except StaleVerdict:
        return Outcome(STALE.format(gw=event.id), False)
    if new is row:
        return Outcome(echo(row, verdict, players), False)
    store.save_executed(new)
    return Outcome(echo(new, verdict, players), True)


def transfers_made(
    cfg: Config,
    client: FplClient,
    sync: StateSync,
    lock_path: Path,
    sent_at: datetime,
    now: datetime | None = None,
) -> str:
    """The inbox's handler for "Transfers made": fetch, lock, record, push.

    Three requests and no more — the bootstrap, the picks and the chip
    history — because the full fetch is two hundred and the inbox runs every
    minute. They are made before the lock is taken: the lock is shared with
    the hourly tick and is held only for what touches the state — the pull,
    the row (a text file, ``state/executed/gw{n}.json``), the commit and the
    push. A push that fails is said on the log and the reply still goes; the
    row is committed locally and the next hourly tick pushes it.
    """
    bootstrap = client.bootstrap()
    current = bootstrap.current_event()
    squad = None
    if current is not None:
        try:
            squad = client.picks(cfg.team_id, current.id)
        except httpx.HTTPStatusError:
            squad = None
    chips_used = [] if squad is None else client.chips_used(cfg.team_id)

    with state_lock(lock_path):
        sync.pull()
        outcome = record_transfers_made(
            Store(cfg.state_dir / DB_NAME),
            bootstrap,
            squad,
            chips_used,
            sent_at,
            now or datetime.now(UTC),
        )
        if outcome.changed:
            sync.publish(PUBLISH_MESSAGE)
    return outcome.reply
