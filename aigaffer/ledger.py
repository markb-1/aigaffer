"""True selling prices, kept by observation.

The public API tells anyone the current price of every player and the bank a
manager holds, and never once what he paid for anyone — and the purchase
price is the number selling actually turns on. FPL's rule: a player whose
price has risen sells at his purchase price plus half the rise, rounded DOWN
to the £0.1m the prices are quoted in; a faller sells at his current price.
A bot that prices every sale at ``now_cost`` therefore overestimates its
budget on every riser it owns, and can recommend a plan the app refuses to
enter. The bank itself is exact — the picks endpoint publishes it — so the
only number this module has to reconstruct is the per-player buy price.

It reconstructs it by watching. The store keeps a ledger of purchases
(:meth:`~aigaffer.store.Store.purchases`), and :func:`observe` maintains it
against every run's picks:

* **The seed.** On the ledger's first ever run, every current squad member is
  recorded at ``now_cost - cost_change_start`` — what he cost when the season
  opened. That is exact for a squad held since GW1, which the owner's
  currently is; a player bought mid-season before the ledger existed would be
  seeded at his season-opening price rather than what was actually paid, and
  reconciliation exists to surface exactly that kind of drift.
* **A first sighting.** A player in the squad the ledger has never seen came
  in by a transfer between runs, and is recorded at his ``now_cost`` on the
  day the run noticed him. Worst case that trails the owner's click by one
  £0.1m price move; reconciliation surfaces it.
* **A departure.** A ledger row for a player no longer held is deleted: his
  sale is settled and the proceeds are already in the published bank.

The one week this must not run is a free hit: the picks endpoint then shows
the chip's one-week temporary team, and a ledger maintained against it would
forget every real buy price and re-learn the standing squad at next week's
prices when it reverts. The whole observation — maintenance, snapshot and
reconciliation — skips that gameweek and picks up on the revert.

**Reconciliation** is the audit. Each run also snapshots what the picks
endpoint published — the gameweek, the bank, the fifteen — and on the first
run after the picks roll to a new gameweek, the previous snapshot plus the
ledger predict what the new bank should be: the old bank, plus the ledger's
selling price for everyone sold, minus the current price of everyone bought.
When the published bank disagrees, the run's report carries ONE line saying
by how much and that the selling estimates may be stale — and nothing is
auto-corrected, because the API's bank is authoritative everywhere already
and a ledger quietly rewriting itself to match would bury the very signal
the line exists to raise. A gameweek with no history behind it is skipped in
silence: a guess is not an audit.

Everything here is in integer tenths of a million, exactly as the API quotes
prices, which is what lets floor division be the game's own rounding.
"""

from dataclasses import dataclass, field

from aigaffer.data.models import Player, Squad
from aigaffer.report.render import price
from aigaffer.store import Store

# What the FPL API calls a free hit in a chip history. The manager package
# keeps the full chip-name translation; this module needs the one chip whose
# week it must sit out, and repeating the API's own spelling here keeps the
# data layer from importing the manager's.
FREE_HIT_API_NAME = "freehit"

# The one line a discrepant reconciliation puts in the report. The amount and
# the direction, in pounds, and the two facts a reader needs: the estimates
# may be stale, and the bank the game publishes is the one that counts — it
# already is, everywhere; this line only says why the two numbers differ.
DISCREPANCY = (
    "Note: the bank is {delta} {direction} what the purchase ledger predicted"
    " for last gameweek's moves — selling-price estimates may be stale; the"
    " API's published bank stays authoritative."
)


def selling_price(buy: int, now: int) -> int:
    """What the game pays when a player bought at ``buy`` is sold at ``now``.

    Integer tenths in, integer tenths out. A faller — or a price that never
    moved — sells at the current price; a riser gives half the rise back,
    rounded DOWN to a whole tenth, which is what integer floor division does
    without being asked: a £0.1m rise halves to £0.05m and rounds to nothing,
    so buy 50, now 51 sells at 50.
    """
    if now <= buy:
        return now
    return buy + (now - buy) // 2


@dataclass(frozen=True)
class Observation:
    """One run's reading of the ledger.

    ``selling_prices`` is what each held player would actually sell for,
    player id to tenths — the dict the solvers put on the sale side of every
    budget. A held player absent from it is one whose current price nobody
    knows (gone from the bootstrap) or, on a skipped free-hit week, one the
    ledger never priced; consumers fall back to ``now_cost`` for him.
    ``note`` is the reconciliation line for the report, or None on the great
    majority of weeks when there is nothing to say.
    """

    selling_prices: dict[int, int] = field(default_factory=dict)
    note: str | None = None


def observe(
    store: Store,
    squad: Squad | None,
    players: dict[int, Player],
    chips_used: list[dict] | tuple = (),
    persist: bool = True,
) -> Observation:
    """Maintain the ledger against this run's picks and price the sales.

    Runs on every mode that fetched a squad — the data is already in hand and
    a reminder that skipped it would leave the ledger a run behind. A draft
    week has no squad and touches nothing; a free-hit gameweek (read off
    ``chips_used``, the API's own chip history) is skipped whole, because its
    picks are the chip's temporary team and not the squad we own.

    ``persist`` is the dry-run gate: False computes everything in memory — the
    solve still needs selling prices — and writes nothing, neither ledger rows
    nor the squad snapshot, so a dry run leaves the store byte-for-byte alone.

    Reconciliation runs before maintenance on purpose: it needs the buy
    prices of the players who just left, and maintenance is about to delete
    those rows.
    """
    if squad is None:
        return Observation()
    if _free_hit_week(chips_used, squad.event):
        ledger = store.purchases()
        return Observation(selling_prices=_priced(ledger, squad, players))

    ledger = store.purchases()
    note = _reconcile(store, squad, players, ledger)

    held = [pid for pid in squad.player_ids if pid in players]
    if not ledger:
        # The seed: the season-opening price, exact for a squad held since GW1.
        additions = {
            pid: players[pid].now_cost - players[pid].cost_change_start
            for pid in held
        }
    else:
        additions = {
            pid: players[pid].now_cost for pid in held if pid not in ledger
        }
    departures = [pid for pid in ledger if pid not in set(squad.player_ids)]

    for pid in departures:
        del ledger[pid]
    ledger.update(additions)

    if persist:
        for pid, buy in additions.items():
            store.record_purchase(pid, buy, squad.event)
        for pid in departures:
            store.forget_purchase(pid)
        store.record_squad(squad.event, squad.bank, squad.player_ids)

    return Observation(selling_prices=_priced(ledger, squad, players), note=note)


def _priced(
    ledger: dict[int, int], squad: Squad, players: dict[int, Player]
) -> dict[int, int]:
    """Each held player's selling price, for everyone the ledger can price.

    After maintenance that is the whole squad; the players it cannot price —
    gone from the bootstrap, or unledgered on a skipped free-hit week — are
    left out rather than guessed at, and every consumer falls back to
    ``now_cost`` for a missing id.
    """
    return {
        pid: selling_price(ledger[pid], players[pid].now_cost)
        for pid in squad.player_ids
        if pid in ledger and pid in players
    }


def _free_hit_week(chips_used: list[dict] | tuple, event: int) -> bool:
    """Was the free hit played in the gameweek these picks belong to?"""
    return any(
        chip.get("name") == FREE_HIT_API_NAME and chip.get("event") == event
        for chip in chips_used
    )


def _reconcile(
    store: Store, squad: Squad, players: dict[int, Player], ledger: dict[int, int]
) -> str | None:
    """The audit line for a gameweek the picks just rolled into, or None.

    Fires at most once per gameweek: only on a run whose gameweek has no
    snapshot yet — the first run after the roll — and only when there is a
    previous snapshot and a ledger to predict from. The prediction prices
    each sale with the ledger's buy price against today's ``now_cost``, which
    is the best reconstruction available a run later; a price that moved
    between the deadline and this run is part of the very staleness the line
    reports. A move involving anyone the ledger or the bootstrap cannot
    price is not audited at all — silence over a guess.
    """
    if not ledger:
        return None
    if store.squad_record(squad.event) is not None:
        return None
    previous = store.last_squad_before(squad.event)
    if previous is None:
        return None

    before, after = set(previous["player_ids"]), set(squad.player_ids)
    sold, bought = before - after, after - before
    if any(pid not in players for pid in sold | bought):
        return None
    if any(pid not in ledger for pid in sold):
        return None

    predicted = (
        previous["bank"]
        + sum(selling_price(ledger[pid], players[pid].now_cost) for pid in sold)
        - sum(players[pid].now_cost for pid in bought)
    )
    if predicted == squad.bank:
        return None
    delta = squad.bank - predicted
    return DISCREPANCY.format(
        delta=price(abs(delta)), direction="above" if delta > 0 else "below"
    )
