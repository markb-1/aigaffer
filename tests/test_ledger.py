"""Tests for the purchase ledger: the selling rule, the bookkeeping, the audit.

Small universes built by hand, because everything here is arithmetic on a few
prices: a player is an id, a current price and — for the seed — how far that
price has moved since the season opened. Prices are integer tenths of a
million throughout, exactly as the API quotes them.
"""

from aigaffer.data.models import Player, Squad
from aigaffer.ledger import observe, selling_price
from aigaffer.store import Store


def player(pid: int, cost: int, change: int = 0) -> Player:
    return Player(
        id=pid,
        web_name=f"P{pid}",
        team=pid,
        element_type=3,
        now_cost=cost,
        status="a",
        minutes=900,
        starts=10,
        total_points=0,
        bonus=0,
        saves=0,
        cost_change_start=change,
    )


def squad_of(ids: list[int], bank: int = 0, event: int = 2) -> Squad:
    return Squad(
        picks=[
            {
                "element": pid,
                "position": index + 1,
                "is_captain": False,
                "is_vice_captain": False,
            }
            for index, pid in enumerate(ids)
        ],
        bank=bank,
        event=event,
    )


def store_at(tmp_path) -> Store:
    return Store(tmp_path / "aigaffer.db")


# --- the selling rule ------------------------------------------------------


def test_a_riser_sells_at_purchase_plus_half_the_rise_rounded_down():
    # Bought at £5.0m, now £5.6m: half of the 0.6 rise is 0.3, sells £5.3m.
    assert selling_price(50, 56) == 53
    # An odd rise floors: half of 0.5 is 0.25, which rounds DOWN to 0.2.
    assert selling_price(50, 55) == 52


def test_a_single_tick_rise_sells_at_the_purchase_price():
    # The £0.1m edge: half of one tick is half a tick, and the game quotes
    # prices in whole tenths, so the profit rounds away to nothing.
    assert selling_price(50, 51) == 50


def test_a_faller_sells_at_the_current_price():
    assert selling_price(50, 45) == 45


def test_an_unmoved_price_sells_at_itself():
    assert selling_price(50, 50) == 50


# --- seeding, sighting, forgetting ----------------------------------------


def test_the_first_run_seeds_the_ledger_from_the_seasons_price_moves(tmp_path):
    # An empty ledger and a squad: every member is seeded at what he cost when
    # the season opened — now_cost minus cost_change_start — which is exact for
    # a squad held since GW1. Player 1 rose two ticks, player 2 fell one,
    # player 3 never moved.
    store = store_at(tmp_path)
    players = {1: player(1, 52, change=2), 2: player(2, 44, change=-1), 3: player(3, 60)}

    observed = observe(store, squad_of([1, 2, 3], event=4), players)

    assert store.purchases() == {1: 50, 2: 45, 3: 60}
    # And the sales are priced off that seed: the riser gives half his rise
    # back (52 -> 51), the faller sells at his current 44, the still man at 60.
    assert observed.selling_prices == {1: 51, 2: 44, 3: 60}


def test_a_player_the_ledger_has_never_seen_is_recorded_at_todays_price(tmp_path):
    # Player 4 arrived by a transfer between runs: his first sighting is
    # recorded at his current price, with the gameweek it happened in.
    store = store_at(tmp_path)
    store.record_purchase(1, buy_price=50, gw_seen=1)
    players = {1: player(1, 50), 4: player(4, 75)}

    observe(store, squad_of([1, 4], event=5), players)

    assert store.purchases() == {1: 50, 4: 75}


def test_a_player_who_left_the_squad_leaves_the_ledger(tmp_path):
    store = store_at(tmp_path)
    store.record_purchase(1, buy_price=50, gw_seen=1)
    store.record_purchase(9, buy_price=80, gw_seen=1)
    players = {1: player(1, 50)}

    observe(store, squad_of([1], event=5), players)

    assert store.purchases() == {1: 50}


def test_a_ledgered_player_missing_from_the_bootstrap_keeps_his_row(tmp_path):
    # Still in the squad, but the bootstrap no longer serves him — a renumbered
    # or suspended element. His buy price is a fact worth keeping; his selling
    # price needs a current price nobody has, so it is simply not offered.
    store = store_at(tmp_path)
    store.record_purchase(1, buy_price=50, gw_seen=1)
    store.record_purchase(2, buy_price=45, gw_seen=1)
    players = {1: player(1, 56)}

    observed = observe(store, squad_of([1, 2], event=5), players)

    assert store.purchases() == {1: 50, 2: 45}
    assert observed.selling_prices == {1: 53}


def test_a_dry_run_prices_the_sales_but_persists_nothing(tmp_path):
    # The solve still needs selling prices, so they are computed in memory;
    # the ledger and the squad memory are left exactly as they were.
    store = store_at(tmp_path)
    players = {1: player(1, 52, change=2), 2: player(2, 44, change=-1)}

    observed = observe(store, squad_of([1, 2], event=4), players, persist=False)

    assert observed.selling_prices == {1: 51, 2: 44}
    assert store.purchases() == {}
    assert store.squad_record(4) is None


def test_a_draft_week_touches_nothing(tmp_path):
    # No squad yet: nothing to seed, nothing to price, nothing to reconcile.
    store = store_at(tmp_path)

    observed = observe(store, None, {1: player(1, 50)})

    assert observed.selling_prices == {}
    assert observed.note is None
    assert store.purchases() == {}


def test_a_free_hit_week_leaves_the_ledger_alone(tmp_path):
    # On a free-hit gameweek the picks endpoint shows the one-week temporary
    # team, not the squad we own. Maintained against it, the ledger would
    # forget every real buy price and re-learn the whole squad at next week's
    # prices — so the week is skipped outright, snapshot included, and the
    # standing squad's ledger waits for the revert.
    store = store_at(tmp_path)
    store.record_purchase(1, buy_price=50, gw_seen=1)
    players = {1: player(1, 56), 8: player(8, 90)}
    chips_used = [{"name": "freehit", "event": 6}]

    observed = observe(store, squad_of([8], event=6), players, chips_used)

    assert store.purchases() == {1: 50}
    assert store.squad_record(6) is None
    assert observed.selling_prices == {}
    assert observed.note is None


def test_every_run_records_what_the_picks_endpoint_said(tmp_path):
    store = store_at(tmp_path)
    players = {1: player(1, 50), 2: player(2, 45)}

    observe(store, squad_of([1, 2], bank=13, event=4), players)

    assert store.squad_record(4) == {"gw": 4, "bank": 13, "player_ids": [1, 2]}


# --- reconciliation --------------------------------------------------------
#
# After a deadline the picks roll to a new gameweek, and the ledger's selling
# estimates meet the one number that can grade them: the bank the game
# actually published. Predicted bank = the previous gameweek's bank, plus what
# the ledger says the sold players raised, minus what the bought ones cost.
# A clean week is silent; a discrepant one earns exactly one line; a week with
# no history to predict from is silent too, because a guess is not an audit.


def audit_board() -> dict[int, Player]:
    """Player 3 was bought at 50 and now stands at 56, so the ledger says his
    sale raised 53; player 4 costs 60 to buy. From a previous bank of 10, the
    swap 3 -> 4 predicts a bank of 10 + 53 - 60 = 3."""
    return {
        1: player(1, 50),
        2: player(2, 45),
        3: player(3, 56),
        4: player(4, 60),
    }


def audited_store(tmp_path) -> Store:
    store = store_at(tmp_path)
    for pid, buy in ((1, 50), (2, 45), (3, 50)):
        store.record_purchase(pid, buy_price=buy, gw_seen=1)
    store.record_squad(1, bank=10, player_ids=[1, 2, 3])
    return store


def test_a_week_the_ledger_predicted_exactly_is_silent(tmp_path):
    store = audited_store(tmp_path)

    observed = observe(store, squad_of([1, 2, 4], bank=3, event=2), audit_board())

    assert observed.note is None


def test_a_week_the_bank_disagrees_with_earns_one_line(tmp_path):
    # The game paid only 52 for the sale — the ledger's buy price was stale —
    # so the published bank lands one tick under the prediction, and the run
    # says so, once, in pounds.
    store = audited_store(tmp_path)

    observed = observe(store, squad_of([1, 2, 4], bank=2, event=2), audit_board())

    assert observed.note is not None
    assert "£0.1m below" in observed.note
    assert "selling-price estimates may be stale" in observed.note


def test_a_surplus_reads_as_above_rather_than_below(tmp_path):
    store = audited_store(tmp_path)

    observed = observe(store, squad_of([1, 2, 4], bank=5, event=2), audit_board())

    assert observed.note is not None
    assert "£0.2m above" in observed.note


def test_the_first_gameweek_with_a_ledger_has_nothing_to_reconcile(tmp_path):
    # A ledger exists but no gameweek was ever recorded — the branch's first
    # run mid-season. There is no previous bank to predict from, so silence.
    store = store_at(tmp_path)
    store.record_purchase(1, buy_price=50, gw_seen=1)

    observed = observe(store, squad_of([1], bank=10, event=2), {1: player(1, 50)})

    assert observed.note is None


def test_a_gameweek_is_reconciled_once_not_once_per_run(tmp_path):
    # The scout run reconciles and records the gameweek; the deadline and
    # reminder runs that follow see the record and stay quiet, so a real
    # discrepancy is one line in one report, not a weekly chorus.
    store = audited_store(tmp_path)
    board = audit_board()

    first = observe(store, squad_of([1, 2, 4], bank=2, event=2), board)
    second = observe(store, squad_of([1, 2, 4], bank=2, event=2), board)

    assert first.note is not None
    assert second.note is None


def test_a_sale_the_ledger_never_priced_is_not_audited(tmp_path):
    # Player 9 left the squad but was never in the ledger — a hole, not data —
    # so the prediction cannot be made and no line pretends it was.
    store = store_at(tmp_path)
    store.record_purchase(1, buy_price=50, gw_seen=1)
    store.record_squad(1, bank=10, player_ids=[1, 9])
    players = {1: player(1, 50), 4: player(4, 60)}

    observed = observe(store, squad_of([1, 4], bank=0, event=2), players)

    assert observed.note is None
