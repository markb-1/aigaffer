"""Tests for recording "Transfers made": which verdict, the position, the echo.

The universe is the pipeline's (tests/fixtures.py): the fifteen of
``PICKS_15_JSON`` with £2.8m in the bank and one free transfer. The moves
used below, priced by hand:

==  ======  ===  ====  ======================================================
id  name    pos  cost  role here
==  ======  ===  ====  ======================================================
4   Dodd    DEF  £4.0m  held; sold for Kelly by a later verdict
6   Grant   MID  £7.5m  held; sold for Reyes
8   Ito     FWD  £5.0m  held; sold for Sarr by a verdict off the API's squad
10  Kelly   DEF  £4.5m  bought by a later verdict
17  Reyes   MID  £9.5m  bought
18  Sarr    FWD  £6.0m  bought by a verdict off the API's squad
==  ======  ===  ====  ======================================================

No ledger has run, so every sale raises the listed price: Grant for Reyes is
28 + 75 - 95 = 8 in the bank.
"""

import fcntl
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from aigaffer.data.fpl_api import FplClient
from aigaffer.data.models import Bootstrap, Player, Squad, Standing
from aigaffer.executed import StaleVerdict, Verdict
from aigaffer.orchestrator import run_pipeline
from aigaffer.recording import (
    CLOSED,
    NOTHING_YET,
    STALE,
    VERDICT_MODES,
    Outcome,
    compose,
    echo,
    record_transfers_made,
    transfers_made,
)
from aigaffer.report.render import entered_moves
from aigaffer.store import Store
from tests.fixtures import (
    HISTORY_PATH,
    PICKS_15_IDS,
    PICKS_15_JSON,
    PICKS_PATH,
    PIPELINE_BOOTSTRAP_JSON,
    PIPELINE_ELEMENTS_JSON,
    CountingTransport,
    config,
    make_client,
    make_executed,
    pipeline_routes,
)

PLAYERS = {e["id"]: Player.model_validate(e) for e in PIPELINE_ELEMENTS_JSON}
REAL = PICKS_15_IDS
REAL_BANK = 28
NOW = datetime(2025, 8, 21, 18, 0, tzinfo=UTC)
SCOUT_TS = "2025-08-19T18:00:00+00:00"  # a Tuesday
DEADLINE_TS = "2025-08-21T17:35:00+00:00"  # a Thursday
LISTED = {pid: player.now_cost for pid, player in PLAYERS.items()}

# A free-hit fifteen and a 4-5-1 eleven from it (over budget at £102.5m, but
# compose never checks budgets, so it is fine for these tests); Jarvis (9), Kelly (10),
# Meier (12), Reyes (17) and Sarr (18) are not in the standing squad.
FH_SQUAD = [1, 9, 3, 4, 10, 12, 13, 5, 6, 11, 14, 17, 7, 15, 18]
FH_XI = [1, 3, 4, 10, 12, 5, 6, 11, 14, 17, 7]


def verdict(mode: str = "deadline", ts: str = DEADLINE_TS, **fields) -> Verdict:
    """A stored decision, Grant-for-Reyes off the API's squad by default."""
    decision = {
        "event": 2,
        "transfers_in": [17],
        "transfers_out": [6],
        "chip": "none",
        "captain": 5,
        "vice": 17,
        "free_transfers": 1,
        "squad_before": sorted(REAL),
        "freehit_squad": None,
        "freehit_xi": None,
    }
    decision.update(fields)
    return Verdict(mode=mode, ts=ts, decision=decision)


def first(mode: str = "scout", ts: str = SCOUT_TS):
    return compose(None, verdict(mode, ts), REAL, REAL_BANK, PLAYERS, LISTED, NOW)


def test_the_first_verdict_records_its_moves_prices_and_bank():
    row = first("deadline", DEADLINE_TS)

    assert row.gw == 2 and row.mode == "deadline"
    assert row.recorded_at == NOW.isoformat()
    assert row.transfers_in == [17] and row.transfers_out == [6]
    assert row.squad_after == sorted(set(REAL) - {6} | {17})
    assert row.bank_after == 8  # 28 + 75 - 95
    assert row.ft_after == 0  # one free transfer, one move
    assert row.ft_before == 1  # the API's count for the week, before it
    assert row.buy_prices == {17: 95} and row.sell_prices == {6: 75}
    assert row.captain == 5 and row.vice == 17
    assert row.chip == "none"
    assert row.verdicts == [["deadline", DEADLINE_TS]]
    assert row.arrival_status == {17: "a"}


def test_the_same_verdict_again_is_a_no_op():
    row = first()

    again = compose(row, verdict("scout", SCOUT_TS), REAL, REAL_BANK, PLAYERS, LISTED, NOW)

    assert again is row


def test_a_second_verdict_composes_onto_the_recorded_position():
    # Tuesday's scout: Grant for Reyes (bank 8, no free transfer left).
    # Thursday's deadline verdict solved from that squad: Dodd for Kelly, a
    # hit — 8 + 40 - 45 = 3 in the bank, free transfers stay at 0.
    row = first()
    deadline = verdict(
        transfers_in=[10], transfers_out=[4], free_transfers=0,
        squad_before=row.squad_after,
    )

    both = compose(row, deadline, REAL, REAL_BANK, PLAYERS, LISTED, NOW)

    assert both.transfers_in == [10, 17] and both.transfers_out == [4, 6]
    assert both.squad_after == sorted(set(REAL) - {4, 6} | {10, 17})
    assert both.bank_after == 3
    assert both.ft_after == 0
    assert both.ft_before == 1, "carried from the first recording"
    assert both.buy_prices == {10: 45, 17: 95}
    assert both.sell_prices == {4: 40, 6: 75}
    assert both.arrival_status == {10: "a", 17: "a"}
    assert both.verdicts == [["scout", SCOUT_TS], ["deadline", DEADLINE_TS]]
    assert both.mode == "deadline"


def test_a_verdict_off_the_apis_squad_replaces_the_row():
    # The other scheduler solved Thursday's verdict before it saw Tuesday's
    # row: its moves are against the API's squad, so they are the position —
    # applying them on top of the row would make Tuesday's moves twice.
    row = first()
    raced = verdict(transfers_in=[18], transfers_out=[8])

    replaced = compose(row, raced, REAL, REAL_BANK, PLAYERS, LISTED, NOW)

    assert replaced.verdicts == [["deadline", DEADLINE_TS]]
    assert replaced.transfers_in == [18] and replaced.transfers_out == [8]
    assert replaced.bank_after == 18  # 28 + 50 - 60
    assert replaced.ft_after == 0
    assert replaced.squad_after == sorted(set(REAL) - {8} | {18})
    assert replaced.buy_prices == {18: 60} and replaced.sell_prices == {8: 50}


def test_a_verdict_off_a_squad_nobody_recorded_is_refused():
    row = first()

    with pytest.raises(StaleVerdict):
        compose(
            row, verdict(squad_before=[1, 2, 3]), REAL, REAL_BANK, PLAYERS, LISTED, NOW
        )


def test_a_record_from_before_squad_before_composes_onto_the_row():
    row = first()
    legacy = verdict(transfers_in=[10], transfers_out=[4], free_transfers=0)
    del legacy.decision["squad_before"]

    both = compose(row, legacy, REAL, REAL_BANK, PLAYERS, LISTED, NOW)

    assert len(both.verdicts) == 2 and both.transfers_in == [10, 17]


def test_a_wildcard_keeps_the_free_transfers():
    # Two moves under a wildcard: 28 + 40 + 75 - 45 - 95 = 3, and the free
    # transfer FPL banks through the chip is still there.
    row = compose(
        None,
        verdict(chip="wildcard", transfers_in=[10, 17], transfers_out=[4, 6]),
        REAL, REAL_BANK, PLAYERS, LISTED, NOW,
    )

    assert row.chip == "wildcard"
    assert row.bank_after == 3
    assert row.ft_after == 1


def test_a_wildcard_after_earlier_moves_gives_the_weeks_free_transfer_back():
    # Tuesday: Grant for Reyes, the one free transfer spent (0 left). Thursday's
    # verdict, solved from that squad, plays the wildcard and adds Dodd for
    # Kelly. FPL folds Tuesday's move into the wildcard, so the week's free
    # transfer stands where it did before any move — 1, not the verdict's 0.
    # The money is real either way: 8 + 40 - 45 = 3.
    row = first()
    wildcard = verdict(
        chip="wildcard", transfers_in=[10], transfers_out=[4], free_transfers=0,
        squad_before=row.squad_after,
    )

    both = compose(row, wildcard, REAL, REAL_BANK, PLAYERS, LISTED, NOW)

    assert both.chip == "wildcard"
    assert both.ft_before == 1 and both.ft_after == 1
    assert both.bank_after == 3
    assert both.transfers_in == [10, 17] and both.transfers_out == [4, 6]


def test_a_free_hit_records_its_team_and_leaves_the_standing_squad():
    row = compose(
        None,
        verdict(
            chip="free_hit", transfers_in=[], transfers_out=[],
            freehit_squad=FH_SQUAD, freehit_xi=FH_XI,
        ),
        REAL, REAL_BANK, PLAYERS, LISTED, NOW,
    )

    assert row.squad_after == sorted(REAL)
    assert row.transfers_in == [] and row.transfers_out == []
    assert row.bank_after == 28 and row.ft_after == 1
    assert row.freehit_squad == FH_SQUAD and row.freehit_xi == FH_XI
    assert row.arrival_status == {9: "a", 10: "a", 12: "a", 17: "a", 18: "a"}


def test_a_free_hit_the_solver_never_planned_records_the_chip_with_no_fifteen():
    # The gaffer chose a free hit the solver had not planned, so the decision
    # carries no fifteen. Nothing to record but the chip: the standing squad
    # stays, no arrivals are flagged, and the echo shows no Free Hit team line.
    row = compose(
        None,
        verdict(chip="free_hit", transfers_in=[], transfers_out=[]),
        REAL, REAL_BANK, PLAYERS, LISTED, NOW,
    )

    assert row.chip == "free_hit"
    assert row.freehit_squad is None and row.freehit_xi is None
    assert row.squad_after == sorted(REAL) and row.arrival_status == {}
    assert "Free Hit team" not in echo(row, verdict(), PLAYERS)


def test_prices_and_flags_are_the_live_boards_at_recording():
    # Reyes ticked up to £9.6m and picked up a knock since the verdict: the
    # owner paid what the board said when he texted, and the flag is today's.
    players = dict(PLAYERS)
    players[17] = players[17].model_copy(update={"now_cost": 96, "status": "d"})

    row = compose(None, verdict(), REAL, REAL_BANK, players, LISTED, NOW)

    assert row.buy_prices == {17: 96}
    assert row.bank_after == 7  # 28 + 75 - 96
    assert row.arrival_status == {17: "d"}


def test_the_moves_read_as_the_owner_entered_them():
    assert entered_moves(make_executed(), PLAYERS) == "Grant → Reyes"
    assert entered_moves(
        make_executed(transfers_in=[], transfers_out=[]), PLAYERS
    ) == "no transfers"
    assert entered_moves(
        make_executed(transfers_in=[10, 17], transfers_out=[4, 6]), PLAYERS
    ) == "Dodd → Kelly, Grant → Reyes"
    assert entered_moves(
        make_executed(chip="wildcard", transfers_in=[10, 17], transfers_out=[4, 6]),
        PLAYERS,
    ) == "Wildcard played (2 transfers)"
    assert entered_moves(
        make_executed(chip="free_hit", transfers_in=[], transfers_out=[]), PLAYERS
    ) == "no transfers; Free Hit played"


def test_the_echo_is_the_whole_new_position():
    row = first("deadline", DEADLINE_TS)

    assert echo(row, verdict(), PLAYERS) == (
        "Recorded for GW2 (from Thursday's deadline verdict):\n"
        "Grant → Reyes. Ferrer (C), Reyes (V).\n"
        "Squad: GKP Alvez, Byrne · DEF Costa, Dodd, Novak, Quill, Tandy"
        " · MID Ferrer, Lozano, Ozturk, Reyes, Voss · FWD Haas, Ito, Pryce\n"
        "Bank £0.8m · 0 free transfers left\n"
        "Later runs this gameweek work from this squad."
    )


def test_the_echo_names_the_early_scout_and_the_free_hit_team():
    row = compose(
        None,
        verdict(
            "early", SCOUT_TS, chip="free_hit", transfers_in=[], transfers_out=[],
            freehit_squad=FH_SQUAD, freehit_xi=FH_XI,
        ),
        REAL, REAL_BANK, PLAYERS, LISTED, NOW,
    )

    text = echo(row, verdict("early", SCOUT_TS), PLAYERS)

    assert text.startswith("Recorded for GW2 (from Tuesday's early scout verdict):")
    assert "No transfers; Free Hit played. Ferrer (C), Reyes (V)." in text
    assert "Free Hit team: GKP Alvez, Jarvis · DEF" in text
    assert "Bank £2.8m · 1 free transfer left" in text


def test_two_free_transfers_run_down_across_recordings():
    # Two free transfers at the API: Grant for Reyes leaves 1, then Dodd for
    # Kelly (solved from that squad, which says 1 left) leaves 0.
    row = compose(None, verdict("scout", SCOUT_TS, free_transfers=2), REAL, REAL_BANK, PLAYERS, LISTED, NOW)
    assert row.ft_before == 2 and row.ft_after == 1

    both = compose(
        row,
        verdict(transfers_in=[10], transfers_out=[4], free_transfers=1,
                squad_before=row.squad_after),
        REAL, REAL_BANK, PLAYERS, LISTED, NOW,
    )

    assert both.ft_before == 2 and both.ft_after == 0
    assert both.bank_after == 3  # 8 + 40 - 45


def test_selling_a_player_signed_earlier_the_same_week_uses_the_selling_rule():
    # Reyes was bought at 95; the board now says 98. The game pays
    # 95 + (98 - 95) // 2 = 96, not the listed 98. Sold for Grant (75) at
    # bank 8: 8 + 96 - 75 = 29. He nets out of the cumulative lists.
    row = first()
    players = dict(PLAYERS)
    players[17] = players[17].model_copy(update={"now_cost": 98})

    back = compose(
        row,
        verdict("deadline", DEADLINE_TS, transfers_in=[6], transfers_out=[17],
                free_transfers=0, squad_before=row.squad_after),
        REAL, REAL_BANK, players, LISTED, NOW,
    )

    assert back.bank_after == 29
    assert 17 not in back.squad_after


def test_a_full_reversal_nets_out_to_nothing():
    # Grant for Reyes, then Reyes for Grant: 8 + 95 - 75 = 28, the bank we began
    # with, and nothing differs from the API's squad.
    row = first()

    back = compose(
        row,
        verdict(transfers_in=[6], transfers_out=[17], free_transfers=0,
                squad_before=row.squad_after),
        REAL, REAL_BANK, PLAYERS, LISTED, NOW,
    )

    assert back.transfers_in == [] and back.transfers_out == []
    assert back.squad_after == sorted(REAL)
    assert back.bank_after == 28
    assert back.buy_prices == {} and back.sell_prices == {}


def test_a_chip_already_played_carries_onto_a_later_ordinary_verdict():
    row = compose(
        None, verdict(chip="wildcard", transfers_in=[10, 17], transfers_out=[4, 6]),
        REAL, REAL_BANK, PLAYERS, LISTED, NOW,
    )

    later = compose(
        row,
        verdict("deadline", DEADLINE_TS, transfers_in=[18], transfers_out=[8],
                squad_before=row.squad_after),
        REAL, REAL_BANK, PLAYERS, LISTED, NOW,
    )

    assert later.chip == "wildcard"
    assert later.ft_after == later.ft_before == 1


def test_a_free_hit_after_moves_undoes_them():
    # Tuesday: Grant for Reyes. Thursday's verdict plays the free hit. FPL
    # reverts to the previous deadline's squad: Tuesday's move is undone, the
    # bank is the API's 28, the free transfer the week began with is kept, and
    # Reyes's purchase is dropped so the ledger never seeds him.
    row = first()
    hit = verdict(
        chip="free_hit", transfers_in=[], transfers_out=[], free_transfers=0,
        freehit_squad=FH_SQUAD, freehit_xi=FH_XI, squad_before=row.squad_after,
    )

    both = compose(row, hit, REAL, REAL_BANK, PLAYERS, LISTED, NOW)

    assert both.chip == "free_hit"
    assert both.squad_after == sorted(REAL) and both.bank_after == 28
    assert both.transfers_in == [] and both.transfers_out == []
    assert both.buy_prices == {} and both.sell_prices == {}
    assert both.ft_before == 1 and both.ft_after == 1
    assert both.verdicts == [["scout", SCOUT_TS], ["deadline", DEADLINE_TS]]
    assert both.freehit_squad == FH_SQUAD and both.freehit_xi == FH_XI
    assert both.arrival_status == {9: "a", 10: "a", 12: "a", 17: "a", 18: "a"}
    assert compose(both, hit, REAL, REAL_BANK, PLAYERS, LISTED, NOW) is both
    assert (
        "Free Hit played: Tuesday's moves (Grant → Reyes) are undone for this"
        " gameweek." in echo(both, hit, PLAYERS)
    )


# --- the handler ---------------------------------------------------------------

BOOTSTRAP = Bootstrap.model_validate(PIPELINE_BOOTSTRAP_JSON)
SQUAD = Squad(
    picks=PICKS_15_JSON["picks"],
    bank=PICKS_15_JSON["entry_history"]["bank"],
    event=1,
    standing=Standing.model_validate(PICKS_15_JSON["entry_history"]),
)
LATER = datetime(2025, 8, 22, 9, 0, tzinfo=UTC)


def run_at(store: Store, gw: int, mode: str, ts: str, decision: dict) -> None:
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "INSERT INTO runs (ts, gw, mode, report_md, decision_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (ts, gw, mode, "# report", json.dumps(decision)),
        )


def test_no_verdict_yet_is_said_so(tmp_path):
    store = Store(tmp_path / "aigaffer.db")

    outcome = record_transfers_made(store, BOOTSTRAP, SQUAD, [], LATER, NOW)

    assert outcome == Outcome(NOTHING_YET.format(gw=2), False)
    assert store.executed(2) is None


def test_a_text_after_the_gameweek_closed_says_so(tmp_path):
    # His text was about GW1's verdict, and GW1's deadline has gone: the API
    # shows what he entered now, and there is nothing for the inbox to add.
    store = Store(tmp_path / "aigaffer.db")
    run_at(store, 1, "deadline", "2025-08-14T17:35:00+00:00", verdict().decision)

    outcome = record_transfers_made(store, BOOTSTRAP, SQUAD, [], LATER, NOW)

    assert outcome == Outcome(CLOSED.format(gw=1), False)


def test_a_verdict_newer_than_the_text_is_not_the_one_he_entered(tmp_path):
    # He texted at 17:35 Thursday about Tuesday's scout. The deadline verdict
    # landed at 17:40 — while the inbox waited on the tick's lock — and he
    # never saw it.
    store = Store(tmp_path / "aigaffer.db")
    run_at(store, 2, "scout", SCOUT_TS, verdict("scout", SCOUT_TS).decision)
    run_at(
        store, 2, "deadline", "2025-08-21T17:40:00+00:00",
        verdict(transfers_in=[18], transfers_out=[8]).decision,
    )

    record_transfers_made(
        store, BOOTSTRAP, SQUAD, [], datetime(2025, 8, 21, 17, 35, tzinfo=UTC), NOW
    )

    row = store.executed(2)
    assert row.verdicts == [["scout", SCOUT_TS]]
    assert row.transfers_in == [17]


def test_a_stale_verdict_is_refused_with_a_reply(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(first())
    run_at(
        store, 2, "deadline", DEADLINE_TS,
        verdict(squad_before=[1, 2, 3]).decision,
    )

    outcome = record_transfers_made(store, BOOTSTRAP, SQUAD, [], LATER, NOW)

    assert outcome == Outcome(STALE.format(gw=2), False)
    assert store.executed(2) == first()


def test_a_re_send_changes_nothing_and_echoes_again(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    run_at(store, 2, "deadline", DEADLINE_TS, verdict().decision)

    once = record_transfers_made(store, BOOTSTRAP, SQUAD, [], LATER, NOW)
    twice = record_transfers_made(store, BOOTSTRAP, SQUAD, [], LATER, NOW)

    assert once.changed is True and twice.changed is False
    assert twice.reply == once.reply
    assert store.executed(2).ft_after == 0, "one free transfer spent once"


def held(path) -> bool:
    """Whether someone holds the flock on ``path`` right now."""
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


class FakeSync:
    """The git sync, recorded: what was called, and whether under the lock."""

    def __init__(self, lock_path) -> None:
        self.lock_path = lock_path
        self.calls: list[tuple[str, bool]] = []

    def pull(self) -> None:
        self.calls.append(("pull", held(self.lock_path)))

    def publish(self, message: str) -> None:
        self.calls.append(("publish", held(self.lock_path)))


@pytest.fixture
def scouted(tmp_path):
    """One real scout run, so the verdict on record is the pipeline's own."""
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    run_pipeline(cfg, make_client(pipeline_routes()), store, "scout", send=False)
    return cfg, store


def test_transfers_made_records_the_scout_he_entered(scouted, tmp_path):
    cfg, store = scouted
    decision = store.latest_verdict(2, VERDICT_MODES).decision
    assert decision["transfers_in"], "the pipeline's scout makes a move"
    transport = CountingTransport(pipeline_routes())
    client = FplClient(http=httpx.Client(transport=transport), sleep=lambda _: None)
    lock = tmp_path / "inbox" / "state.lock"
    sync = FakeSync(lock)

    reply = transfers_made(
        cfg, client, sync, lock, sent_at=datetime.now(UTC) + timedelta(seconds=1)
    )

    row = store.executed(2)
    ins, outs = decision["transfers_in"], decision["transfers_out"]
    assert row.transfers_in == sorted(ins) and row.transfers_out == sorted(outs)
    # A fresh ledger sells at the listed price: bank 28 + sales - buys.
    assert row.bank_after == (
        28 + sum(PLAYERS[p].now_cost for p in outs) - sum(PLAYERS[p].now_cost for p in ins)
    )
    assert row.ft_after == max(0, 1 - len(ins))
    assert reply.startswith("Recorded for GW2 (from ")
    # Recording asks the API for three things, not two hundred.
    # Each exactly once: a retry loop or a stray extra fetch would show here.
    assert dict(transport.counts) == {
        "/api/bootstrap-static/": 1, PICKS_PATH: 1, HISTORY_PATH: 1,
    }
    # Pull, write and push all happen under the shared lock.
    assert sync.calls == [("pull", True), ("publish", True)]


def test_a_second_text_publishes_nothing(scouted, tmp_path):
    cfg, store = scouted
    lock = tmp_path / "inbox" / "state.lock"
    sync = FakeSync(lock)
    sent = datetime.now(UTC) + timedelta(seconds=1)

    first_reply = transfers_made(cfg, make_client(pipeline_routes()), sync, lock, sent)
    second_reply = transfers_made(cfg, make_client(pipeline_routes()), sync, lock, sent)

    assert second_reply == first_reply
    assert [name for name, _ in sync.calls] == ["pull", "publish", "pull"]


class LockProbeTransport(CountingTransport):
    """Records, per request, whether the state lock was held at that moment."""

    def __init__(self, routes: dict, lock_path) -> None:
        super().__init__(routes)
        self.lock_path = lock_path
        self.held_during: list[tuple[str, bool]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.held_during.append((request.url.path, held(self.lock_path)))
        return super().handle_request(request)


def test_the_fetches_are_made_outside_the_lock(scouted, tmp_path):
    # The lock is shared with the hourly tick; holding it across the network
    # would stall the tick behind three slow requests. If any client.* call
    # moved inside ``with state_lock`` this fails.
    cfg, _ = scouted
    lock = tmp_path / "inbox" / "state.lock"
    lock.parent.mkdir()  # the probe opens the file before the first lock exists
    transport = LockProbeTransport(pipeline_routes(), lock)
    client = FplClient(http=httpx.Client(transport=transport), sleep=lambda _: None)

    transfers_made(
        cfg, client, FakeSync(lock), lock, sent_at=datetime.now(UTC) + timedelta(seconds=1)
    )

    assert len(transport.held_during) == 3
    assert all(not was_held for _, was_held in transport.held_during)


def test_the_row_is_read_after_the_pull(tmp_path):
    # Tuesday's scout was entered and its row pushed by the other scheduler;
    # this machine's checkout does not have it until the pull brings it in.
    # Thursday's deadline verdict was solved from the squad Tuesday left, so it
    # composes only onto that row: read before the pull, the store has no row
    # and the verdict is "stale" (neither the API's squad nor the row's).
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    tuesday = first()  # Grant (6) out, Reyes (17) in
    run_at(store, 2, "scout", SCOUT_TS, verdict("scout", SCOUT_TS).decision)
    run_at(
        store, 2, "deadline", DEADLINE_TS,
        verdict(
            transfers_in=[10], transfers_out=[4], squad_before=tuesday.squad_after
        ).decision,
    )
    lock = tmp_path / "inbox" / "state.lock"

    class PullingSync(FakeSync):
        def pull(self) -> None:
            Store(cfg.state_dir / "aigaffer.db").save_executed(tuesday)
            super().pull()

    sync = PullingSync(lock)

    reply = transfers_made(
        cfg, make_client(pipeline_routes()), sync, lock, sent_at=LATER, now=NOW
    )

    row = store.executed(2)
    assert not reply.startswith("Not recorded"), reply
    # Cumulative: Tuesday's Grant-for-Reyes plus Thursday's Dodd-for-Kelly.
    assert row.verdicts == [["scout", SCOUT_TS], ["deadline", DEADLINE_TS]]
    assert sorted(row.transfers_in) == [10, 17]
    assert sorted(row.transfers_out) == [4, 6]
    assert sync.calls == [("pull", True), ("publish", True)]
