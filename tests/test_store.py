import sqlite3
from datetime import datetime, timedelta

from aigaffer.store import Store


def test_saved_run_is_recorded(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.save_run(12, "deadline", "# report", {"transfers": []})
    assert store.has_run(12, "deadline") is True


def test_has_run_false_on_empty_store(tmp_path):
    assert Store(tmp_path / "aigaffer.db").has_run(12, "deadline") is False


def test_has_run_is_false_for_other_gw_and_mode(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.save_run(12, "deadline", "# report", {})
    assert store.has_run(13, "deadline") is False
    assert store.has_run(12, "scout") is False


def test_last_runs_round_trips_the_decision(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    decision = {"transfers": [{"out": 1, "in": 2}], "captain": 3, "hit": -4}
    store.save_run(12, "deadline", "# report", decision)

    runs = store.last_runs()

    assert len(runs) == 1
    assert runs[0]["gw"] == 12
    assert runs[0]["mode"] == "deadline"
    assert runs[0]["decision"] == decision


def test_last_runs_returns_most_recent_first_capped_at_n(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    for gw in range(1, 5):
        store.save_run(gw, "scout", "# report", {"gw": gw})

    runs = store.last_runs(2)

    assert [run["gw"] for run in runs] == [4, 3]


def test_last_runs_defaults_to_five(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    for gw in range(1, 8):
        store.save_run(gw, "scout", "# report", {})

    assert len(store.last_runs()) == 5


def test_timestamp_is_utc_iso_format(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.save_run(12, "deadline", "# report", {})

    ts = datetime.fromisoformat(store.last_runs()[0]["ts"])

    assert ts.utcoffset() == timedelta(0)


def test_report_markdown_is_stored(tmp_path):
    # Nothing reads report_md back yet, so check the column directly.
    db_path = tmp_path / "aigaffer.db"
    Store(db_path).save_run(12, "deadline", "# AI Gaffer\n\nSell everyone.", {})

    with sqlite3.connect(db_path) as conn:
        stored = conn.execute("SELECT report_md FROM runs").fetchone()[0]

    assert stored == "# AI Gaffer\n\nSell everyone."


def test_reopening_the_same_path_keeps_the_data(tmp_path):
    db_path = tmp_path / "aigaffer.db"
    Store(db_path).save_run(12, "deadline", "# report", {})

    reopened = Store(db_path)

    assert reopened.has_run(12, "deadline") is True
    assert len(reopened.last_runs()) == 1


def test_decision_reads_back_the_newest_record_for_a_gw_and_mode(tmp_path):
    # A --force rerun records a second row for the same gw and mode, and the
    # newest is the operative one: it is what the person overruled the dedup
    # to produce.
    store = Store(tmp_path / "aigaffer.db")
    store.save_run(2, "deadline", "# first", {"captain": 8})
    store.save_run(2, "deadline", "# forced rerun", {"captain": 13})
    store.save_run(2, "scout", "# scout", {"captain": 1})

    assert store.decision(2, "deadline") == {"captain": 13}


def test_decision_is_none_for_a_run_that_never_happened(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.save_run(2, "scout", "# scout", {"captain": 1})

    assert store.decision(2, "deadline") is None
    assert store.decision(3, "scout") is None


def test_creates_missing_parent_directories(tmp_path):
    store = Store(tmp_path / "state" / "nested" / "aigaffer.db")
    store.save_run(12, "deadline", "# report", {})
    assert store.has_run(12, "deadline") is True


# --- the purchase ledger ---------------------------------------------------


def test_a_recorded_purchase_reads_back_as_a_buy_price(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.record_purchase(5, buy_price=125, gw_seen=1)
    store.record_purchase(7, buy_price=105, gw_seen=3)

    assert store.purchases() == {5: 125, 7: 105}


def test_an_empty_ledger_is_an_empty_dict(tmp_path):
    assert Store(tmp_path / "aigaffer.db").purchases() == {}


def test_recording_a_purchase_twice_keeps_the_newer_price(tmp_path):
    # Sold and re-bought: the second purchase is the one the selling rule
    # measures rises against, so it replaces the first rather than erroring.
    store = Store(tmp_path / "aigaffer.db")
    store.record_purchase(5, buy_price=125, gw_seen=1)
    store.record_purchase(5, buy_price=128, gw_seen=9)

    assert store.purchases() == {5: 128}


def test_a_forgotten_purchase_is_gone_and_the_rest_stay(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.record_purchase(5, buy_price=125, gw_seen=1)
    store.record_purchase(7, buy_price=105, gw_seen=1)

    store.forget_purchase(5)

    assert store.purchases() == {7: 105}


def test_forgetting_a_player_the_ledger_never_held_is_quiet(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.forget_purchase(999)
    assert store.purchases() == {}


def test_the_ledger_survives_reopening_the_store(tmp_path):
    db_path = tmp_path / "aigaffer.db"
    Store(db_path).record_purchase(5, buy_price=125, gw_seen=1)

    assert Store(db_path).purchases() == {5: 125}


# --- the squad memory ------------------------------------------------------


def test_a_recorded_squad_reads_back_whole(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.record_squad(3, bank=13, player_ids=[1, 2, 3])

    assert store.squad_record(3) == {"gw": 3, "bank": 13, "player_ids": [1, 2, 3]}


def test_a_gameweek_never_recorded_is_none(tmp_path):
    assert Store(tmp_path / "aigaffer.db").squad_record(3) is None


def test_recording_a_gameweek_again_replaces_it(tmp_path):
    # Several runs serve one gameweek — scout, deadline, reminder — and each
    # writes what the picks endpoint said. Latest wins; they say the same thing.
    store = Store(tmp_path / "aigaffer.db")
    store.record_squad(3, bank=13, player_ids=[1, 2, 3])
    store.record_squad(3, bank=13, player_ids=[1, 2, 4])

    assert store.squad_record(3)["player_ids"] == [1, 2, 4]


def test_last_squad_before_finds_the_newest_earlier_gameweek(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.record_squad(1, bank=10, player_ids=[1])
    store.record_squad(2, bank=20, player_ids=[2])
    store.record_squad(4, bank=40, player_ids=[4])

    # GW3 was never recorded — a skipped week — so GW4's predecessor is GW2.
    assert store.last_squad_before(4) == {"gw": 2, "bank": 20, "player_ids": [2]}
    assert store.last_squad_before(2) == {"gw": 1, "bank": 10, "player_ids": [1]}


def test_last_squad_before_the_first_record_is_none(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.record_squad(2, bank=20, player_ids=[2])

    assert store.last_squad_before(2) is None
