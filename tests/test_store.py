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


def test_creates_missing_parent_directories(tmp_path):
    store = Store(tmp_path / "state" / "nested" / "aigaffer.db")
    store.save_run(12, "deadline", "# report", {})
    assert store.has_run(12, "deadline") is True
