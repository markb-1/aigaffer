import json
import sqlite3

import pytest
from datetime import UTC, datetime, timedelta

from aigaffer.executed import Verdict
from aigaffer.store import Store
from tests.fixtures import make_executed


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


def test_a_withheld_report_is_remembered_by_its_reason(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    assert store.withheld_before(2, "deadline", "AuthenticationError") is False

    store.record_withheld(2, "deadline", "AuthenticationError")

    assert store.withheld_before(2, "deadline", "AuthenticationError") is True
    assert store.withheld_before(2, "deadline", "RateLimitError") is False
    assert store.withheld_before(2, "scout", "AuthenticationError") is False
    assert store.withheld_before(3, "deadline", "AuthenticationError") is False


def test_a_withheld_report_has_not_run(tmp_path):
    store = Store(tmp_path / "aigaffer.db")

    store.record_withheld(2, "deadline", "AuthenticationError")

    assert store.has_run(2, "deadline") is False
    assert store.last_runs() == []


# --- what the owner entered ---------------------------------------------------


def run_at(store: Store, gw: int, mode: str, ts: str, decision: dict) -> None:
    """A run row with a timestamp of the test's choosing — save_run stamps now."""
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "INSERT INTO runs (ts, gw, mode, report_md, decision_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (ts, gw, mode, "# report", json.dumps(decision)),
        )


def test_an_executed_row_round_trips_with_its_integer_keys(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    row = make_executed(
        chip="free_hit",
        freehit_squad=[1, 9, 3],
        freehit_xi=[1, 3],
        arrival_status={17: "d", 9: "a"},
    )

    store.save_executed(row)

    back = store.executed(2)
    assert back == row
    assert back.buy_prices == {17: 95}, "JSON's string keys come back as ints"
    assert back.arrival_status == {17: "d", 9: "a"}


def test_a_week_with_no_free_hit_keeps_its_none(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed())

    back = store.executed(2)
    assert back.freehit_squad is None and back.freehit_xi is None


def test_the_row_is_a_text_file_beside_the_store(tmp_path):
    # A text file so that the inbox's commit and the other scheduler's
    # database commit touch different files and always rebase cleanly.
    store = Store(tmp_path / "aigaffer.db")

    store.save_executed(make_executed())

    path = tmp_path / "executed" / "gw2.json"
    text = path.read_text()
    assert json.loads(text)["gw"] == 2
    assert json.loads(text)["buy_prices"] == {"17": 95}
    assert text.endswith("\n")
    assert [p.name for p in (tmp_path / "executed").iterdir()] == ["gw2.json"], (
        "the temp file was renamed into place, not left behind"
    )


def test_a_second_recording_replaces_the_gameweeks_row(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed())
    store.save_executed(make_executed(bank_after=3, verdicts=[["scout", "a"], ["deadline", "b"]]))

    assert store.executed(2).bank_after == 3
    assert [p.name for p in (tmp_path / "executed").iterdir()] == ["gw2.json"]


def test_recording_never_touches_the_database(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    before = (tmp_path / "aigaffer.db").read_bytes()

    store.save_executed(make_executed())

    assert (tmp_path / "aigaffer.db").read_bytes() == before


def test_a_store_never_recorded_into_has_no_executed_directory(tmp_path):
    store = Store(tmp_path / "aigaffer.db")

    assert store.executed(2) is None
    assert not (tmp_path / "executed").exists()


def test_no_row_for_another_gameweek(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed())

    assert store.executed(3) is None


def _drifted(field: str) -> str:
    """A whole, valid row with one key gone — a file written by another
    version of the code, or edited by hand."""
    data = json.loads(json.dumps(make_executed().to_json()))
    del data[field]
    return json.dumps(data)


@pytest.mark.parametrize(
    ("text", "fault"),
    [
        ('{"gw": 2, "mode": "dead', "JSONDecodeError"),  # truncated mid-write
        (_drifted("bank_after"), "KeyError"),  # schema drift
        (json.dumps({**json.loads(_drifted("ft_after")), "gw": "two"}), "ValueError"),
        ("[1, 2, 3]", "TypeError"),  # JSON, but not a row
        (b"\xff\xfe{", "UnicodeDecodeError"),  # not even text
    ],
    ids=["truncated", "missing-key", "bad-value", "not-a-row", "not-utf8"],
)
def test_an_unreadable_recorded_week_reads_as_none_with_one_line(
    tmp_path, capsys, text, fault
):
    # A recorded week is a convenience on top of the API's squad, never a
    # dependency of it: a gw2.json that will not parse must cost the owner
    # his "Transfers made", not the deadline report. One line names the
    # class of the fault — never its words, which could quote the file.
    store = Store(tmp_path / "aigaffer.db")
    (tmp_path / "executed").mkdir()
    path = tmp_path / "executed" / "gw2.json"
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)

    assert store.executed(2) is None

    assert capsys.readouterr().out.splitlines() == [
        f"recorded week gw2 unreadable: {fault} — running from the API's squad"
    ]


MODES = ("early", "scout", "deadline")


def test_the_latest_verdict_is_the_newest_of_the_three_report_modes(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    run_at(store, 2, "early", "2025-08-17T18:00:00+00:00", {"n": 1})
    run_at(store, 2, "deadline", "2025-08-21T17:35:00+00:00", {"n": 3})
    run_at(store, 2, "scout", "2025-08-20T17:35:00+00:00", {"n": 2})
    # The reminder is never a verdict to enter, however new.
    run_at(store, 2, "reminder", "2025-08-22T14:35:00+00:00", {"n": 4})

    verdict = store.latest_verdict(2, MODES)

    assert verdict == Verdict(
        mode="deadline", ts="2025-08-21T17:35:00+00:00", decision={"n": 3}
    )


def test_a_tie_on_the_timestamp_goes_to_the_row_written_last(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    run_at(store, 2, "scout", "2025-08-20T17:35:00+00:00", {"n": 1})
    run_at(store, 2, "deadline", "2025-08-20T17:35:00+00:00", {"n": 2})

    assert store.latest_verdict(2, MODES).decision == {"n": 2}


def test_a_verdict_newer_than_the_cutoff_is_not_one_he_saw(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    run_at(store, 2, "scout", "2025-08-20T17:35:00+00:00", {"n": 1})
    run_at(store, 2, "deadline", "2025-08-21T17:35:00+00:00", {"n": 2})

    cutoff = datetime(2025, 8, 21, 17, 0, tzinfo=UTC)

    assert store.latest_verdict(2, MODES, at_or_before=cutoff).decision == {"n": 1}
    assert store.latest_verdict(2, MODES, at_or_before=datetime(2025, 8, 1, tzinfo=UTC)) is None


def test_no_verdict_for_a_gameweek_with_none(tmp_path):
    assert Store(tmp_path / "aigaffer.db").latest_verdict(2, MODES) is None


def test_a_decision_is_found_by_its_timestamp(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    run_at(store, 2, "deadline", "2025-08-21T17:35:00+00:00", {"n": 1})
    run_at(store, 2, "deadline", "2025-08-21T18:35:00+00:00", {"n": 2})

    assert store.decision_at(2, "deadline", "2025-08-21T17:35:00+00:00") == {"n": 1}
    assert store.decision_at(2, "deadline", "2025-08-21T19:00:00+00:00") is None
