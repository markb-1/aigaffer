"""Tests for the what-if log: two lines per what-if, read back by the cap, the
resume after a kill, and the recorder's heads-up — and a line it cannot read
never stops any of them."""

import json
from datetime import UTC, datetime, timedelta, timezone

from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.whatif import HOLD, MARGINAL, PLAY
from aigaffer.whatif_log import LOG_FILE, WhatIfLog, numbers_entry, opinion_entry
from tests.test_chip_opinion import sample_opinion, sample_whatif

NOW = datetime(2026, 10, 8, 14, 2, tzinfo=UTC)


def log(tmp_path) -> WhatIfLog:
    return WhatIfLog(tmp_path / LOG_FILE)


def numbers(event: int = 6, band: str = PLAY, ts: datetime = NOW, chip: str = WILDCARD) -> dict:
    return {"ts": ts.isoformat(), "event": event, "band": band, "chip": chip}


def lines(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / LOG_FILE).read_text().splitlines()]


def test_each_append_is_one_line_with_its_kind_and_update_id(tmp_path):
    book = log(tmp_path)
    book.append_numbers(101, numbers())
    book.append_opinion(101, {"ts": NOW.isoformat(), "verdict": HOLD})

    written = lines(tmp_path)
    assert [line["kind"] for line in written] == ["numbers", "opinion"]
    assert all(line["update_id"] == 101 for line in written)
    assert written[0]["band"] == PLAY


def test_the_directory_is_made_on_first_append(tmp_path):
    book = WhatIfLog(tmp_path / "inbox" / LOG_FILE)
    book.append_numbers(1, numbers())

    assert (tmp_path / "inbox" / LOG_FILE).exists()


def test_a_missing_file_reads_as_empty(tmp_path):
    book = log(tmp_path)

    assert book.numbers_today(NOW) == 0
    assert book.numbers(1) is None
    assert book.has_opinion(1) is False
    assert book.latest_play(6, NOW.isoformat()) is None


def test_the_cap_counts_numbers_lines_of_the_utc_day(tmp_path):
    book = log(tmp_path)
    midnight = datetime(2026, 10, 8, tzinfo=UTC)
    book.append_numbers(1, numbers(ts=midnight - timedelta(seconds=1)))  # yesterday
    book.append_numbers(2, numbers(ts=midnight))  # today's first second
    book.append_numbers(3, numbers(ts=NOW))
    book.append_opinion(3, {"ts": NOW.isoformat(), "verdict": HOLD})  # not a numbers line
    # 00:30 in UK summer time is still 23:30 UTC yesterday.
    bst = timezone(timedelta(hours=1))
    book.append_numbers(4, numbers(ts=datetime(2026, 10, 8, 0, 30, tzinfo=bst)))

    assert book.numbers_today(NOW) == 2
    assert book.numbers_today(NOW.astimezone(bst)) == 2


def test_numbers_and_opinion_are_found_by_update_id(tmp_path):
    book = log(tmp_path)
    book.append_numbers(7, numbers(band=MARGINAL))
    book.append_numbers(8, numbers(band=HOLD))
    book.append_opinion(8, {"ts": NOW.isoformat(), "verdict": HOLD})

    assert book.numbers(7)["band"] == MARGINAL
    assert book.numbers(9) is None
    assert book.has_opinion(8) is True
    assert book.has_opinion(7) is False


def test_lines_it_cannot_read_are_skipped(tmp_path):
    path = tmp_path / LOG_FILE
    path.write_text(
        "not json\n"
        "[1, 2, 3]\n"
        '{"kind": "numbers"}\n'
        '{"kind": "numbers", "update_id": 5, "ts": "yesterday", "event": 6, "band": "play"}\n'
        "\n"
    )
    book = WhatIfLog(path)
    book.append_numbers(6, numbers())

    assert book.numbers_today(NOW) == 1
    assert book.numbers(6)["event"] == 6
    # Readable enough to find by id, even with a timestamp nobody can parse.
    assert book.numbers(5)["band"] == "play"
    assert book.latest_play(6, (NOW - timedelta(days=1)).isoformat())["update_id"] == 6


def test_a_half_written_last_line_does_not_hide_the_ones_before(tmp_path):
    book = log(tmp_path)
    book.append_numbers(1, numbers())
    with (tmp_path / LOG_FILE).open("a") as handle:
        handle.write('{"kind": "numbers", "update_id": 2, "ts": ')  # killed mid-write

    assert book.numbers(1) is not None
    book.append_numbers(3, numbers())  # a fresh line after the torn one is still read
    assert book.numbers(3) is not None


def test_latest_play_is_the_newest_play_for_that_gameweek_after_the_verdict(tmp_path):
    book = log(tmp_path)
    verdict_at = NOW - timedelta(hours=2)
    book.append_numbers(1, numbers(ts=NOW - timedelta(hours=3)))  # before the verdict
    book.append_numbers(2, numbers(ts=NOW - timedelta(hours=1)))
    book.append_numbers(3, numbers(ts=NOW, band=HOLD))  # newer, but a hold
    book.append_numbers(4, numbers(ts=NOW - timedelta(minutes=30), event=7))  # other gameweek
    book.append_numbers(5, numbers(ts=NOW - timedelta(minutes=10), chip=FREE_HIT))

    found = book.latest_play(6, verdict_at.isoformat())
    assert found["update_id"] == 5 and found["chip"] == FREE_HIT
    assert book.latest_play(6, NOW.isoformat()) is None
    assert book.latest_play(8, verdict_at.isoformat()) is None


def test_latest_play_orders_by_time_not_by_line(tmp_path):
    # Lines are appended in handling order, which a resumed what-if can break.
    book = log(tmp_path)
    book.append_numbers(2, numbers(ts=NOW))
    book.append_numbers(1, numbers(ts=NOW - timedelta(minutes=5)))

    assert book.latest_play(6, (NOW - timedelta(hours=1)).isoformat())["update_id"] == 2


def test_the_numbers_entry_has_exactly_the_specs_fields():
    whatif = sample_whatif()
    entry = numbers_entry(whatif, ts=NOW.isoformat(), total_seconds=187.4)

    assert set(entry) == {
        "ts", "event", "chip", "band", "net", "gain", "bars_diff", "margin",
        "proven_on", "proven_off", "weekly_on", "weekly_off", "chips_on",
        "chips_off", "squad_on", "captain_on", "vice_on", "moves_on",
        "off_summary", "minutes_source", "solve_seconds", "total_seconds",
    }
    assert entry["event"] == 2 and entry["chip"] == WILDCARD and entry["band"] == MARGINAL
    assert (entry["net"], entry["gain"], entry["bars_diff"], entry["margin"]) == (2.0, 4.5, 2.5, 8.0)
    assert (entry["proven_on"], entry["proven_off"]) == (True, False)
    assert entry["weekly_on"] == {"2": 61.0, "3": 70.0, "4": 58.0}
    assert entry["chips_on"] == {"2": WILDCARD, "3": "bench_boost"}
    assert entry["chips_off"] == {"3": "bench_boost"}
    assert entry["squad_on"] == sorted(whatif.on.plan.squad)
    assert (entry["captain_on"], entry["vice_on"]) == (8, 13)
    assert entry["moves_on"] == {"in": [17, 18], "out": [7, 15]}
    assert entry["off_summary"] == {"in": [], "out": [], "hits": 0}
    assert entry["solve_seconds"] == {"on": 41.2, "off": 60.0}
    assert entry["total_seconds"] == 187.4
    assert entry["minutes_source"] == "Wednesday's scout (Gale 0)"
    json.dumps(entry)  # everything in it is JSON


def test_a_free_hit_entry_keeps_the_fifteen_he_would_field():
    whatif = sample_whatif()
    fh_path = whatif.on.path.__class__(
        **{**whatif.on.path.__dict__, "week1_chip": FREE_HIT, "week1_freehit_squad": [18, 1, 3]}
    )
    on = whatif.on.__class__(**{**whatif.on.__dict__, "path": fh_path})
    fh = whatif.__class__(**{**whatif.__dict__, "kind": FREE_HIT, "on": on})

    assert numbers_entry(fh, ts=NOW.isoformat(), total_seconds=1.0)["squad_on"] == [1, 3, 18]


def test_the_opinion_entry_carries_the_verdict_the_cost_and_the_tokens():
    entry = opinion_entry(sample_opinion(), ts=NOW.isoformat(), model="claude-opus-5-5")

    assert set(entry) == {
        "ts", "verdict", "better_week", "new_fact", "searches", "source",
        "turns", "seconds", "tokens", "est_usd",
    }
    assert entry["tokens"] == {"input": 12000, "cache_read": 40000, "cache_write": 9000, "output": 2500}
    # 12k×4 + 2.5k×20 + 40k×0.20 + 9k×5, per million
    assert entry["est_usd"] == 0.151
    assert opinion_entry(sample_opinion(), ts=NOW.isoformat(), model="other")["est_usd"] is None
