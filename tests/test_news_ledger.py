"""The news ledger: what earlier runs learned, judged fresh or due a re-check.

Every board here is hand-built. The clock is a Friday, 17:30 UTC — the T-18h
deadline run for a Saturday 11:30 deadline — so rule 3's deadline clause is
in play for every test that does not move it.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from aigaffer.data.models import Event, Fixture, Player
from aigaffer.news_ledger import (
    CATEGORIES,
    CLUB_PLAYED,
    FPL_MOVED,
    NEW_GAMEWEEK,
    ON_THE_DAY,
    PRESSER_SINCE,
    RETURN_DUE,
    SEARCH_BUDGET,
    TIERS,
    TIME_LIMIT,
    UNREADABLE,
    Entry,
    LedgerView,
    evaluate_ledger,
    ledger_path,
    read_ledger,
    write_ledger,
)

DEADLINE = datetime(2026, 10, 17, 11, 30, tzinfo=UTC)  # Saturday
NOW = DEADLINE - timedelta(hours=18)                    # Friday 17:30, the deadline run
GW = 7
HOME, AWAY, SUNDAY, MONDAY = 1, 2, 3, 4                 # club ids


def player(pid: int, team: int = HOME, **fields) -> Player:
    return Player(
        id=pid, web_name=f"P{pid}", team=team, element_type=3, now_cost=60,
        status=fields.pop("status", "a"), minutes=0, starts=0, total_points=0,
        bonus=0, saves=0, **fields,
    )


def fixture(fid: int, home: int, away: int, kickoff: datetime, gw: int = GW,
            played: bool = False) -> Fixture:
    return Fixture(
        id=fid, event=gw, team_h=home, team_a=away,
        kickoff_time=kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        finished_provisional=played,
    )


EVENT = Event(id=GW, deadline_time=DEADLINE, is_next=True, is_current=False, finished=False)
FIXTURES = [
    fixture(1, HOME, AWAY, datetime(2026, 10, 17, 14, 0, tzinfo=UTC)),    # Saturday 15:00 UK
    fixture(2, SUNDAY, 9, datetime(2026, 10, 18, 13, 0, tzinfo=UTC)),     # Sunday
    fixture(3, MONDAY, 8, datetime(2026, 10, 19, 19, 0, tzinfo=UTC)),     # Monday night
]


def snapshot(p: Player) -> dict:
    return {
        "status": p.status,
        "chance_of_playing_next_round": p.chance_of_playing_next_round,
        "news": p.news,
        "news_added": p.news_added,
    }


def entry(p: Player, category: str = "nailed", *, gw: int = GW, checked: datetime = NOW - timedelta(hours=2),
          quote: str = "2026-10-17", minutes: float = 90.0, return_gw: int | None = None,
          fpl: dict | None = None) -> dict:
    return Entry(
        player_id=p.id, gw=gw, category=category, expected_minutes=minutes, tier=1,
        quote_date=quote, source="presser", note="said so", return_gw=return_gw,
        checked_at=checked.isoformat(), run="scout",
        fpl=snapshot(p) if fpl is None else fpl,
    ).to_json()


def judge(players: list[Player], raw: dict, fixtures=FIXTURES, event=EVENT, now=NOW) -> LedgerView:
    return evaluate_ledger(raw, SimpleNamespace(elements=players), fixtures, event, now)


def test_the_vocabulary_is_the_specs():
    assert CATEGORIES == ("nailed", "rotation", "doubt", "test_on_day", "injured", "suspended", "ill")
    assert TIERS == (1, 2, 3)
    assert SEARCH_BUDGET == {"early": 6, "scout": 6, "deadline": 10}


def test_a_player_gains_fpls_news_fields_with_quiet_defaults():
    plain = player(5)
    assert plain.news == "" and plain.news_added is None
    flagged = Player.model_validate({**plain.model_dump(), "news": "Knee - 75%", "news_added": "2026-10-05T18:12:00Z"})
    assert flagged.news_added == "2026-10-05T18:12:00Z"  # kept as the API's string


def test_a_nailed_starter_checked_today_is_fresh():
    p = player(5)
    view = judge([p], {"5": entry(p)})
    assert view.entries[5].fresh is True and view.entries[5].reason is None


def test_fpl_moving_the_flag_makes_it_a_re_check():
    before = player(5)
    after = player(5, status="d", chance_of_playing_next_round=75, news="Knock - 75%")
    view = judge([after], {"5": entry(before)})
    assert view.entries[5].fresh is False and view.entries[5].reason == FPL_MOVED


def test_a_match_his_club_has_played_since_makes_it_a_re_check():
    p = player(5)
    played = fixture(9, HOME, 7, NOW - timedelta(hours=1), gw=GW, played=True)
    view = judge([p], {"5": entry(p, checked=NOW - timedelta(hours=3))}, fixtures=[*FIXTURES, played])
    assert view.entries[5].reason == CLUB_PLAYED


def test_a_match_played_before_the_check_changes_nothing():
    p = player(5)
    earlier = fixture(9, HOME, 7, NOW - timedelta(days=2), gw=GW, played=True)
    view = judge([p], {"5": entry(p, checked=NOW - timedelta(hours=3))}, fixtures=[*FIXTURES, earlier])
    assert view.entries[5].fresh is True


def test_a_doubt_from_the_last_gameweek_is_a_re_check_and_an_injury_carries_over():
    doubt, hurt = player(5), player(6)
    raw = {
        "5": entry(doubt, "doubt", gw=GW - 1, checked=NOW - timedelta(hours=10)),
        "6": entry(hurt, "injured", gw=GW - 1, checked=NOW - timedelta(days=3), minutes=0.0, return_gw=GW + 3),
    }
    view = judge([doubt, hurt], raw)
    assert view.entries[5].reason == NEW_GAMEWEEK
    assert view.entries[6].fresh is True


def test_a_test_on_the_day_is_never_fresh_on_read():
    p = player(5)
    view = judge([p], {"5": entry(p, "test_on_day", checked=NOW - timedelta(minutes=5))})
    assert view.entries[5].reason == ON_THE_DAY


def test_an_injury_is_fresh_until_his_return_gameweek():
    p = player(5)
    due = judge([p], {"5": entry(p, "injured", gw=GW - 2, checked=NOW - timedelta(days=5), minutes=0.0, return_gw=GW)})
    assert due.entries[5].reason == RETURN_DUE
    out = judge([p], {"5": entry(p, "injured", gw=GW - 2, checked=NOW - timedelta(days=5), minutes=0.0, return_gw=GW + 1)})
    assert out.entries[5].fresh is True


def test_each_category_has_its_time_limit():
    # 24h for a doubt, 72h for rotation, 7 days for a nailed starter, 14 for an injury with no date.
    # Saturday club with a Tuesday quote: rule 3 would fire too; the time limit is the earlier reason.
    p = player(5, team=AWAY)
    cases = [
        ("doubt", timedelta(hours=25)), ("rotation", timedelta(hours=73)),
        ("nailed", timedelta(days=8)), ("injured", timedelta(days=15)),
    ]
    for category, age in cases:
        view = judge([p], {"5": entry(p, category, checked=NOW - age, quote="2026-10-13")})
        assert view.entries[5].reason == TIME_LIMIT, category


def test_a_nailed_entry_never_outlives_the_gameweek_after_it():
    p = player(5)
    view = judge([p], {"5": entry(p, gw=GW - 2, checked=NOW - timedelta(days=2))})
    assert view.entries[5].reason == TIME_LIMIT


def test_a_wednesday_doubt_is_a_re_check_once_the_press_conference_is_done():
    # Saturday 14:00 UTC kickoff: K − 72h is Wednesday 14:00, so a Tuesday quote predates
    # the presser window; at the Friday 17:30 run K − 18h has not come, but the deadline
    # run has (deadline − 18h), so the presser counts as done.
    p = player(5, team=HOME)
    view = judge([p], {"5": entry(p, "doubt", checked=NOW - timedelta(hours=20), quote="2026-10-13")})
    assert view.entries[5].reason == PRESSER_SINCE


def test_a_sunday_and_a_monday_club_count_as_having_spoken_at_the_deadline_run():
    # Their pressers may come Saturday — after the FPL deadline — so the deadline run
    # is the last useful moment and the clause treats them as done (spec §5 rule 3).
    sun, mon = player(5, team=SUNDAY), player(6, team=MONDAY)
    raw = {
        "5": entry(sun, "rotation", checked=NOW - timedelta(hours=40), quote="2026-10-14"),
        "6": entry(mon, "rotation", checked=NOW - timedelta(hours=40), quote="2026-10-14"),
    }
    view = judge([sun, mon], raw)
    assert view.entries[5].reason == PRESSER_SINCE
    assert view.entries[6].reason == PRESSER_SINCE


def test_before_the_deadline_run_a_monday_club_has_not_spoken_yet():
    mon = player(6, team=MONDAY)
    scout_time = DEADLINE - timedelta(hours=60)  # Wednesday night
    view = judge([mon], {"6": entry(mon, "rotation", checked=scout_time - timedelta(hours=2), quote="2026-10-13")}, now=scout_time)
    assert view.entries[6].fresh is True


def test_a_quote_inside_the_presser_window_is_not_pre_presser():
    # (K − 72h).date() is Wednesday 14 Oct for the Saturday club: a Thursday quote is
    # on or after it, so rule 3 leaves it alone. (The accepted edge, spec §5.)
    p = player(5, team=HOME)
    view = judge([p], {"5": entry(p, "doubt", checked=NOW - timedelta(hours=20), quote="2026-10-15")})
    assert view.entries[5].fresh is True


def test_a_club_with_no_fixture_this_gameweek_has_no_presser_rule():
    p = player(5, team=12)
    view = judge([p], {"5": entry(p, "rotation", checked=NOW - timedelta(hours=40), quote="2026-10-12")})
    assert view.entries[5].fresh is True


def test_players_who_left_and_malformed_entries_are_dropped():
    p, nine = player(5), player(9)
    raw = {
        "5": entry(p), "6": entry(player(6)), "7": {"player_id": 7}, "8": "rubbish",
        "9": {**entry(nine), "category": "knackered"},
    }
    view = judge([p, nine], raw)  # 6 is not on the board any more; 9's category is not ours
    assert list(view.entries) == [5]


def test_a_missing_file_reads_as_empty_and_quietly(tmp_path, capsys):
    assert read_ledger(tmp_path / "nope.json") == {}
    assert capsys.readouterr().out == ""


def test_a_file_that_is_not_utf8_reads_as_empty_with_one_line(tmp_path, capsys):
    # read_text raises UnicodeDecodeError, a ValueError and not an OSError.
    path = tmp_path / "ledger.json"
    path.write_bytes(b"\xff\xfe{\x80")
    assert read_ledger(path) == {}
    lines = capsys.readouterr().out.splitlines()
    assert lines == [UNREADABLE.format(reason="UnicodeDecodeError")]


def test_an_unreadable_file_reads_as_empty_with_one_line(tmp_path, capsys):
    for text in ("{not json", "[1, 2]", ""):
        path = tmp_path / "ledger.json"
        path.write_text(text, encoding="utf-8")
        assert read_ledger(path) == {}
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and all(line.startswith(UNREADABLE.split("(")[0]) for line in lines)
    assert all("Expecting" not in line for line in lines)  # the class name, never the words


def test_the_ledger_lives_under_state_news(tmp_path):
    assert ledger_path(tmp_path / "state") == tmp_path / "state" / "news" / "ledger.json"


def record(pid: int, minutes: float, category: str, reason: str) -> dict:
    return {
        "player_id": pid, "expected_minutes": minutes, "reason": reason, "category": category,
        "tier": 1, "quote_date": "2026-10-16", "source": "presser", "return_gw": None,
    }


def test_the_write_keeps_the_view_adds_the_records_and_the_later_call_wins(tmp_path):
    kept, changed = player(5), player(6, status="d", chance_of_playing_next_round=50)
    view = judge([kept, changed], {"5": entry(kept), "6": entry(player(6))})
    path = ledger_path(tmp_path)
    write_ledger(
        path, view,
        [record(6, 60.0, "doubt", "a doubt (paper)"), record(6, 0.0, "injured", "out (club)")],
        gw=GW, run="deadline", now=NOW, players={5: kept, 6: changed},
    )
    written = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(written) == ["5", "6"]
    assert written["5"] == entry(kept)
    assert written["6"]["category"] == "injured" and written["6"]["expected_minutes"] == 0.0
    assert written["6"]["note"] == "out (club)" and written["6"]["run"] == "deadline"
    assert written["6"]["checked_at"] == NOW.isoformat() and written["6"]["gw"] == GW
    assert written["6"]["fpl"] == snapshot(changed)
    assert not any(p.name.endswith(".tmp") for p in path.parent.iterdir())


def test_the_write_drops_players_who_left_and_skips_unknown_records(tmp_path):
    p = player(5)
    view = judge([p], {"5": entry(p), "9": entry(player(9))})  # 9 has left: not in the view
    path = ledger_path(tmp_path)
    write_ledger(path, view, [record(42, 0.0, "doubt", "who")], gw=GW, run="scout", now=NOW, players={5: p})
    assert sorted(json.loads(path.read_text(encoding="utf-8"))) == ["5"]


def test_wrongly_typed_fields_are_dropped_never_a_crash():
    # Each entry is otherwise valid; one field has the wrong type. JSON can hold
    # any of these, a hand-edit or a bad write could leave them, and the run must
    # carry on with the entries it can read (player 5, who is intact).
    good, bad = player(5), player(6)
    broken = {
        "gw as text": {**entry(bad), "gw": "7"},
        "return_gw as text": {**entry(bad, "injured", minutes=0.0), "return_gw": "9"},
        "player_id as list": {**entry(bad), "player_id": [6]},
        "minutes as text": {**entry(bad), "expected_minutes": "90"},
        "tier as bool": {**entry(bad), "tier": True},
        "fpl as list": {**entry(bad), "fpl": []},
        "note as number": {**entry(bad), "note": 3},
        "source as number": {**entry(bad), "source": 3},
    }
    view = judge([good, bad], {"5": entry(good), **broken})
    assert list(view.entries) == [5]
