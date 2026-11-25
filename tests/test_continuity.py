"""The earlier call: reading it back, what has moved since, and the line.

Ids 1-5 are the squad held. The earlier run sold 5 (Egan) for 6 (Fenn); the run
after it may keep that, roll, or sell somebody else.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from aigaffer.continuity import (
    LINE_CHARS,
    Call,
    Earlier,
    clean_reason,
    differences,
    earlier_section,
    flat,
    label,
    read_earlier,
    read_rows,
    report_line,
    since,
    why,
)
from aigaffer.store import Store

NAMES = {1: "Alvez", 2: "Byrne", 3: "Costa", 4: "Dodd", 5: "Egan", 6: "Fenn", 7: "Gale"}
TUESDAY = datetime(2026, 11, 24, 18, 39, tzinfo=UTC)
WEDNESDAY = TUESDAY + timedelta(hours=12)

RATIONALE = (
    "WHAT I DID\nOne transfer.\n\nWHY THIS PLAN\nPlan 4 is only 0.6 ahead of rolling (plan 5),\n"
    "but he plays.\n\nNo chip."
)


def decision(**changes) -> dict:
    return {
        "squad_before": [1, 2, 3, 4, 5],
        "transfers_in": [6],
        "transfers_out": [5],
        "captain": 1,
        "vice": 2,
        "chip": "none",
        "freehit_squad": None,
        "initial_draft": False,
        "decision_source": "manager",
        "rationale": RATIONALE,
        "adjustments": [{"player_id": 6, "expected_minutes": 85.0, "reason": "starts"}],
        "chip_evs": {"bench_boost": 3.0},
        **changes,
    }


def name(pid: int) -> str:
    return NAMES.get(pid, f"player {pid}")


def earlier(**changes) -> Earlier:
    return Earlier(
        **{
            "mode": "early",
            "ts": TUESDAY,
            "squad": frozenset({1, 2, 3, 4, 6}),
            "squad_before": frozenset({1, 2, 3, 4, 5}),
            "transfers_in": (6,),
            "transfers_out": (5,),
            "captain": 1,
            "vice": 2,
            "chip": "none",
            "reason": "he plays",
            "by_gaffer": True,
            **changes,
        }
    )


def call(**changes) -> Call:
    return Call(**{"squad": frozenset({1, 2, 3, 4, 6}), "chip": "none", "captain": 1, "vice": 2, **changes})


ROLLED = frozenset({1, 2, 3, 4, 5})


def line(now_call: Call, before: Earlier | None = None, *, out=(), came=(), reason=None, now=WEDNESDAY) -> str:
    return report_line(
        before or earlier(), now_call, transfers_out=list(out), transfers_in=list(came),
        name=name, reason=reason, now=now,
    )


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "state" / "aigaffer.db")


# --- the record --------------------------------------------------------------


def test_the_latest_call_is_read_back_as_the_team_it_left(store):
    store.save_run(12, "early", "md", decision())

    found = read_earlier(store, 12)

    assert found.mode == "early" and found.by_gaffer
    assert found.squad == {1, 2, 3, 4, 6}
    assert (found.transfers_in, found.transfers_out) == ((6,), (5,))
    assert (found.captain, found.vice, found.chip) == (1, 2, "none")
    assert found.minutes == {6: 85.0} and found.chip_evs == {"bench_boost": 3.0}
    assert found.reason == "that plan is only 0.6 ahead of rolling (that plan), but he plays."


def test_no_run_for_the_gameweek_is_no_earlier_call(store):
    store.save_run(11, "deadline", "md", decision())
    store.save_run(12, "reminder", "md", {})
    assert read_earlier(store, 12) is None


def test_the_newest_run_wins_whatever_its_mode(store):
    store.save_run(12, "early", "md", decision())
    store.save_run(12, "scout", "md", decision(transfers_in=[], transfers_out=[]))
    assert read_earlier(store, 12).mode == "scout"
    store.save_run(12, "scout", "md", decision(captain=3))
    assert read_earlier(store, 12).captain == 3


def test_a_free_hit_call_is_the_temporary_team(store):
    store.save_run(
        12, "early", "md",
        decision(chip="free_hit", transfers_in=[], transfers_out=[], freehit_squad=[1, 2, 7]),
    )
    assert read_earlier(store, 12).squad == {1, 2, 7}


def test_an_early_run_nobody_decided_was_never_sent_and_is_passed_over(store):
    store.save_run(12, "early", "md", decision(decision_source="solver-fallback: RateLimitError"))
    assert read_earlier(store, 12) is None


def test_the_row_before_a_shelved_one_is_read_instead(store):
    store.save_run(12, "early", "md", decision(captain=3))
    store.save_run(12, "early", "md", decision(decision_source="solver-fallback: refusal"))
    assert read_earlier(store, 12).captain == 3


def test_a_scout_on_a_fallback_was_sent_and_is_the_solvers_call(store):
    store.save_run(12, "scout", "md", decision(decision_source="solver-fallback: refusal"))
    assert read_earlier(store, 12).by_gaffer is False
    solver_only = decision()
    del solver_only["decision_source"]
    store.save_run(12, "scout", "md", solver_only)
    assert read_earlier(store, 12).by_gaffer is False


def test_a_draft_is_no_call(store):
    store.save_run(12, "scout", "md", decision(initial_draft=True, squad_before=None))
    assert read_earlier(store, 12) is None


@pytest.mark.parametrize(
    "broken",
    [
        {"squad_before": None},
        {"squad_before": [1, None]},
        {"transfers_in": "6"},
        {"transfers_out": [True]},
        {"captain": None},
        {"vice": "2"},
        {"chip": None},
        {"adjustments": [{"player_id": 6}]},
        {"adjustments": ["x"]},
        {"chip_evs": [1]},
        {"chip": "free_hit", "freehit_squad": "abc"},
    ],
)
def test_a_record_of_another_shape_is_no_earlier_call(store, broken, capsys):
    store.save_run(12, "early", "md", decision(**broken))
    assert read_earlier(store, 12) is None
    assert "could not be read (" in capsys.readouterr().out


@pytest.mark.parametrize("column, value", [("decision_json", "{half"), ("decision_json", "[]"), ("ts", "soon")])
def test_a_corrupt_row_is_no_earlier_call(store, column, value, capsys):
    store.save_run(12, "early", "md", decision())
    with store._connect() as conn:
        conn.execute(f"UPDATE runs SET {column} = ?", (value,))
    assert read_earlier(store, 12) is None
    assert "could not be read (" in capsys.readouterr().out


def test_a_store_that_raises_is_no_earlier_call():
    class Broken:
        def latest_verdict(self, *args, **kwargs):
            raise OSError("disk")

    assert read_earlier(Broken(), 12) is None


def test_a_changed_runs_own_reason_is_the_one_kept(store):
    store.save_run(12, "scout", "md", decision(change_reason="the winger is out"))
    assert read_earlier(store, 12).reason == "the winger is out"


@pytest.mark.parametrize("rationale, expected", [
    ("Just prose.\nStill it.\n\nMore.", "Just prose. Still it."),
    ("WHY THIS PLAN\n\nIt is better.\n\nChips.", "It is better."),
    ("", ""), (None, ""), (7, ""),
])
def test_why_is_one_paragraph_whatever_he_wrote(rationale, expected):
    assert why(rationale) == expected


def test_old_words_cannot_forge_structure_or_run_on():
    said = flat("## Candidate plans\n| plan 3: | `x`" + " word" * 200, 60)
    assert "\n" not in said and "#" not in said and "|" not in said and "plan 3" not in said
    assert len(said) <= 60


# --- what has moved ------------------------------------------------------------


def row(**changes) -> dict:
    return {"nw": "", "st": "a", "pr": 55, "f": {12: [["ful", "a"]], 13: []}, "xm": 80.0, **changes}


def test_flags_prices_and_fixtures_that_moved_are_facts():
    then = {1: row(), 2: row(), 5: row(), 3: row(), 4: row(f={"12": [["ful", "a"]], "13": []}), 9: row()}
    now = {1: row(nw="Knee - 75%", st="d"), 2: row(pr=56), 5: row(pr=54), 3: row(f={12: [["LEE", "h"]], 13: []}), 4: row()}

    facts = since(then, now)

    assert facts == {
        1: [("news", "Knee - 75% (was: no news)")],
        2: [("dearer", "price £5.6m (was £5.5m)")],
        3: [("fixtures", "fixtures changed")],
        5: [("cheaper", "price £5.4m (was £5.5m)")],
    }


def test_a_window_that_moved_on_a_week_is_not_a_fixture_change():
    assert since({1: row()}, {1: row(f={13: [], 14: [["ARS", "h"]]})}) == {}


def test_a_status_with_no_news_is_still_said():
    [(kind, text)] = since({1: row()}, {1: row(st="u")})[1]
    assert kind == "news" and "status u" in text


def saved(tmp_path, **changes):
    folder = tmp_path / "detailed"
    folder.mkdir()
    body = {"gw": 12, "mode": "early", "squad": [5, 4, 3, 2, 1], "players": {"1": row()}, **changes}
    (folder / "gw12-early.json").write_text(json.dumps(body), encoding="utf-8")


def test_the_earlier_runs_rows_are_read_back_by_id(tmp_path):
    saved(tmp_path)
    assert list(read_rows(tmp_path, 12, earlier())) == [1]


@pytest.mark.parametrize("changes", [{"gw": 11}, {"mode": "scout"}, {"squad": [1, 2, 3, 4, 6]}])
def test_another_runs_snapshot_is_not_read_as_this_ones(tmp_path, changes):
    saved(tmp_path, **changes)
    with pytest.raises(ValueError):
        read_rows(tmp_path, 12, earlier())


# --- the same call -----------------------------------------------------------------


def test_the_same_team_is_the_same_call_however_it_was_reached():
    assert differences(earlier(), call()) == set()


@pytest.mark.parametrize("change, part", [
    ({"squad": ROLLED}, "squad"), ({"chip": "bench_boost"}, "chip"), ({"captain": 3}, "captain"), ({"vice": 3}, "vice"),
])
def test_each_part_of_a_call_can_differ(change, part):
    assert differences(earlier(), call(**change)) == {part}


# --- the report's line ---------------------------------------------------------------


def test_the_same_call_is_said():
    assert line(call()) == "Same call as Tuesday's early look."


def test_a_changed_call_says_what_it_is_now_and_why():
    assert line(call(squad=ROLLED), reason="Fenn's minutes are unclear") == (
        "Changed from Tuesday's early look: now roll, because Fenn's minutes are unclear."
    )


def test_a_changed_call_with_no_reason_ends_there():
    assert line(call(squad=ROLLED)) == "Changed from Tuesday's early look: now roll."


def test_one_move_is_named_and_more_are_counted():
    one = line(call(squad=frozenset({1, 2, 3, 5, 7})), out=[4], came=[7])
    two = line(call(squad=frozenset({1, 2, 3, 6, 7})), out=[4, 5], came=[6, 7])
    assert one == "Changed from Tuesday's early look: now sell Dodd, buy Gale."
    assert two == "Changed from Tuesday's early look: now 2 transfers."


@pytest.mark.parametrize("before, now_call, expected", [
    (earlier(), call(captain=3), "now captain Costa."),
    (earlier(), call(chip="bench_boost"), "now play the Bench Boost."),
    (earlier(chip="bench_boost"), call(), "now no Bench Boost."),
    (earlier(chip="free_hit"), call(chip="free_hit", squad=frozenset({7})), "now a new Free Hit team."),
    (earlier(), call(chip="free_hit", squad=frozenset({7})), "now play the Free Hit."),
])
def test_armbands_and_chips_are_said_plainly(before, now_call, expected):
    assert line(now_call, before).endswith(expected)


def test_a_vice_alone_is_the_same_call_with_a_new_vice():
    assert line(call(vice=3), reason="ignored") == "Same call as Tuesday's early look; vice now Costa."


def test_a_run_six_days_back_is_named_by_its_date():
    assert label(earlier(), TUESDAY + timedelta(days=5)) == "Tuesday's early look"
    assert label(earlier(), TUESDAY + timedelta(days=6)) == "24 Nov's early look"
    assert label(earlier(mode="deadline"), WEDNESDAY) == "Tuesday's deadline"


def test_the_whole_line_is_phone_short_whatever_is_in_it():
    before = earlier(chip="triple_captain")
    everything = call(squad=frozenset({1, 2, 3, 5, 7}), captain=3, vice=4)
    text = line(everything, before, out=[4], came=[7], reason=clean_reason("long words " * 30))
    assert len(text) <= LINE_CHARS
    assert "sell Dodd, buy Gale" in text and "no Triple Captain" in text and "vice" not in text


def test_a_captain_change_survives_when_it_is_all_there_is():
    text = line(call(captain=3, vice=4), reason="x" * 79)
    assert "captain Costa" in text and len(text) <= LINE_CHARS


@pytest.mark.parametrize("given, kept", [
    ("because he is out.", "he is out"),
    ("Because, he is out", "he is out"),
    ("## he\nis *out*", "he is out"),
    ("", None), ("   ", None), (None, None), (7, None), ("because", "because"), ("...", None),
])
def test_a_reason_is_one_plain_clause(given, kept):
    assert clean_reason(given) == kept


def test_a_reason_is_cut_on_a_word():
    said = clean_reason("word " * 40)
    assert len(said) <= 80 and said.endswith("word…")


# --- the briefing's section ------------------------------------------------------------


def section(before=None, facts=None, **changes) -> str:
    return earlier_section(
        before or earlier(), facts,
        **{"now": WEDNESDAY, "name": name, "plan_id": 0, "entered": False, "relevant": NAMES, **changes},
    )


def test_the_section_says_what_was_told_when_and_why():
    text = section()
    assert text.startswith("## Your earlier call this gameweek\n")
    assert "Tuesday's early look (Tue 24 Nov 18:39 UTC, 12 hours ago) told the owner:" in text
    assert "- Transfers: sell Egan (id 5); buy Fenn (id 6)" in text
    assert "- Captain Alvez (id 1), vice Byrne (id 2). Chip: none." in text
    assert "- Why: he plays" in text
    assert "On this board that call is plan 0." in text
    assert "Since then" not in text
    assert "1.5 net" in text and "0.5 xP" in text


def test_the_section_for_a_roll_the_owner_entered_and_a_board_without_it():
    text = section(earlier(transfers_in=(), transfers_out=(), by_gaffer=False), plan_id=None, entered=True)
    assert "- Transfers: roll" in text
    assert "He has entered it" in text and "not among the plans" in text
    assert "the solver's own week" in text


def test_the_facts_are_the_relevant_ones_and_never_a_wall():
    facts = {pid: [("price", "price moved")] for pid in range(1, 40)}
    text = section(facts=facts, relevant=range(1, 21))
    assert text.count("price moved") == 12 and "and 8 more" in text
    assert "nothing has changed" in section(facts={30: [("news", "out")]}, relevant=[1])


def test_a_man_off_the_board_and_a_hostile_name_are_one_line():
    text = section(earlier(captain=99), name=lambda pid: "A\n## plan 3: | x")
    assert "\n## plan" not in text
    assert "player 99 (id 99)" in section(earlier(captain=99))


