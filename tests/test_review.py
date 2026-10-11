"""The gameweek review: facts, rendering, the file, the tick (aigaffer/review.py)."""

import copy

import httpx
import pytest

from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.data.models import Bootstrap, EntryWeek, HistoryRow, LiveStats
from aigaffer.executed import Verdict
from aigaffer.review import (
    MINUTES_MISS,
    REVIEW_ATTEMPTS,
    ReviewUnavailable,
    Written,
    append_review,
    build_facts,
    render_review,
    review_facts,
    review_record,
    review_due,
    review_target,
    review_tick,
    run_review,
)
from aigaffer.store import Store
from tests.fixtures import BOOTSTRAP_JSON, PICKS_JSON, fake_fpl_transport

# The fixture universe's players (tests/fixtures.py ELEMENTS_JSON): ids 1-8.
BOARD = Bootstrap.model_validate(BOOTSTRAP_JSON)
NAMES = {p.id: p.web_name for p in BOARD.elements}


def pick(element, position, multiplier=1, captain=False, vice=False):
    return {"element": element, "position": position, "multiplier": multiplier,
            "is_captain": captain, "is_vice_captain": vice}


def entry(picks=None, *, chip=None, subs=(), points=60, cost=0, bench=5, rank=900_000):
    return EntryWeek(
        picks=picks or [
            pick(1, 1), pick(3, 2), pick(4, 3), pick(5, 4, 2, captain=True),
            pick(6, 5), pick(7, 6, vice=True), pick(2, 12, 0), pick(8, 13, 0),
        ],
        active_chip=chip, automatic_subs=list(subs), points=points,
        event_transfers_cost=cost, points_on_bench=bench, overall_rank=rank,
    )


HISTORY = [HistoryRow(event=1, points=50, overall_rank=1_200_000), HistoryRow(event=2, points=60, overall_rank=900_000)]
LIVE = {pid: LiveStats(minutes=90, total_points=pid) for pid in range(1, 9)}  # player n scores n


def verdict(**decision):
    base = {"squad_before": [1, 2, 3, 4, 5, 6, 7, 8], "transfers_in": [], "transfers_out": [],
            "captain": 5, "chip": "none", "decision_source": "manager",
            "rationale": "Hold; captain Five.", "chip_justification": "", "adjustments": []}
    return Verdict(mode="deadline", ts="2025-08-22T00:00:00+00:00", decision={**base, **decision})


def facts(**kwargs):
    args = dict(gw=2, bootstrap=BOARD, entry=entry(), history=HISTORY, live=LIVE,
                transfers=[], verdict=verdict())
    args.update(kwargs)
    return build_facts(**args)


def test_the_target_is_the_latest_week_that_is_finished_and_checked():
    board = BOARD.model_copy(deep=True)
    assert review_target(board) is None  # GW1 finished, never data-checked
    board.events[0].data_checked = True
    assert review_target(board) == 1
    board.events[1].finished = True  # GW2 finished but not checked yet
    assert review_target(board) == 1
    board.events[1].data_checked = True
    assert review_target(board) == 2


def test_the_headline_numbers():
    f = facts(entry=entry(points=68, cost=4, bench=9))
    assert (f.points, f.hits, f.net_points, f.bench_points) == (68, 4, 64, 9)
    assert (f.rank_before, f.rank_after) == (1_200_000, 900_000)
    assert f.average == BOARD.events[1].average_entry_score


def test_the_rank_before_is_the_row_for_the_week_before_not_an_index():
    # A history with a gap (a week with no row) must not shift the lookup.
    f = facts(history=[HistoryRow(event=2, points=60, overall_rank=900_000)])
    assert f.rank_before is None and f.rank_after == 900_000


def test_the_served_multipliers_decide_the_starters_and_subs_are_only_narrated():
    # Player 2 came on for 6: the endpoint already moved him, and the subs list
    # only tells the story. Applying it again would bench him a second time.
    picks = [pick(1, 1), pick(3, 2), pick(4, 3), pick(5, 4, 2, captain=True),
             pick(2, 5), pick(7, 6, vice=True), pick(6, 12, 0), pick(8, 13, 0)]
    f = facts(entry=entry(picks, subs=[(2, 6)]))
    assert [s.player_id for s in f.starters] == [1, 3, 4, 5, 2, 7]
    assert [(i.name, o.name) for i, o in f.auto_subs] == [(NAMES[2], NAMES[6])]


def test_the_captain_counts_double_and_the_best_in_the_eleven_is_named():
    f = facts()
    assert (f.captain.player_id, f.captain_points) == (5, 10)
    assert f.best_in_xi.player_id == 7


def test_a_triple_captain_counts_three_times():
    picks = [pick(1, 1), pick(3, 2), pick(4, 3), pick(5, 4, 3, captain=True),
             pick(6, 5), pick(7, 6, vice=True), pick(2, 12, 0), pick(8, 13, 0)]
    f = facts(entry=entry(picks, chip="3xc"))
    assert (f.captain_points, f.chip) == (15, "triple_captain")


def test_a_captain_who_did_not_play_leaves_the_armband_to_the_vice():
    picks = [pick(1, 1), pick(3, 2), pick(4, 3), pick(5, 4, 0, captain=True),
             pick(6, 5), pick(7, 6, 2, vice=True), pick(2, 12, 0), pick(8, 13, 0)]
    f = facts(entry=entry(picks))
    assert (f.captain.player_id, f.captain_points) == (7, 14)


def test_neither_armband_playing_scores_the_captain_nothing():
    picks = [pick(1, 1), pick(3, 2), pick(4, 3), pick(5, 4, 0, captain=True),
             pick(6, 5), pick(7, 6, 1, vice=True), pick(2, 12, 0), pick(8, 13, 0)]
    f = facts(entry=entry(picks))
    assert (f.captain.player_id, f.captain.multiplier, f.captain_points) == (5, 0, 0)


def test_a_bench_boost_week_starts_all_fifteen():
    picks = [pick(pid, pos, 2 if pid == 5 else 1, captain=pid == 5) for pos, pid in enumerate(range(1, 9), 1)]
    f = facts(entry=entry(picks, chip="bboost"))
    assert len(f.starters) == 8 and f.chip == "bench_boost"


def test_transfers_net_of_hits_and_a_player_missing_from_live_scores_nothing():
    rows = [{"element_in": 7, "element_out": 99, "event": 2},
            {"element_in": 6, "element_out": 2, "event": 2},
            {"element_in": 3, "element_out": 4, "event": 1}]  # last week's: ignored
    f = facts(transfers=rows, entry=entry(cost=4))
    assert [s.points for s in f.transfers_in] == [7, 6]
    assert [s.points for s in f.transfers_out] == [0, 2]
    assert f.transfers_out[0].name == "id 99"
    assert f.transfers_net == 7 + 6 - 0 - 2 - 4


def test_followed_compares_the_fifteen_and_the_captain_never_the_transfer_lists():
    # A midweek "Transfers made" then a rolled deadline verdict: the record is a
    # hold on the recorded squad, while the API shows the moves. The fifteen
    # match, so he followed it.
    squad = [1, 2, 3, 4, 5, 6, 7, 8]
    f = facts(verdict=verdict(squad_before=squad, transfers_in=[], transfers_out=[]),
              transfers=[{"element_in": 7, "element_out": 99, "event": 2}])
    assert (f.followed.squad, f.followed.captain) == (True, True)


def test_a_different_fifteen_or_captain_is_named():
    f = facts(verdict=verdict(squad_before=[1, 2, 3, 4, 5, 6, 7, 99], captain=7))
    assert f.followed.squad is False
    assert f.followed.not_fielded == ("id 99",) and f.followed.fielded_instead == (NAMES[8],)
    assert f.followed.captain is False and f.followed.captain_wanted == NAMES[7]


def test_a_free_hit_week_is_measured_against_the_free_hit_fifteen():
    f = facts(entry=entry(chip="freehit"),
              verdict=verdict(chip="free_hit", freehit_squad=[1, 2, 3, 4, 5, 6, 7, 8], squad_before=[9]))
    assert f.followed.squad is True and f.free_hit is True


def test_a_solver_week_is_marked_so():
    f = facts(verdict=verdict(decision_source="solver-fallback: out of time"))
    assert f.gaffer_decided is False


def test_no_verdict_leaves_the_comparisons_out():
    f = facts(verdict=None)
    assert (f.followed, f.minute_calls, f.gaffer_decided, f.verdict_mode) == (None, None, None, None)


def test_an_old_record_missing_its_keys_leaves_the_comparisons_out():
    f = facts(verdict=Verdict(mode="deadline", ts="t", decision={"captain": 5}))
    assert f.followed is None and f.minute_calls is None and f.gaffer_decided is False


def test_minute_calls_keep_the_last_word_per_player_and_only_the_misses():
    adjustments = [
        {"player_id": 3, "expected_minutes": 90.0, "reason": "fit"},
        {"player_id": 3, "expected_minutes": 0.0, "reason": "banned"},   # last word: 0, played 90
        {"player_id": 4, "expected_minutes": 75.0, "reason": "fine"},    # 15 off: under the bar
    ]
    f = facts(verdict=verdict(adjustments=adjustments))
    assert [(c.name, c.called, c.played) for c in f.minute_calls] == [(NAMES[3], 0.0, 90)]
    assert MINUTES_MISS == 30


def test_the_facts_text_says_what_the_writer_needs():
    text = review_facts(facts(entry=entry(points=68, cost=4)))
    assert text.startswith("GW2 final: 68 points (64 after 4 in hits)")
    assert f"- {NAMES[5]} (" in text and "captain ×2" in text
    assert "Verdict followed: yes" in text


def test_the_facts_round_trip_as_json():
    import json

    assert json.loads(json.dumps(facts().to_json()))["gw"] == 2


def test_a_no_view_rationale_is_the_solvers_even_under_the_manager_source():
    from aigaffer.manager.agent import NO_VIEW

    assert facts(verdict=verdict(decision_source="manager", rationale=NO_VIEW)).gaffer_decided is False
    assert facts(verdict=verdict(decision_source="manager")).gaffer_decided is True


def test_the_manager_source_is_the_managers_own_constant():
    from aigaffer import review
    from aigaffer.manager.agent import MANAGER

    assert review.GAFFER_SOURCE == MANAGER


def test_no_highest_score_leaves_the_clause_out():
    assert BOARD.events[1].highest_score is None
    text = review_facts(facts())
    assert "; highest" not in text and text.splitlines()[0].endswith(".")


def test_a_minutes_miss_is_worded_with_the_call_and_the_minutes():
    text = review_facts(facts(verdict=verdict(
        adjustments=[{"player_id": 3, "expected_minutes": 0.0, "reason": "banned"}])))
    assert f"{NAMES[3]} called 0, played 90" in text


def test_neither_armband_playing_is_said_so():
    picks = [pick(1, 1), pick(3, 2), pick(4, 3), pick(5, 4, 0, captain=True),
             pick(6, 5), pick(7, 6, 1, vice=True), pick(2, 12, 0), pick(8, 13, 0)]
    assert "did not play, and neither did the vice: 0." in review_facts(facts(entry=entry(picks)))


def test_a_free_hit_weeks_transfers_are_marked_temporary():
    text = review_facts(facts(entry=entry(chip="freehit"),
                              transfers=[{"element_in": 7, "element_out": 2, "event": 2}]))
    assert "(free hit: the moves were for this week only)" in text


WRITTEN = Written(
    went_well=("Captaincy paid.",), went_wrong=("The hit lost points.",),
    lessons=("Avoid -4 for a one-week punt.",), tokens={"input": 1, "output": 2}, turns=1,
)


def test_the_section_reads_as_the_spec_shows():
    f = facts(entry=entry(points=68, cost=4, bench=9))
    section = render_review(f, WRITTEN)
    lines = section.splitlines()
    assert lines[0] == "<!-- looking-back gw2 -->"
    assert lines[1] == "---"
    assert lines[3] == "## Looking back — GW2"
    avg = BOARD.events[1].average_entry_score
    assert lines[5] == f"**68 pts** (64 after the hit) · average {avg} · overall rank 1,200,000 → 900,000 (▲ 300,000)"
    assert lines[6] == f"Captain {NAMES[5]} 5 ×2 (best in the XI: {NAMES[7]} 7) · bench 9 · hits −4 · no chip"
    assert "### What went well\n- Captaincy paid." in section
    assert "### What went wrong\n- The hit lost points." in section
    assert section.endswith("### Lessons\n- Avoid -4 for a one-week punt.\n")


def test_the_rank_falls_and_the_first_week_has_no_rank_before():
    worse = render_review(facts(history=[HistoryRow(event=1, points=1, overall_rank=800_000),
                                          HistoryRow(event=2, points=1, overall_rank=900_000)]), WRITTEN)
    assert "overall rank 800,000 → 900,000 (▼ 100,000)" in worse
    first = render_review(facts(history=[HistoryRow(event=2, points=1, overall_rank=900_000)]), WRITTEN)
    assert "· overall rank 900,000\n" in first


def test_no_hits_no_net_and_the_captain_best_is_left_out_when_he_was_the_best():
    picks = [pick(1, 1), pick(3, 2), pick(4, 3), pick(5, 4), pick(6, 5),
             pick(7, 6, 2, captain=True), pick(2, 12, 0), pick(8, 13, 0)]
    section = render_review(facts(entry=entry(picks)), WRITTEN)
    line = section.splitlines()[6]
    assert line == f"Captain {NAMES[7]} 7 ×2 · bench 5 · no chip"
    assert "after the hit" not in section


def test_an_empty_list_reads_as_none():
    section = render_review(facts(), Written(("Fine.",), (), (), {}, 1))
    assert "### What went wrong\n- None.\n" in section
    assert section.endswith("### Lessons\n- None.\n")


def test_appending_keeps_every_byte_above_the_marker(tmp_path):
    original = "# GW2 report\r\n\r\nPlan: hold.\r\nno trailing newline"
    (tmp_path / "GW2.md").write_bytes(original.encode("utf-8"))
    append_review(tmp_path, 2, "<!-- looking-back gw2 -->\nsection\n")
    written = (tmp_path / "GW2.md").read_bytes()
    assert written.startswith(original.encode("utf-8"))
    assert written == (original + "\n\n<!-- looking-back gw2 -->\nsection\n").encode("utf-8")


def test_a_second_review_replaces_the_first_through_the_marker(tmp_path):
    (tmp_path / "GW2.md").write_text("report\n", encoding="utf-8")
    append_review(tmp_path, 2, "<!-- looking-back gw2 -->\nfirst\n")
    append_review(tmp_path, 2, "<!-- looking-back gw2 -->\nsecond\n")
    assert (tmp_path / "GW2.md").read_text(encoding="utf-8") == "report\n\n\n<!-- looking-back gw2 -->\nsecond\n"


def test_a_week_with_no_report_gets_the_section_alone(tmp_path):
    append_review(tmp_path, 2, "<!-- looking-back gw2 -->\nsection\n")
    assert (tmp_path / "GW2.md").read_text(encoding="utf-8") == "<!-- looking-back gw2 -->\nsection\n"


def test_the_readme_is_never_touched(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("📋 **Latest verdict: [GW3](GW3.md)** <!-- latest-verdict -->\n", encoding="utf-8")
    append_review(tmp_path, 2, "<!-- looking-back gw2 -->\nsection\n")
    assert readme.read_text(encoding="utf-8").endswith("[GW3](GW3.md)** <!-- latest-verdict -->\n")


def test_the_record_keeps_the_lessons_and_the_facts():
    record = review_record(facts(), WRITTEN)
    assert record["mode"] == "review" and record["event"] == 2
    assert record["lessons"] == ["Avoid -4 for a one-week punt."]
    assert record["facts"]["gw"] == 2 and record["tokens"] == {"input": 1, "output": 2}


TEAM = 42


def final_board(gw=1):
    payload = copy.deepcopy(BOOTSTRAP_JSON)
    payload["events"][gw - 1].update(finished=True, data_checked=True, highest_score=99)
    return payload


def routes(board=None, **overrides):
    picks = copy.deepcopy(PICKS_JSON)
    picks["entry_history"]["points_on_bench"] = 3
    base = {
        "/api/bootstrap-static/": board or final_board(),
        f"/api/entry/{TEAM}/event/1/picks/": picks,
        f"/api/entry/{TEAM}/history/": {"current": [{"event": 1, "points": 57, "overall_rank": 1234567}], "past": [], "chips": []},
        f"/api/entry/{TEAM}/transfers/": [],
        "/api/event/1/live/": {"elements": [{"id": pid, "stats": {"minutes": 90, "total_points": pid}} for pid in range(1, 9)]},
    }
    base.update(overrides)
    return base


def api(route_map, statuses=None):
    # No sleeping through the client's retry backoff on a scripted 500.
    return FplClient(
        http=httpx.Client(transport=fake_fpl_transport(route_map, statuses)),
        sleep=lambda seconds: None,
    )


def cfg(tmp_path, **kwargs):
    return Config(team_id=TEAM, state_dir=tmp_path / "state", anthropic_api_key="sk-test", **kwargs)


def fake_write(cfg, facts):
    return WRITTEN


def test_a_review_appends_writes_its_file_and_stores_its_row(tmp_path):
    c = cfg(tmp_path)
    store = Store(tmp_path / "state" / "aigaffer.db")
    (tmp_path / "GW1.md").write_text("report\n", encoding="utf-8")
    client = api(routes())
    section = run_review(c, client, store, client.bootstrap(), 1, write=fake_write)
    assert (tmp_path / "GW1.md").read_text(encoding="utf-8") == "report\n\n\n" + section
    assert (tmp_path / "state" / "reports" / "gw1-review.md").read_text(encoding="utf-8") == section
    assert store.has_run(1, "review") and store.decision(1, "review")["lessons"] == list(WRITTEN.lessons)


def test_a_dry_run_writes_nothing(tmp_path):
    c = cfg(tmp_path)
    store = Store(tmp_path / "state" / "aigaffer.db")
    client = api(routes())
    section = run_review(c, client, store, client.bootstrap(), 1, dry_run=True, write=fake_write)
    assert "## Looking back — GW1" in section
    assert not (tmp_path / "GW1.md").exists() and not store.has_run(1, "review")


def test_a_week_that_is_not_final_is_unavailable(tmp_path):
    client = api(routes(board=copy.deepcopy(BOOTSTRAP_JSON)))
    with pytest.raises(ReviewUnavailable, match="not final"):
        run_review(cfg(tmp_path), client, Store(tmp_path / "s.db"), client.bootstrap(), 1, write=fake_write)


def test_a_team_with_no_entry_is_unavailable(tmp_path):
    client = api(routes(), statuses={f"/api/entry/{TEAM}/event/1/picks/": 404})
    with pytest.raises(ReviewUnavailable, match="no entry for GW1"):
        run_review(cfg(tmp_path), client, Store(tmp_path / "s.db"), client.bootstrap(), 1, write=fake_write)


def test_due_needs_a_final_week_unreviewed_a_manager_and_attempts_left(tmp_path):
    store = Store(tmp_path / "s.db")
    board = Bootstrap.model_validate(final_board())
    assert review_due(cfg(tmp_path), store, board) == 1
    assert review_due(cfg(tmp_path, manager_enabled=False), store, board) is None
    assert review_due(cfg(tmp_path), store, BOARD) is None  # nothing final
    for _ in range(REVIEW_ATTEMPTS):
        store.record_withheld(1, "review", "x")
    assert review_due(cfg(tmp_path), store, board) is None
    store2 = Store(tmp_path / "s2.db")
    store2.save_run(1, "review", "md", {})
    assert review_due(cfg(tmp_path), store2, board) is None


def test_the_target_moves_on_when_the_next_week_is_final(tmp_path):
    payload = final_board()
    payload["events"][1].update(finished=True, data_checked=True)
    store = Store(tmp_path / "s.db")
    store.save_run(1, "review", "md", {})
    assert review_due(cfg(tmp_path), store, Bootstrap.model_validate(payload)) == 2


def test_the_tick_records_a_failure_and_never_raises(tmp_path, capsys, monkeypatch):
    store = Store(tmp_path / "state" / "aigaffer.db")
    client = api(routes(), statuses={"/api/event/1/live/": 500})
    monkeypatch.setattr("aigaffer.review._writer", lambda: fake_write)
    review_tick(cfg(tmp_path), client, store)
    assert store.withheld_count(1, "review") == 1
    assert "review: GW1 failed (HTTPStatusError)" in capsys.readouterr().out


def test_the_sixth_failure_says_it_gave_up(tmp_path, capsys, monkeypatch):
    store = Store(tmp_path / "state" / "aigaffer.db")
    for _ in range(REVIEW_ATTEMPTS - 1):
        store.record_withheld(1, "review", "x")
    client = api(routes(), statuses={"/api/event/1/live/": 500})
    monkeypatch.setattr("aigaffer.review._writer", lambda: fake_write)
    review_tick(cfg(tmp_path), client, store)
    out = capsys.readouterr().out
    assert "review: GW1 gave up after 6 attempts" in out
    assert "will try again" not in out


def test_a_store_that_fails_while_recording_does_not_escape_the_tick(tmp_path, capsys, monkeypatch):
    import sqlite3

    store = Store(tmp_path / "state" / "aigaffer.db")

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk is full")

    monkeypatch.setattr(store, "record_withheld", broken)
    monkeypatch.setattr("aigaffer.review._writer", lambda: fake_write)
    client = api(routes(), statuses={"/api/event/1/live/": 500})
    review_tick(cfg(tmp_path), client, store)  # must not raise
    assert "review: GW1 failed (HTTPStatusError)" in capsys.readouterr().out


def test_the_tick_stands_down_when_the_peer_wrote_it(tmp_path, monkeypatch):
    store = Store(tmp_path / "state" / "aigaffer.db")
    called = []
    monkeypatch.setattr("aigaffer.review._writer", lambda: lambda c, f: called.append(f) or WRITTEN)
    review_tick(cfg(tmp_path), api(routes()), store, overtaken=lambda gw, mode: True)
    assert called == [] and not store.has_run(1, "review")
    assert store.withheld_count(1, "review") == 0


def test_the_tick_writes_the_review_when_due(tmp_path, monkeypatch):
    store = Store(tmp_path / "state" / "aigaffer.db")
    monkeypatch.setattr("aigaffer.review._writer", lambda: fake_write)
    review_tick(cfg(tmp_path), api(routes()), store, overtaken=lambda gw, mode: False)
    assert store.has_run(1, "review")
