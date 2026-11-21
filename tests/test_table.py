"""The league table, form and model view, on invented leagues."""

import json
from datetime import UTC, datetime
from math import exp, log, prod

from aigaffer.data.models import Bootstrap, Event, Fixture, Team
from aigaffer.data.table import league_table, table_lines, table_record
from aigaffer.model.strength import TeamStrengths


def team(tid: int, short: str, code: int = 0, name: str | None = None) -> Team:
    return Team(
        id=tid, name=name or f"{short} FC", short_name=short, code=code,
        strength_attack_home=1000, strength_attack_away=1000,
        strength_defence_home=1000, strength_defence_away=1000,
        strength_overall_home=1000, strength_overall_away=1000,
    )


def board(*teams: Team) -> Bootstrap:
    event = Event(
        id=5, deadline_time=datetime(2026, 9, 20, tzinfo=UTC),
        is_next=True, is_current=False, finished=False,
    )
    return Bootstrap(events=[event], teams=list(teams), elements=[])


_ids = iter(range(1, 10_000))


def played(h: int, a: int, hg: int, ag: int, day: int = 1, event: int | None = 1,
           finished: bool = True, provisional: bool = False) -> Fixture:
    return Fixture(
        id=next(_ids), event=event, team_h=h, team_a=a,
        kickoff_time=f"2026-08-{day:02d}T15:00:00Z",
        team_h_score=hg, team_a_score=ag,
        finished=finished, finished_provisional=provisional,
    )


THREE = board(team(1, "AAA", 101), team(2, "BBB", 102), team(3, "CCC", 103))


def by_short(rows):
    return {row.short: row for row in rows}


def test_nothing_played_is_no_table():
    unplayed = Fixture(id=1, event=1, team_h=1, team_a=2)
    assert league_table(THREE, [unplayed]) is None
    assert league_table(THREE, []) is None


def test_points_then_goal_difference_then_goals_scored():
    # Everyone wins one and loses one: 3 points each. AAA's +2 beats BBB's
    # +1, which beats CCC's -3.
    fixtures = [
        played(1, 2, 3, 0, day=1),   # AAA +3, BBB -3
        played(2, 3, 4, 0, day=2),   # BBB +4, CCC -4
        played(3, 1, 5, 4, day=3),   # CCC +1, AAA -1
    ]
    rows = league_table(THREE, fixtures)
    # AAA: pts 3, gd +2, gf 7. BBB: pts 3, gd +1, gf 4. CCC: pts 3, gd -3.
    assert [r.short for r in rows] == ["AAA", "BBB", "CCC"]
    assert [r.position for r in rows] == [1, 2, 3]
    aaa = rows[0]
    assert (aaa.played, aaa.won, aaa.drawn, aaa.lost) == (2, 1, 0, 1)
    assert (aaa.gf, aaa.ga, aaa.gd, aaa.pts) == (7, 5, 2, 3)


def test_goals_scored_breaks_a_tie_on_goal_difference():
    four = board(team(1, "AAA"), team(2, "BBB"), team(3, "CCC"), team(4, "DDD"))
    fixtures = [played(1, 3, 3, 2), played(2, 4, 1, 0)]   # both win by one
    rows = league_table(four, fixtures)
    assert [r.short for r in rows][:2] == ["AAA", "BBB"]
    assert not rows[0].level and not rows[1].level


def test_teams_level_on_everything_share_the_place():
    fixtures = [played(1, 2, 1, 1), played(2, 3, 1, 1), played(3, 1, 1, 1)]
    rows = league_table(THREE, fixtures)
    assert [r.position for r in rows] == [1, 1, 1]
    assert all(r.level for r in rows)
    # Stable within the tie: by short name.
    assert [r.short for r in rows] == ["AAA", "BBB", "CCC"]


def test_a_team_yet_to_play_sorts_by_the_plain_rules():
    four = board(team(1, "AAA"), team(2, "BBB"), team(3, "CCC"), team(4, "DDD"))
    # CCC loses twice; DDD has not played. DDD (0 pts, GD 0) is above CCC.
    rows = league_table(four, [played(1, 3, 3, 0), played(2, 3, 1, 0, day=2)])
    order = [r.short for r in rows]
    assert order.index("DDD") < order.index("CCC")
    ddd = by_short(rows)["DDD"]
    assert ddd.played == 0 and ddd.last5 == ()


def test_teams_yet_to_play_are_level_with_each_other():
    four = board(team(1, "AAA"), team(2, "BBB"), team(3, "CCC"), team(4, "DDD"))
    rows = by_short(league_table(four, [played(1, 2, 2, 0)]))
    assert rows["CCC"].position == rows["DDD"].position == 2
    assert rows["CCC"].level and rows["DDD"].level


def test_full_time_counts_before_the_data_check():
    provisional = played(1, 2, 2, 0, finished=False, provisional=True)
    rows = by_short(league_table(THREE, [provisional]))
    assert rows["AAA"].pts == 3


def test_unplayed_postponed_and_scoreless_fixtures_do_not_count():
    postponed = Fixture(id=900, event=None, team_h=1, team_a=3)
    no_score = Fixture(id=901, event=2, team_h=1, team_a=3, finished=True)
    rows = by_short(league_table(THREE, [played(1, 2, 1, 0), postponed, no_score]))
    assert rows["AAA"].played == 1
    assert rows["CCC"].played == 0
    assert len(rows["AAA"].last5) == 1


def test_a_fixture_for_a_team_not_on_the_board_is_ignored():
    rows = by_short(league_table(THREE, [played(1, 2, 1, 0), played(1, 99, 5, 0, day=2)]))
    assert rows["AAA"].played == 1


def test_last_five_is_the_latest_five_oldest_first():
    two = board(team(1, "AAA"), team(2, "BBB"))
    fixtures = [played(1, 2, n, 0, day=n + 1) for n in range(7)]
    aaa = by_short(league_table(two, fixtures))["AAA"]
    assert [r.gf for r in aaa.last5] == [2, 3, 4, 5, 6]
    assert aaa.last5[-1].opp == "BBB" and aaa.last5[-1].venue == "h"
    bbb = by_short(league_table(two, fixtures))["BBB"]
    assert bbb.last5[-1].venue == "a" and bbb.last5[-1].outcome == "L"


def test_a_rearranged_match_lands_by_kickoff_not_by_round():
    two = board(team(1, "AAA"), team(2, "BBB"))
    # The round-1 match was postponed and played on day 20, after round 3.
    late = played(1, 2, 9, 0, day=20, event=1)
    on_time = [played(1, 2, 1, 0, day=10, event=2), played(1, 2, 2, 0, day=15, event=3)]
    aaa = by_short(league_table(two, [late, *on_time]))["AAA"]
    assert [r.gf for r in aaa.last5] == [1, 2, 9]


def test_home_and_away_splits():
    fixtures = [played(1, 2, 2, 0), played(3, 1, 1, 1, day=2), played(1, 3, 0, 1, day=3)]
    aaa = by_short(league_table(THREE, fixtures))["AAA"]
    assert (aaa.home.w, aaa.home.d, aaa.home.l, aaa.home.gf, aaa.home.ga) == (1, 0, 1, 2, 1)
    assert (aaa.away.w, aaa.away.d, aaa.away.l, aaa.away.gf, aaa.away.ga) == (0, 1, 0, 1, 1)


def fitted(attack: dict[int, float], defence: dict[int, float]) -> TeamStrengths:
    return TeamStrengths(mu=0.3, home=0.2, attack=attack, defence=defence)


def test_no_fit_means_no_model_view():
    rows = league_table(THREE, [played(1, 2, 1, 0)])
    assert all(r.model_att is None and r.model_def is None for r in rows)
    record = table_record(rows)
    assert all("ma" not in row and "md" not in row for row in record["rows"])
    assert not any("model" in line for line in table_lines(record))


def test_the_model_view_joins_through_code_not_id():
    # Ids 1..3 carry codes 101..103; the fit knows the codes only. A fit
    # that also held ids 1..3 as codes would answer the wrong clubs.
    fit = fitted(
        {101: 0.4, 102: 0.0, 103: -0.4, 1: -2.0, 2: 2.0, 3: 0.0},
        {101: -0.3, 102: 0.0, 103: 0.3, 1: 2.0, 2: -2.0, 3: 0.0},
    )
    rows = by_short(league_table(THREE, [played(1, 2, 1, 0)], fit))
    assert rows["AAA"].model_att > 1 > rows["CCC"].model_att
    assert rows["AAA"].model_def < 1 < rows["CCC"].model_def


def test_the_model_view_is_centred_on_this_seasons_clubs():
    # A relegated side (code 999) still in the fit with a weak attack and
    # a leaky defence drags the fit's own centre down; the live three must
    # still average exactly 1 on both columns, geometrically.
    fit = fitted(
        {101: 0.5, 102: 0.3, 103: 0.1, 999: -0.9},
        {101: -0.4, 102: -0.2, 103: 0.0, 999: 0.6},
    )
    rows = league_table(THREE, [played(1, 2, 1, 0)], fit)
    assert abs(log(prod(r.model_att for r in rows))) < 1e-9
    assert abs(log(prod(r.model_def for r in rows))) < 1e-9
    aaa = by_short(rows)["AAA"]
    assert abs(aaa.model_att - exp(0.5 - 0.3)) < 1e-9


def test_a_club_the_fit_never_saw_has_no_model_view():
    fit = fitted({101: 0.2, 102: -0.2}, {101: 0.0, 102: 0.0})
    rows = by_short(league_table(THREE, [played(1, 2, 1, 0)], fit))
    assert rows["CCC"].model_att is None
    assert rows["AAA"].model_att is not None


def test_the_record_survives_json_and_carries_the_season_scale():
    fit = fitted({101: 0.2, 102: 0.0, 103: -0.2}, {101: -0.2, 102: 0.0, 103: 0.2})
    fixtures = [played(1, 2, 3, 0), played(2, 3, 1, 1, day=2), played(3, 1, 0, 2, day=3)]
    record = table_record(league_table(THREE, fixtures, fit))
    assert json.loads(json.dumps(record)) == record
    # 7 goals over 6 team-games.
    assert record["avg"] == round(7 / 6, 2)
    aaa = record["rows"][0]
    assert aaa["c"] == "AAA" and aaa["cf"] == "AAA FC" and aaa["pos"] == 1
    assert aaa["h"] == [1, 0, 0, 3, 0] and aaa["a"] == [1, 0, 0, 2, 0]
    assert aaa["l5"] == [["W", 3, 0, "BBB", "h"], ["W", 2, 0, "CCC", "a"]]
    assert aaa["ma"] == round(exp(0.2), 2)


def test_a_line_per_team_with_the_figures_a_reader_needs():
    fit = fitted({101: 0.2, 102: 0.0, 103: -0.2}, {101: -0.2, 102: 0.0, 103: 0.2})
    fixtures = [played(1, 2, 3, 0), played(2, 3, 1, 1, day=2), played(3, 1, 0, 2, day=3)]
    lines = table_lines(table_record(league_table(THREE, fixtures, fit)), {"BBB"})
    rows = [line for line in lines if line[:1].isdigit()]
    assert len(rows) == 3
    aaa = rows[0]
    assert aaa.startswith("1 AAA P2 6pts W2 D0 L0 GF5 GA0 GD+5")
    assert "home 1-0-0 3:0" in aaa and "away 1-0-0 2:0" in aaa
    assert "last 2: W3-0 BBB(h) W2-0 CCC(a), 2.5-0.0 a game" in aaa
    assert "model att 1.22 def 0.82" in aaa
    # Season: 5 scored in 2 against a league 7/6 a team-game.
    assert f"season att {2.5 / (7 / 6):.2f} def 0.00" in aaa
    bbb = next(row for row in rows if " BBB " in row)
    assert bbb.startswith("3 BBB [ours] P2") and "[ours]" not in aaa
    assert any("[ours]" in line for line in lines if not line[:1].isdigit())


def test_level_places_are_marked():
    fixtures = [played(1, 2, 1, 1), played(2, 3, 1, 1), played(3, 1, 1, 1)]
    lines = table_lines(table_record(league_table(THREE, fixtures)))
    assert sum(line.startswith("1= ") for line in lines) == 3


def test_a_team_yet_to_play_reads_as_such():
    four = board(team(1, "AAA"), team(2, "BBB"), team(3, "CCC"), team(4, "DDD"))
    lines = table_lines(table_record(league_table(four, [played(1, 2, 2, 0)])))
    ddd = next(line for line in lines if " DDD " in line)
    assert "P0" in ddd and "season" not in ddd and "last 5: none" in ddd


def test_names_off_the_wire_cannot_forge_lines_or_columns():
    hostile = board(team(1, "A\n## X|Y [ours]", 101), team(2, "BBB", 102))
    lines = table_lines(table_record(league_table(hostile, [played(1, 2, 1, 0)])))
    assert not any(line.startswith("##") for line in lines)
    row = next(line for line in lines if line[:1].isdigit() and "BBB" not in line.split()[1])
    assert "|" not in row.split(" P1 ")[0]
    assert "[ours]" not in row
