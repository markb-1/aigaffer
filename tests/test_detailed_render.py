"""The detailed report's renderer: a pure function from snapshots to markdown.

Every snapshot here is hand-built and small. The window is gameweeks 8 and 9,
and the players are:

==  ========  ===  =====  ===========================  ==============
id  name      pos  price  fixtures                     xP (GW8, GW9)
==  ========  ===  =====  ===========================  ==============
1   Saka      MID  10.1   GW8 LEE(h)  GW9 LEE(h)+wol(a)  4.12, 7.30
2   Salah     MID  12.9   GW8 wol(a)  GW9 blank         5.00, 0.00
3   Haaland   FWD  14.0   GW8 LEE(h)  GW9 LEE(h)        6.00, 3.00
4   Justin    DEF   4.6   GW8 LEE(h)  GW9 blank         3.00, 0.00
5   Xhaka     MID   5.0   GW8 wol(a)  GW9 LEE(h)        2.00, 4.00
6   Raya      GKP   5.6   GW8 LEE(h)  GW9 LEE(h)        3.00, 3.00
==  ========  ===  =====  ===========================  ==============

Window totals: Saka 11.42, Salah 5.00, Haaland 9.00, Justin 3.00, Xhaka 6.00.
"""

from aigaffer.detailed.render import (
    NO_GAFFER,
    REASON_UNAVAILABLE,
    render_detailed,
    what_changed,
)


def row(name, pos, pr, f8, f9, x8, x9, *, st="a", nw="", xm=90.0, club="ARS"):
    return {
        "n": name, "p": pos, "c": club, "cf": club, "pr": pr, "sp": None,
        "st": st, "nw": nw, "xm": xm,
        "f": {8: f8, 9: f9}, "x": {8: x8, 9: x9}, "t": round(x8 + x9, 2),
    }


PLAYERS = {
    1: row("Saka", "MID", 101, [["LEE", "h"]], [["LEE", "h"], ["WOL", "a"]], 4.12, 7.30),
    2: row("Salah", "MID", 129, [["WOL", "a"]], [], 5.0, 0.0),
    3: row("Haaland", "FWD", 140, [["LEE", "h"]], [["LEE", "h"]], 6.0, 3.0),
    4: row("Justin", "DEF", 46, [["LEE", "h"]], [], 3.0, 0.0),
    5: row("Xhaka", "MID", 50, [["WOL", "a"]], [["LEE", "h"]], 2.0, 4.0),
    6: row("Raya", "GKP", 56, [["LEE", "h"]], [["LEE", "h"]], 3.0, 3.0),
}


def plan(pid, ins, outs, *, hits=0, xp=20.0, net=19.0, resolved=False):
    return {
        "in": ins, "out": outs, "hits": hits, "xp": xp, "net": net, "path": [],
        "ft_kept": 1, "ft_credit": 0.0, "transfer_cost": 0.0,
        "bench_charge": 0.0, "resolved": resolved,
    }


def snap(mode="deadline", ts="2026-10-17T10:00Z", **over):
    base = {
        "gw": 8, "mode": mode, "ts": ts, "events": [8, 9], "bank": 5, "ft": 1,
        "players": PLAYERS,
        "squad": [1, 2, 3, 4, 6],
        "xi": [1, 2, 3, 4, 6],
        "bench": [5],
        "captain": 1, "vice": 2,
        "plans": {0: plan(0, [5], [4]), 1: plan(1, [], [], xp=18.0, net=18.0)},
        "chosen": 0,
        "decision": {
            "rationale": "Xhaka's fixtures beat Justin's.",
            "adjustments": [], "chip": "none", "chip_justification": "",
            "source": "manager",
        },
        "chip_evs": {"bench_boost": 6.1},
        "calendar": ["Bench boost: GW9, the double"],
        "freehit": None,
        "skipped": [
            {"id": 3, "pos": "FWD", "replaces": 4, "gap": 6.0,
             "price_diff": 94, "tag": "unaffordable"},
        ],
    }
    base.update(over)
    return base


def test_the_sections_appear_in_order():
    text = render_detailed([snap()], "Sat 18 Oct 11:00")

    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings == [
        "# GW8 Detailed Report",
        "## What changed",
        "## The plan",
        "## Transfers",
        "## Your squad",
        "## Captain and vice",
        "## Transfers skipped",
        "## The options weighed",
        "## Chips",
    ]
    assert "Deadline: Sat 18 Oct 11:00 · Last updated: deadline run 2026-10-17T10:00Z" in text


def test_a_transfer_shows_both_mens_fixtures_side_by_side_and_the_gap():
    text = render_detailed([snap()], "d")

    transfers = text.split("## Transfers\n")[1].split("\n## ")[0]
    # Justin out: LEE(h) 3.00 then a blank; Xhaka in: wol(a) 2.00 then LEE(h) 4.00.
    # Totals 3.00 and 6.00, so the gap is +3.00.
    assert "Justin" in transfers and "Xhaka" in transfers
    assert "LEE(h) 3.00" in transfers and "— 0.00" in transfers
    assert "wol(a) 2.00" in transfers and "LEE(h) 4.00" in transfers
    assert "+3.00" in transfers


def test_double_and_blank_cells():
    text = render_detailed([snap()], "d")

    assert "LEE(h)+wol(a) 7.30" in text
    assert "— 0.00" in text
    assert "LEE(h) 4.12" in text


def test_what_changed_names_old_and_new_and_first_look_first():
    early = snap("early", "T1")
    later = snap(
        "scout", "T2",
        captain=3,
        bench=[4],
        plans={0: plan(0, [4], [5]), 1: plan(1, [], [])},
        decision={**snap()["decision"], "chip": "bench_boost"},
    )

    # Early planned Justin -> Xhaka; scout plans Xhaka -> Justin.
    lines = what_changed(early, later)

    assert "Transfers: Xhaka → Justin (was: Justin → Xhaka)" in lines
    assert "Captain: Haaland (was: Saka)" in lines
    assert "Chip: bench_boost (was: none)" in lines
    assert "Bench: Justin (was: Xhaka)" in lines
    assert what_changed(None, early) == ["First look"]
    assert what_changed(early, snap("scout", "T2")) == ["No change to the plan"]

    text = render_detailed([early, later], "d")
    section = text.split("## What changed\n")[1].split("\n## ")[0]
    assert section.index("**scout — T2**") < section.index("**early — T1**")
    assert "First look" in section.split("**early — T1**")[1]


def test_what_changed_lists_squad_news_changes():
    early = snap("early", "T1")
    injured = {**PLAYERS, 1: {**PLAYERS[1], "st": "d", "nw": "knock — 75%"}}
    scout = snap("scout", "T2", players=injured)

    lines = what_changed(early, scout)

    assert "Saka: news 'knock — 75%' (was none)" in lines


def test_what_changed_lists_xmins_for_adjusted_players():
    early = snap("early", "T1")
    cut = {**PLAYERS, 1: {**PLAYERS[1], "xm": 70.0}}
    scout = snap(
        "scout", "T2", players=cut,
        decision={**snap()["decision"], "adjustments": [
            {"player_id": 1, "expected_minutes": 70.0, "reason": "knock"}]},
    )

    assert "Saka: xMins 90 → 70" in what_changed(early, scout)


def test_a_rerun_entry_says_so():
    first = snap("deadline", "T1")
    again = snap("deadline", "T2", reran=True)

    text = render_detailed([first, again], "d")

    assert "**deadline — T2 (re-run)**" in text
    assert "**deadline — T1**" in text


def test_no_gaffer_prints_the_solvers_pick_line():
    solver = snap(
        decision={**snap()["decision"], "source": "solver"}, reasons=None,
    )
    manager = snap(reasons=None)

    assert NO_GAFFER in render_detailed([solver], "d")
    assert NO_GAFFER == "Solver's pick — no gaffer this run"
    unavailable = render_detailed([manager], "d")
    assert REASON_UNAVAILABLE in unavailable
    assert NO_GAFFER not in unavailable
    # And a snapshot with no reasons key at all reads the same as None.
    bare = snap()
    assert REASON_UNAVAILABLE in render_detailed([bare], "d")


def test_a_free_hit_week_titles_the_squad_as_the_free_hit_team():
    hit = snap(
        freehit={"squad": [1, 2, 3, 5, 6], "xi": [1, 2, 3, 5, 6]},
        decision={**snap()["decision"], "chip": "free_hit"},
    )

    text = render_detailed([hit], "d")

    assert "## Your free-hit team" in text
    assert "## Your squad" not in text
    # The standing squad is named; Justin is in it, and not in the free-hit team.
    assert "Standing squad: Saka, Salah, Haaland, Justin, Raya" in text
