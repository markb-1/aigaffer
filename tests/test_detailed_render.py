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
7   Cucurella DEF   6.1   GW8 wol(a)  GW9 LEE(h)        2.00, 4.00
6   Raya      GKP   5.6   GW8 LEE(h)  GW9 LEE(h)        3.00, 3.00
==  ========  ===  =====  ===========================  ==============

Window totals: Saka 11.42, Salah 5.00, Haaland 9.00, Justin 3.00, Xhaka 6.00.
"""

from aigaffer.report.render import ADVISORY, NOT_APPLIED, OVERRULED
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
    7: row("Cucurella", "DEF", 61, [["WOL", "a"]], [["LEE", "h"]], 2.0, 4.0),
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
        "plans": {0: plan(0, [7], [4]), 1: plan(1, [], [], xp=18.0, net=18.0)},
        "chosen": 0,
        "decision": {
            "rationale": "Cucurella's fixtures beat Justin's.",
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
        "### Justin → Cucurella",
        "## Your squad",
        "## Captain and vice",
        "## Transfers skipped",
        "## The options weighed",
        "## The road ahead",
        "## Chips",
    ]
    assert "Deadline: Sat 18 Oct 11:00 · Last updated: deadline run 2026-10-17T10:00Z" in text


def test_a_transfer_shows_both_mens_fixtures_side_by_side_and_the_gap():
    text = render_detailed([snap()], "d")

    transfers = text.split("## Transfers\n")[1].split("\n## ")[0]
    # Justin out: LEE(h) 3.00 then a blank; Cucurella in: wol(a) 2.00 then LEE(h) 4.00.
    # Totals 3.00 and 6.00, so the gap is +3.00.
    assert "Justin" in transfers and "Cucurella" in transfers
    assert "LEE(h) 3.00" in transfers and "— 0.00" in transfers
    assert "wol(a) 2.00" in transfers and "LEE(h) 4.00" in transfers
    assert "Cucurella over Justin: +3.00 weighted (+3.00 plain) over the window." in transfers


def test_totals_are_plain_and_weighted_and_the_gap_gives_both():
    # Two men whose weighted total t is below their plain sum, as in a real
    # decayed window. Out (Justin): x 3.00 + 0.00, plain 3.00, t 2.50.
    # In (Cucurella): x 2.00 + 4.00, plain 6.00, t 4.00.
    # Weighted gap 4.00 - 2.50 = +1.50; plain gap 6.00 - 3.00 = +3.00.
    players = {
        **PLAYERS,
        4: {**PLAYERS[4], "t": 2.5},
        7: {**PLAYERS[7], "t": 4.0},
    }
    text = render_detailed([snap(players=players)], "d")

    transfers = _section(text, "Transfers")
    assert "| Total | Weighted |" in transfers
    assert "Window" not in text
    assert "Cucurella over Justin: +1.50 weighted (+3.00 plain) over the window." in transfers
    # Cucurella's row ends with plain 6.00 then weighted 4.00; Justin's 3.00, 2.50.
    assert any(l.startswith("| In |") and l.endswith("| 6.00 | 4.00 |") for l in transfers.splitlines())
    assert any(l.startswith("| Out |") and l.endswith("| 3.00 | 2.50 |") for l in transfers.splitlines())
    caption = ("Weighted counts later gameweeks for less, as the planner does;"
               " it is the total decisions are made on.")
    assert caption in transfers
    assert caption in _section(text, "Your squad")


def test_the_skipped_gap_header_says_weighted():
    section = _section(render_detailed([snap()], "d"), "Transfers skipped")

    assert "| Gap (weighted) |" in section


def test_the_plan_line_reads_in_full():
    # snap(): xp 20.00, net 19.00, 0 hits, bank 5 tenths = 0.5m.
    section = _section(render_detailed([snap()], "d"), "The plan")

    assert (
        "Expected points over the window: 20.00 xP; net 19.00 (after hits and charges,"
        " plus the credit for free transfers kept); 0 hits; bank £0.5m."
    ) in section


def test_the_chips_section_uses_chip_labels_and_signs():
    none = _section(render_detailed([snap()], "d"), "Chips")
    assert "No chip this week." in none
    assert "Chip played: none" not in none
    assert "Points each chip would add this week:" in none
    assert "- Bench Boost: +6.10" in none

    played = snap(
        decision={**snap()["decision"], "chip": "bench_boost"},
        chip_evs={"bench_boost": 14.6, "triple_captain": -1.0, "wildcard": 12.0},
    )
    text = _section(render_detailed([played], "d"), "Chips")
    assert "Chip played: Bench Boost." in text
    assert "- Bench Boost: +14.60" in text
    assert "- Triple Captain: -1.00" in text
    # The wildcard is priced over the window, and says so as the GW report does.
    assert "- Wildcard (over the window): +12.00" in text


def test_the_captain_line_shows_the_plain_total_labelled():
    # Saka: 4.12 + 7.30 = 11.42 plain (t is 11.42 too here, so make t differ).
    players = {**PLAYERS, 1: {**PLAYERS[1], "t": 9.0}}
    section = _section(render_detailed([snap(players=players)], "d"), "Captain and vice")

    assert "- **Captain: Saka** (11.42 over the window)" in section


def test_the_rivals_leave_out_the_vice_and_say_what_they_are_ranked_on():
    # XI 1, 2, 3, 4, 5, 6 with Saka (1) captain and Haaland (3) vice. The XI's
    # other attackers by plain window total: Haaland 6.00 + 3.00 = 9.00,
    # Xhaka 2.00 + 4.00 = 6.00, Salah 5.00 + 0.00 = 5.00. Haaland is the
    # strongest of them, but he is the vice and has his own line above, so the
    # two rivals are Xhaka and Salah.
    shot = snap(xi=[1, 2, 3, 4, 5, 6], bench=[], captain=1, vice=3)
    section = _section(render_detailed([shot], "d"), "Captain and vice")

    assert "- **Vice: Haaland** (9.00 over the window)" in section
    assert "- Rivals (by total xP over the window): Xhaka (6.00), Salah (5.00)" in section
    assert section.count("Haaland") == 1


def test_the_squad_lists_the_xi_by_position_then_total_and_the_bench_in_order():
    # XI given deliberately jumbled: Haaland FWD, Saka MID 11.42, Raya GKP,
    # Salah MID 5.00, Justin DEF. Expected: Raya, Justin, Saka, Salah, Haaland,
    # then bench 5 (Xhaka) after.
    cur = snap(xi=[3, 1, 6, 2, 4])
    squad = _section(render_detailed([cur], "d"), "Your squad")

    names = [l.split("|")[2].split(" (")[0].strip() for l in squad.splitlines()
             if l.startswith("| XI") or l.startswith("| Bench")]
    assert names == ["Raya", "Justin", "Saka", "Salah", "Haaland", "Xhaka"]
    assert "| Bench 1 |" in squad


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
        plans={0: plan(0, [4], [7]), 1: plan(1, [], [])},
        decision={**snap()["decision"], "chip": "bench_boost"},
    )

    # Early planned Justin -> Cucurella; scout plans Cucurella -> Justin.
    lines = what_changed(early, later)

    assert "Transfers: Cucurella → Justin (was: Justin → Cucurella)" in lines
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


def assert_tables_well_formed(text):
    """Every table in the document: a header and separator lead its rows, and
    every row has the header's column count (escaped pipes not counted)."""
    lines = text.splitlines()
    index = 0
    tables = 0
    while index < len(lines):
        if lines[index].startswith("|"):
            assert index + 1 < len(lines) and set(lines[index + 1]) <= set("|-"), (
                f"table rows without a header and separator at line {index}: {lines[index]}"
            )
            width = _columns(lines[index])
            while index < len(lines) and lines[index].startswith("|"):
                assert _columns(lines[index]) == width, lines[index]
                index += 1
            tables += 1
        else:
            index += 1
    return tables


def _columns(line):
    return len(line.replace("\\|", "").strip("|").split("|"))


def _reasoned():
    return snap(reasons={
        "transfers": [{"out": 4, "in": 7, "reason": "Cucurella has two games to Justin's one."}],
        "captain": "Saka has the double.",
        "chips": "Hold the bench boost for the double.",
        "plan": "The transfer beats standing pat by two points.",
        "skipped": [{"id": 3, "reason": "Haaland costs more than the bank allows."}],
    })


def _section(text, heading):
    return text.split(f"## {heading}\n")[1].split("\n## ")[0]


def test_the_transfer_reason_is_the_entry_matching_out_and_in():
    text = render_detailed([_reasoned()], "d")

    assert "Cucurella has two games to Justin's one." in _section(text, "Transfers")
    # An entry for a different move does not match this one.
    other = snap(reasons={"transfers": [{"out": 4, "in": 5, "reason": "wrong man"}]})
    shown = _section(render_detailed([other], "d"), "Transfers")
    assert "wrong man" not in shown
    assert REASON_UNAVAILABLE in shown


def test_the_captain_reason_shows_and_the_vice_has_none():
    section = _section(render_detailed([_reasoned()], "d"), "Captain and vice")

    captain = next(l for l in section.splitlines() if "Captain: Saka" in l)
    vice = next(l for l in section.splitlines() if "Vice: Salah" in l)
    assert "Saka has the double." in captain
    assert "—" not in vice


def test_the_chips_reason_shows_in_the_chips_section():
    chip = snap(
        reasons=_reasoned()["reasons"],
        decision={**snap()["decision"], "chip": "bench_boost"},
    )

    assert "Hold the bench boost for the double." in _section(
        render_detailed([chip], "d"), "Chips")


def test_the_plan_reason_ends_the_options_weighed():
    section = _section(render_detailed([_reasoned()], "d"), "The options weighed")

    assert section.strip().splitlines()[-1] == "The transfer beats standing pat by two points."


def test_a_skipped_row_gets_its_reason_and_a_missing_one_stays_blank():
    text = render_detailed([_reasoned()], "d")
    row = next(l for l in _section(text, "Transfers skipped").splitlines() if "Haaland" in l)
    assert row.endswith("| Haaland costs more than the bank allows. |")

    bare = snap(reasons={"skipped": []})
    row = next(l for l in _section(render_detailed([bare], "d"), "Transfers skipped").splitlines()
               if "Haaland" in l)
    assert row.endswith("| unaffordable |  |")
    assert REASON_UNAVAILABLE not in row


def test_transfers_pair_by_position_not_by_sorted_id():
    # Ids sorted: out [5, 9], in [3, 12]. Zipping would pair 5 (DEF) with
    # 3 (MID) and 9 (MID) with 12 (DEF). By position: DEF 5 -> DEF 12 and
    # MID 9 -> MID 3 (listed by ascending outgoing id: Dee 5, then Emm 9).
    players = {
        **PLAYERS,
        5: row("Dee", "DEF", 50, [], [], 1.0, 1.0),
        9: row("Emm", "MID", 60, [], [], 2.0, 2.0),
        3: row("Emmtwo", "MID", 60, [], [], 5.0, 5.0),
        12: row("Deetwo", "DEF", 50, [], [], 4.0, 4.0),
    }
    cur = snap(players=players, plans={0: plan(0, [3, 12], [5, 9])})

    lines = what_changed(snap("early", "T1"), cur)

    assert "Transfers: Dee → Deetwo, Emm → Emmtwo (was: Justin → Cucurella)" in lines


def test_every_transfer_has_its_own_table():
    players = {
        **PLAYERS,
        8: row("Eight", "MID", 60, [["WOL", "a"]], [], 1.0, 0.0),
        9: row("Nine", "MID", 60, [["LEE", "h"]], [["LEE", "h"]], 3.0, 3.0),
    }
    cur = snap(players=players, plans={0: plan(0, [7, 9], [4, 8])})

    text = render_detailed([cur], "d")
    section = _section(text, "Transfers")

    assert section.count("### ") == 2
    assert "### Justin → Cucurella" in section and "### Eight → Nine" in section
    for part in section.split("### ")[1:]:
        lines = part.splitlines()
        assert lines[2].startswith("| | Player") and lines[3].startswith("|---")
        assert lines[4].startswith("| Out") and lines[5].startswith("| In")
    assert assert_tables_well_formed(text) == 5  # two transfers, squad, skipped, options


def test_squad_and_skipped_tables_are_well_formed():
    text = render_detailed([_reasoned()], "d")

    assert assert_tables_well_formed(text) >= 3


def test_free_text_cannot_break_a_table_or_a_line():
    nasty = _reasoned()
    nasty["reasons"]["skipped"] = [{"id": 3, "reason": "too dear | by a lot\nreally"}]
    nasty["reasons"]["transfers"][0]["reason"] = "a | b\nc"

    text = render_detailed([nasty], "d")

    assert "too dear \\| by a lot really" in text
    assert "a \\| b c" in text
    assert assert_tables_well_formed(text) >= 3
    early = snap("early", "T1")
    news = snap("scout", "T2", players={**PLAYERS, 1: {**PLAYERS[1], "nw": "knock\nout"}})
    assert "Saka: news 'knock out' (was none)" in what_changed(early, news)


def test_an_unpairable_move_is_noted_not_dropped_silently():
    # Two incoming defenders but only one outgoing man to pair with.
    cur = snap(plans={0: plan(0, [7, 4], [6])})

    section = _section(render_detailed([cur], "d"), "Transfers")

    assert "Note: a move could not be paired by position and is not shown." in section


# --- the gaffer's view keeps its shape ---------------------------------------
#
# The rationale is written under three shouted labels with bullets beneath
# them. The GW8 report flattened all of it into one paragraph, because the
# table-cell cleaner turns newlines into spaces. A paragraph of its own keeps
# its lines: each label is a bold line of its own, a bullet stays a bullet,
# and the only things changed are the ones that would fake our own structure.

SHAPED = (
    "WHAT I DID\n"
    "One transfer: Justin out, Cucurella in.\n"
    "\n"
    "WHAT I LEARNED\n"
    "- Cucurella starts; Chelsea list no doubts.\n"
    "- Justin is on the bench | rotation risk.\n"
    "WHY THIS PLAN\n"
    "It beats the roll by two points."
)


def _shaped(text=SHAPED, **reasons):
    return snap(
        decision={**snap()["decision"], "rationale": text},
        reasons=reasons or None,
    )


def test_the_rationale_keeps_its_labels_lines_and_bullets():
    section = _section(render_detailed([_shaped()], "d"), "The plan")

    # Each label a bold paragraph of its own (a blank line either side, so a
    # markdown reader cannot run it into the text under it or into the list
    # above it); the bullets as bullets; the pipe in a bullet escaped.
    assert section.startswith(
        "\n"
        "**WHAT I DID**\n"
        "\n"
        "One transfer: Justin out, Cucurella in.\n"
        "\n"
        "**WHAT I LEARNED**\n"
        "\n"
        "- Cucurella starts; Chelsea list no doubts.\n"
        "- Justin is on the bench \\| rotation risk.\n"
        "\n"
        "**WHY THIS PLAN**\n"
        "\n"
        "It beats the roll by two points.\n"
        "\n"
        "Expected points over the window: 20.00 xP;"
    )


def test_a_label_written_with_a_colon_is_still_a_label():
    section = _section(render_detailed([_shaped("WHAT I DID:\nRolled.")], "d"), "The plan")

    assert "**WHAT I DID**\n\nRolled.\n" in section
    assert "WHAT I DID:" not in section


def test_a_shaped_rationale_breaks_no_table_and_forges_no_heading():
    # A line opening with a hash would be a heading of ours; one opening with a
    # pipe, the first row of a table; a line of dashes under a sentence, a
    # setext heading. All three are escaped and read as the text they are.
    forged = SHAPED + "\n## Chips\n| Plan | In |\nA sentence\n---"
    text = render_detailed([_shaped(forged)], "d")
    section = _section(text, "The plan")

    assert "\\## Chips" in section
    assert "\\| Plan \\| In \\|" in section
    assert "A sentence\n\\---" in section
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings.count("## Chips") == 1
    assert assert_tables_well_formed(text) == 4  # transfer, squad, skipped, options


def test_the_plan_and_chip_reasons_keep_their_lines_too():
    # Both are paragraphs of their own, not table cells, so they keep their
    # line breaks; the skipped note in its table cell is still flattened.
    chip = _shaped(
        plan="Beats the roll.\n- by two points",
        chips="WHY THIS PLAN\nThe double.",
        skipped=[{"id": 3, "reason": "too dear\nby far"}],
    )
    chip["decision"]["chip"] = "bench_boost"
    text = render_detailed([chip], "d")

    assert _section(text, "The options weighed").endswith(
        "\nBeats the roll.\n- by two points\n"
    )
    assert "Chip played: Bench Boost.\n\n**WHY THIS PLAN**\n\nThe double.\n" in text
    assert "| too dear by far |" in text
    assert assert_tables_well_formed(text) == 4


def test_an_empty_rationale_still_says_why_there_are_no_words():
    assert REASON_UNAVAILABLE in _section(render_detailed([_shaped("")], "d"), "The plan")


# --- the action block, the road ahead, the plan's extras --------------------


def _with_path(path, **over):
    chosen = {**plan(0, [7], [4]), "path": path}
    return snap(plans={0: chosen, 1: plan(1, [], [])}, **over)


def test_do_this_prints_verbatim_after_the_deadline_line_before_what_changed():
    block = "## Do this\n\n⏰ Make these by Sat 18 Oct 11:00 — GW8\n\n1. SELL Justin"
    text = render_detailed([snap(do_this=block)], "Sat 18 Oct 11:00")

    deadline = "Deadline: Sat 18 Oct 11:00 · Last updated: deadline run 2026-10-17T10:00Z"
    assert f"{deadline}\n\n{block}\n\n## What changed" in text


def test_a_snapshot_without_do_this_prints_nothing_there():
    for shot in (snap(), snap(do_this=None)):
        text = render_detailed([shot], "d")
        assert "## Do this" not in text
        assert "Last updated: deadline run 2026-10-17T10:00Z\n\n## What changed" in text


def test_the_road_ahead_reads_like_the_gameweek_reports_with_spend_and_chip():
    # GW9: Justin (4.6) out, Cucurella (6.1) in, no hits: 1 move, 1 free, no hit.
    # GW10: Salah (12.9) out, Haaland (14.0) in, one hit: 1 - 1 = 0 free, -4 pts,
    # and a bench boost that gameweek.
    path = [
        {"gw": 9, "in": [7], "out": [4], "chip": "none", "hits": 0},
        {"gw": 10, "in": [3], "out": [2], "chip": "bench_boost", "hits": 1},
    ]
    section = _section(render_detailed([_with_path(path)], "d"), "The road ahead")

    assert section.splitlines() == [
        "",
        "- GW9: out Justin (£4.6m), in Cucurella (£6.1m) — 1 FT, no hit",
        "- GW10: out Salah (£12.9m), in Haaland (£14.0m) — 0 FTs, -4 pts in hits — Bench Boost",
        "",
        ADVISORY,
    ]


def test_the_road_ahead_without_hits_omits_the_spend_clause():
    # An older snapshot: the moves carry no hits, and none is guessed.
    path = [{"gw": 9, "in": [7], "out": [4], "chip": "none"}]
    section = _section(render_detailed([_with_path(path)], "d"), "The road ahead")

    assert "- GW9: out Justin (£4.6m), in Cucurella (£6.1m)\n" in section + "\n"
    assert "FT" not in section and "hit" not in section.replace("Advisory", "")


def test_a_chip_only_week_on_the_road_is_the_chip_alone():
    path = [{"gw": 9, "in": [], "out": [], "chip": "bench_boost", "hits": 0}]
    section = _section(render_detailed([_with_path(path)], "d"), "The road ahead")

    assert "- GW9: Bench Boost\n" in section + "\n"


def test_a_path_player_with_no_row_is_named_by_id_and_given_no_price():
    path = [{"gw": 9, "in": [99], "out": [4], "chip": "none", "hits": 0}]
    section = _section(render_detailed([_with_path(path)], "d"), "The road ahead")

    assert "- GW9: out Justin (£4.6m), in #99 — 1 FT, no hit" in section


def test_an_empty_path_says_nothing_is_planned_and_gives_no_advisory():
    section = _section(render_detailed([snap()], "d"), "The road ahead")

    assert section.strip() == "No moves planned after this week."
    assert ADVISORY not in section


def _minutes(adjustments=None, unapplied=None, **extra):
    decision = {**snap()["decision"], **extra}
    if adjustments is not None:
        decision["adjustments"] = adjustments
    if unapplied is not None:
        decision["unapplied"] = unapplied
    return snap(decision=decision)


def test_overruled_minutes_are_listed_last_word_per_player():
    # Saka was set to 60 then 75: only 75 stands. Order is first appearance.
    adjustments = [
        {"player_id": 1, "expected_minutes": 60.0, "reason": "knock"},
        {"player_id": 4, "expected_minutes": 0.0, "reason": "suspended"},
        {"player_id": 1, "expected_minutes": 75.0, "reason": "trained Friday"},
    ]
    section = _section(render_detailed([_minutes(adjustments)], "d"), "The plan")

    assert f"\n{OVERRULED}\n\n- Set Saka to 75 mins — trained Friday\n- Set Justin to 0 mins — suspended" in section
    assert "60 mins" not in section


def test_minutes_he_only_noted_are_listed_unless_the_player_was_applied():
    # Saka was applied (above) and noted later: listed once, as overruled.
    # Xhaka (5) was only ever written down.
    adjustments = [{"player_id": 1, "expected_minutes": 70.0, "reason": "knock"}]
    unapplied = [
        {"player_id": 1, "expected_minutes": 20.0, "reason": "late thought"},
        {"player_id": 5, "expected_minutes": 30.0, "reason": "rotation risk"},
    ]
    section = _section(render_detailed([_minutes(adjustments, unapplied)], "d"), "The plan")

    assert f"{NOT_APPLIED}\n\n- Xhaka at 30 mins — rotation risk" in section
    assert "late thought" not in section


def test_a_gaffer_run_prints_its_provenance_line():
    section = _section(render_detailed([_minutes(searches=8)], "d"), "The plan")

    assert section.rstrip().endswith("8 web searches. Decided by the gaffer.")
    one = _section(render_detailed([_minutes(searches=1)], "d"), "The plan")
    assert one.rstrip().endswith("1 web search. Decided by the gaffer.")


def test_a_fallback_run_names_why_in_its_provenance():
    shot = _minutes(searches=3, source="solver-fallback: Timeout")
    section = _section(render_detailed([shot], "d"), "The plan")

    assert section.rstrip().endswith(
        "3 web searches. The gaffer was unavailable (Timeout); this is the solver's pick."
    )


def test_a_solver_only_run_prints_no_provenance_even_with_a_stray_count():
    section = _section(render_detailed([_minutes(source="solver", searches=0)], "d"), "The plan")

    assert "web search" not in section and "Decided by" not in section


def test_old_snapshots_without_the_new_keys_still_render():
    old = snap()
    for key in ("unapplied", "searches"):
        assert key not in old["decision"]
    section = _section(render_detailed([old], "d"), "The plan")

    assert OVERRULED not in section and NOT_APPLIED not in section
    assert "web search" not in section
