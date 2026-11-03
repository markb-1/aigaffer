"""The lineup table: the eleven and the bench, each with this week's fixture
and xP, in one fenced block.

Boards are built by hand; the column widths are worked in comments.
"""

from aigaffer.report.lineup_table import lineup_table

NAMES = {
    1: "Raya",
    2: "Gabriel",
    3: "Calvert-Lewin",
    4: "Rice",
    5: "Haaland",
    6: "Guéhi",
    7: "Pedro",
}
ROWS = {
    1: {"f": {10: [("BRE", "h")]}, "x": {10: 4.04}},
    2: {"f": {10: [("HUL", "h")]}, "x": {10: 5.82}},
    3: {"f": {10: [("TOT", "a")]}, "x": {10: 4.9}},
    4: {"f": {10: [("BHA", "h"), ("ARS", "a")]}, "x": {10: 10.25}},  # a double
    5: {"f": {10: []}, "x": {10: 0.0}},  # a blank
    6: {"f": {10: [("NFO", "a")]}, "x": {10: 3.5}},
    7: {"f": {10: [("SUN", "a")]}, "x": {10: 0.0}},
}
GROUPS = [("GK", [1]), ("DEF", [2, 3]), ("MID", [4]), ("FWD", [5])]


def table(rows=ROWS, gw=10, notes=None, bench=(6, 7)):
    return lineup_table(
        GROUPS,
        bench,
        captain=5,
        vice=2,
        name=NAMES.__getitem__,
        rows=rows,
        gw=gw,
        notes=notes,
    )


def test_the_table_matches_the_approved_shape():
    # Widths across XI and bench together: the name column is Calvert-Lewin's
    # 13 (wider than "Gabriel V" and "Haaland C", 9 each); the fixture column
    # is the double's 7 ("BHA+ars"); the xP column is 10.25's 4 ("10.2"),
    # so every other number is right-aligned under it with a leading space.
    # Haaland blanks: his fixture is the dash and his xP slot is empty, so the
    # row ends at the dash. The first row is never padded.
    assert table() == [
        "```",
        "GW10 fixture, xP",
        "GK  Raya           BRE      4.0",
        "DEF Gabriel V      HUL      5.8",
        "    Calvert-Lewin  tot      4.9",
        "MID Rice           BHA+ars 10.2",
        "FWD Haaland C      —",
        "",
        "B1  Guéhi          nfo      3.5",
        "B2  Pedro          sun      0.0",
        "```",
    ]


def test_a_player_with_no_row_prints_his_name_alone():
    rows = {pid: row for pid, row in ROWS.items() if pid != 6}

    lines = table(rows=rows)

    assert lines[8] == "B1  Guéhi"
    assert lines[:8] == table()[:8]
    assert lines[9:] == table()[9:]


def test_no_rows_is_names_only():
    names_only = [
        "```",
        "GK  Raya",
        "DEF Gabriel V",
        "    Calvert-Lewin",
        "MID Rice",
        "FWD Haaland C",
        "",
        "B1  Guéhi",
        "B2  Pedro",
        "```",
    ]

    assert table(rows=None, gw=None) == names_only
    assert table(rows={}, gw=10) == names_only
    # Rows only for men not in the eleven: no numbers to show for the team.
    assert table(rows={99: ROWS[1]}, gw=10) == names_only
    # Rows but no gameweek to read them for.
    assert table(gw=None) == names_only


def test_notes_are_a_last_column():
    # One space after the xP column, whose padding lines the notes up.
    lines = table(notes={2: "75%", 6: "60 mins", 99: "x"})

    assert "DEF Gabriel V      HUL      5.8 75%" in lines
    assert "B1  Guéhi          nfo      3.5 60 mins" in lines
    assert not any(line.endswith(" x") for line in lines)


def test_notes_without_numbers():
    # "Gabriel V" (9) is padded to Calvert-Lewin's 13, then one space.
    lines = table(rows=None, gw=None, notes={2: "75%"}, bench=(6,))

    assert "DEF Gabriel V     75%" in lines


def test_an_empty_bench_has_no_blank_row():
    lines = table(bench=())

    assert lines[-2] == "FWD Haaland C      —"
    assert "" not in lines


def test_an_ordinary_week_fits_the_measured_width():
    # A 13-letter name with an armband is 15 wide; single fixtures are 3; the
    # widest number ("12.0") is 4. 3 + 1 + 15 + 2 + 3 + 1 + 4 = 29.
    names = {1: "Calvert-Lewin", 2: "Vuskovic"}
    rows = {
        1: {"f": {10: [("BRE", "h")]}, "x": {10: 4.0}},
        2: {"f": {10: [("BRE", "h")]}, "x": {10: 12.0}},
    }

    lines = lineup_table(
        [("FWD", [1, 2])],
        [],
        captain=1,
        vice=2,
        name=names.__getitem__,
        rows=rows,
        gw=10,
    )

    assert "FWD Calvert-Lewin C  BRE  4.0" in lines
    assert "    Vuskovic V       BRE 12.0" in lines
    assert max(len(line) for line in lines) <= 32


def test_the_first_row_is_shorter_than_any_numbered_row():
    # Telegram's badge sits over the block's top right, so the label row must
    # not reach as far as a player's xP.
    lines = table()
    numbered = [line for line in lines[2:] if line and line != "```" and line[-1].isdigit()]

    assert all(len(lines[1]) < len(line) for line in numbered)

