"""Tests for the Chip forecast: the newest saved plan's chip thinking, instantly.

``render_forecast`` is pure, so each rule of a chip's line is pinned on a
hand-built record with the exact text worked out in a comment. ``chip_forecast``
is then run end to end on the pipeline fakes (tests/fixtures.py) to pin what it
reads: a stored record, the bootstrap and the chip history, and nothing else.
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx

from aigaffer.chip_forecast import (
    CALENDAR_UNAVAILABLE,
    FOOTER,
    NO_CURRENT_CHIPS,
    NO_GAMEWEEK,
    NO_PLAN,
    chip_forecast,
    render_forecast,
)
from aigaffer.chips import (
    BENCH_BOOST,
    FREE_HIT,
    TRIPLE_CAPTAIN,
    WILDCARD,
    HeldChip,
)
from aigaffer.data.fpl_api import FplClient
from aigaffer.store import Store
from tests.fixtures import (
    HISTORY_PATH,
    PICKS_PATH,
    CountingTransport,
    config,
    make_client,
    make_executed,
    pipeline_routes,
)

# 2026-10-06 is a Tuesday; 23:30 UTC is 00:30 on Wednesday in Dublin (summer
# time runs to 25 October), which is the day the owner would call it.
TS = "2026-10-06T23:30:00+00:00"

BB, TC, WC, FH = BENCH_BOOST, TRIPLE_CAPTAIN, WILDCARD, FREE_HIT

# The first set to GW19 and the second GW20-38, one of each kind, as held at GW7.
FIRST = (
    HeldChip(BB, 1, 19), HeldChip(TC, 1, 19), HeldChip(WC, 2, 19), HeldChip(FH, 2, 19),
)
SECOND = (
    HeldChip(BB, 20, 38), HeldChip(TC, 20, 38), HeldChip(WC, 20, 38), HeldChip(FH, 20, 38),
)


def entry(chip_id, saved_for=None, value=None, bars=None):
    return {"chip": chip_id, "saved_for": saved_for, "value": value, "bars": bars or {}}


def calendar(*entries, fell_back=False):
    return {"fell_back": fell_back, "entries": list(entries)}


def path(*events, chip_at=None):
    """A path over ``events`` that plays ``chip_at`` ({event: chip}) and no other."""
    chip_at = chip_at or {}
    return [
        {"event": e, "in": [], "out": [], "hits": 0, "chip": chip_at.get(e, "none")}
        for e in events
    ]


def record(chip="none", path_=None, calendar_=None):
    return {"chip": chip, "path": path_, "chip_calendar": calendar_}


def render(rec, *, held=FIRST, used=(), gw=7, record_gw=7, mode="scout"):
    return render_forecast(rec, mode, TS, record_gw, gw, held, list(used))


def test_the_whole_message_is_laid_out_as_the_owner_asked():
    # GW7. Bench boost is this week's chip (rule 1). Triple captain and free
    # hit are saved for GW14 / GW16 by the calendar (rule 3; the path plays
    # neither). The wildcard is saved for nothing: the path runs GW7-12, so
    # "nothing in GW7-12 beats keeping it" (rule 4). The second set is the
    # four chips of GW20-38, and a wildcard was played in GW2.
    rec = record(
        chip=BB,
        path_=path(*range(7, 13)),
        calendar_=calendar(
            entry("bench_boost@19", saved_for=7, value=11.0),
            entry("triple_captain@19", saved_for=14, value=9.2),
            entry("free_hit@19", saved_for=16, value=15.3),
            entry("wildcard@19"),
        ),
    )

    text = render(rec, held=FIRST + SECOND,
                  used=[{"name": "wildcard", "event": 2}])

    assert text == (
        "🗓 Chip forecast — as of Wednesday's scout (GW7)\n"
        "Bench boost — GW7 (this week) · expires GW19\n"
        "Triple captain — GW14 · worth ~+9 · expires GW19\n"
        "Wildcard — no week yet: nothing in GW7–12 beats keeping it · expires GW19\n"
        "Free hit — GW16 · worth ~+15 · expires GW19\n"
        "Second set (GW20–38): bench boost, triple captain, wildcard, free hit\n"
        "Played: wildcard GW2\n"
        "Re-planned every run — this is the current thinking, not a commitment."
    )
    assert text.splitlines()[-1] == FOOTER


def test_a_chip_the_record_plays_this_week_says_so():
    # The record's own week-1 chip is the triple captain and it is GW7.
    text = render(record(chip=TC), held=(FIRST[1],))

    assert text.splitlines()[1] == "Triple captain — GW7 (this week) · expires GW19"


def test_this_week_needs_the_chip_to_be_playable_this_week():
    # A held triple captain whose window opens in GW9 cannot be this week's,
    # even if a record says so: it is the second set's, not a current chip.
    text = render(record(chip=TC, calendar_=calendar()), held=(HeldChip(TC, 9, 19),))

    assert text.splitlines()[1:3] == [
        NO_CURRENT_CHIPS,
        "Second set (GW9–19): triple captain",
    ]
    assert "(this week)" not in text


def test_a_chip_the_path_plays_later_shows_its_earliest_week():
    # The path plays a bench boost in GW11 and GW9: the earliest is GW9. The
    # path wins over the calendar's own saved-for week (GW14).
    rec = record(
        path_=path(7, 8, 9, 10, 11, chip_at={11: BB, 9: BB}),
        calendar_=calendar(entry("bench_boost@19", saved_for=14, value=6.0)),
    )

    text = render(rec, held=(FIRST[0],))

    assert text.splitlines()[1] == "Bench boost — GW9 · expires GW19"


def test_a_path_chip_outside_the_chips_window_is_ignored():
    # The held chip is the first set's bench boost (GW1-19). The path plays a
    # bench boost in GW25, which is the second set's chip to play, not this
    # one's, so it is not named: the calendar's saved-for week (GW12) is.
    rec = record(
        path_=path(7, 25, chip_at={25: BB}),
        calendar_=calendar(entry("bench_boost@19", saved_for=12, value=5.4)),
    )

    text = render(rec, held=(FIRST[0],))

    assert text.splitlines()[1] == "Bench boost — GW12 · worth ~+5 · expires GW19"


def test_a_saved_for_week_without_a_value_omits_the_worth():
    rec = record(calendar_=calendar(entry("free_hit@19", saved_for=16)))

    text = render(rec, held=(FIRST[3],))

    assert text.splitlines()[1] == "Free hit — GW16 · expires GW19"


def test_a_wildcard_is_never_saved_for_a_week():
    # The calendar can hold a saved_for for a wildcard; the report's own line
    # shows a dash for it, and so do we: wildcards are not scheduled.
    rec = record(
        path_=path(7, 8, 9),
        calendar_=calendar(entry("wildcard@19", saved_for=9, value=30.0)),
    )

    text = render(rec, held=(FIRST[2],))

    assert text.splitlines()[1] == (
        "Wildcard — no week yet: nothing in GW7–9 beats keeping it · expires GW19"
    )


def test_a_wildcard_without_a_path_reads_its_window_off_the_calendar_bars():
    # No path: the last bar week in the wildcard's calendar entry is the end.
    rec = record(calendar_=calendar(entry("wildcard@19", bars={"7": 1.0, "8": 2.0, "10": 3.0})))

    text = render(rec, held=(FIRST[2],))

    assert text.splitlines()[1] == (
        "Wildcard — no week yet: nothing in GW7–10 beats keeping it · expires GW19"
    )


def test_a_wildcard_with_no_known_window_just_has_no_week():
    rec = record(calendar_=calendar(entry("wildcard@19")))

    text = render(rec, held=(FIRST[2],))

    assert text.splitlines()[1] == "Wildcard — no week yet · expires GW19"


def test_a_chip_the_calendar_does_not_save_for_has_no_week_yet():
    rec = record(calendar_=calendar(entry("bench_boost@19")))

    text = render(rec, held=(FIRST[0],))

    assert text.splitlines()[1] == "Bench boost — no week yet · expires GW19"


def test_a_fell_back_calendar_is_unavailable_but_the_path_still_speaks():
    # fell_back: rules 3-4 read "calendar unavailable". The path chip (rule 2)
    # and this-week chip (rule 1) are the plan itself and still show.
    rec = record(
        chip=BB,
        path_=path(7, 8, 9, chip_at={9: TC}),
        calendar_=calendar(
            entry("free_hit@19", saved_for=16, value=15.0), fell_back=True
        ),
    )

    text = render(rec, held=(FIRST[0], FIRST[1], FIRST[2], FIRST[3]))

    assert text.splitlines()[1:5] == [
        "Bench boost — GW7 (this week) · expires GW19",
        "Triple captain — GW9 · expires GW19",
        f"Wildcard — {CALENDAR_UNAVAILABLE} · expires GW19",
        f"Free hit — {CALENDAR_UNAVAILABLE} · expires GW19",
    ]
    assert CALENDAR_UNAVAILABLE == "calendar unavailable"


def test_no_calendar_at_all_is_unavailable_too():
    text = render(record(calendar_=None), held=(FIRST[3],))

    assert text.splitlines()[1] == "Free hit — calendar unavailable · expires GW19"


def test_chips_are_ordered_by_expiry_then_the_usual_chip_order():
    # A first-set chip expiring GW19 comes before one expiring GW38 even
    # though the latter is handed in first; within a set, BB, TC, WC, FH.
    held = (HeldChip(BB, 1, 38), FIRST[3], FIRST[0])
    text = render(record(calendar_=calendar()), held=held)
    # (Bench boost 1-38 is current-set here: its window has opened.)

    assert [line.split(" — ")[0] for line in text.splitlines()[1:4]] == [
        "Bench boost", "Free hit", "Bench boost",
    ]
    assert [line.rsplit(" · ", 1)[1] for line in text.splitlines()[1:4]] == [
        "expires GW19", "expires GW19", "expires GW38",
    ]


def test_a_second_set_chip_gets_one_line_not_a_chip_line():
    # held at GW7: the second set opens GW20. A free hit shifted a week by the
    # back-to-back rule opens GW21: the line spans the whole set.
    held = (FIRST[0], HeldChip(BB, 20, 38), HeldChip(FH, 21, 38))

    text = render(record(calendar_=calendar()), held=held)

    assert text.splitlines()[2] == "Second set (GW20–38): bench boost, free hit"
    assert "Free hit —" not in text


def test_played_chips_are_named_with_their_gameweeks_in_order():
    used = [
        {"name": "bboost", "event": 5},
        {"name": "wildcard", "event": 2},
        {"name": "3xc", "event": 6},
        {"name": "freehit", "event": 3},
        {"name": "mystery", "event": 4},  # not ours: left out
    ]

    text = render(record(calendar_=calendar()), held=(), used=used)

    assert (
        "Played: wildcard GW2, free hit GW3, bench boost GW5, triple captain GW6"
        in text.splitlines()
    )


def test_no_played_chips_means_no_played_line():
    assert "Played" not in render(record(calendar_=calendar()), held=FIRST)


def test_a_record_from_last_gameweek_names_itself_and_is_not_this_week():
    # Now GW8, the plan is GW7's deadline verdict. Its week-1 chip (bench boost)
    # was GW7's, not GW8's: this-week is only for a record of this gameweek.
    # The path week GW7 is already gone; GW10 is the forecast.
    rec = record(chip=BB, path_=path(7, 8, 9, 10, chip_at={10: BB, 7: BB}))

    text = render(rec, held=(FIRST[0],), gw=8, record_gw=7, mode="deadline")

    assert text.splitlines()[:2] == [
        "🗓 Chip forecast — as of Wednesday's deadline (GW7)",
        "Bench boost — GW10 · expires GW19",
    ]


def test_the_early_scout_is_named_as_such():
    text = render(record(calendar_=calendar()), held=(), mode="early")

    assert text.splitlines()[0] == "🗓 Chip forecast — as of Wednesday's early scout (GW7)"


def test_no_chips_in_the_current_set_says_so_and_still_lists_the_rest():
    text = render(record(calendar_=calendar()), held=SECOND, used=[{"name": "wildcard", "event": 2}])

    assert text.splitlines() == [
        "🗓 Chip forecast — as of Wednesday's scout (GW7)",
        NO_CURRENT_CHIPS,
        "Second set (GW20–38): bench boost, triple captain, wildcard, free hit",
        "Played: wildcard GW2",
        FOOTER,
    ]


def test_a_path_written_before_it_recorded_chips_is_tolerated():
    # Records saved before the path carried a chip have moves of exactly
    # {"event","in","out","hits"}: no chip key. They name no chip week, so
    # the calendar speaks, and nothing raises.
    old = [{"event": e, "in": [1], "out": [2], "hits": 0} for e in (7, 8, 9)]
    rec = record(
        path_=old, calendar_=calendar(entry("triple_captain@19", saved_for=14, value=9.2))
    )

    text = render(rec, held=(FIRST[1],))

    assert text.splitlines()[1] == "Triple captain — GW14 · worth ~+9 · expires GW19"


def test_a_saved_for_week_already_gone_is_no_week():
    # A record from GW7 saved the bench boost for GW7; it is GW8 now.
    rec = record(calendar_=calendar(entry("bench_boost@19", saved_for=7, value=6.0)))

    text = render(rec, held=(FIRST[0],), gw=8, record_gw=7)

    assert text.splitlines()[1] == "Bench boost — no week yet · expires GW19"


# --- chip_forecast, on the pipeline fakes ---------------------------------------
#
# The pipeline universe's next gameweek is GW2; the chip history has a
# wildcard played in GW1, before the first-set wildcard's window (GW2-19)
# opens: it spends nothing, so all eight chips are still held.


def stored(tmp_path, gw=2, mode="scout", decision=None):
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    store.save_run(
        gw, mode, "report", decision or record(chip=BB, calendar_=calendar())
    )
    return cfg, store


def counting_client(transport):
    return FplClient(http=httpx.Client(transport=transport), sleep=lambda _: None)


def test_the_forecast_reads_the_newest_report_and_asks_for_two_things(tmp_path):
    cfg, store = stored(tmp_path)
    stamp = store.latest_verdict(2, ("scout",)).ts
    day = datetime.fromisoformat(stamp).astimezone(
        ZoneInfo("Europe/Dublin")
    ).strftime("%A")
    transport = CountingTransport(pipeline_routes())

    reply = chip_forecast(cfg, counting_client(transport), datetime.now(UTC))

    assert reply.splitlines() == [
        f"🗓 Chip forecast — as of {day}'s scout (GW2)",
        "Bench boost — GW2 (this week) · expires GW19",
        "Triple captain — no week yet · expires GW19",
        "Wildcard — no week yet · expires GW19",
        "Free hit — no week yet · expires GW19",
        "Second set (GW20–38): bench boost, triple captain, wildcard, free hit",
        "Played: wildcard GW1",
        FOOTER,
    ]
    # The bootstrap and the chip history, each once; no picks, no fixtures.
    assert dict(transport.counts) == {
        "/api/bootstrap-static/": 1, HISTORY_PATH: 1,
    }
    assert PICKS_PATH not in transport.counts


def test_a_report_from_last_gameweek_is_used_when_this_one_has_none(tmp_path):
    cfg, _ = stored(tmp_path, gw=1, mode="deadline")

    reply = chip_forecast(cfg, make_client(pipeline_routes()), datetime.now(UTC))

    assert reply.splitlines()[0].endswith("'s deadline (GW1)")


def test_no_report_yet_says_so(tmp_path):
    cfg = config(state_dir=tmp_path / "state")
    Store(cfg.state_dir / "aigaffer.db")

    reply = chip_forecast(cfg, make_client(pipeline_routes()), datetime.now(UTC))

    assert reply == NO_PLAN.format(gw=2)
    assert reply == "No plan yet — the first report for GW2 hasn't run."


def test_a_missing_store_is_no_plan_and_is_not_created(tmp_path):
    # During the tick's pull the database can be momentarily absent; opening
    # a Store would create an empty one and wedge the rebase.
    cfg = config(state_dir=tmp_path / "state")

    reply = chip_forecast(cfg, make_client(pipeline_routes()), datetime.now(UTC))

    assert reply == NO_PLAN.format(gw=2)
    assert not (tmp_path / "state").exists()


def test_a_recorded_chip_the_api_already_lists_is_played_once(tmp_path):
    cfg, store = stored(tmp_path)
    store.save_executed(make_executed(chip=BB, transfers_in=[], transfers_out=[]))
    routes = pipeline_routes()
    routes[HISTORY_PATH] = {
        **routes[HISTORY_PATH],
        "chips": [
            {"name": "wildcard", "time": "2025-08-15T10:00:00Z", "event": 1},
            {"name": "bboost", "time": "2025-08-20T10:00:00Z", "event": 2},
        ],
    }

    reply = chip_forecast(cfg, make_client(routes), datetime.now(UTC))

    assert "Played: wildcard GW1, bench boost GW2" in reply.splitlines()


def test_no_gameweek_ahead_says_so(tmp_path):
    routes = pipeline_routes()
    bootstrap = dict(routes["/api/bootstrap-static/"])
    bootstrap["events"] = [
        {**e, "is_next": False} for e in bootstrap["events"]
    ]
    routes["/api/bootstrap-static/"] = bootstrap
    cfg = config(state_dir=tmp_path / "state")

    reply = chip_forecast(cfg, make_client(routes), datetime.now(UTC))

    assert reply == NO_GAMEWEEK == "No gameweek ahead — nothing to forecast."


def test_a_chip_recorded_this_week_counts_as_played_not_held(tmp_path):
    # "Transfers made" with a bench boost this week: the chip history does not
    # have it until the deadline, but it is spent now, and shows as played in
    # GW2. The others keep their windows: still the current set.
    cfg, store = stored(tmp_path)
    store.save_executed(make_executed(chip=BB, transfers_in=[], transfers_out=[]))

    reply = chip_forecast(cfg, make_client(pipeline_routes()), datetime.now(UTC))

    lines = reply.splitlines()
    assert not any(line.startswith("Bench boost —") for line in lines)
    assert "Played: wildcard GW1, bench boost GW2" in lines
    assert lines[1].startswith("Triple captain —")
    # The others keep their windows: still current, not shoved to a "second
    # set" by the one-chip-a-week shift.
    assert not any(line.startswith("Second set (GW3") for line in lines)
    assert [line.split(" — ")[0] for line in lines[1:4]] == [
        "Triple captain", "Wildcard", "Free hit",
    ]
