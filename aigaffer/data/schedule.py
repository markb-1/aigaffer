"""Who each club plays, gameweek by gameweek, read off the fixture list.

Two documents need the same answer to "whom do they play and where": the
detailed report's snapshot, which prints it beside every player's projection,
and the manager's briefing, which has to tell the gaffer the same thing in
words he can quote. They once sat in the first module, which left the second
either importing from a report package it has no other business with or
rewriting the lookup, and a venue written twice is a venue that can be wrong
once. It lives here, below both, with nothing but the data models above it.
"""

from collections import defaultdict

from aigaffer.data.models import Fixture


def opponents(
    fixtures: list[Fixture], clubs: dict, window: set[int]
) -> dict[tuple[int, int], list[tuple[str, str]]]:
    """Each club's opponents in each window gameweek, as ``(short name, h|a)``.

    ``h`` means *this* club is at home: the home side's entry names the away
    side with ``h``, and the away side's names the home side with ``a``. A
    fixture with no gameweek (``event`` None) has not been scheduled and is
    no one's game yet; one outside the window is not this run's. Kick-off
    order within a gameweek, then fixture id, so a double reads in the order
    it is played. A club with no entry for a gameweek has a blank.
    """
    schedule: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
    ordered = sorted(
        (f for f in fixtures if f.event is not None and f.event in window),
        key=lambda f: (f.kickoff_time or "", f.id),
    )
    for fixture in ordered:
        schedule[(fixture.event, fixture.team_h)].append((clubs[fixture.team_a].short_name, "h"))
        schedule[(fixture.event, fixture.team_a)].append((clubs[fixture.team_h].short_name, "a"))
    return schedule


def games_text(games: list[tuple[str, str]]) -> str:
    """One gameweek as the house writes it: ``LEE(h)``, a double joined with
    ``+``, a blank as ``—``, away opponents in lower case (``lee(a)``) so the
    venue is visible at a glance as well as in the brackets."""
    if not games:
        return "—"
    return "+".join(f"{opp if where == 'h' else opp.lower()}({where})" for opp, where in games)
