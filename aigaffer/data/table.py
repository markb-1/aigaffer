"""The league table, each club's form, and what the model thinks of it.

The projections already price every fixture through the fitted team
strengths (:mod:`aigaffer.model.strength`), but nothing a language model
reads showed where the clubs actually stand. A gaffer asked to explain a
fixture call could argue only from his players' totals, and could not say
whether the ratings behind them agreed with the table. This module is that
missing page, built from what every run already holds — the bootstrap, the
fixture list and the run's fit — with no network and no I/O.

Three layers, so that every reader prints the same lines:

* :func:`league_table` computes the rows: the table sorted by the Premier
  League's tie-breaks (points, then goal difference, then goals scored; a tie
  on all three shares the place, shown ``4=``, and head-to-head is
  deliberately not applied), each club's last five, the home and away splits,
  and the model's view.
* :func:`table_record` is the run snapshot's form of them, in the snapshot's
  short-key style.
* :func:`table_lines` renders the record, one line a club plus a legend, so
  the weekly briefing (record built in memory) and the question handler
  (record read off disk) print exactly the same thing.

**Which matches count.** Every fixture that has been played — full time, not
the data check a day later (:attr:`~aigaffer.data.models.Fixture.played`) —
with both scores filled in. The strength fit waits for ``finished``, so the
table can run up to a day ahead of the model; a result is a result.

**The model view, on the season's scale.** ``exp(attack)`` and
``exp(defence)`` from the fit, joined through each club's permanent ``code``
as the fit itself is, and re-centred over this season's clubs: the fit
centres over every team it saw, relegated sides included, and those sides'
weak attacks would otherwise flatter every live attack by a few percent.
After the re-centring 1.00 means "average among this season's clubs" — the
same meaning the season column has, which is each club's goals scored and
conceded a game divided by the league's goals per team per game. In the
fit's terms ``defence`` multiplies the goals a team concedes, so under 1 is
the better defence. The model is opponent-adjusted and deliberately shrunk
toward the average, so it always sits nearer 1 than the raw season figure;
the legend says so, because without it every row reads as the model being
wrong.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import exp

from aigaffer.data.models import Bootstrap, Fixture
from aigaffer.model.strength import TeamStrengths

LAST_N = 5

# The longest club name off the wire before it is cut: the briefing's own cap.
MAX_FIELD = 60

# The flag a reader's clubs carry. Not a leading "*", which a model reads as
# a markdown bullet.
OURS = "[ours]"


@dataclass(frozen=True)
class Record:
    """Won, drawn, lost, goals for and against, over some matches."""

    w: int = 0
    d: int = 0
    l: int = 0  # noqa: E741 — the table's own letter
    gf: int = 0
    ga: int = 0

    def plus(self, gf: int, ga: int) -> "Record":
        return Record(
            self.w + (gf > ga), self.d + (gf == ga), self.l + (gf < ga),
            self.gf + gf, self.ga + ga,
        )

    def as_list(self) -> list[int]:
        return [self.w, self.d, self.l, self.gf, self.ga]


@dataclass(frozen=True)
class Result:
    """One played match from one club's side: ``venue`` is ``h`` or ``a``,
    ``opp`` the opponent's short name."""

    gf: int
    ga: int
    opp: str
    venue: str

    @property
    def outcome(self) -> str:
        return "W" if self.gf > self.ga else "D" if self.gf == self.ga else "L"


@dataclass(frozen=True)
class TeamRow:
    """One club's line. ``level`` says another club shares ``position``.
    ``model_att`` and ``model_def`` are None with no fit, or for a club the
    fit never saw."""

    short: str
    name: str
    position: int
    level: bool
    home: Record
    away: Record
    last5: tuple[Result, ...]
    model_att: float | None = None
    model_def: float | None = None

    @property
    def won(self) -> int:
        return self.home.w + self.away.w

    @property
    def drawn(self) -> int:
        return self.home.d + self.away.d

    @property
    def lost(self) -> int:
        return self.home.l + self.away.l

    @property
    def played(self) -> int:
        return self.won + self.drawn + self.lost

    @property
    def gf(self) -> int:
        return self.home.gf + self.away.gf

    @property
    def ga(self) -> int:
        return self.home.ga + self.away.ga

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    @property
    def pts(self) -> int:
        return 3 * self.won + self.drawn


def league_table(
    bootstrap: Bootstrap,
    fixtures: Iterable[Fixture],
    strengths: TeamStrengths | None = None,
) -> list[TeamRow] | None:
    """The table as it stands, best first; None before anything is played.

    Every club on the bootstrap has a row, one yet to play included. A
    fixture naming a club the bootstrap does not carry is passed over.
    """
    clubs = {t.id: t for t in bootstrap.teams}
    games = sorted(
        (
            f
            for f in fixtures
            if f.played
            and f.team_h_score is not None
            and f.team_a_score is not None
            and f.team_h in clubs
            and f.team_a in clubs
        ),
        # By kickoff, so a rearranged match lands when it was played, not in
        # the round it was first given.
        key=lambda f: (f.kickoff_time or "", f.event or 0, f.id),
    )
    if not games:
        return None

    home = {tid: Record() for tid in clubs}
    away = {tid: Record() for tid in clubs}
    form: dict[int, list[Result]] = {tid: [] for tid in clubs}
    for f in games:
        hg, ag = f.team_h_score, f.team_a_score
        home[f.team_h] = home[f.team_h].plus(hg, ag)
        away[f.team_a] = away[f.team_a].plus(ag, hg)
        form[f.team_h].append(Result(hg, ag, clubs[f.team_a].short_name, "h"))
        form[f.team_a].append(Result(ag, hg, clubs[f.team_h].short_name, "a"))

    model = _model_view(bootstrap, strengths)
    rows = [
        TeamRow(
            short=t.short_name,
            name=t.name,
            position=0,
            level=False,
            home=home[tid],
            away=away[tid],
            last5=tuple(form[tid][-LAST_N:]),
            model_att=model.get(tid, (None, None))[0],
            model_def=model.get(tid, (None, None))[1],
        )
        for tid, t in clubs.items()
    ]

    def standing(row: TeamRow) -> tuple[int, int, int]:
        return (row.pts, row.gd, row.gf)

    rows.sort(key=lambda r: (*(-v for v in standing(r)), r.short))
    keys = [standing(r) for r in rows]
    return [
        replace(
            row,
            position=1 + sum(k > standing(row) for k in keys),
            level=keys.count(standing(row)) > 1,
        )
        for row in rows
    ]


def _model_view(
    bootstrap: Bootstrap, strengths: TeamStrengths | None
) -> dict[int, tuple[float, float]]:
    """Each live club's fitted multipliers, re-centred over the live clubs
    the fit knows (see the module docstring). Keyed by bootstrap id, joined
    through ``code``: ids reset every July and the fit never sees them."""
    if strengths is None:
        return {}
    known = {
        t.id: t.code
        for t in bootstrap.teams
        if t.code and t.code in strengths.attack and t.code in strengths.defence
    }
    if not known:
        return {}
    mean_att = sum(strengths.attack[c] for c in known.values()) / len(known)
    mean_def = sum(strengths.defence[c] for c in known.values()) / len(known)
    return {
        tid: (
            exp(strengths.attack[code] - mean_att),
            exp(strengths.defence[code] - mean_def),
        )
        for tid, code in known.items()
    }


def table_record(rows: list[TeamRow]) -> dict:
    """The snapshot's form: short keys, the totals left to be derived from
    the splits, the model's two figures only where there are any.

    ``avg`` is the league's goals per team per game, the season column's
    yardstick. Everything is a list or a string-keyed dict, so the record
    reads back from JSON exactly as it was written.
    """
    team_games = sum(r.played for r in rows)
    record_rows = []
    for r in rows:
        row = {
            "c": r.short,
            "cf": r.name,
            "pos": r.position,
            "lvl": r.level,
            "h": r.home.as_list(),
            "a": r.away.as_list(),
            "l5": [[g.outcome, g.gf, g.ga, g.opp, g.venue] for g in r.last5],
        }
        if r.model_att is not None and r.model_def is not None:
            row["ma"] = round(r.model_att, 2)
            row["md"] = round(r.model_def, 2)
        record_rows.append(row)
    return {
        "avg": round(sum(r.gf for r in rows) / team_games, 2) if team_games else 0.0,
        "rows": record_rows,
    }


def table_lines(record: dict, flagged: Iterable[str] = ()) -> list[str]:
    """A legend, then one line a club in table order.

    ``flagged`` is short names to mark :data:`OURS`. Every string off the
    wire is flattened to one line first: these lines sit in documents whose
    structure is carried by line breaks and pipes.
    """
    marked = set(flagged)
    rows = record.get("rows", [])
    modelled = any("ma" in row for row in rows)
    # The yardstick, derived unrounded from the splits: the stored ``avg`` is
    # two decimals, and dividing by it would shade every season figure.
    team_games = sum(sum(row["h"][:3]) + sum(row["a"][:3]) for row in rows)
    goals = sum(row["h"][3] + row["a"][3] for row in rows)
    avg = goals / team_games if team_games else 0.0
    legend = [
        "Place (= shares it), club, played, points, won-drawn-lost, goals for"
        " and against and the difference | home and away won-drawn-lost and"
        " goals | the last five, oldest first, and their goals for and against"
        " a game"
        + (
            " | model: the fitted attack and defence ratings | season: goals"
            f" scored and conceded a game over the league's {avg:.2f} a team a game."
            if modelled
            else "."
        ),
    ]
    if modelled:
        legend.append(
            "On both scales 1.00 is average among this season's clubs; attack"
            " over 1 scores more, defence under 1 concedes less. The model's"
            " ratings are opponent-adjusted and deliberately shrunk toward the"
            " average, so they sit nearer 1 than the season's: only a large"
            " gap, or two figures pointing opposite ways, is worth a second look."
        )
    if marked:
        legend.append(f"{OURS} marks a club you hold a player of or a plan would buy or sell from.")
    return [*legend, "", *(_line(row, row["c"] in marked, avg, modelled) for row in rows)]


def _line(row: dict, ours: bool, avg: float, modelled: bool) -> str:
    h, a = row["h"], row["a"]
    w, d, l, gf, ga = (h[i] + a[i] for i in range(5))
    played = w + d + l
    place = f"{row['pos']}{'=' if row['lvl'] else ''}"
    club = _flat(row["c"]) + (f" {OURS}" if ours else "")
    parts = [
        f"{place} {club} P{played} {3 * w + d}pts W{w} D{d} L{l}"
        f" GF{gf} GA{ga} GD{gf - ga:+d}",
        f"home {h[0]}-{h[1]}-{h[2]} {h[3]}:{h[4]} away {a[0]}-{a[1]}-{a[2]} {a[3]}:{a[4]}",
    ]
    last = row["l5"]
    if last:
        games = " ".join(f"{o}{f}-{g} {_flat(opp)}({v})" for o, f, g, opp, v in last)
        per_for = sum(g[1] for g in last) / len(last)
        per_against = sum(g[2] for g in last) / len(last)
        parts.append(f"last {len(last)}: {games}, {per_for:.1f}-{per_against:.1f} a game")
    else:
        parts.append("last 5: none")
    if modelled:
        view = []
        if "ma" in row:
            view.append(f"model att {row['ma']:.2f} def {row['md']:.2f}")
        if played and avg > 0:
            view.append(f"season att {gf / played / avg:.2f} def {ga / played / avg:.2f}")
        if view:
            parts.append(", ".join(view))
    return " | ".join(parts)


def _flat(text: str) -> str:
    """Somebody else's string, safe on one of our lines: one line, capped,
    with the pipe (our column separator) and the brackets (our flag) turned
    into punctuation that reads as theirs. The briefing's ``_safe``, kept
    here because this package must not import the manager's."""
    cleaned = text.replace("|", "/").replace("[", "(").replace("]", ")")
    return " ".join(cleaned.split())[:MAX_FIELD]
