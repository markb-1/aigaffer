"""The review on the command line: which ticks run it, and the hand command."""

import copy

import pytest

from aigaffer import __main__ as cli
from aigaffer import review
from aigaffer.store import Store
from tests.test_orchestrator import (  # noqa: F401 — store and the chips switch are fixtures
    bootstrap_due_in,
    chips_off_from_the_environment,
    decided,
    pipeline_routes,
    serve,
    store,
    stub_gaffer,
)
from tests.test_review import WRITTEN

# The CLI's configured universe (tests/test_orchestrator.py ``store`` fixture):
# GW1 is current and finished; it is made final here by adding data_checked.


def final(payload):
    payload = copy.deepcopy(payload)
    payload["events"][0].update(data_checked=True)
    return payload


def spy(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "review_tick", lambda *a, **k: calls.append((a, k)))
    return calls


def test_a_stand_down_tick_runs_the_review(monkeypatch, store):
    calls = spy(monkeypatch)
    serve(monkeypatch, pipeline_routes(bootstrap=final(bootstrap_due_in(100))))
    store.save_run(2, "early", "md", {})  # the early scout has gone: the tick stands down
    assert cli.main(["auto"]) == 0
    assert len(calls) == 1


def test_a_scout_or_deadline_tick_skips_the_review(monkeypatch, store):
    calls = spy(monkeypatch)
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: "report")
    for hours in (30, 10, 2):  # scout, deadline, reminder->deadline windows
        serve(monkeypatch, pipeline_routes(bootstrap=final(bootstrap_due_in(hours))))
        assert cli.main(["auto"]) == 0
    assert calls == []


def test_an_early_tick_runs_the_review_before_the_report(monkeypatch, store):
    order = []
    monkeypatch.setattr(cli, "review_tick", lambda *a, **k: order.append("review"))
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: order.append("report") or "report")
    monkeypatch.setattr(cli, "_mode", lambda args, client, store: ("early", 2))
    serve(monkeypatch, pipeline_routes(bootstrap=final(bootstrap_due_in(100))))
    assert cli.main(["auto"]) == 0
    assert order == ["review", "report"]


def test_a_dry_run_tick_skips_the_review(monkeypatch, store):
    calls = spy(monkeypatch)
    store.save_run(2, "early", "md", {})
    serve(monkeypatch, pipeline_routes(bootstrap=final(bootstrap_due_in(100))))
    assert cli.main(["auto", "--dry-run"]) == 0
    assert calls == []


def test_the_tick_asks_the_api_for_the_bootstrap_once(monkeypatch, store):
    store.save_run(2, "early", "md", {})  # a stand-down tick: no fixtures fetched
    serve(monkeypatch, pipeline_routes(bootstrap=final(bootstrap_due_in(100))))
    client = cli.FplClient()
    fetched = []
    real = client.bootstrap
    monkeypatch.setattr(client, "bootstrap", lambda: fetched.append(1) or real())
    monkeypatch.setattr(cli, "FplClient", lambda: client)
    seen = []
    monkeypatch.setattr(
        cli, "review_tick", lambda cfg, ticking, st, overtaken=None: seen.append(ticking.bootstrap())
    )
    assert cli.main(["auto"]) == 0
    assert len(seen) == 1 and len(fetched) == 1


def test_the_hand_command_prints_on_a_dry_run_and_writes_nothing(monkeypatch, store, capsys, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(review, "_writer", lambda: lambda cfg, facts: WRITTEN)
    serve(monkeypatch, final_routes())
    assert cli.main(["review", "--dry-run"]) == 0
    assert "## Looking back — GW1" in capsys.readouterr().out
    assert not store.has_run(1, "review")


def test_the_hand_command_refuses_a_second_review_without_force(monkeypatch, store, capsys, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(review, "_writer", lambda: lambda cfg, facts: WRITTEN)
    serve(monkeypatch, final_routes())
    store.save_run(1, "review", "md", {})
    assert cli.main(["review"]) == 0
    assert "has run already" in capsys.readouterr().out
    assert cli.main(["review", "--force"]) == 0
    assert len([r for r in store.last_runs(10) if r["mode"] == "review"]) == 2
    # Forced, it replaced the section rather than adding a second.
    assert (tmp_path / "GW1.md").read_text(encoding="utf-8").count("<!-- looking-back gw1 -->") == 1


def test_the_hand_command_needs_the_manager(monkeypatch, store, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    serve(monkeypatch, final_routes())
    assert cli.main(["review"]) == 1
    assert "needs the manager" in capsys.readouterr().out


@pytest.mark.xfail(reason="needs Task 6: the Last week briefing section", strict=True)
def test_an_early_tick_reviews_first_and_the_scout_reads_the_lessons(monkeypatch, store):
    # The behaviour the feature exists for, end to end across the CLI, the
    # review and the briefing: GW1 goes final, the early scout's tick reviews
    # it, and the scout's gaffer reads its lessons.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "_mode", lambda args, client, store: ("early", 2))
    monkeypatch.setattr(cli, "peer_has_run", lambda *a: False)
    monkeypatch.setattr(review, "_writer", lambda: lambda cfg, facts: WRITTEN)
    gaffer = stub_gaffer(monkeypatch, decided)
    # pipeline_routes last, so its fifteen-man picks win at the shared URL and
    # the run is not a draft (a draft never consults the gaffer).
    serve(monkeypatch, {**final_routes(), **pipeline_routes(bootstrap=final(bootstrap_due_in(100)))})
    assert cli.main(["auto"]) == 0
    assert store.has_run(1, "review")
    assert "## Last week (GW1)" in gaffer.consults[0].briefing


def final_routes():
    """The CLI universe with GW1 final and everything the review reads."""
    from tests.test_orchestrator import TEAM_ID
    from tests.test_review import routes as review_routes

    board = final(bootstrap_due_in(100))
    base = review_routes(board=board)
    # tests.test_review.routes is keyed to its own team id; re-key to the CLI's.
    return {path.replace("/entry/42/", f"/entry/{TEAM_ID}/"): payload for path, payload in base.items()}
