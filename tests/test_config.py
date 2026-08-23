from pathlib import Path

import pytest

from aigaffer.config import Config


@pytest.fixture
def manager_env(monkeypatch):
    """A clean slate for the manager knobs: the real environment may hold a key."""
    monkeypatch.setenv("FPL_TEAM_ID", "1")
    for name in ("ANTHROPIC_API_KEY", "AIGAFFER_MANAGER", "AIGAFFER_MANAGER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def planner_env(monkeypatch):
    """A clean slate for the planner knob: the real environment may pin it."""
    monkeypatch.setenv("FPL_TEAM_ID", "1")
    monkeypatch.delenv("AIGAFFER_PLANNER", raising=False)
    return monkeypatch


@pytest.fixture
def chips_env(monkeypatch):
    """A clean slate for the chip knob: the real environment may pin it."""
    monkeypatch.setenv("FPL_TEAM_ID", "1")
    monkeypatch.delenv("AIGAFFER_CHIPS", raising=False)
    return monkeypatch


def test_from_env_reads_values(monkeypatch):
    monkeypatch.setenv("FPL_TEAM_ID", "1234567")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("AIGAFFER_STATE_DIR", "/tmp/gaffer-state")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = Config.from_env()
    assert cfg.team_id == 1234567
    assert cfg.telegram_token == "tok"
    assert cfg.telegram_chat_id == "42"
    assert cfg.anthropic_api_key == "sk-ant-test"
    assert cfg.state_dir == Path("/tmp/gaffer-state")
    assert cfg.horizon == 6


def test_defaults(monkeypatch):
    monkeypatch.setenv("FPL_TEAM_ID", "1")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("AIGAFFER_STATE_DIR", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = Config.from_env()
    assert cfg.telegram_token is None
    assert cfg.telegram_chat_id is None
    assert cfg.state_dir == Path("state")
    assert cfg.decay == 0.85
    assert cfg.anthropic_api_key is None


def test_manager_enabled_by_a_key(manager_env):
    manager_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert Config.from_env().manager_enabled is True


def test_manager_switched_off_by_env_despite_a_key(manager_env):
    manager_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    manager_env.setenv("AIGAFFER_MANAGER", "0")
    assert Config.from_env().manager_enabled is False


def test_manager_disabled_without_a_key(manager_env):
    assert Config.from_env().manager_enabled is False


def test_manager_cannot_be_switched_on_without_a_key(manager_env):
    manager_env.setenv("AIGAFFER_MANAGER", "1")
    assert Config.from_env().manager_enabled is False


def test_an_empty_key_is_no_key(manager_env):
    """An unconfigured CI secret arrives exported and empty, not absent."""
    manager_env.setenv("ANTHROPIC_API_KEY", "")
    manager_env.setenv("AIGAFFER_MANAGER", "1")
    cfg = Config.from_env()
    assert cfg.anthropic_api_key is None
    assert cfg.manager_enabled is False


def test_manager_enabled_for_any_value_but_zero(manager_env):
    manager_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    manager_env.setenv("AIGAFFER_MANAGER", "yes")
    assert Config.from_env().manager_enabled is True


def test_manager_model_default_and_override(manager_env):
    assert Config.from_env().manager_model == "claude-opus-5"
    manager_env.setenv("AIGAFFER_MANAGER_MODEL", "claude-sonnet-5")
    assert Config.from_env().manager_model == "claude-sonnet-5"


def test_the_planner_looks_over_the_window_by_default(planner_env):
    assert Config.from_env().planner == "multi"
    assert Config(team_id=1).planner == "multi"


def test_the_planner_can_be_pinned_to_the_single_week_solver(planner_env):
    planner_env.setenv("AIGAFFER_PLANNER", "single")
    assert Config.from_env().planner == "single"


def test_only_the_word_single_switches_the_planner(planner_env):
    """The AIGAFFER_MANAGER convention: one literal value, everything else the
    default. A typo is a run of the planner that was already chosen, not a
    silent fallback to the other engine."""
    for value in ("Single", "SINGLE", "0", "greedy", ""):
        planner_env.setenv("AIGAFFER_PLANNER", value)
        assert Config.from_env().planner == "multi"


def test_the_planner_schedules_chips_by_default(chips_env):
    assert Config.from_env().chips is True
    assert Config(team_id=1).chips is True


def test_chips_are_switched_off_by_the_literal_off(chips_env):
    chips_env.setenv("AIGAFFER_CHIPS", "off")
    assert Config.from_env().chips is False


def test_only_the_word_off_switches_the_chips(chips_env):
    """The AIGAFFER_PLANNER convention: one literal value disables, everything
    else leaves the feature on. A typo does not silently turn off chip
    planning."""
    for value in ("Off", "OFF", "0", "no", "", "on"):
        chips_env.setenv("AIGAFFER_CHIPS", value)
        assert Config.from_env().chips is True


def test_manager_enabled_follows_the_key_when_constructed_directly():
    key = "sk-ant-test"
    assert Config(team_id=1).manager_enabled is False
    assert Config(team_id=1, anthropic_api_key=key).manager_enabled is True
    assert Config(team_id=1, anthropic_api_key=key, manager_enabled=False).manager_enabled is False
    assert Config(team_id=1, manager_enabled=True).manager_enabled is False
