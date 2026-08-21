from pathlib import Path

from aigaffer.config import Config


def test_from_env_reads_values(monkeypatch):
    monkeypatch.setenv("FPL_TEAM_ID", "1234567")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("AIGAFFER_STATE_DIR", "/tmp/gaffer-state")
    cfg = Config.from_env()
    assert cfg.team_id == 1234567
    assert cfg.telegram_token == "tok"
    assert cfg.telegram_chat_id == "42"
    assert cfg.state_dir == Path("/tmp/gaffer-state")
    assert cfg.horizon == 6


def test_defaults(monkeypatch):
    monkeypatch.setenv("FPL_TEAM_ID", "1")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("AIGAFFER_STATE_DIR", raising=False)
    cfg = Config.from_env()
    assert cfg.telegram_token is None
    assert cfg.telegram_chat_id is None
    assert cfg.state_dir == Path("state")
    assert cfg.decay == 0.85
