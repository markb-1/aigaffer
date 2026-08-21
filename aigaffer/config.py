import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    team_id: int
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    state_dir: Path = field(default_factory=lambda: Path("state"))
    horizon: int = 6
    decay: float = 0.85
    anthropic_api_key: str | None = None
    # An opt-out, not an opt-in: the manager runs whenever there is a key to run
    # it with, and __post_init__ clamps this to the key either way.
    manager_enabled: bool = True
    manager_model: str = "claude-opus-5"

    def __post_init__(self) -> None:
        # No key, no manager. Asking for one without credentials is a run that
        # would die at the first API call, so it is a run without a manager.
        self.manager_enabled = self.manager_enabled and self.anthropic_api_key is not None

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            team_id=int(os.environ["FPL_TEAM_ID"]),
            telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
            state_dir=Path(os.environ.get("AIGAFFER_STATE_DIR", "state")),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            # Only "0" turns it off; unset or anything else leaves the key to decide.
            manager_enabled=os.environ.get("AIGAFFER_MANAGER") != "0",
            manager_model=os.environ.get("AIGAFFER_MANAGER_MODEL", "claude-opus-5"),
        )
