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

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            team_id=int(os.environ["FPL_TEAM_ID"]),
            telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
            state_dir=Path(os.environ.get("AIGAFFER_STATE_DIR", "state")),
        )
