"""Settings loaded from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so the project has no extra dependency for it."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    chat_webhook_url: str | None = field(default_factory=lambda: os.getenv("CHAT_WEBHOOK_URL") or None)
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./tickets.db"))
    api_key: str | None = field(default_factory=lambda: os.getenv("TRIAGE_API_KEY") or None)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    return Settings()
