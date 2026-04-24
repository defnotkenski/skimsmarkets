from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Default horizon window — markets whose game_start_time sits further out than
# this are left out of the slate. 24h catches "today's slate"; use 48-72 on the
# CLI to pull in tomorrow. Enforced server-side via Polymarket's startTimeMax,
# so events outside the window never hit the matcher/LLM path.
DEFAULT_HORIZON_HOURS = 12

# Concurrency caps. See plan for rationale.
SPECIALIST_SEM = 16
DIRECTOR_SEM = 2
# Per-event BBO fan-out against Polymarket. Each event can trigger N parallel
# BBO lookups (one per tradable side); this caps aggregate concurrency.
POLYMARKET_FETCH_SEM = 8


@dataclass(frozen=True)
class Config:
    xai_api_key: str
    anthropic_api_key: str

    @classmethod
    def from_env(cls) -> "Config":
        # Reads .env from the current directory (and parents) if present. Does not
        # override vars that are already set in the shell, so explicit exports win.
        load_dotenv()
        xai = os.environ.get("XAI_API_KEY", "").strip()
        anth = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        missing: list[str] = []
        if not xai:
            missing.append("XAI_API_KEY")
        if not anth:
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required env var(s): {', '.join(missing)}. "
                "Add them to a .env file at the project root or export them in your shell."
            )
        return cls(xai_api_key=xai, anthropic_api_key=anth)
