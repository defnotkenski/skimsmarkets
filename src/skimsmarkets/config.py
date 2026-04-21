from __future__ import annotations

import os
from dataclasses import dataclass

# Seed list of sports-series ticker prefixes. The Kalshi client ALSO dynamically discovers
# series via /series?category=Sports, so this list is a hint, not a hard filter. It's
# deliberately broad and includes tickers that may return zero live events off-season.
SPORTS_SERIES_SEED: tuple[str, ...] = (
    "KXNBAGAME",
    "KXMLBGAME",
    "KXNHLGAME",
    "KXNFLGAME",
    "KXMLSGAME",
    "KXUFCFIGHT",
    "KXATPMATCH",
    "KXWTAMATCH",
)

# Markets whose expected_expiration_time is more than this many hours out are skipped.
# (expected_expiration_time sits ~shortly after game end, so 24h catches "today's slate".)
MAX_HOURS_UNTIL_EXPIRATION = 24

# Concurrency caps. See plan for rationale.
EVENT_SEM = 4
SPECIALIST_SEM = 16
DIRECTOR_SEM = 2


@dataclass(frozen=True)
class Config:
    xai_api_key: str
    anthropic_api_key: str

    @classmethod
    def from_env(cls) -> "Config":
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
                "Export them or use `uv run --env-file .env`."
            )
        return cls(xai_api_key=xai, anthropic_api_key=anth)
