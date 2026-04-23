from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

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
SPECIALIST_SEM = 16
DIRECTOR_SEM = 2
# Per-event BBO fan-out against Polymarket. Each matched Kalshi event can trigger
# N parallel BBO lookups (one per matched side); this caps aggregate concurrency.
POLYMARKET_FETCH_SEM = 8

# Kalshi series prefix → Polymarket league slug. Unmapped series skip Polymarket
# enrichment (logged at info level, not an error). Slugs are best-guesses based
# on the Polymarket US sports API; verify and correct on first live run.
KALSHI_SERIES_TO_POLYMARKET_LEAGUE: dict[str, str] = {
    "KXNBAGAME": "nba",
    "KXMLBGAME": "mlb",
    "KXNHLGAME": "nhl",
    "KXNFLGAME": "nfl",
    "KXMLSGAME": "mls",
    "KXUFCFIGHT": "ufc",
    "KXATPMATCH": "atp",
    "KXWTAMATCH": "wta",
    "KXLALIGAGAME": "lal",
}

# Emergency kill-switch for the Polymarket overlay. "0" / "false" / "False" disable
# it for the run; anything else (including unset) leaves it on. The CLI also exposes
# --no-polymarket which forces this to False for a single invocation.
_POLY_DISABLED = {"0", "false", "False", "no", "NO"}


def polymarket_enabled() -> bool:
    return os.environ.get("POLYMARKET_ENABLED", "1").strip() not in _POLY_DISABLED


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
