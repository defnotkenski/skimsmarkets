"""Kalshi execution venue — async client + Pydantic models + tennis matcher.

Polymarket is the data source; Kalshi is the execution venue. Public
`/events` / `/markets` reads need no auth. `POST /portfolio/orders` is
RSA-PSS-signed against the API key configured in `Config`.

The package's `__init__` deliberately re-exports only the lightweight
pieces (models + matcher). `KalshiClient` is in `kalshi.client` —
import it explicitly when you need it; eager re-export here would
pull `cryptography` into every code path that touches the matcher.
"""

from skimsmarkets.kalshi.matcher import (
    MatchOutcome,
    extract_match_players,
    find_kalshi_match,
    last_token,
)
from skimsmarkets.kalshi.models import (
    KalshiEvent,
    KalshiMarket,
    OrderRequest,
    OrderResponse,
)

__all__ = [
    "KalshiEvent",
    "KalshiMarket",
    "MatchOutcome",
    "OrderRequest",
    "OrderResponse",
    "extract_match_players",
    "find_kalshi_match",
    "last_token",
]
