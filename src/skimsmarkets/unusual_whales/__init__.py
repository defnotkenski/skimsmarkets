"""Unusual Whales prediction-market integration.

`UnusualWhalesClient` fetches per-asset flow detail (tag scores, MCI,
liquidity, recent smart/contrarian/insider activity) and
`GammaTokenResolver` bridges our Polymarket slug to the ERC-1155
`asset_id` UW is keyed by. Data attaches to each `PolymarketEvent` as
`uw_context` and is rendered only into the `market_context` specialist's
prompt.
"""

from skimsmarkets.unusual_whales.client import UnusualWhalesClient
from skimsmarkets.unusual_whales.gamma import (
    GammaMarketSnapshot,
    GammaTokenResolver,
    fetch_gamma_event,
    list_gamma_events,
)
from skimsmarkets.unusual_whales.models import (
    UnusualWhalesContext,
    UWInsider,
    UWLiquidity,
    UWMci,
    UWTagScores,
    UWTrade,
)
from skimsmarkets.unusual_whales.rendering import render_uw_block

__all__ = [
    "GammaMarketSnapshot",
    "GammaTokenResolver",
    "UWInsider",
    "UWLiquidity",
    "UWMci",
    "UWTagScores",
    "UWTrade",
    "UnusualWhalesClient",
    "UnusualWhalesContext",
    "fetch_gamma_event",
    "list_gamma_events",
    "render_uw_block",
]
