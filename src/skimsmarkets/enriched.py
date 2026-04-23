"""Pipeline-internal data carriers that span vendor packages.

Lives here rather than in `kalshi/` or `polymarket/` because both sides have a
say: the enriched record bundles a Kalshi event with its optional Polymarket
overlay and is threaded through the specialist → director → sizing stages.

Dataclass (not Pydantic) per CLAUDE.md convention — Pydantic is reserved for
external payloads and LLM-structured output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skimsmarkets.kalshi.models import KalshiEvent
from skimsmarkets.polymarket.matching import SideMatch
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket


@dataclass
class EnrichedEvent:
    """A Kalshi event plus any matched Polymarket overlay.

    - `polymarket` and `side_map` come from the matcher (`match_event`).
    - `polymarket_price_by_kalshi_side` is populated after the per-side BBO
      fan-out: keyed by the Kalshi `yes_sub_title`, values are PolymarketMarkets
      with `yes_bid_dollars` / `yes_ask_dollars` filled in. A side present in
      `side_map` but absent from this dict means the BBO call failed for it.
    """

    kalshi: KalshiEvent
    polymarket: PolymarketEvent | None = None
    side_map: dict[str, SideMatch] = field(default_factory=dict)
    polymarket_price_by_kalshi_side: dict[str, PolymarketMarket] = field(default_factory=dict)

    @property
    def has_polymarket(self) -> bool:
        return self.polymarket is not None and bool(self.polymarket_price_by_kalshi_side)

    def poly_market_for(self, kalshi_yes_sub_title: str) -> PolymarketMarket | None:
        return self.polymarket_price_by_kalshi_side.get(kalshi_yes_sub_title)
