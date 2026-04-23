"""Thin async wrapper over `polymarket_us.AsyncPolymarketUS`.

Why a wrapper when CLAUDE.md permits the SDK directly:
- One place to normalize SDK response shapes before they bleed into our Pydantic
  models (the SDK's public shapes are still in flux per the fresh docs).
- Consistent `async with` semantics that mirror KalshiClient, so the pipeline
  can open both clients under the same context-manager pattern.
- Central seam for test doubles / fault injection.

All calls here are read-only (public market data). No credentials are needed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from types import TracebackType
from typing import Any, Self

from polymarket_us import AsyncPolymarketUS  # type: ignore[import-not-found]

from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket

log = logging.getLogger(__name__)


def _extract_value(obj: Any, key: str) -> Any:
    """Fetch a nested `{key: {"value": ...}}` or flat `{key: ...}` field.

    The BBO response is documented as `{"bestBid": {"value": "0.55"}}`, but the
    SDK may normalize it to `{"bestBid": 0.55}`. Handle both.
    """
    if obj is None:
        return None
    raw = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw.get("value")
    return raw


class PolymarketClient:
    """Async context-managed wrapper. `async with PolymarketClient() as c: ...`."""

    def __init__(self, *, timeout: float = 30.0) -> None:
        # Lazy: instantiate now, enter it on __aenter__ so cleanup is symmetric.
        self._sdk = AsyncPolymarketUS(timeout=timeout)
        self._entered = False

    async def __aenter__(self) -> Self:
        # The SDK is documented as `async with AsyncPolymarketUS() as client`, so
        # we delegate enter/exit to it. If a future SDK version happens to be
        # non-context-managed, the hasattr check keeps us compatible.
        if hasattr(self._sdk, "__aenter__"):
            await self._sdk.__aenter__()
            self._entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._entered and hasattr(self._sdk, "__aexit__"):
            await self._sdk.__aexit__(exc_type, exc, tb)
        elif hasattr(self._sdk, "close"):
            # Fallback for SDKs that expose close() instead of context-manager.
            result = self._sdk.close()
            if hasattr(result, "__await__"):
                await result

    async def list_sports_events(
        self,
        *,
        series_prefix: str | None = None,
        limit: int = 500,
        start_time_min: datetime | None = None,
        start_time_max: datetime | None = None,
    ) -> list[PolymarketEvent]:
        """Fetch open sports events, optionally filtered to a series-slug prefix.

        The SDK's `events.list` accepts `categories`, `active`, `closed`,
        `archived`, `limit`, `offset`, `seriesId`, `startTimeMin`, `startTimeMax` —
        but NOT a league filter. We pass `active=True, closed=False` to get
        live-tradable events and then filter client-side by `seriesSlug` prefix
        (e.g. 'nba' matches 'nba-2025'). Response envelope observed as
        `{"events": [...]}`; we fall back to `{"data": [...]}` defensively.

        `start_time_min`/`start_time_max` bound the Polymarket event pool by
        game-start time. Passing them lets the caller drop season-winner futures
        and far-out playoff games from the matcher's candidate pool, which both
        cuts network work and removes ambiguous-futures collision cases where
        the matcher could pair a daily game to a season-winner slug.
        """
        params: dict[str, Any] = {
            "categories": ["sports"],
            "active": True,
            "closed": False,
            "limit": limit,
        }
        if start_time_min is not None:
            params["startTimeMin"] = start_time_min.isoformat().replace("+00:00", "Z")
        if start_time_max is not None:
            params["startTimeMax"] = start_time_max.isoformat().replace("+00:00", "Z")
        raw = await self._sdk.events.list(params)
        items = self._unwrap_list(raw, "events")
        events: list[PolymarketEvent] = []
        for item in items:
            try:
                ev = PolymarketEvent.model_validate(item)
            except Exception as e:  # noqa: BLE001 — skip one bad event, keep going
                log.debug(
                    "failed to parse polymarket event: %s (raw keys=%s)",
                    e, list(item.keys()) if isinstance(item, dict) else type(item),
                )
                continue
            if series_prefix:
                slug = ev.series_slug or ""
                # Match 'nba' against 'nba-2025' but also guard against 'nba'
                # matching 'nba-league-stuff-2025' incorrectly.
                if not (slug == series_prefix or slug.startswith(series_prefix + "-")):
                    continue
            events.append(ev)
        return events

    async def get_bbo(self, market_slug: str) -> PolymarketMarket | None:
        """Fetch best bid/offer for a market and return a PolymarketMarket.

        Returns None if the call fails or the response shape isn't recognizable
        — callers should treat None as "no Polymarket price available" and fall
        back to the Kalshi-only path.
        """
        try:
            raw = await self._sdk.markets.bbo(market_slug)
        except Exception as e:  # noqa: BLE001
            log.warning("polymarket bbo(%s) failed: %s", market_slug, e)
            return None
        if raw is None:
            return None
        md = raw.get("marketData") if isinstance(raw, dict) else None
        if md is None:
            # Some SDK shapes flatten it; try the raw response.
            md = raw if isinstance(raw, dict) else {}
        bid = _extract_value(md, "bestBid")
        ask = _extract_value(md, "bestAsk")
        last = _extract_value(md, "lastTradePrice")
        volume = md.get("volume") if isinstance(md, dict) else None
        liquidity = md.get("liquidity") if isinstance(md, dict) else None
        return PolymarketMarket(
            slug=market_slug,
            yes_bid_dollars=bid,
            yes_ask_dollars=ask,
            last_trade_price_dollars=last,
            volume_dollars=volume,
            liquidity_dollars=liquidity,
        )

    @staticmethod
    def _unwrap_list(raw: Any, primary_key: str) -> list[Any]:
        """Return a list of items from either `{key: [...]}`, `{"data": [...]}`, or a raw list."""
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if primary_key in raw and isinstance(raw[primary_key], list):
                return raw[primary_key]
            if "data" in raw and isinstance(raw["data"], list):
                return raw["data"]
        return []
