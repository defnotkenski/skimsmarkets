"""Thin async wrapper over `polymarket_us.AsyncPolymarketUS`.

Why a wrapper when CLAUDE.md permits the SDK directly:
- One place to normalize SDK response shapes before they bleed into our Pydantic
  models (the SDK's public shapes are still in flux per the fresh docs).
- Consistent `async with` semantics for the pipeline's context-manager pattern.
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


def _coerce_float_safe(v: Any) -> float | None:
    """Best-effort float coercion that returns None on anything unparseable.

    polymarket-us returns numeric stats as JSON strings (e.g. `"114353.000"`),
    and occasionally as `None` or absent. Centralize the try/except so callers
    stay terse.
    """
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_book_side(levels: Any) -> list[tuple[float, float]]:
    """Turn a list of `{px: {value, currency}, qty: "..."}` levels into
    `(price, qty)` float tuples, dropping any level with unparseable parts.

    Order is preserved — the SDK returns bids best-first (highest price)
    and offers best-first (lowest price), and downstream consumers
    (`bid_levels[0]` → top of book, `sum(qty × px)` → full-book dollars)
    rely on that ordering.
    """
    if not isinstance(levels, list):
        return []
    out: list[tuple[float, float]] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        px = _coerce_float_safe(_extract_value(level, "px"))
        qty = _coerce_float_safe(level.get("qty"))
        if px is None or qty is None:
            continue
        out.append((px, qty))
    return out


def _reference_price(
    bid: Any, ask: Any, last: Any
) -> float | None:
    """Pick a single price to multiply share counts by, in preference order:
    bid/ask midpoint → last trade → whichever of bid/ask is present.

    Used only for deriving dollar figures from `sharesTraded` / `openInterest`;
    callers that need bid and ask individually should still read those fields.
    """
    b = _coerce_float_safe(bid)
    a = _coerce_float_safe(ask)
    if b is not None and a is not None:
        return (b + a) / 2.0
    lt = _coerce_float_safe(last)
    if lt is not None:
        return lt
    return b if b is not None else a


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
                    e,
                    list(item.keys()) if isinstance(item, dict) else type(item),
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

    async def get_book(self, market_slug: str) -> PolymarketMarket | None:
        """Fetch the order book for a market and return a PolymarketMarket.

        Why `markets.book` instead of `markets.bbo`: the book response is a
        strict superset of BBO — same one HTTP call, but it also carries the
        full bids/offers ladders, intraday `stats` (open/high/low/close +
        `notionalTraded`, the *true* USD volume), an authoritative `state`
        flag (OPEN / SUSPENDED / HALTED / …) and `lastTradeQty`. See
        `/tmp/skimsmarkets_probes/poly_us_markets_book.json` for the shape.

        Returns None if the call fails or the response shape isn't recognizable
        — callers should treat None as "no live book" and fall back to whatever
        snapshot prices the events.list response already carried.

        Dollar volume: when `notionalTraded` is present we use it as
        `volume_dollars` directly (Polymarket's own Σ(price_at_fill × qty),
        which captures intraday price drift). When absent, we fall back to
        the legacy `sharesTraded × reference_price` derivation.

        `open_interest_dollars` continues to mean "outstanding shares × ref
        price" — a market-size measure, NOT order-book depth. Real book
        depth is `yes_bid_book_dollars` / `yes_ask_book_dollars`.
        """
        try:
            raw = await self._sdk.markets.book(market_slug)
        except Exception as e:  # noqa: BLE001
            log.warning("polymarket book(%s) failed: %s", market_slug, e)
            return None
        if raw is None:
            return None
        md = raw.get("marketData") if isinstance(raw, dict) else None
        if md is None:
            # Some SDK shapes flatten it; try the raw response.
            md = raw if isinstance(raw, dict) else {}
        if not isinstance(md, dict):
            return None
        stats = md.get("stats") if isinstance(md.get("stats"), dict) else {}

        # Best bid/ask come from the top of the bids/offers ladders. The
        # ladders are sorted best-first (highest bid first, lowest ask
        # first), and each level is `{px: {value, currency}, qty: "..."}`.
        bids_raw = md.get("bids") if isinstance(md.get("bids"), list) else []
        offers_raw = md.get("offers") if isinstance(md.get("offers"), list) else []
        bid_levels = _parse_book_side(bids_raw)
        ask_levels = _parse_book_side(offers_raw)
        bid = bid_levels[0][0] if bid_levels else None
        ask = ask_levels[0][0] if ask_levels else None
        bid_size_top = bid_levels[0][1] if bid_levels else None
        ask_size_top = ask_levels[0][1] if ask_levels else None
        # Total $ resting on each side across all visible levels.
        bid_book_dollars = (
            sum(px * qty for px, qty in bid_levels) if bid_levels else None
        )
        ask_book_dollars = (
            sum(px * qty for px, qty in ask_levels) if ask_levels else None
        )

        last = _extract_value(stats, "lastTradePx")
        # Intraday range — all four are value-wrapped Amounts in `stats`.
        high_px = _extract_value(stats, "highPx")
        low_px = _extract_value(stats, "lowPx")
        open_px = _extract_value(stats, "openPx")
        close_px = _extract_value(stats, "closePx")
        last_trade_qty = _coerce_float_safe(stats.get("lastTradeQty"))
        # `notionalTraded` is value-wrapped; `sharesTraded` and `openInterest`
        # are bare strings (legacy from the BBO shape).
        notional_traded = _coerce_float_safe(_extract_value(stats, "notionalTraded"))
        shares_traded = _coerce_float_safe(stats.get("sharesTraded"))
        open_interest = _coerce_float_safe(stats.get("openInterest"))
        state = md.get("state") if isinstance(md.get("state"), str) else None

        ref_price = _reference_price(bid, ask, last)
        # Prefer the true notional Polymarket computed; only derive from
        # shares × ref when notional isn't published. The derived number
        # is wrong whenever price moved during the session — see
        # PolymarketMarket field comments.
        volume_dollars = notional_traded
        if volume_dollars is None and shares_traded is not None and ref_price is not None:
            volume_dollars = shares_traded * ref_price
        oi_dollars = (
            open_interest * ref_price
            if open_interest is not None and ref_price is not None
            else None
        )

        return PolymarketMarket(
            slug=market_slug,
            yes_bid_dollars=bid,
            yes_ask_dollars=ask,
            yes_bid_depth=len(bid_levels) if bid_levels else None,
            yes_ask_depth=len(ask_levels) if ask_levels else None,
            yes_bid_size_top=bid_size_top,
            yes_ask_size_top=ask_size_top,
            yes_bid_book_dollars=bid_book_dollars,
            yes_ask_book_dollars=ask_book_dollars,
            market_state=state,
            last_trade_price_dollars=last,
            last_trade_qty=last_trade_qty,
            notional_traded_dollars=notional_traded,
            high_px_dollars=high_px,
            low_px_dollars=low_px,
            open_px_dollars=open_px,
            close_px_dollars=close_px,
            volume_dollars=volume_dollars,
            open_interest_dollars=oi_dollars,
            liquidity_dollars=oi_dollars,
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
