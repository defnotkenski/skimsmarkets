"""Kalshi-sourced enrichment stages — replaces the Polymarket CLOB
book + price-history fetchers in `pipeline.py`.

Two stages, each adapter to a Kalshi endpoint:

- `enrich_kalshi_book(events, client, sem)` — full bid/ask depth
  per market via `/markets/{ticker}/orderbook`. Replaces
  `pipeline.enrich_clob_book`. NO-side semantics differ: Kalshi
  exposes both sides as independent books, so each
  `PolymarketMarket` (whether `is_no_side=True` or False) reads its
  OWN ticker's book — never inverts the favorite's depth.

- `enrich_kalshi_history(events, client, sem)` — sparkline + 1h /
  4h / 24h price moves per market via
  `/series/{s}/markets/{t}/candlesticks?period_interval=60`.
  Replaces `pipeline.enrich_price_history`. The candle bucket size
  is `cfg.KALSHI_SLATE_HISTORY_INTERVAL_MINUTES` (currently 60 —
  the only legal value besides 1).

The output field names stay `clob_*` (e.g.
`clob_price_path_sparkline`, `yes_bid_size_top`) so the JSONL
schema is preserved and the director's rendering at
`agents/director.py:121-126` keeps working unchanged. The SOURCE
shifts from Polymarket CLOB to Kalshi; the SCHEMA does not.

Reuses `summarize_history` from `clob/__init__.py` for the
sparkline + windowed-scalar derivation since it operates on
`[{"t": ..., "p": ...}]` shape — the candlestick adapter just
maps Kalshi buckets into that shape before summarising.
"""

from __future__ import annotations

import asyncio
import logging
import time

from skimsmarkets import config as cfg
from skimsmarkets.clob import summarize_history
from skimsmarkets.kalshi.client import KalshiClient
from skimsmarkets.kalshi.models import KalshiCandle
from skimsmarkets.polymarket.models import PolymarketEvent

log = logging.getLogger(__name__)


# --- book enrichment -------------------------------------------------------


async def enrich_kalshi_book(
    events: list[PolymarketEvent],
    client: KalshiClient,
    sem: asyncio.Semaphore,
) -> None:
    """Attach full-book depth + size + dollars to each market.

    Iteration is per-market-slug (= Kalshi ticker), deduplicated
    across events. Each market — including the `is_no_side=True`
    underdog — reads its OWN ticker's `/orderbook`, NOT the
    favorite's inverted book. This is the load-bearing semantic
    difference vs `enrich_clob_book`: Kalshi exposes both sides as
    independent native YES books.

    Output field set mirrors `clob/__init__.py:summarize_book` so
    the director's rendering at `agents/director.py:76-102` reads
    Kalshi-sourced numbers without code changes.
    """
    refs: dict[str, list[tuple[PolymarketEvent, int]]] = {}
    for ev in events:
        for i, m in enumerate(ev.markets):
            if m.slug:
                refs.setdefault(m.slug, []).append((ev, i))
    if not refs:
        return

    async def _one(ticker: str, mks: list[tuple[PolymarketEvent, int]]) -> None:
        async with sem:
            ob = await client.fetch_orderbook(ticker=ticker)
            if ob is None:
                return
            bid_levels = ob.yes_levels
            ask_levels = ob.no_levels
            # `yes_levels` from Kalshi IS the YES-side book (bids to
            # buy YES + asks to sell YES). For binary mutually-
            # exclusive markets like tennis, the YES-side asks live
            # in the orderbook's `no_dollars` array (someone else
            # buying NO is offering YES at the inverse price).
            # Empirically the wire shape splits into "bids on YES"
            # (yes_dollars) and "bids on NO" (no_dollars) — the YES
            # asks are the NO bids' inversion, which Kalshi has
            # already done in the `no_dollars` array.
            #
            # For our purposes (top-of-book + total $ resting): the
            # YES side's "bid" is anyone willing to BUY YES at price
            # P; the YES side's "ask" is anyone willing to SELL YES
            # at price P. On Kalshi's CLOB, selling YES @ P is
            # equivalent to buying NO @ (1-P). So we use yes_dollars
            # for our bid side and no_dollars (priced as buy-NO)
            # converted via 1-P for our ask side.
            bid_top, bid_top_size, bid_book_dollars, bid_depth = _summarize_side(
                bid_levels
            )
            # The "ask" book in YES-equivalent prices: each NO bid at
            # P reads as a YES ask at (1-P), with the same size.
            ask_inverted = [(1.0 - px, sz) for px, sz in ask_levels]
            # Re-sort ascending by inverted price so top-of-ask is the
            # cheapest price to BUY YES.
            ask_inverted.sort(key=lambda pair: pair[0])
            ask_top, ask_top_size, ask_book_dollars, ask_depth = _summarize_side(
                ask_inverted
            )
            for ev, idx in mks:
                m = ev.markets[idx]
                ev.markets[idx] = m.model_copy(
                    update={
                        "yes_bid_size_top": bid_top_size,
                        "yes_ask_size_top": ask_top_size,
                        "yes_bid_book_dollars": bid_book_dollars,
                        "yes_ask_book_dollars": ask_book_dollars,
                        "yes_bid_depth": bid_depth,
                        "yes_ask_depth": ask_depth,
                    }
                )

    await asyncio.gather(*(_one(s, refs_) for s, refs_ in refs.items()))
    enriched = sum(
        1
        for ev in events
        for m in ev.markets
        if m.yes_bid_book_dollars is not None or m.yes_ask_book_dollars is not None
    )
    total = sum(len(ev.markets) for ev in events)
    log.info("attached kalshi book to %d/%d markets", enriched, total)


def _summarize_side(
    levels: list[tuple[float, float]],
) -> tuple[float | None, float | None, float | None, int | None]:
    """`(top_px, top_size, total_dollars, depth)` for one book side.

    `levels` is best-first (highest bid or lowest ask at index 0).
    Empty side returns four Nones — a one-sided book is a real
    market state and should still produce a usable per-market row.
    """
    if not levels:
        return None, None, None, None
    top_px, top_size = levels[0]
    book_dollars = sum(px * sz for px, sz in levels)
    return top_px, top_size, book_dollars, len(levels)


# --- price-history enrichment ----------------------------------------------


async def enrich_kalshi_history(
    events: list[PolymarketEvent],
    client: KalshiClient,
    sem: asyncio.Semaphore,
) -> None:
    """Attach `clob_price_*` sparkline + recency moves to each market.

    Per ticker: fetch 24h of `period_interval=
    cfg.KALSHI_SLATE_HISTORY_INTERVAL_MINUTES` candlesticks, project
    each bucket's `yes_bid.close_dollars` into the
    `[{"t": end_ts, "p": close}, ...]` shape `summarize_history`
    expects, then attach the sparkline + 30m/1h/4h/24h windowed
    scalars. Field names stay `clob_*` so JSONL schema and director
    rendering are unchanged.

    On Kalshi both sides have their own price history, so the
    `is_no_side` clone reads its OWN ticker's candlesticks — no
    sign-flip on the scalars (mirrors the book-enrichment posture
    above).
    """
    refs: dict[str, list[tuple[PolymarketEvent, int, str]]] = {}
    for ev in events:
        for i, m in enumerate(ev.markets):
            if not m.slug:
                continue
            # series_ticker is encoded in the Kalshi ticker prefix
            # (`KXATPMATCH-...` → `KXATPMATCH`). Splitting on `-`
            # is safe because Kalshi tickers don't carry dashes in
            # the series part.
            series_ticker = m.slug.split("-", 1)[0]
            refs.setdefault(m.slug, []).append((ev, i, series_ticker))
    if not refs:
        return

    now_ts = int(time.time())
    start_ts = now_ts - 24 * 60 * 60

    async def _one(
        ticker: str, mks: list[tuple[PolymarketEvent, int, str]]
    ) -> None:
        async with sem:
            series_ticker = mks[0][2]
            sticks = await client.fetch_candlesticks(
                series_ticker=series_ticker,
                ticker=ticker,
                period_interval=cfg.KALSHI_SLATE_HISTORY_INTERVAL_MINUTES,
                start_ts=start_ts,
                end_ts=now_ts,
            )
            history = _candles_to_history(sticks)
            if not history:
                return
            summary = summarize_history(history)
            if summary is None:
                return
            for ev, idx, _ in mks:
                m = ev.markets[idx]
                ev.markets[idx] = m.model_copy(
                    update={
                        "clob_price_change_30m": summary.change_30m,
                        "clob_price_change_1h": summary.change_1h,
                        "clob_price_change_4h": summary.change_4h,
                        "clob_price_change_24h": summary.change_24h,
                        "clob_price_path_sparkline": summary.sparkline,
                        "clob_price_history": summary.raw_points,
                    }
                )

    await asyncio.gather(*(_one(s, refs_) for s, refs_ in refs.items()))
    enriched = sum(
        1
        for ev in events
        for m in ev.markets
        if m.clob_price_path_sparkline is not None
    )
    total = sum(len(ev.markets) for ev in events)
    log.info("attached kalshi history to %d/%d markets", enriched, total)


def _candles_to_history(
    sticks: list[KalshiCandle] | None,
) -> list[dict[str, float | int]] | None:
    """Project `KalshiCandle` buckets into the `[{"t":..., "p":...}]`
    shape `clob.summarize_history` expects.

    Each candle's `yes_bid.close_dollars` becomes the price; bucket
    `end_period_ts` is the timestamp. Buckets without a parseable
    bid close are dropped (early-life markets often ship empty
    `yes_bid` / `yes_ask` blocks until the first quote lands).
    """
    if not sticks:
        return None
    out: list[dict[str, float | int]] = []
    for c in sticks:
        if c.end_period_ts is None:
            continue
        bid = c.yes_bid.get("close_dollars")
        try:
            p = float(bid)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        out.append({"t": c.end_period_ts, "p": p})
    return out or None
