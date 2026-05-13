"""Polymarket CLOB enrichment stages — book depth + price history.

Two stages, each attaching freshly-fetched detail to every
`PolymarketMarket` in the slate:

- `enrich_clob_book` — full bid/ask depth + size + book-$ totals via
  `clob.polymarket.com/book?token_id=...`. Replaces the per-market
  book snapshot that gamma's listing payload doesn't carry.

- `enrich_price_history` — sparkline + 30m/1h/4h/24h windowed moves
  via `clob.polymarket.com/prices-history?market=...&interval=1d`.
  Output fields stay `clob_*` so JSONL schema and downstream
  rendering are unchanged.

Both stages share `GammaTokenResolver` (cached per-run) to resolve
`market.slug` → `(yes_token_id, no_token_id)` via gamma `/markets?slug=`.
The resolver coalesces concurrent lookups per slug so we don't pay the
same gamma call twice across these two stages and the UW bridge.

NO-side semantics. Polymarket head-to-heads ship as a single binary
market with `from_gamma` synthesizing a NO clone via `inverted_no_side`.
The CLOB endpoint exposes per-token data, and the YES token's bid book
is the implied NO ask book (and vice versa). Both stages handle the
inversion in-place when `market.is_no_side=True`:
  - Book: swap bid/ask sides (no value flip — `yes_bid book` reads as
    `no_ask book`).
  - History: negate the windowed scalars (sparkline rendering on the
    NO clone is the caller's problem — `clob/__init__.py:invert_sparkline`
    is the helper, but we leave it to the renderer because the raw
    history is shared across YES + NO copies).
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from skimsmarkets.clob import (
    fetch_book,
    fetch_price_history,
    summarize_book,
    summarize_history,
)
from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.unusual_whales import GammaTokenResolver

log = logging.getLogger(__name__)


async def enrich_clob_book(
    events: list[PolymarketEvent],
    resolver: GammaTokenResolver,
    http: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> None:
    """Attach CLOB order-book size + depth + book-$ to each market.

    Per unique market slug (deduped across events): resolve YES token_id
    via gamma, fetch the full CLOB book, summarize, and attach to every
    `(event, market_index)` reference sharing that slug.

    NO-side clones receive the same fetch result with bid/ask sides
    swapped — YES bid book is the implied NO ask book and vice versa.
    No value flip, just label swap.
    """
    by_market_slug: dict[str, list[tuple[PolymarketEvent, int]]] = {}
    for ev in events:
        for i, m in enumerate(ev.markets):
            if m.slug:
                by_market_slug.setdefault(m.slug, []).append((ev, i))
    if not by_market_slug:
        return

    async def _one(market_slug: str, refs: list[tuple[PolymarketEvent, int]]) -> None:
        async with sem:
            snap = await resolver.resolve_snapshot(market_slug)
            if snap is None or snap.clob_token_ids is None:
                return
            yes_token_id, _no_token_id = snap.clob_token_ids
            book = await fetch_book(http, yes_token_id)
            summary = summarize_book(book)
            if summary is None:
                return
            for evt, idx in refs:
                mkt = evt.markets[idx]
                if mkt.is_no_side:
                    # NO clone: bid/ask sides swap (YES bid book = implied
                    # NO ask book) but values themselves don't flip.
                    evt.markets[idx] = mkt.model_copy(
                        update={
                            "yes_bid_size_top": summary.ask_top_size,
                            "yes_ask_size_top": summary.bid_top_size,
                            "yes_bid_book_dollars": summary.ask_book_dollars,
                            "yes_ask_book_dollars": summary.bid_book_dollars,
                            "yes_bid_depth": summary.ask_depth,
                            "yes_ask_depth": summary.bid_depth,
                        }
                    )
                else:
                    evt.markets[idx] = mkt.model_copy(
                        update={
                            "yes_bid_size_top": summary.bid_top_size,
                            "yes_ask_size_top": summary.ask_top_size,
                            "yes_bid_book_dollars": summary.bid_book_dollars,
                            "yes_ask_book_dollars": summary.ask_book_dollars,
                            "yes_bid_depth": summary.bid_depth,
                            "yes_ask_depth": summary.ask_depth,
                        }
                    )

    await asyncio.gather(*(_one(s, refs) for s, refs in by_market_slug.items()))
    enriched = sum(
        1
        for ev in events
        for m in ev.markets
        if m.yes_bid_book_dollars is not None or m.yes_ask_book_dollars is not None
    )
    total = sum(len(ev.markets) for ev in events)
    log.info("attached clob book to %d/%d markets", enriched, total)


async def enrich_price_history(
    events: list[PolymarketEvent],
    resolver: GammaTokenResolver,
    http: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> None:
    """Attach `clob_price_*` sparkline + recency scalars to each market.

    Per unique market slug: resolve YES token_id via gamma, fetch ~24h
    of mid prices from `clob.polymarket.com/prices-history`, summarize,
    and attach.

    Iteration is per **market slug**, not per event slug. Soccer-style
    3-way events split each outcome into its own market slug, and gamma's
    `/markets?slug=` only resolves on the per-outcome slug. Tennis/UFC
    binary head-to-heads share one slug across YES + inverted-NO clones;
    per-market iteration picks up both clones, and the NO record carries
    the negated windowed scalars (positive YES move = negative NO move).
    """
    by_market_slug: dict[str, list[tuple[PolymarketEvent, int]]] = {}
    for ev in events:
        for i, m in enumerate(ev.markets):
            if m.slug:
                by_market_slug.setdefault(m.slug, []).append((ev, i))
    if not by_market_slug:
        return

    async def _one(market_slug: str, refs: list[tuple[PolymarketEvent, int]]) -> None:
        async with sem:
            snap = await resolver.resolve_snapshot(market_slug)
            if snap is None or snap.clob_token_ids is None:
                return
            yes_token_id, _no_token_id = snap.clob_token_ids
            history = await fetch_price_history(http, yes_token_id)
            if not history:
                return
            summary = summarize_history(history)
            if summary is None:
                return
            for evt, idx in refs:
                mkt = evt.markets[idx]
                if mkt.is_no_side:
                    evt.markets[idx] = mkt.model_copy(
                        update={
                            "clob_price_change_30m": (
                                -summary.change_30m
                                if summary.change_30m is not None
                                else None
                            ),
                            "clob_price_change_1h": (
                                -summary.change_1h
                                if summary.change_1h is not None
                                else None
                            ),
                            "clob_price_change_4h": (
                                -summary.change_4h
                                if summary.change_4h is not None
                                else None
                            ),
                            "clob_price_change_24h": (
                                -summary.change_24h
                                if summary.change_24h is not None
                                else None
                            ),
                            # Sparkline stays as the YES-side rendering;
                            # the renderer flips it to the NO side via
                            # `clob.invert_sparkline` when needed.
                            "clob_price_path_sparkline": summary.sparkline,
                            "clob_price_history": summary.raw_points,
                        }
                    )
                else:
                    evt.markets[idx] = mkt.model_copy(
                        update={
                            "clob_price_change_30m": summary.change_30m,
                            "clob_price_change_1h": summary.change_1h,
                            "clob_price_change_4h": summary.change_4h,
                            "clob_price_change_24h": summary.change_24h,
                            "clob_price_path_sparkline": summary.sparkline,
                            "clob_price_history": summary.raw_points,
                        }
                    )

    await asyncio.gather(*(_one(s, refs) for s, refs in by_market_slug.items()))
    enriched = sum(
        1
        for ev in events
        for m in ev.markets
        if m.clob_price_path_sparkline is not None
    )
    total = sum(len(ev.markets) for ev in events)
    log.info("attached clob history to %d/%d markets", enriched, total)
