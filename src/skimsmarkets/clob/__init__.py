"""Polymarket CLOB price-history fetcher and summarization helpers.

The CLOB endpoint `https://clob.polymarket.com/prices-history?market=<token>`
is public and unauthed. It returns `{history: [{t: epoch_seconds, p: mid}]}`
— a stream of mid-price points covering whatever window/fidelity the caller
requested. With `interval=1d&fidelity=5` it returns ~288 points (one per
~5min) over the past 24h; with `interval=max&fidelity=1` it returns the full
lifetime at one point per minute (thousands of points).

Two callers ride this module:
- The live pipeline (gated by `CLOB_HISTORY_ENABLED`) calls it per ranked
  slug for short-window momentum signal feeding the director.
- The backtest module (`backtest/clob_history.py`) wraps the same fetcher
  with on-disk caching and uses the full lifetime for retrospective
  analysis.

This module owns the bare HTTP loop and the LLM-facing summarization
(sparkline + time-windowed scalars). Concurrency / caching / retry are
caller concerns — keep this module side-effect-free so both callers can
compose it differently.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

_CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
_CLOB_BOOK_URL = "https://clob.polymarket.com/book"

# Time-window targets for the LLM-facing scalars, in seconds.
_WINDOW_30M = 30 * 60
_WINDOW_1H = 60 * 60
_WINDOW_4H = 4 * 60 * 60
_WINDOW_24H = 24 * 60 * 60

# When picking the point nearest to "t_now − window_seconds", reject any
# match further than this tolerance from the requested target. Five minutes
# is loose enough for `fidelity=5` (one point per 5 min) but tight enough to
# avoid pretending we have a 30-minute reading when the nearest sample is
# actually 25 minutes off-target.
_NEAREST_TOLERANCE_S = 5 * 60


@dataclass(frozen=True, slots=True)
class PriceHistorySummary:
    """LLM-ready compression of a CLOB price-history series.

    `sparkline` is an arrow-joined string of N evenly-spaced samples
    (`"0.520→0.554→0.601→0.612→0.620"`) — 5 points fits one LLM line and
    captures direction plus roughly one inflection. `change_*` scalars
    are signed (positive = price moved up over that window) and `None`
    when the history doesn't cover the requested window. `raw_points`
    keeps the full input around for backtest / debug consumers; live
    rendering only reads `sparkline` and the scalars.
    """

    sparkline: str
    raw_points: list[tuple[int, float]] = field(default_factory=list)
    change_30m: float | None = None
    change_1h: float | None = None
    change_4h: float | None = None
    change_24h: float | None = None


async def fetch_price_history(
    client: httpx.AsyncClient,
    token_id: str,
    *,
    interval: str = "1d",
    fidelity: int = 5,
) -> list[dict[str, Any]] | None:
    """GET clob.polymarket.com/prices-history. Returns the raw history list
    (each item `{"t": <epoch_s>, "p": <mid>}`) or None on any failure.

    Public, unauthed. No internal semaphore — the caller owns concurrency
    (live pipeline uses `CLOB_FETCH_SEM`; backtest uses its own).

    Defaults are tuned for live use: `interval=1d` covers the past 24h,
    `fidelity=5` gives one sample per ~5min (~288 points). Backtest
    overrides with `interval=max, fidelity=1` for the full lifetime at
    minute resolution.
    """
    params: dict[str, str] = {
        "market": token_id,
        "interval": interval,
        "fidelity": str(fidelity),
    }
    try:
        resp = await client.get(_CLOB_HISTORY_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001 — degrade gracefully, log + skip
        log.warning("clob prices-history token=%s failed: %s", token_id, e)
        return None
    history = data.get("history") if isinstance(data, dict) else None
    if not isinstance(history, list):
        log.warning("clob prices-history token=%s: bad shape", token_id)
        return None
    return history


def summarize_history(
    history: list[dict[str, Any]] | None,
    *,
    sparkline_points: int = 5,
) -> PriceHistorySummary | None:
    """Reduce a raw history list to a `PriceHistorySummary`.

    Returns None when the history is empty or every point is unparseable.
    Scalars come out as `None` individually when the series doesn't reach
    far enough back (e.g. 1h history → only `change_30m` and `change_1h`
    are populated, the rest stay None). The sparkline is always populated
    when at least one point survives parsing.

    `sparkline_points` controls how many evenly-spaced samples make up the
    arrow-joined string. Five is the default — fits one LLM line, shows
    direction and roughly one inflection.
    """
    if not history:
        return None
    points = _parse_points(history)
    if not points:
        return None

    sparkline = _build_sparkline(points, sparkline_points)
    now_t, now_p = points[-1]

    return PriceHistorySummary(
        sparkline=sparkline,
        raw_points=points,
        change_30m=_change_over_window(points, now_t, now_p, _WINDOW_30M),
        change_1h=_change_over_window(points, now_t, now_p, _WINDOW_1H),
        change_4h=_change_over_window(points, now_t, now_p, _WINDOW_4H),
        change_24h=_change_over_window(points, now_t, now_p, _WINDOW_24H),
    )


def _parse_points(history: list[dict[str, Any]]) -> list[tuple[int, float]]:
    """Turn raw `[{"t":..., "p":...}, ...]` into sorted `(t, p)` tuples,
    dropping any item whose `t` or `p` doesn't parse cleanly. Sorted by `t`
    ascending so the latest point is `points[-1]` and bisect lookups behave.
    """
    out: list[tuple[int, float]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        t_raw = item.get("t")
        p_raw = item.get("p")
        try:
            t = int(t_raw)  # type: ignore[arg-type]
            p = float(p_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        out.append((t, p))
    out.sort(key=lambda tp: tp[0])
    return out


def _change_over_window(
    points: list[tuple[int, float]],
    now_t: int,
    now_p: float,
    window_s: int,
) -> float | None:
    """Return `now_p − p_at(now_t − window_s)` when a point exists within
    ±`_NEAREST_TOLERANCE_S` of the target timestamp. None when the series
    doesn't reach that far back, or the nearest point is outside tolerance.
    """
    target_t = now_t - window_s
    # Bisect on the timestamp-only index. Build it once per call — cheap
    # for the few hundred points the live path sees, and the backtest
    # caller doesn't go through this helper.
    timestamps = [t for t, _ in points]
    idx = bisect_left(timestamps, target_t)
    # Compare both neighbors and pick the closer one in time.
    candidates: list[tuple[int, int, float]] = []  # (abs_dt, t, p)
    if idx < len(points):
        t, p = points[idx]
        candidates.append((abs(t - target_t), t, p))
    if idx > 0:
        t, p = points[idx - 1]
        candidates.append((abs(t - target_t), t, p))
    if not candidates:
        return None
    candidates.sort()
    nearest_dt, _t, p = candidates[0]
    if nearest_dt > _NEAREST_TOLERANCE_S:
        return None
    return now_p - p


def _build_sparkline(points: list[tuple[int, float]], n: int) -> str:
    """Take `n` evenly-spaced samples from `points` (by index, not by time)
    and join them with `→` formatted to 3 decimals.

    By-index sampling is intentional: it gives a fixed-width sparkline
    regardless of how many raw points came back, and the LLM doesn't need
    minute-precision uniformity to read the shape.
    """
    if n < 1:
        return ""
    if len(points) <= n:
        return "→".join(f"{p:.3f}" for _, p in points)
    # n samples: indices 0, len/(n-1), 2·len/(n-1), ..., len-1.
    last = len(points) - 1
    step = last / (n - 1)
    sampled = [points[round(i * step)][1] for i in range(n)]
    return "→".join(f"{p:.3f}" for p in sampled)


@dataclass(frozen=True, slots=True)
class BookSummary:
    """Compact view of a CLOB order book for one side of a market.

    Mirrors the per-side fields the pipeline writes onto `PolymarketMarket`
    (`yes_bid_*` / `yes_ask_*`). All fields are optional so that a one-sided
    book (only bids resting, or only offers) still produces a usable summary
    rather than `None`.
    """

    bid_top: float | None
    bid_top_size: float | None
    bid_book_dollars: float | None
    bid_depth: int | None
    ask_top: float | None
    ask_top_size: float | None
    ask_book_dollars: float | None
    ask_depth: int | None


async def fetch_book(
    client: httpx.AsyncClient,
    token_id: str,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    """GET clob.polymarket.com/book. Returns `(bid_levels, ask_levels)` or
    None on any failure.

    Each level is a `(price, size)` tuple of floats. Levels are returned
    best-first: bids ordered descending by price (highest bid first), asks
    ordered ascending by price (lowest ask first). The CLOB endpoint
    actually returns bids ascending and asks descending — we reverse both
    here so callers don't have to remember which way the raw shape goes.

    Public, unauthed. No internal semaphore — the caller owns concurrency
    via `CLOB_FETCH_SEM`. Same degrade-on-failure pattern as
    `fetch_price_history`: warn + return None.
    """
    try:
        resp = await client.get(_CLOB_BOOK_URL, params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001 — degrade gracefully, log + skip
        log.warning("clob book token=%s failed: %s", token_id, e)
        return None
    if not isinstance(data, dict):
        log.warning("clob book token=%s: bad shape", token_id)
        return None
    bid_levels = _parse_book_side(data.get("bids"))
    ask_levels = _parse_book_side(data.get("asks"))
    # Both sides arrive worst-first from the CLOB (bids ascending by price,
    # asks descending). Reverse so consumers can read `levels[0]` as
    # top-of-book without thinking about it.
    bid_levels.reverse()
    ask_levels.reverse()
    return bid_levels, ask_levels


def _parse_book_side(levels: Any) -> list[tuple[float, float]]:
    """Turn `[{"price": "...", "size": "..."}, ...]` into `(px, sz)` tuples,
    dropping any level whose fields don't parse cleanly.
    """
    if not isinstance(levels, list):
        return []
    out: list[tuple[float, float]] = []
    for level in levels:
        if not isinstance(level, dict):
            continue
        try:
            px = float(level["price"])
            sz = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append((px, sz))
    return out


def summarize_book(
    book: tuple[list[tuple[float, float]], list[tuple[float, float]]] | None,
) -> BookSummary | None:
    """Reduce raw `(bid_levels, ask_levels)` to a `BookSummary`.

    Returns None when the book is None (fetch failed). Returns a `BookSummary`
    with per-side fields set to None when one side is empty — a one-sided
    book is still informative ("nothing offered" is a real market state).

    `*_book_dollars` is `Σ price × size` across all visible levels on that
    side — the user asked for total $ resting, not just top-of-book qty.
    """
    if book is None:
        return None
    bid_levels, ask_levels = book

    def _side(levels: list[tuple[float, float]]) -> tuple[
        float | None, float | None, float | None, int | None
    ]:
        if not levels:
            return None, None, None, None
        top_px, top_size = levels[0]
        book_dollars = sum(px * sz for px, sz in levels)
        return top_px, top_size, book_dollars, len(levels)

    bid_top, bid_top_size, bid_book_dollars, bid_depth = _side(bid_levels)
    ask_top, ask_top_size, ask_book_dollars, ask_depth = _side(ask_levels)
    return BookSummary(
        bid_top=bid_top,
        bid_top_size=bid_top_size,
        bid_book_dollars=bid_book_dollars,
        bid_depth=bid_depth,
        ask_top=ask_top,
        ask_top_size=ask_top_size,
        ask_book_dollars=ask_book_dollars,
        ask_depth=ask_depth,
    )


def invert_sparkline(sparkline: str | None) -> str | None:
    """Flip a YES-side sparkline (`"0.520→0.612"`) into the implied NO-side
    series (`"0.480→0.388"`). Used by the pipeline to produce the NO clone
    of an enriched market without re-fetching.
    """
    if not sparkline:
        return None
    parts: list[str] = []
    for chunk in sparkline.split("→"):
        try:
            parts.append(f"{1.0 - float(chunk):.3f}")
        except ValueError:
            return None
    return "→".join(parts)
