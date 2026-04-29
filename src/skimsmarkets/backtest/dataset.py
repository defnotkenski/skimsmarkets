"""Assemble per-outcome feature rows from cached gamma + CLOB data.

For each closed soccer match we emit one row per market (= per outcome:
home / draw / away on a 3-way; sometimes 2-way). The row carries:

  - identity: event_slug, market_slug, league, kickoff (UTC), outcome label
  - settlement: 1 if YES resolved (this side won), else 0
  - liquidity: volumeNum on the market, openInterest if present
  - prices at fixed lookback offsets from kickoff:
      open    (first price in series)
      t24h    (~24 hours pre-kickoff)
      t1h     (~1 hour pre-kickoff)
      t1m     (~1 minute pre-kickoff = closing line)
  - movement: (t1m - t24h)
  - volatility: stdev of 1-min returns over the last 24h pre-kickoff

Three-way derived columns (event-level) are computed in a second pass:
  - sum_close: home_t1m + draw_t1m + away_t1m  (overround proxy)

Output: a pandas DataFrame, also persisted to `backtest_cache/dataset.parquet`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import pandas as pd

from . import cache, clob_history, gamma_history

log = logging.getLogger(__name__)


@dataclass
class Row:
    event_slug: str
    market_slug: str
    league: str
    outcome: str
    kickoff_ts: int
    settled_yes: int
    volume_num: float | None
    open_p: float | None
    t24h_p: float | None
    t1h_p: float | None
    t1m_p: float | None
    movement_24h: float | None
    vol_24h: float | None
    n_points: int


def _league_from_slug(slug: str) -> str:
    return slug.split("-", 1)[0]


def _kickoff_ts(event: dict[str, Any]) -> int | None:
    """Use endDate as kickoff. CLAUDE.md: gamma's endDate IS the tipoff analog."""
    raw = event.get("endDate")
    if not isinstance(raw, str):
        return None
    try:
        # endDate format: "2026-04-27T19:00:00Z"
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except ValueError:
        return None


def _yes_token_id(market: dict[str, Any]) -> str | None:
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        parsed = raw
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
        return parsed[0]
    return None


def _settled_yes(market: dict[str, Any]) -> int | None:
    """Read `outcomePrices`; YES side resolved => 1 else 0. None if missing."""
    raw = market.get("outcomePrices")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        parsed = raw
    if not isinstance(parsed, list) or not parsed:
        return None
    try:
        yes = float(parsed[0])
    except (TypeError, ValueError):
        return None
    if yes >= 0.999:
        return 1
    if yes <= 0.001:
        return 0
    # ambiguous resolution (50/50 cancellation, etc.) — drop
    return None


def _price_at_or_before(history: list[dict[str, Any]], ts: int) -> float | None:
    """Return the most recent price at or before `ts`. None if all points after."""
    last: float | None = None
    for pt in history:
        t = pt.get("t")
        p = pt.get("p")
        if not isinstance(t, (int, float)) or not isinstance(p, (int, float)):
            continue
        if t > ts:
            break
        last = float(p)
    return last


def _stdev_returns(history: list[dict[str, Any]], start_ts: int, end_ts: int) -> float | None:
    """Stdev of 1-min log-ish returns within [start_ts, end_ts]. None if <5 points."""
    pts = [
        pt for pt in history
        if isinstance(pt.get("t"), (int, float)) and start_ts <= pt["t"] <= end_ts
        and isinstance(pt.get("p"), (int, float)) and 0.0 < pt["p"] < 1.0
    ]
    if len(pts) < 5:
        return None
    rets: list[float] = []
    for a, b in zip(pts[:-1], pts[1:]):
        # use simple p-difference rather than log — prices near 0/1 blow up logs
        # and we just want a noise/volatility proxy.
        rets.append(b["p"] - a["p"])
    if not rets:
        return None
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    return var ** 0.5


def _build_row(
    event: dict[str, Any],
    market: dict[str, Any],
    history: list[dict[str, Any]],
    kickoff_ts: int,
) -> Row | None:
    settled = _settled_yes(market)
    if settled is None:
        return None
    open_p = history[0].get("p") if history else None
    t24h = _price_at_or_before(history, kickoff_ts - 24 * 3600)
    t1h = _price_at_or_before(history, kickoff_ts - 3600)
    t1m = _price_at_or_before(history, kickoff_ts - 60)
    movement = (t1m - t24h) if (t1m is not None and t24h is not None) else None
    vol = _stdev_returns(history, kickoff_ts - 24 * 3600, kickoff_ts - 60)
    return Row(
        event_slug=event.get("slug", ""),
        market_slug=market.get("slug", ""),
        league=_league_from_slug(event.get("slug", "")),
        outcome=market.get("groupItemTitle") or "",
        kickoff_ts=kickoff_ts,
        settled_yes=settled,
        volume_num=market.get("volumeNum"),
        open_p=float(open_p) if isinstance(open_p, (int, float)) else None,
        t24h_p=t24h,
        t1h_p=t1h,
        t1m_p=t1m,
        movement_24h=movement,
        vol_24h=vol,
        n_points=len(history),
    )


async def build_dataset(*, max_events: int = 800) -> pd.DataFrame:
    """End-to-end: fetch events, pull histories, assemble rows."""
    events = await gamma_history.fetch_closed_soccer_events(max_events=max_events)
    log.info("closed soccer events collected: %d", len(events))
    # Collect every YES token id we need to fetch.
    token_to_ctx: dict[str, tuple[dict[str, Any], dict[str, Any], int]] = {}
    for ev in events:
        ts = _kickoff_ts(ev)
        if ts is None:
            continue
        for m in ev.get("markets", []) or []:
            tid = _yes_token_id(m)
            if tid is None:
                continue
            token_to_ctx[tid] = (ev, m, ts)
    log.info("unique YES tokens to fetch: %d", len(token_to_ctx))
    histories = await clob_history.fetch_many(list(token_to_ctx))
    rows: list[Row] = []
    for tid, (ev, m, ts) in token_to_ctx.items():
        hist = histories.get(tid)
        if not hist:
            continue
        row = _build_row(ev, m, hist, ts)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame([asdict(r) for r in rows])
    if not df.empty:
        out = cache.cache_path("dataset.parquet")
        df.to_parquet(out, index=False)
        log.info("dataset rows: %d, written to %s", len(df), out)
    return df
