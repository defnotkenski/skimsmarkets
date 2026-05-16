"""How quickly does the market converge to its closing-line accuracy?

For a sample of events, snapshot each market's mid at a grid of pre-kickoff
offsets (48h, 24h, 12h, 6h, 3h, 1h, 30m, 10m, 1m), compute Brier vs.
settlement at each offset. This tells us the optimal *entry window* — too
early and the price is uninformative, too late and the spread widens.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from skimsmarkets.backtest import cache
from skimsmarkets.backtest.dataset import (
    _kickoff_ts,
    _price_at_or_before,
    _settled_yes,
    _yes_token_id,
)
from skimsmarkets.backtest.gamma_history import is_moneyline_game_event

OFFSETS_SECONDS: list[tuple[str, int]] = [
    ("T-7d",  7 * 86400),
    ("T-3d",  3 * 86400),
    ("T-48h", 48 * 3600),
    ("T-24h", 24 * 3600),
    ("T-12h", 12 * 3600),
    ("T-6h",  6 * 3600),
    ("T-3h",  3 * 3600),
    ("T-1h",  3600),
    ("T-30m", 1800),
    ("T-10m", 600),
    ("T-1m",  60),
]


def _events_iter():
    """Re-read cached gamma pages and yield the same moneyline events used
    elsewhere. Avoids holding 1k events in memory."""
    page_dir = cache.cache_path("gamma_closed_soccer", "_dummy").parent
    for f in sorted(page_dir.glob("page_*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for ev in data:
            if isinstance(ev, dict) and is_moneyline_game_event(ev.get("slug", "")):
                yield ev


def _load_history(token_id: str) -> list[dict[str, Any]] | None:
    """Read cached price history for a token; None if missing."""
    return cache.load("prices", f"{token_id}_f1.json")


def build_convergence_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ev in _events_iter():
        kickoff = _kickoff_ts(ev)
        if kickoff is None:
            continue
        for m in ev.get("markets", []) or []:
            settled = _settled_yes(m)
            tid = _yes_token_id(m)
            if settled is None or tid is None:
                continue
            hist = _load_history(tid)
            if not hist:
                continue
            for label, dt in OFFSETS_SECONDS:
                p = _price_at_or_before(hist, kickoff - dt)
                rows.append({
                    "market_slug": m.get("slug"),
                    "league": (ev.get("slug") or "").split("-", 1)[0],
                    "offset": label,
                    "p": p,
                    "settled_yes": settled,
                })
    return pd.DataFrame(rows)


def convergence_summary(df: pd.DataFrame) -> str:
    """Brier of the mid at each offset. Lower = sharper.

    Coverage = fraction of rows that have a price at this offset (early
    offsets miss markets that hadn't opened yet).
    """
    total_per_offset = df.groupby("offset", observed=True).size()
    sub = df.dropna(subset=["p"]).copy()
    sub["sq"] = (sub.p - sub.settled_yes) ** 2
    g = sub.groupby("offset", observed=True).agg(
        n=("sq", "size"),
        brier=("sq", "mean"),
    )
    g["coverage_pct"] = (100 * g["n"] / total_per_offset).round(1)
    g = g.round(4)
    order = [label for label, _ in OFFSETS_SECONDS]
    g = g.reindex(order)
    return g[["n", "coverage_pct", "brier"]].to_string()
