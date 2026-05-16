"""Pull per-token mid-price history from the public CLOB endpoint, cached.

Backtest needs full-lifetime price history for thousands of settled tokens.
The bare HTTP fetch lives in `skimsmarkets.clob.fetch_price_history` (shared
with the live pipeline); this module wraps it with on-disk caching and a
module-level concurrency cap so repeated backtest builds don't re-fetch.

The cache is keyed by `<token_id>_f<fidelity>.json` so different fidelities
coexist (though backtest sticks to `fidelity=1` — one point per minute over
the full lifetime, ~1.5–3k points per pre-game window).

We hedge concurrent calls behind a small semaphore — the CLOB endpoint is
generous but not unlimited, and this dataset is a few thousand tokens.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from skimsmarkets.clob import fetch_price_history as _fetch_core

from . import cache

_FETCH_SEM = asyncio.Semaphore(8)


async def fetch_price_history(
    client: httpx.AsyncClient,
    token_id: str,
    *,
    fidelity: int = 1,
    interval: str = "max",
    force: bool = False,
) -> list[dict[str, float]] | None:
    """Return `[{"t": unix, "p": mid}, ...]` for a CLOB token, cached by id.

    `None` on hard failure (logged by the core fetcher); empty list on a
    token with no history. Cache files live under
    `backtest_cache/prices/<token_id>.json` keyed only by token id —
    `fidelity` is baked into the filename so different fidelities co-exist.
    """
    cache_key = ("prices", f"{token_id}_f{fidelity}.json")
    if not force:
        cached = cache.load(*cache_key)
        if cached is not None:
            return cached
    async with _FETCH_SEM:
        history = await _fetch_core(
            client, token_id, interval=interval, fidelity=fidelity
        )
    if history is None:
        return None
    cache.save(history, *cache_key)
    return history


async def fetch_many(
    token_ids: list[str], *, fidelity: int = 1
) -> dict[str, list[dict[str, Any]] | None]:
    """Fetch many tokens concurrently. Returns dict keyed by token id."""
    timeout = httpx.Timeout(60.0)
    out: dict[str, list[dict[str, Any]] | None] = {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def _one(tid: str) -> None:
            out[tid] = await fetch_price_history(client, tid, fidelity=fidelity)
        await asyncio.gather(*(_one(t) for t in token_ids))
    return out
