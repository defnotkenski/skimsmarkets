"""Pull per-token mid-price history from the public CLOB endpoint.

`https://clob.polymarket.com/prices-history?market=<clobTokenId>&fidelity=N`
returns `{"history": [{"t": <unix>, "p": <mid>}, ...]}`. We pull
`fidelity=1` (one point per minute) over the full lifetime of the token. For
a 2-week pre-game soccer market that's ~1.5–3k points; per-token JSON is
small enough to dump as-is to disk.

We hedge concurrent calls behind a small semaphore — the CLOB endpoint is
generous but not unlimited, and this dataset is a few thousand tokens.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from . import cache

log = logging.getLogger(__name__)

_CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
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

    `None` on hard failure (logged); empty list on a token with no history.
    Cache files live under `backtest_cache/prices/<token_id>.json` keyed only
    by token id — `fidelity` is baked into the filename so different
    fidelities co-exist.
    """
    cache_key = ("prices", f"{token_id}_f{fidelity}.json")
    if not force:
        cached = cache.load(*cache_key)
        if cached is not None:
            return cached
    params: dict[str, str] = {
        "market": token_id,
        "fidelity": str(fidelity),
        "interval": interval,
    }
    async with _FETCH_SEM:
        try:
            resp = await client.get(_CLOB_HISTORY_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            log.warning("clob prices-history token=%s failed: %s", token_id, e)
            return None
    history = data.get("history") if isinstance(data, dict) else None
    if not isinstance(history, list):
        log.warning("clob prices-history token=%s: bad shape", token_id)
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
