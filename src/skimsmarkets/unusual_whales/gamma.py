"""Polymarket gamma-api helper — resolve a market slug to its clob token IDs.

Unusual Whales indexes everything by the ERC-1155 `asset_id`, but our
`PolymarketMarket` only carries the human slug (the `polymarket-us` SDK's
`MarketDetail` TypedDict doesn't expose token IDs). Polymarket's public
gamma-api fills the gap: `https://gamma-api.polymarket.com/markets?slug=X`
returns `clobTokenIds` as a JSON-stringified `[yes_token_id, no_token_id]`
pair. No auth required. One lookup per unique slug, cached per run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

_GAMMA_URL = "https://gamma-api.polymarket.com/markets"


class GammaTokenResolver:
    """Async cache over gamma-api's slug → (yes_token_id, no_token_id) lookup.

    Instantiate once per pipeline run (cache has run-scoped lifetime); each
    unique slug is fetched at most once even under concurrent access thanks
    to a per-slug lock. Failures cache as `None` so we don't retry a bad slug.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self._cache: dict[str, tuple[str, str] | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def resolve(self, slug: str) -> tuple[str, str] | None:
        """Return `(yes_asset_id, no_asset_id)` for `slug`, or None on any failure."""
        if slug in self._cache:
            return self._cache[slug]
        # Per-slug lock coalesces concurrent requests for the same slug so we
        # don't fire two gamma-api calls while the first is in flight.
        lock = self._locks.setdefault(slug, asyncio.Lock())
        async with lock:
            if slug in self._cache:
                return self._cache[slug]
            result = await self._fetch(slug)
            self._cache[slug] = result
            return result

    async def _fetch(self, slug: str) -> tuple[str, str] | None:
        try:
            resp = await self._client.get(_GAMMA_URL, params={"slug": slug})
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            log.warning("gamma-api slug=%s failed: %s", slug, e)
            return None
        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            log.warning("gamma-api slug=%s non-json response: %s", slug, e)
            return None
        record = _first_record(data)
        if record is None:
            return None
        return _parse_clob_token_ids(record.get("clobTokenIds"))


def _first_record(data: Any) -> dict[str, Any] | None:
    """gamma-api returns a list (even for a single-slug query). Take the first."""
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return first
    if isinstance(data, dict):
        return data
    return None


def _parse_clob_token_ids(raw: Any) -> tuple[str, str] | None:
    """`clobTokenIds` arrives as a JSON-encoded string of a 2-element list."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        parsed = raw
    if not isinstance(parsed, list) or len(parsed) < 2:
        return None
    yes_id, no_id = parsed[0], parsed[1]
    if not isinstance(yes_id, str) or not isinstance(no_id, str):
        return None
    return yes_id, no_id
