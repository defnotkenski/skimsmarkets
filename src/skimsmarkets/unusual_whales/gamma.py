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
_GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"


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


async def list_gamma_events(
    client: httpx.AsyncClient,
    *,
    tag_slug: str | None = "sports",
    page_size: int = 500,
    max_pages: int = 6,
) -> list[dict[str, Any]]:
    """List upcoming offshore-Polymarket events from gamma-api, paginated.

    Returns raw event dicts (slug, markets, endDate, etc.) ordered by
    soonest game-time first via `order=endDate&ascending=true`. Used by the
    `--gamma-league <prefix>` path to filter client-side by slug prefix —
    gamma has no `seriesSlug` field, so league filtering rides the slug
    prefix convention (`lib-`, `ucl-`, `arg-`, `epl-`, `spl-`, etc.).

    Pagination is necessary because esports (cs2, lol, dota2) and
    high-volume markets crowd out actual sports leagues in page 1 — Copa
    Libertadores (`lib-`) lives in page 2, etc. Walks up to `max_pages` ×
    `page_size` events serially (capped to keep latency bounded) and stops
    early on an empty/short page.

    `tag_slug` defaults to 'sports' because the gamma global feed is
    dominated by crypto / news / `will-X` futures whose endDates crowd out
    actual sports events under `order=endDate`. Set to None to fetch the
    unfiltered feed. `closed=false` keeps already-resolved events out.

    Failures degrade to whatever pages succeeded (with a warning) — same
    posture as `fetch_gamma_event`.
    """
    all_events: list[dict[str, Any]] = []
    for page in range(max_pages):
        params: dict[str, str] = {
            "closed": "false",
            "order": "endDate",
            "ascending": "true",
            "limit": str(page_size),
            "offset": str(page * page_size),
        }
        if tag_slug:
            params["tag_slug"] = tag_slug
        try:
            resp = await client.get(_GAMMA_EVENTS_URL, params=params)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            log.warning("gamma-api events list page=%d failed: %s", page, e)
            break
        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "gamma-api events list page=%d non-json response: %s", page, e
            )
            break
        if not isinstance(data, list):
            log.warning(
                "gamma-api events list page=%d: unexpected shape %s",
                page, type(data),
            )
            break
        all_events.extend(item for item in data if isinstance(item, dict))
        # Short page = end of feed; no point asking for the next offset.
        if len(data) < page_size:
            break
    return all_events


async def fetch_gamma_event(
    client: httpx.AsyncClient, slug: str
) -> dict[str, Any] | None:
    """Fetch a single offshore-Polymarket event by slug from gamma-api.

    Gamma's `/events?slug=<X>` returns a list (length 1 on hit, 0 on miss).
    We return the raw event dict so callers can either feed it to
    `PolymarketEvent.from_gamma()` or read fields ad-hoc. Public, unauthed
    endpoint — same `httpx.AsyncClient` pattern as `GammaTokenResolver`.

    Failure semantics mirror the token resolver: log a warning, return None,
    never raise. The pipeline treats a missing offshore event the same way
    it treats a settled US event (skip, continue).
    """
    try:
        resp = await client.get(_GAMMA_EVENTS_URL, params={"slug": slug})
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        log.warning("gamma-api events slug=%s failed: %s", slug, e)
        return None
    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        log.warning("gamma-api events slug=%s non-json response: %s", slug, e)
        return None
    record = _first_record(data)
    if record is None:
        log.warning("gamma-api events slug=%s: no records returned", slug)
    return record


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
