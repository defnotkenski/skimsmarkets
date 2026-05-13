"""Polymarket gamma-api helper — resolve a market slug to its clob token IDs.

Unusual Whales indexes everything by the ERC-1155 `asset_id`, but our
`PolymarketMarket` only carries the human slug (the `polymarket-us` SDK's
`MarketDetail` TypedDict doesn't expose token IDs). Polymarket's public
gamma-api fills the gap: `https://gamma-api.polymarket.com/markets?slug=X`
returns `clobTokenIds` as a JSON-stringified `[yes_token_id, no_token_id]`
pair. No auth required. One lookup per unique slug, cached per run.

The same `/markets?slug=` response also carries supplementary fields the
polymarket-us SDK doesn't surface — `oneDayPriceChange`, `competitive`,
`spread`, `liquidityClob`, `acceptingOrders`. Since we already pay for the
HTTP call to get clobTokenIds, the resolver caches the full snapshot and
exposes both: `resolve()` keeps the legacy token-tuple shape for callers
that only need IDs (UW path), and `resolve_snapshot()` returns the full
record for the pipeline's gamma-piggyback merge.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
_GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# Polite, identifiable User-Agent for gamma requests. Polymarket's gamma
# API is fronted by Cloudflare which sometimes serves 403 to requests
# carrying httpx's default `python-httpx/X.Y.Z` UA — a generic Python
# signature that shared-IP cloud sandboxes can get flagged on. Sending
# an identifiable UA (project name + repo URL for contact) preempts the
# WAF block in practice and is the standard "be a good citizen" pattern
# for unauthenticated public APIs.
_GAMMA_HEADERS = {
    "User-Agent": "skimsmarkets/1.0 (+https://github.com/defnotkenski/skimsmarkets)",
}


@dataclass(frozen=True, slots=True)
class GammaMarketSnapshot:
    """A single gamma `/markets?slug=` payload reduced to the fields we use.

    All fields are optional — gamma occasionally returns markets without
    book state (settled, unfunded) or without the computed momentum
    fields. None means "absent or unparseable"; renderers should skip
    rather than substitute a default.

    `clob_token_ids` is the original purpose of the call (UW asset-id
    lookup); the rest are free riders piggybacked on the same response.
    """

    clob_token_ids: tuple[str, str] | None
    spread: float | None
    one_day_price_change: float | None
    one_month_price_change: float | None
    competitive: float | None
    liquidity_clob: float | None
    volume_clob: float | None
    accepting_orders: bool | None
    enable_order_book: bool | None


def _coerce_float(v: Any) -> float | None:
    """Best-effort float coercion. Returns None on anything unparseable."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce_bool(v: Any) -> bool | None:
    """Pass through real bools; reject anything else (don't coerce 0/1/"true")."""
    if isinstance(v, bool):
        return v
    return None


class GammaTokenResolver:
    """Async cache over gamma-api's slug → market-snapshot lookup.

    Instantiate once per pipeline run (cache has run-scoped lifetime); each
    unique slug is fetched at most once even under concurrent access thanks
    to a per-slug lock. Failures cache as `None` so we don't retry a bad slug.

    Two read APIs share the same cache:
    - `resolve(slug)` returns just the `(yes_asset_id, no_asset_id)` tuple
      for backwards compatibility with the UW token-resolution path.
    - `resolve_snapshot(slug)` returns the full `GammaMarketSnapshot` for
      the pipeline's gamma-piggyback enrichment.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self._cache: dict[str, GammaMarketSnapshot | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def resolve(self, slug: str) -> tuple[str, str] | None:
        """Return `(yes_asset_id, no_asset_id)` for `slug`, or None on any failure."""
        snap = await self.resolve_snapshot(slug)
        return snap.clob_token_ids if snap is not None else None

    async def resolve_snapshot(self, slug: str) -> GammaMarketSnapshot | None:
        """Return the full gamma snapshot for `slug`, or None on any failure."""
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

    async def _fetch(self, slug: str) -> GammaMarketSnapshot | None:
        try:
            resp = await self._client.get(
                _GAMMA_URL, params={"slug": slug}, headers=_GAMMA_HEADERS,
            )
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
        return GammaMarketSnapshot(
            clob_token_ids=_parse_clob_token_ids(record.get("clobTokenIds")),
            spread=_coerce_float(record.get("spread")),
            one_day_price_change=_coerce_float(record.get("oneDayPriceChange")),
            one_month_price_change=_coerce_float(record.get("oneMonthPriceChange")),
            competitive=_coerce_float(record.get("competitive")),
            liquidity_clob=_coerce_float(record.get("liquidityClob")),
            volume_clob=_coerce_float(record.get("volumeClob")),
            accepting_orders=_coerce_bool(record.get("acceptingOrders")),
            enable_order_book=_coerce_bool(record.get("enableOrderBook")),
        )


async def list_gamma_events(
    client: httpx.AsyncClient,
    *,
    tag_slug: str | None = "sports",
    page_size: int = 500,
    max_pages: int = 6,
) -> list[dict[str, Any]]:
    """List upcoming Polymarket events from gamma-api, paginated.

    Returns raw event dicts (slug, markets, endDate, etc.) ordered by
    soonest game-time first via `order=endDate&ascending=true`. Used by
    `fetch_gamma_slate` to filter client-side by slug prefix — gamma has
    no canonical `seriesSlug` field, so league filtering rides the slug
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
        # Retry transient HTTP errors (403/429/5xx) with exponential
        # backoff (1s, 2s, 4s). Polymarket gamma is fronted by Cloudflare
        # and occasionally serves 403 to rapid sequential calls from the
        # same IP — observed during cloud-routine fires where `skims
        # fetch` and `skims rank` hit the endpoint within seconds of
        # each other. Without this loop, a single transient 403 empties
        # the slate to 0 and silently aborts the routine. Mirrors the
        # Kalshi client's `list_events` 429 backoff pattern.
        backoff = 1.0
        resp: httpx.Response | None = None
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = await client.get(
                    _GAMMA_EVENTS_URL, params=params, headers=_GAMMA_HEADERS,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < 3:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                break
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                if attempt < 3:
                    log.warning(
                        "gamma-api events list page=%d got HTTP %d, "
                        "retrying in %.1fs (attempt %d/4)",
                        page, resp.status_code, backoff, attempt + 1,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
            break
        if resp is None:
            log.warning(
                "gamma-api events list page=%d failed after retries: %s",
                page, last_err,
            )
            break
        try:
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
        resp = await client.get(
            _GAMMA_EVENTS_URL,
            params={"slug": slug},
            headers=_GAMMA_HEADERS,
        )
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
