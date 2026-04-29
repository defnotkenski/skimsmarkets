"""Pull closed soccer events from gamma-api with disk caching.

We page through gamma's `/events?closed=true&tag_slug=soccer` feed sorted by
recency (`order=endDate&ascending=false`) and keep only actual game events —
slugs like `epl-mun-bre-2026-04-27`, not season-long futures or
`next-X-manager` markets. The same league-prefix convention used by the live
pipeline applies here.

Each closed event arrives with `outcomePrices` populated (e.g. `["1","0"]`),
giving us ground truth without any extra round-trip. We persist the raw event
payload — downstream feature-building re-derives everything from it.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from . import cache

log = logging.getLogger(__name__)

_GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

# League slug prefixes that identify soccer game events. Sourced from observed
# gamma slugs — the live pipeline already keys on these for offshore fetches.
SOCCER_LEAGUE_PREFIXES: tuple[str, ...] = (
    "epl-",   # English Premier League
    "ucl-",   # UEFA Champions League
    "uel-",   # UEFA Europa League
    "uecl-",  # UEFA Conference League
    "spl-",   # Spanish La Liga
    "bun-",   # German Bundesliga
    "lig-",   # French Ligue 1
    "lib-",   # Copa Libertadores
    "sud-",   # Copa Sudamericana
    "seri-",  # Italian Serie A
    "mls-",   # Major League Soccer
    "mex-",   # Liga MX
    "aus-",   # A-League
    "arg-",   # Argentine Primera
    "bra-",   # Brasileirão
    "tur-",   # Turkish Süper Lig
    "ned-",   # Eredivisie
    "por-",   # Primeira Liga
    "sue-",   # Allsvenskan
    "jpn-",   # J-League
    "kor-",   # K-League
    "chn-",   # Chinese Super League
    "sau-",   # Saudi Pro League
    "egy-",   # Egyptian Premier
)

# Game-event slugs end with a YYYY-MM-DD date. Manager / futures / season slugs
# do not. This is the cleanest way to separate the two without a per-league
# allowlist.
_GAME_DATE_RE = re.compile(r"-\d{4}-\d{2}-\d{2}(?:-|$)")

# "More markets", spreads/totals, exact score, BTTS, player props, halftime,
# corners — all variant slugs hanging off the same kickoff. We want only the
# canonical 3-way moneyline event (slug ends with the date, no suffix).
_VARIANT_SUFFIXES: tuple[str, ...] = (
    "-more-markets",
    "-spread",
    "-total",
    "-exact-score",
    "-btts",
    "-player-props",
    "-halftime-result",
    "-total-corners",
    "-double-chance",
    "-clean-sheet",
)


def is_moneyline_game_event(slug: str) -> bool:
    """True iff `slug` is a canonical 3-way moneyline soccer match.

    Encodes the conventions documented in CLAUDE.md: gamma omits
    sportsMarketType, so we rely on slug shape — league prefix, ends with
    date, no variant suffix.
    """
    if not any(slug.startswith(p) for p in SOCCER_LEAGUE_PREFIXES):
        return False
    if not _GAME_DATE_RE.search(slug):
        return False
    return not any(slug.endswith(s) for s in _VARIANT_SUFFIXES)


async def fetch_closed_soccer_events(
    *,
    max_events: int = 2000,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    """Page through closed soccer events newest-first, return moneyline games.

    Cached as one JSON file per page so re-runs only fetch new pages. Stops
    when an empty/short page is returned or when we've collected `max_events`
    moneyline events.
    """
    collected: list[dict[str, Any]] = []
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for page in range((max_events // page_size) + 4):
            cache_key = ("gamma_closed_soccer", f"page_{page:03d}.json")
            data = cache.load(*cache_key)
            if data is None:
                params = {
                    "closed": "true",
                    "tag_slug": "soccer",
                    "order": "endDate",
                    "ascending": "false",
                    "limit": str(page_size),
                    "offset": str(page * page_size),
                }
                try:
                    resp = await client.get(_GAMMA_EVENTS_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:  # noqa: BLE001
                    log.warning("gamma closed soccer page=%d failed: %s", page, e)
                    break
                if not isinstance(data, list):
                    log.warning("gamma closed soccer page=%d: bad shape", page)
                    break
                cache.save(data, *cache_key)
            if not data:
                break
            for ev in data:
                if not isinstance(ev, dict):
                    continue
                slug = ev.get("slug", "")
                if is_moneyline_game_event(slug):
                    collected.append(ev)
            if len(collected) >= max_events:
                break
            if len(data) < page_size:
                break
            # Be polite — small inter-page delay even on cache hits is harmless.
            await asyncio.sleep(0.05)
    return collected[:max_events]
