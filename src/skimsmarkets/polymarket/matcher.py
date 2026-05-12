"""Reverse matcher: Kalshi event → Polymarket slug.

Mirror of `kalshi/matcher.py:find_kalshi_match`, but pointed the
other way. Used by `pipeline.bridge_uw_context` to discover the
matching Polymarket market for a Kalshi-sourced ranker event so
Unusual Whales wallet flow (which is keyed by Polymarket
`asset_id`) can still attach.

Match key:
- Tour matches (`atp-` ↔ `atp-`, `wta-` ↔ `wta-` slug prefixes)
- Both player surnames present (in either order) after
  `_normalize_name` canonicalisation
- Polymarket `gameStartTime` within ±36h of the Kalshi event's
  earliest market `game_start_time` (absorbs reschedule drift
  between venues — Kalshi tends to lock the time earlier)

Returns the unique matching Polymarket slug, or None on:
- No tour match
- No matching surnames
- Multiple matches (ambiguous)
- Any extraction failure (silent — UW just doesn't attach)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from skimsmarkets.polymarket.models import (
    PolymarketEvent,
    _coerce_time,
    _parse_h2h_question,
)
from skimsmarkets.tennis.matchstat import _normalize_name

log = logging.getLogger(__name__)


# Window absorbing reschedule drift between venues. Kalshi tends to
# lock match times earlier than Polymarket; ±36h is wide enough to
# cover all observed drift without crossing day-boundary collisions
# for tournaments that schedule sequential same-venue matches.
_TIPOFF_WINDOW = timedelta(hours=36)


def find_polymarket_slug(
    kalshi_event: PolymarketEvent,
    gamma_payloads: list[dict[str, Any]],
) -> str | None:
    """Find the Polymarket slug matching this Kalshi event.

    `kalshi_event` is the adapter-built `PolymarketEvent` whose
    `slug` is `{tour}-{tournament}-{lastA}-{lastB}-{date}`. Surnames
    are extracted from the slug's player tokens (matches the
    canonicalisation used by `kalshi/slate.py:_surname_from_yes_sub_title`),
    so the join doesn't depend on title parsing of the Kalshi side.
    """
    surnames = _extract_kalshi_surnames(kalshi_event)
    if surnames is None:
        return None
    fav, dog = surnames
    tour = _kalshi_tour(kalshi_event)
    if tour is None:
        return None
    tipoff = _kalshi_tipoff(kalshi_event)

    matches: list[str] = []
    for payload in gamma_payloads:
        slug = payload.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        if not slug.startswith(f"{tour}-"):
            continue
        # Pull surname pair from any market in the gamma event whose
        # question parses cleanly. Tennis events on gamma typically
        # have one binary market per matchup (tour-style), so the
        # first parseable hit is authoritative.
        gamma_surnames = _extract_gamma_surnames(payload)
        if gamma_surnames is None:
            continue
        if {fav, dog} != gamma_surnames:
            continue
        # Tipoff window — only enforce when both sides have a tipoff.
        # Kalshi always has one (load-bearing for the slate filter),
        # but gamma occasionally drops it on rescheduled fixtures.
        if tipoff is not None:
            gamma_tipoff = _gamma_tipoff(payload)
            if gamma_tipoff is not None and abs(gamma_tipoff - tipoff) > _TIPOFF_WINDOW:
                continue
        matches.append(slug)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        log.debug(
            "polymarket reverse match for %s ambiguous (%d candidates)",
            kalshi_event.slug,
            len(matches),
        )
    return None


def _extract_kalshi_surnames(
    ev: PolymarketEvent,
) -> tuple[str, str] | None:
    """Pull both surnames from the slate adapter's synthesized slug
    (e.g. `wta-rome-andreeva-gauff-2026-05-12` → `("andreeva", "gauff")`).

    Falls back to parsing market `yes_sub_title` if the slug shape is
    unexpected (defensive — the adapter is the only writer of these
    slugs today, but a future rename shouldn't silently break UW).
    """
    if ev.slug:
        # Slug format: {tour}-{tournament...}-{lastA}-{lastB}-{yyyy-mm-dd}
        # Date suffix is always 3 dash-separated numeric tokens (e.g.
        # `2026-05-12`). The two surnames sit just before that.
        parts = ev.slug.split("-")
        if len(parts) >= 5 and all(p.isdigit() for p in parts[-3:]):
            fav = parts[-5]
            dog = parts[-4]
            if fav and dog:
                return fav, dog
    # Fallback: parse from market labels directly.
    fav_market = next((m for m in ev.markets if not m.is_no_side and m.yes_sub_title), None)
    dog_market = next((m for m in ev.markets if m.is_no_side and m.yes_sub_title), None)
    if fav_market is None or dog_market is None:
        return None
    fav = _last_normalized_token(fav_market.yes_sub_title or "")
    dog = _last_normalized_token(dog_market.yes_sub_title or "")
    if not fav or not dog:
        return None
    return fav, dog


def _kalshi_tour(ev: PolymarketEvent) -> str | None:
    if not ev.slug:
        return None
    if ev.slug.startswith("atp-"):
        return "atp"
    if ev.slug.startswith("wta-"):
        return "wta"
    return None


def _kalshi_tipoff(ev: PolymarketEvent) -> datetime | None:
    starts = [t for m in ev.markets if (t := m.game_start_time) is not None]
    return min(starts) if starts else None


def _extract_gamma_surnames(payload: dict[str, Any]) -> set[str] | None:
    """Pull both surnames from the first parseable market `question`
    in a gamma payload, normalized to lowercase last-token form so
    the comparison against Kalshi's adapter slug surnames is direct.
    """
    raw_markets = payload.get("markets")
    if not isinstance(raw_markets, list):
        return None
    for raw in raw_markets:
        if not isinstance(raw, dict):
            continue
        h2h = _parse_h2h_question(raw.get("question"))
        if h2h is None:
            continue
        a, b = h2h
        a_token = _last_normalized_token(a)
        b_token = _last_normalized_token(b)
        if not a_token or not b_token or a_token == b_token:
            continue
        return {a_token, b_token}
    return None


def _gamma_tipoff(payload: dict[str, Any]) -> datetime | None:
    """Earliest `gameStartTime` across the event's markets, falling
    back to the event-level `startTime`. Mirrors gamma's precedence
    order in `pipeline.fetch_gamma_slate`.
    """
    times: list[datetime] = []
    raw_markets = payload.get("markets")
    if isinstance(raw_markets, list):
        for raw in raw_markets:
            if not isinstance(raw, dict):
                continue
            t = _coerce_time(raw.get("gameStartTime"))
            if t is not None:
                times.append(t)
    if times:
        return min(times)
    return _coerce_time(payload.get("startTime"))


def _last_normalized_token(name: str) -> str:
    """Lowercased, normalized last whitespace-separated token. Mirrors
    `kalshi/slate.py:_surname_from_yes_sub_title` so the Kalshi adapter
    slug and the gamma reverse-match use the same canonicalisation.
    """
    if not name:
        return ""
    norm = _normalize_name(name)
    if not norm:
        return ""
    last = norm.split()[-1] if norm.split() else norm
    # Strip non-alnum the same way the slug synthesizer does.
    return "".join(c for c in last if c.isalnum())
