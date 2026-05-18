"""Polymarket gamma-sourced slate builder.

Replaces the Kalshi adapter that briefly held the slate role
(2026-05-11 → 2026-05-12). Polymarket is back as the data source for
its broader sport coverage and more consistent per-market metadata;
Kalshi survives only as the execution venue (`skims execute`).

Two entry points share the file:

- `fetch_gamma_slate` — default browse via gamma `/events?tag_slug=...`,
  filtered by leagues, horizon, tradability, blowout, and (optionally)
  min open interest. Returns `PolymarketEvent` objects via
  `PolymarketEvent.from_gamma` so the rest of the pipeline is unchanged.

- `fetch_gamma_events` — per-slug lookup bypassing the horizon filter,
  used for explicit `--slug` CLI requests.

The MAX_SLATE_EVENTS cap is NOT applied here — it moved to
`selection.select_top_events` so the selection stage scores the entire
population by fundamental imbalance rather than tipoff alone.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import httpx

from skimsmarkets import config as cfg
from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.unusual_whales import fetch_gamma_event, list_gamma_events

log = logging.getLogger(__name__)


async def fetch_gamma_slate(
    http: httpx.AsyncClient,
    leagues: list[str],
    horizon_hours: int,
    *,
    sports: list[str] | None = None,
    max_implied_probability: float = cfg.MAX_IMPLIED_PROBABILITY,
    min_favorite_probability: float | None = cfg.MIN_FAVORITE_PROBABILITY,
    min_open_interest_dollars: float = cfg.MIN_OPEN_INTEREST_DOLLARS,
) -> list[PolymarketEvent]:
    """Fetch the Polymarket sports slate from gamma-api.

    Single source of truth for the default browse. Filters layered top
    to bottom:

    1. **Bulk listing.** `list_gamma_events` pages through gamma's
       `/events?tag_slug=<tag>&order=endDate&ascending=true` for upcoming
       events soonest-first. Pagination is necessary because esports
       (cs2, lol, dota2) and high-volume markets crowd out actual sports
       leagues in page 1. When `sports` is non-empty, one listing call
       fans out per sport tag (gamma's `tag_slug` query param accepts
       only a single value), and the per-sport payloads are unioned and
       deduped by event slug. Empty `sports` falls back to gamma's
       umbrella `tag_slug=sports`.
    2. **Variant-bundle drop.** `PolymarketEvent.from_gamma` skips
       `-more-markets` / `-halftime-result` / `-exact-score` /
       `-total-corners` / `-player-props` event-level variants and
       non-moneyline market slugs inline.
    3. **League prefix filter.** When `leagues` is non-empty, keep only
       events whose slug starts with `<league>-` for any of them. Empty
       list = no filter (browse all sports). Anchored on the dash so
       `arg` doesn't accidentally swallow `argf-`.
    4. **Horizon time window.** Keep events whose earliest market
       `gameStartTime` falls within
       `[now - HORIZON_BACKSTOP_HOURS, now + horizon_hours]`. The
       backstop catches long-tail endings (overtime, weather delays)
       that haven't settled yet. **Critical:** filter on per-market
       `gameStartTime`, NOT event `endDate` — `endDate` is frozen at
       market creation and lags rescheduled fixtures by days.
    5. **Tradability filter.** Drop markets without `yes_sub_title` or
       bid/ask. `from_gamma` already pre-filters bid/ask presence, so
       this is a belt-and-suspenders check that mostly catches
       label-less futures placeholders.
    6. **Blowout filter.** Drop events whose favorite (highest YES mid
       across markets) sits at or above `max_implied_probability`.
    7. **Floor filter.** Optional — when `min_favorite_probability` is
       set, drop events whose favorite is priced BELOW the floor.
       Inverse-shaped counterpart of the blowout filter. Default None
       (disabled). Primary use case: tail mode wants to LLM-evaluate
       only events that can produce Prime EV through the asymmetric-
       payoff path (favorite ≥ ~0.75, underdog ≤ ~0.25).
    8. **Min OI filter.** Optional — when `min_open_interest_dollars > 0`,
       drop events where the MIN OI across markets is below the floor.
       Default is OFF (0.0); see config docs for the rationale.

    The horizon filter applied here is the slate-level cut on raw gamma
    `gameStartTime`. `pipeline.apply_horizon_filter` runs the same cut
    again later, AFTER `overlay_matchstats_tipoffs` refines tennis tipoffs
    via the MatchStats vendor — events whose precise tipoff falls outside
    the window get dropped there. Both cuts use the same `HORIZON_BACKSTOP
    _HOURS` constant for symmetry.
    """
    now = datetime.now(tz=UTC)
    horizon_start = now - timedelta(hours=cfg.HORIZON_BACKSTOP_HOURS)
    horizon_end = now + timedelta(hours=horizon_hours)

    sports = sports or []
    if sports:
        # Gamma's `tag_slug` query param is single-valued — fan out one
        # listing per sport tag and union the payloads. Dedupe by event
        # slug because tags overlap (e.g. `ufc` ⊂ `mma`).
        page_lists = await asyncio.gather(
            *(list_gamma_events(http, tag_slug=s) for s in sports)
        )
        seen: set[str] = set()
        payloads: list[dict] = []
        for plist in page_lists:
            for p in plist:
                slug = p.get("slug")
                if not isinstance(slug, str) or slug in seen:
                    continue
                seen.add(slug)
                payloads.append(p)
        log.info(
            "fetched %d gamma payloads across %d sport tag(s) [%s] "
            "(leagues=%s, horizon=%sh)",
            len(payloads),
            len(sports),
            ",".join(sports),
            leagues or "all",
            horizon_hours,
        )
    else:
        payloads = await list_gamma_events(http)
        log.info(
            "fetched %d gamma payloads (leagues=%s, horizon=%sh)",
            len(payloads),
            leagues or "all",
            horizon_hours,
        )

    # League prefixes are anchored on dash so `arg` doesn't swallow `argf-`.
    league_prefixes = [f"{p}-" for p in leagues]

    kept: list[PolymarketEvent] = []
    dropped_blowout = 0
    dropped_too_competitive = 0
    dropped_thin_oi = 0
    for payload in payloads:
        slug = payload.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        if league_prefixes and not any(slug.startswith(p) for p in league_prefixes):
            continue
        ev = PolymarketEvent.from_gamma(payload)
        if ev is None:
            # `from_gamma` already drops -more-markets variants, settled
            # markets, etc. Silently skip those.
            continue
        starts = [t for m in ev.markets if (t := m.game_start_time) is not None]
        if starts:
            tipoff = min(starts)
            if not (horizon_start <= tipoff <= horizon_end):
                continue
        # Belt-and-suspenders tradability filter — `from_gamma` already
        # drops bid/ask=None markets, but a missing `yes_sub_title` would
        # leave the LLM unable to identify which side is which.
        live_markets = [
            m
            for m in ev.markets
            if m.yes_sub_title
            and m.yes_bid_dollars is not None
            and m.yes_ask_dollars is not None
        ]
        if not live_markets:
            continue
        # Blowout filter — drop events whose favorite is priced at or above
        # `max_implied_probability` on the YES mid. Mid is the cleanest
        # consensus implied prob; `max` across markets identifies the
        # favorite uniformly across binary head-to-heads and 3-way soccer
        # (max of home/draw/away).
        favorite_mid = max(
            (m.yes_bid_dollars + m.yes_ask_dollars) / 2.0  # type: ignore[operator]
            for m in live_markets
        )
        if favorite_mid >= max_implied_probability:
            dropped_blowout += 1
            continue
        # Floor filter — drop events whose favorite is priced BELOW
        # `min_favorite_probability` (i.e. too competitive for tail mode's
        # asymmetric-payoff strategy to fire on). None = disabled, which is
        # the default for confidence / ev modes. Tail mode sets ~0.75 so
        # the LLM doesn't burn tokens on 0.55/0.45 coin-flips that can't
        # produce Prime EV through the asymmetric-payoff path.
        if (
            min_favorite_probability is not None
            and favorite_mid < min_favorite_probability
        ):
            dropped_too_competitive += 1
            continue
        # Min OI floor — same posture as the Kalshi adapter: drop events
        # where ANY side has thin resting interest. Off by default
        # (MIN_OPEN_INTEREST_DOLLARS = 0.0) because Polymarket's CLOB book
        # builds up over hours and a non-zero default would mask fresh
        # markets that are tradable but pre-resting-interest.
        if min_open_interest_dollars > 0:
            min_oi = min(
                (m.open_interest_dollars or 0.0) for m in live_markets
            )
            if min_oi < min_open_interest_dollars:
                dropped_thin_oi += 1
                continue
        if len(live_markets) != len(ev.markets):
            ev = ev.model_copy(update={"markets": live_markets})
        kept.append(ev)

    floor_desc = (
        f" + floor (>={min_favorite_probability:.2f})"
        if min_favorite_probability is not None else ""
    )
    log.info(
        "kept %d gamma events after league + horizon + tradability + "
        "blowout (>=%.2f)%s + min_oi (>=$%.0f) filters; dropped blowouts=%d, "
        "too-competitive=%d, thin-oi=%d",
        len(kept),
        max_implied_probability,
        floor_desc,
        min_open_interest_dollars,
        dropped_blowout,
        dropped_too_competitive,
        dropped_thin_oi,
    )

    # Sort by earliest market tipoff ascending. Sort is unconditional
    # because gamma's listing is ordered by `endDate` (settlement
    # window), not tipoff — the two diverge on tours like ATP where
    # settlement lags match end by days. Events without any market
    # `game_start_time` sort last.
    _far_future = datetime.max.replace(tzinfo=UTC)

    def _tipoff(ev: PolymarketEvent) -> datetime:
        starts = [t for m in ev.markets if (t := m.game_start_time) is not None]
        return min(starts) if starts else _far_future

    kept.sort(key=_tipoff)
    return kept


async def fetch_gamma_events(
    http: httpx.AsyncClient,
    slugs: list[str],
    sem: asyncio.Semaphore,
) -> list[PolymarketEvent]:
    """Fetch specific Polymarket events by slug from gamma-api.

    Each slug fetched in parallel under `sem`; failures degrade per-slug
    — bogus slugs log a warning and drop out.

    No horizon filter is applied here, unlike `fetch_gamma_slate`. Slugs
    reach this function only via explicit `--slug` CLI args, so the user
    has already opted in to that specific event — second-guessing with a
    horizon check produces surprising drops when gamma's `endDate` is a
    settlement window rather than a tipoff.
    """
    if not slugs:
        return []

    async def _one(slug: str) -> PolymarketEvent | None:
        async with sem:
            payload = await fetch_gamma_event(http, slug)
            if payload is None:
                return None
            event = PolymarketEvent.from_gamma(payload)
            if event is None:
                log.warning(
                    "gamma slug=%s: no tradable moneyline markets after filter",
                    slug,
                )
            return event

    raw_events = await asyncio.gather(*(_one(s) for s in slugs))
    kept = [ev for ev in raw_events if ev is not None]

    log.info("fetched %d/%d events from gamma-api by slug", len(kept), len(slugs))
    return kept
