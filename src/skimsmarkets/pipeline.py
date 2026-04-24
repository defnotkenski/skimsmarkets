from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from anthropic import AsyncAnthropic
from xai_sdk import AsyncClient as XAIAsyncClient

from skimsmarkets import config as cfg
from skimsmarkets.agents.director import synthesize_prediction
from skimsmarkets.agents.schemas import SizedMarketPrediction, SpecialistReport
from skimsmarkets.agents.specialists import SPECIALISTS
from skimsmarkets.polymarket import PolymarketClient
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket
from skimsmarkets.unusual_whales import GammaTokenResolver, UnusualWhalesClient

log = logging.getLogger(__name__)

# Only include sides that look like "which team/outcome wins this specific game" —
# drop futures, spreads, totals, MVP markets. `None` is kept defensively in case
# the SDK hasn't populated `sportsMarketType` for a given market.
_ALLOWED_MARKET_TYPES: frozenset[str | None] = frozenset(
    {"moneyline", "drawable_outcome", None}
)


@dataclass
class ErrorRecord:
    event_id: str
    stage: str  # "specialist:<name>" or "director"
    error: str


@dataclass
class RunResult:
    run_id: str
    predictions: list[SizedMarketPrediction] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    fetched_events: int = 0
    considered_events: int = 0


async def fetch_polymarket_slate(
    pm: PolymarketClient,
    league: str | None,
    horizon_hours: int,
) -> list[PolymarketEvent]:
    """Fetch the Polymarket sports slate inside the time horizon.

    The SDK's `events.list` accepts `startTimeMin` / `startTimeMax`, so the
    horizon filter is pushed server-side — we never pay to parse or BBO games
    outside the window. `start_time_min` sits 6h in the past to cover overtime
    and long-tail game endings that haven't settled yet. `start_time_max` is
    `now + horizon_hours` for the upcoming slate.

    Each returned event is trimmed to its moneyline / drawable_outcome markets
    (head-to-head NO-side inversion is already baked in by `PolymarketEvent`'s
    validator). Events with no allowed markets are dropped entirely — futures,
    spreads, and totals shouldn't hit the specialist pipeline.
    """
    now = datetime.now(tz=UTC)
    start_time_min = now - timedelta(hours=6)
    start_time_max = now + timedelta(hours=horizon_hours)

    events = await pm.list_sports_events(
        series_prefix=league,
        start_time_min=start_time_min,
        start_time_max=start_time_max,
    )
    log.info(
        "fetched %d polymarket events (league=%s, horizon=%sh)",
        len(events),
        league or "all",
        horizon_hours,
    )

    kept: list[PolymarketEvent] = []
    for ev in events:
        allowed_markets = [
            m for m in ev.markets if m.sports_market_type in _ALLOWED_MARKET_TYPES
        ]
        if not allowed_markets:
            continue
        # Rebuild the event with the filtered market list so downstream stages
        # don't have to re-filter. Pydantic model_copy is the cheapest way to
        # get an updated copy without re-validating every other field.
        kept.append(ev.model_copy(update={"markets": allowed_markets}))

    log.info(
        "kept %d events with moneyline/drawable_outcome markets (dropped %d)",
        len(kept),
        len(events) - len(kept),
    )
    return kept


async def resolve_market_prices(
    pm: PolymarketClient,
    events: list[PolymarketEvent],
    sem: asyncio.Semaphore,
) -> None:
    """Refresh each event's market prices with live BBO, mutating in place.

    Head-to-head events carry two records per slug after NO-side expansion
    (YES + inverted NO). A single `get_bbo(slug)` call covers both: the NO-side
    record's bid/ask is derived by inverting the YES book (`no_bid = 1 − yes_ask`,
    `no_ask = 1 − yes_bid`). Slugs with an already-populated snapshot keep those
    prices when the BBO call returns None — a fallback that matters in practice
    because BBO is the flakier of the two endpoints.
    """
    unique_slugs: set[str] = {m.slug for ev in events for m in ev.markets}

    async def _one(slug: str) -> tuple[str, PolymarketMarket | None]:
        async with sem:
            return slug, await pm.get_bbo(slug)

    bbo_results = await asyncio.gather(*(_one(s) for s in sorted(unique_slugs)))
    # Explicit comprehension instead of `dict(...)` so the checker keeps the
    # PolymarketMarket|None value type through to the lookup site below.
    bbo_by_slug: dict[str, PolymarketMarket | None] = {
        slug: bbo for slug, bbo in bbo_results
    }

    for ev in events:
        for i, m in enumerate(ev.markets):
            bbo = bbo_by_slug.get(m.slug)
            if bbo is None:
                continue
            yes_bid = bbo.yes_bid_dollars
            yes_ask = bbo.yes_ask_dollars
            if yes_bid is None and yes_ask is None:
                continue
            if m.is_no_side:
                new_bid = 1.0 - yes_ask if yes_ask is not None else None
                new_ask = 1.0 - yes_bid if yes_bid is not None else None
            else:
                new_bid = yes_bid
                new_ask = yes_ask
            ev.markets[i] = m.model_copy(update={
                "yes_bid_dollars": new_bid,
                "yes_ask_dollars": new_ask,
                # last_trade is YES-directional — drop on the NO clone to avoid
                # misleading the reader.
                "last_trade_price_dollars": (
                    None if m.is_no_side
                    else bbo.last_trade_price_dollars or m.last_trade_price_dollars
                ),
                "volume_dollars": bbo.volume_dollars or m.volume_dollars,
                "liquidity_dollars": bbo.liquidity_dollars or m.liquidity_dollars,
            })


async def resolve_unusual_whales(
    uw: UnusualWhalesClient,
    events: list[PolymarketEvent],
    sem: asyncio.Semaphore,
) -> None:
    """Fetch Unusual Whales flow context per event and attach to `event.uw_context`.

    For each event: resolve `event.slug` → YES-side ERC-1155 `asset_id` via
    Polymarket's gamma-api (the polymarket-us SDK and gamma-api are two distinct
    venues, but `PolymarketEvent.slug` is the gamma-api market slug verbatim —
    polymarket-us `market.slug` adds a category prefix like `aec-`, so the
    event-level slug is what we need), then GET UW's
    `/predictions/market/{asset_id}` for the compact flow snapshot. Failures at
    any step leave `uw_context=None` and don't abort the run — this is an
    enrichment stage, not a dependency.

    Events with no valid slug or no gamma-api match are silently skipped; UW
    only indexes the public/gamma Polymarket, so cross-venue matching is
    best-effort.
    """
    if not uw.enabled:
        return

    slugged = [(ev.slug, ev) for ev in events if ev.slug]
    if not slugged:
        return

    # Dedupe by slug — for our data model event.slug is unique per event, but
    # guard against future multi-event-per-slug shapes regardless.
    by_slug: dict[str, list[PolymarketEvent]] = {}
    for slug, ev in slugged:
        by_slug.setdefault(slug, []).append(ev)

    resolver = GammaTokenResolver(uw.http)

    async def _one(slug: str, evs: list[PolymarketEvent]) -> None:
        async with sem:
            token_ids = await resolver.resolve(slug)
            if token_ids is None:
                return
            yes_asset_id, _no_asset_id = token_ids
            ctx = await uw.get_market_detail(yes_asset_id)
            if ctx is None:
                return
            for e in evs:
                e.uw_context = ctx

    await asyncio.gather(*(_one(s, evs) for s, evs in by_slug.items()))
    attached = sum(1 for ev in events if ev.uw_context is not None)
    log.info("attached unusual-whales context to %d/%d events", attached, len(events))


async def _run_specialists(
    xai: XAIAsyncClient,
    event: PolymarketEvent,
    sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> dict[str, SpecialistReport] | None:
    """Run all specialists for one event. Return None if any specialist failed
    (the event then skips director)."""

    async def _one(specialist: str) -> tuple[str, "SpecialistReport | Exception"]:
        async with sem:
            try:
                return specialist, await SPECIALISTS[specialist](xai, event)
            except Exception as e:  # noqa: BLE001
                return specialist, e

    results = await asyncio.gather(*(_one(n) for n in SPECIALISTS))
    reports: dict[str, SpecialistReport] = {}
    failed = False
    for name, result in results:
        if isinstance(result, Exception):
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage=f"specialist:{name}",
                    error=f"{type(result).__name__}: {result}",
                )
            )
            failed = True
        else:
            reports[name] = result
    if failed:
        return None
    return reports


async def process_event(
    xai: XAIAsyncClient,
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    specialist_sem: asyncio.Semaphore,
    director_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> SizedMarketPrediction | None:
    log.info("processing event %s (%s)", event.id, event.title)
    reports = await _run_specialists(xai, event, specialist_sem, errors)
    if reports is None:
        return None

    async with director_sem:
        try:
            return await synthesize_prediction(anthropic, event, reports)
        except Exception as e:  # noqa: BLE001
            errors.append(
                ErrorRecord(
                    event_id=event.id,
                    stage="director",
                    error=f"{type(e).__name__}: {e}",
                )
            )
            return None


async def run_pipeline(
    *,
    league: str | None = None,
    dry_run: bool = False,
    horizon_hours: int = cfg.DEFAULT_HORIZON_HOURS,
) -> RunResult:
    """End-to-end: fetch the Polymarket sports slate inside the horizon, refresh
    each event's BBO, then run 4 specialists + director per event. Returns a
    leaderboard-ready `RunResult` sorted downstream by predicted probability.
    """
    config = cfg.Config.from_env()
    run_id = uuid.uuid4().hex[:8]
    result = RunResult(run_id=run_id)

    specialist_sem = asyncio.Semaphore(cfg.SPECIALIST_SEM)
    director_sem = asyncio.Semaphore(cfg.DIRECTOR_SEM)
    poly_sem = asyncio.Semaphore(cfg.POLYMARKET_FETCH_SEM)
    uw_sem = asyncio.Semaphore(cfg.UW_FETCH_SEM)

    async with (
        PolymarketClient() as pm,
        UnusualWhalesClient(config.unusual_whales_api_key) as uw,
    ):
        events = await fetch_polymarket_slate(pm, league, horizon_hours)
        result.fetched_events = len(events)

        await resolve_market_prices(pm, events, poly_sem)

        if dry_run:
            events = events[:1]
        result.considered_events = len(events)
        log.info("considering %d events (dry_run=%s)", len(events), dry_run)

        if not events:
            return result

        # UW enrichment runs after the slate is finalized (post dry-run trim)
        # so we don't waste API budget on events we're going to drop.
        await resolve_unusual_whales(uw, events, uw_sem)

        xai = XAIAsyncClient(api_key=config.xai_api_key)
        anthropic = AsyncAnthropic(api_key=config.anthropic_api_key)
        try:
            predictions = await asyncio.gather(
                *(
                    process_event(
                        xai,
                        anthropic,
                        e,
                        specialist_sem,
                        director_sem,
                        result.errors,
                    )
                    for e in events
                )
            )
        finally:
            await xai.close()

        for p in predictions:
            if p is not None:
                result.predictions.append(p)

    return result
