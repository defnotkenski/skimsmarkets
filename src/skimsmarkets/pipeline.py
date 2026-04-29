from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic
from xai_sdk import AsyncClient as XAIAsyncClient

from skimsmarkets import config as cfg
from skimsmarkets.agents.director import synthesize_prediction
from skimsmarkets.agents.schemas import MarketPrediction, SpecialistReport
from skimsmarkets.agents.specialists import SPECIALISTS
from skimsmarkets.polymarket import PolymarketClient
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket
from skimsmarkets.unusual_whales import (
    GammaTokenResolver,
    UnusualWhalesClient,
    fetch_gamma_event,
    list_gamma_events,
)

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
    predictions: list[MarketPrediction] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    fetched_events: int = 0
    considered_events: int = 0


async def fetch_polymarket_slate(
    pm: PolymarketClient,
    league: str | None,
    horizon_hours: int,
    poly_sem: asyncio.Semaphore,
) -> list[PolymarketEvent]:
    """Fetch the Polymarket sports slate, refresh BBO, and return the tradable
    set.

    This is the single source of truth for "what's in today's slate" — both the
    --fetch-only display path and the full pipeline iterate exactly the events
    returned here, with no further filtering. Filters layered server→client:

    1. **Server-side time window.** The SDK's `events.list` accepts
       `startTimeMin` / `startTimeMax`, so we never parse or BBO games outside
       the window. `start_time_min` sits 6h in the past to catch overtime and
       long-tail endings that haven't settled yet; `start_time_max` is
       `now + horizon_hours` for the upcoming slate.
    2. **Market-type filter.** Keep only moneyline / drawable_outcome markets
       (head-to-head NO-side inversion is already baked in by
       `PolymarketEvent`'s validator). Futures, spreads, totals are dropped —
       predicting "will this team win" on a season-winner market poisons the
       output.
    3. **BBO refresh.** Live bid/ask from `markets.bbo(slug)` overwrites the
       events.list snapshot prices.
    4. **Tradability filter.** Drop markets without a `yes_sub_title` label
       (can't tell the LLM which side is which) or without bid/ask (no live
       prices to reason about). This catches ended/settled games naturally —
       once a market settles its BBO disappears. Drop the event entirely if no
       markets survive.

    Filtering tradability AFTER BBO refresh is load-bearing: the events.list
    snapshot is often missing prices that the live BBO does provide, so we'd
    drop too many events if we filtered before the refresh.
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

    type_filtered: list[PolymarketEvent] = []
    for ev in events:
        allowed_markets = [
            m for m in ev.markets if m.sports_market_type in _ALLOWED_MARKET_TYPES
        ]
        if not allowed_markets:
            continue
        # Rebuild the event with the filtered market list so downstream stages
        # don't have to re-filter. Pydantic model_copy is the cheapest way to
        # get an updated copy without re-validating every other field.
        type_filtered.append(ev.model_copy(update={"markets": allowed_markets}))

    log.info(
        "kept %d events after market-type filter (dropped %d)",
        len(type_filtered),
        len(events) - len(type_filtered),
    )

    await resolve_market_prices(pm, type_filtered, poly_sem)

    # Tradability filter: a market is tradable when it has a labeled side AND a
    # live bid/ask. After BBO refresh, settled/ended games and unpriced lines
    # fail this and get dropped here so neither --fetch-only nor the pipeline
    # has to re-filter downstream.
    tradable: list[PolymarketEvent] = []
    for ev in type_filtered:
        live_markets = [
            m
            for m in ev.markets
            if m.yes_sub_title
            and m.yes_bid_dollars is not None
            and m.yes_ask_dollars is not None
        ]
        if not live_markets:
            continue
        tradable.append(ev.model_copy(update={"markets": live_markets}))

    log.info(
        "kept %d events with live tradable markets (dropped %d)",
        len(tradable),
        len(type_filtered) - len(tradable),
    )
    return tradable


async def fetch_gamma_events(
    http: httpx.AsyncClient,
    slugs: list[str],
    horizon_hours: int,
    sem: asyncio.Semaphore,
) -> list[PolymarketEvent]:
    """Fetch specific offshore-Polymarket events by slug from gamma-api.

    Opt-in fallback for events that don't list on polymarket-us (mostly
    international soccer, niche sports). Each slug fetched in parallel under
    `sem`; failures degrade per-slug — bogus slugs log a warning and drop out.

    Applies the same horizon window as `fetch_polymarket_slate` (now-6h to
    now+horizon_hours) on each event's `game_start_time` so a stale CLI
    argument for a past game doesn't sneak into the slate.
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

    now = datetime.now(tz=UTC)
    horizon_start = now - timedelta(hours=6)
    horizon_end = now + timedelta(hours=horizon_hours)
    kept: list[PolymarketEvent] = []
    for ev in raw_events:
        if ev is None:
            continue
        # Use the earliest market `game_start_time` we have. Gamma populates
        # this from the event's `endDate` (= game time) inside `from_gamma`.
        starts = [t for m in ev.markets if (t := m.game_start_time) is not None]
        if starts:
            tipoff = min(starts)
            if not (horizon_start <= tipoff <= horizon_end):
                log.info(
                    "gamma slug=%s: tipoff %s outside horizon [%s, %s], dropping",
                    ev.slug,
                    tipoff.isoformat(),
                    horizon_start.isoformat(),
                    horizon_end.isoformat(),
                )
                continue
        kept.append(ev)

    log.info(
        "fetched %d/%d offshore events from gamma-api",
        len(kept),
        len(slugs),
    )
    return kept


async def fetch_gamma_league_slate(
    http: httpx.AsyncClient,
    prefixes: list[str],
    horizon_hours: int,
) -> list[PolymarketEvent]:
    """Fetch all near-term offshore-Polymarket events whose slug starts with
    any of `prefixes` (e.g. ['lib', 'ucl'] → Copa Libertadores + Champions
    League). Mirrors the polymarket-us `--league` prefix-match pattern, but
    matches on slug prefix rather than `seriesSlug` since gamma omits that
    field.

    One HTTP call lists up to 200 upcoming events ordered by soonest tipoff;
    we filter client-side. Pagination isn't worth the complexity here —
    single-league horizons rarely exceed 200 events. If a user wants more,
    they can add `--gamma-slug` for specific stragglers.
    """
    if not prefixes:
        return []

    payloads = await list_gamma_events(http)
    if not payloads:
        return []

    # Match against `<prefix>-` so `arg` doesn't accidentally swallow `argf-`
    # or any other longer prefix that happens to share the leading letters.
    match_prefixes = [f"{p}-" for p in prefixes]

    now = datetime.now(tz=UTC)
    horizon_start = now - timedelta(hours=6)
    horizon_end = now + timedelta(hours=horizon_hours)

    kept: list[PolymarketEvent] = []
    for payload in payloads:
        slug = payload.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        if not any(slug.startswith(p) for p in match_prefixes):
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
        kept.append(ev)

    log.info(
        "fetched %d offshore events matching prefixes %s",
        len(kept),
        prefixes,
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

    Offshore events (`venue == "offshore"`) are skipped — gamma-api populates
    bid/ask on the `/events` payload directly, and there's no `markets.bbo()`
    analog on the gamma side. Trying to call it via the polymarket-us SDK would
    404 because gamma slugs don't exist on the US venue.
    """
    us_events = [ev for ev in events if ev.venue == "us"]
    if not us_events:
        return
    unique_slugs: set[str] = {m.slug for ev in us_events for m in ev.markets}

    async def _one(slug: str) -> tuple[str, PolymarketMarket | None]:
        async with sem:
            return slug, await pm.get_bbo(slug)

    bbo_results = await asyncio.gather(*(_one(s) for s in sorted(unique_slugs)))
    # Explicit comprehension instead of `dict(...)` so the checker keeps the
    # PolymarketMarket|None value type through to the lookup site below.
    bbo_by_slug: dict[str, PolymarketMarket | None] = {
        slug: bbo for slug, bbo in bbo_results
    }

    for ev in us_events:
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
            ev.markets[i] = m.model_copy(
                update={
                    "yes_bid_dollars": new_bid,
                    "yes_ask_dollars": new_ask,
                    # last_trade is YES-directional — drop on the NO clone to avoid
                    # misleading the reader.
                    "last_trade_price_dollars": (
                        None
                        if m.is_no_side
                        else bbo.last_trade_price_dollars or m.last_trade_price_dollars
                    ),
                    "volume_dollars": bbo.volume_dollars or m.volume_dollars,
                    "liquidity_dollars": bbo.liquidity_dollars or m.liquidity_dollars,
                }
            )


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

    async def _one(s: str, evs: list[PolymarketEvent]) -> None:
        async with sem:
            token_ids = await resolver.resolve(s)
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


# Logs live at <repo-root>/logs/runs/<run_id>.jsonl. Resolved at module-load
# so behaviour doesn't drift with cwd. `parents[2]` walks
# src/skimsmarkets/pipeline.py → src/skimsmarkets → src → repo-root.
_LOG_ROOT = Path(__file__).resolve().parents[2] / "logs" / "runs"


def _persist_run(result: RunResult) -> None:
    """Write each prediction (with its entry decision) to a per-run JSONL.

    One file per run named `<run_id>.jsonl`. Best-effort: any I/O failure is
    logged and swallowed — persistence must never abort the run, matching the
    enrichment-stage posture.

    Why JSONL not parquet: lines are easy to tail, easy to grep, easy to feed
    into a future grading script that joins against gamma settlement after
    kickoff. Volume is tiny (one line per ranked event, ~30/day).
    """
    try:
        _LOG_ROOT.mkdir(parents=True, exist_ok=True)
        path = _LOG_ROOT / f"{result.run_id}.jsonl"
        logged_at = datetime.now(UTC).isoformat()
        with path.open("w") as f:
            for p in result.predictions:
                payload = {
                    "run_id": result.run_id,
                    "logged_at_utc": logged_at,
                    "event_id": p.event_id,
                    "event_title": p.event_title,
                    "market_slug": p.market_slug,
                    "venue": p.venue,
                    "predicted_winner": p.predicted_winner,
                    "predicted_yes_probability": p.predicted_yes_probability,
                    "polymarket_implied_probability": p.polymarket_implied_probability,
                    "confidence": p.confidence,
                    "headline": p.headline,
                }
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        log.info("persisted %d predictions to %s", len(result.predictions), path)
    except Exception as e:  # noqa: BLE001
        log.warning("run-log persistence failed: %s", e)


async def process_event(
    xai: XAIAsyncClient,
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    specialist_sem: asyncio.Semaphore,
    director_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> MarketPrediction | None:
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
    gamma_slugs: list[str] | None = None,
    gamma_leagues: list[str] | None = None,
    skip_us: bool = False,
) -> RunResult:
    """End-to-end: fetch the Polymarket sports slate inside the horizon, refresh
    each event's BBO, then run 4 specialists + director per event. Returns a
    leaderboard-ready `RunResult` sorted downstream by predicted probability.

    Offshore fallback (gamma-api):
    - `gamma_slugs`: explicit list of offshore event slugs to include.
    - `gamma_leagues`: list of slug prefixes (e.g. 'lib', 'ucl') to bulk-pull
      all near-term offshore events for that league. Independent of `--league`
      because US and offshore use different naming conventions for league codes.
    Both add events tagged `venue="offshore"` to the slate.

    `skip_us=True` bypasses the polymarket-us fetch entirely — only the
    offshore events from `gamma_slugs`/`gamma_leagues` will reach the slate.
    `league` is silently ignored under `skip_us=True` (it's a US-only filter).
    """
    config = cfg.Config.from_env()
    run_id = uuid.uuid4().hex[:8]
    result = RunResult(run_id=run_id)

    specialist_sem = asyncio.Semaphore(cfg.SPECIALIST_SEM)
    director_sem = asyncio.Semaphore(cfg.DIRECTOR_SEM)
    poly_sem = asyncio.Semaphore(cfg.POLYMARKET_FETCH_SEM)
    uw_sem = asyncio.Semaphore(cfg.UW_FETCH_SEM)
    gamma_sem = asyncio.Semaphore(cfg.GAMMA_FETCH_SEM)

    async with (
        PolymarketClient() as pm,
        UnusualWhalesClient(config.unusual_whales_api_key) as uw,
    ):
        # `fetch_polymarket_slate` is the single source of truth for the US
        # slate: it fetches, refreshes BBO, and applies the tradability filter
        # so the slate matches what --fetch-only displays. `skip_us` bypasses
        # it for offshore-only runs.
        if skip_us:
            events = []
        else:
            events = await fetch_polymarket_slate(pm, league, horizon_hours, poly_sem)
        # Offshore fallback: gamma events arrive with bid/ask already populated
        # and venue="offshore", so they bypass `resolve_market_prices`. Slugs
        # and leagues compose — both lists union into the slate.
        offshore: list[PolymarketEvent] = []
        if gamma_slugs:
            offshore += await fetch_gamma_events(
                uw.http, gamma_slugs, horizon_hours, gamma_sem
            )
        if gamma_leagues:
            offshore += await fetch_gamma_league_slate(
                uw.http, gamma_leagues, horizon_hours
            )
        if offshore:
            # Dedupe by event id so a slug supplied via both --gamma-slug and
            # --gamma-league only lands once. Preserve first-seen order.
            seen: set[str] = {ev.id for ev in events}
            for ev in offshore:
                if ev.id in seen:
                    continue
                seen.add(ev.id)
                events.append(ev)
        result.fetched_events = len(events)

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

    if result.predictions:
        _persist_run(result)
    return result
