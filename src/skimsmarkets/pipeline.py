from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from anthropic import AsyncAnthropic
from xai_sdk import AsyncClient as XAIAsyncClient

from skimsmarkets import config as cfg
from skimsmarkets.agents.director import synthesize_prediction
from skimsmarkets.agents.schemas import SizedMarketPrediction, SpecialistReport
from skimsmarkets.agents.specialists import SPECIALISTS
from skimsmarkets.enriched import EnrichedEvent
from skimsmarkets.kalshi import KalshiClient
from skimsmarkets.kalshi.models import KalshiEvent, KalshiMarket
from skimsmarkets.polymarket import PolymarketClient
from skimsmarkets.polymarket.matching import match_event
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket

log = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    event_ticker: str
    stage: str  # "specialist:<name>" or "director"
    error: str


@dataclass
class RunResult:
    run_id: str
    predictions: list[SizedMarketPrediction] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    fetched_events: int = 0
    considered_events: int = 0
    polymarket_matched: int = 0
    polymarket_unmatched: int = 0


async def fetch_live_sports(
    kalshi: KalshiClient,
    series_filter: str | None,
) -> list[KalshiEvent]:
    """Return live events. If series_filter is given, only that series; otherwise union of
    the seed list + dynamically-discovered sports series."""
    if series_filter:
        return await kalshi.list_open_events(series_ticker=series_filter)

    discovered = await kalshi.list_sports_series()
    series_tickers = {s.ticker for s in discovered} | set(cfg.SPORTS_SERIES_SEED)
    log.info("discovered %d sports series (seed+dynamic)", len(series_tickers))

    async def _fetch_one(ticker: str) -> list[KalshiEvent]:
        try:
            return await kalshi.list_open_events(series_ticker=ticker)
        except Exception as e:  # noqa: BLE001
            log.warning("series %s fetch failed: %s", ticker, e)
            return []

    batches = await asyncio.gather(*(_fetch_one(t) for t in sorted(series_tickers)))
    return [e for batch in batches for e in batch]


async def _fetch_polymarket_series(
    polymarket: PolymarketClient,
    series_prefixes: set[str],
) -> dict[str, list[PolymarketEvent]]:
    """Fetch polymarket events for each mapped series prefix concurrently.

    One failure per prefix is isolated (warning log + empty list) so a single
    flaky league doesn't block cross-venue enrichment for the others.
    """

    async def _one(prefix: str) -> tuple[str, list[PolymarketEvent]]:
        try:
            return prefix, await polymarket.list_sports_events(series_prefix=prefix)
        except Exception as e:  # noqa: BLE001
            log.warning("polymarket series=%s fetch failed: %s", prefix, e)
            return prefix, []

    pairs = await asyncio.gather(*(_one(lg) for lg in sorted(series_prefixes)))
    # Explicit typed construction so static checkers infer dict[str, list[PolymarketEvent]]
    # rather than falling back to asyncio.gather's variadic-Any return type.
    result: dict[str, list[PolymarketEvent]] = {
        prefix: events for prefix, events in pairs
    }
    return result


def _kalshi_series_to_poly_prefix(series_ticker: str | None) -> str | None:
    if series_ticker is None:
        return None
    return cfg.KALSHI_SERIES_TO_POLYMARKET_LEAGUE.get(series_ticker)


async def _resolve_polymarket_prices(
    polymarket: PolymarketClient,
    enriched: EnrichedEvent,
    sem: asyncio.Semaphore,
) -> None:
    """Populate `polymarket_price_by_kalshi_side` with authoritative prices.

    BBO is preferred because it's the live quote; when BBO is unavailable or
    returns an empty shape we fall back to the snapshot prices the events.list
    response already embedded. Head-to-head markets can map two Kalshi sides
    to a single slug (YES direction + NO direction), so BBO is deduped by slug
    and the NO-side result is derived by inverting the YES bid/ask. Mutates
    `enriched` in place.
    """
    # Index snapshots by (slug, is_no_side) so we can retrieve the right
    # direction's pre-parsed fallback when BBO is empty.
    snapshot_by_key: dict[tuple[str, bool], PolymarketMarket] = {
        (m.slug, m.is_no_side): m
        for m in (enriched.polymarket.markets if enriched.polymarket else [])
    }

    # Dedupe BBO calls: two Kalshi sides pairing to YES+NO of the same slug
    # should only trigger one network call.
    unique_slugs = {m.polymarket_market_slug for m in enriched.side_map.values()}

    async def _fetch_bbo(slug: str) -> tuple[str, PolymarketMarket | None]:
        async with sem:
            return slug, await polymarket.get_bbo(slug)

    bbo_results = await asyncio.gather(*(_fetch_bbo(s) for s in unique_slugs))
    bbo_by_slug: dict[str, PolymarketMarket | None] = dict(bbo_results)

    for kalshi_side, match in enriched.side_map.items():
        slug = match.polymarket_market_slug
        snap = snapshot_by_key.get((slug, match.is_no_side))
        bbo = bbo_by_slug.get(slug)
        if bbo is not None and (
            bbo.yes_bid_dollars is not None or bbo.yes_ask_dollars is not None
        ):
            if match.is_no_side:
                # BBO returns YES-direction bid/ask; invert for the NO side.
                yes_bid = bbo.yes_bid_dollars
                yes_ask = bbo.yes_ask_dollars
                inv_bid = 1.0 - yes_ask if yes_ask is not None else None
                inv_ask = 1.0 - yes_bid if yes_bid is not None else None
                resolved = bbo.model_copy(update={
                    "is_no_side": True,
                    "yes_sub_title": (snap.yes_sub_title if snap else None)
                    or match.kalshi_yes_sub_title,
                    "team_aliases": snap.team_aliases if snap else [],
                    "yes_bid_dollars": inv_bid,
                    "yes_ask_dollars": inv_ask,
                    "last_trade_price_dollars": None,
                })
            else:
                # Carry the snapshot's display metadata forward — BBO doesn't
                # echo team names back on its own.
                if snap and snap.yes_sub_title and bbo.yes_sub_title is None:
                    bbo = bbo.model_copy(update={"yes_sub_title": snap.yes_sub_title})
                resolved = bbo
            enriched.polymarket_price_by_kalshi_side[kalshi_side] = resolved
        elif snap is not None:
            enriched.polymarket_price_by_kalshi_side[kalshi_side] = snap


async def enrich_with_polymarket(
    polymarket: PolymarketClient | None,
    kalshi_events: list[KalshiEvent],
    *,
    poly_sem: asyncio.Semaphore,
) -> tuple[list[EnrichedEvent], int, int]:
    """Attach Polymarket overlay to each Kalshi event. Returns (enriched_list,
    matched_count, unmatched_count). When `polymarket` is None, every Kalshi
    event becomes a bare EnrichedEvent with no overlay — the pipeline then
    behaves identically to the Kalshi-only flow."""
    if polymarket is None:
        return [EnrichedEvent(kalshi=e) for e in kalshi_events], 0, len(kalshi_events)

    # Group Kalshi events by their mapped Polymarket series prefix so we fetch
    # each series' event pool exactly once.
    events_by_prefix: dict[str, list[KalshiEvent]] = defaultdict(list)
    skipped: list[KalshiEvent] = []
    for ev in kalshi_events:
        prefix = _kalshi_series_to_poly_prefix(ev.series_ticker)
        if prefix is None:
            skipped.append(ev)
            continue
        events_by_prefix[prefix].append(ev)

    if not events_by_prefix:
        log.info(
            "polymarket enrichment: 0/%d kalshi events have a mapped series prefix",
            len(kalshi_events),
        )
        return [EnrichedEvent(kalshi=e) for e in kalshi_events], 0, len(kalshi_events)

    pm_by_prefix = await _fetch_polymarket_series(polymarket, set(events_by_prefix))

    enriched_list: list[EnrichedEvent] = []
    matched = 0
    unmatched = 0

    for prefix, kalshi_batch in events_by_prefix.items():
        pool = pm_by_prefix.get(prefix, [])
        for ev in kalshi_batch:
            em = match_event(ev, pool)
            if em is None:
                enriched_list.append(EnrichedEvent(kalshi=ev))
                unmatched += 1
                continue
            enriched = EnrichedEvent(
                kalshi=ev,
                polymarket=em.polymarket_event,
                side_map=em.side_map,
            )
            try:
                await _resolve_polymarket_prices(polymarket, enriched, poly_sem)
            except Exception as e:  # noqa: BLE001
                # A BBO fan-out failure at this level is unusual — individual
                # failures are already caught inside get_bbo. Log and continue
                # with whatever sides did resolve.
                log.warning(
                    "polymarket BBO fan-out failed for %s: %s",
                    ev.event_ticker,
                    e,
                )
            matched += 1
            enriched_list.append(enriched)

    # Carry through the skipped events so the caller sees a 1:1 mapping with kalshi_events.
    for ev in skipped:
        enriched_list.append(EnrichedEvent(kalshi=ev))
        unmatched += 1

    log.info(
        "polymarket enrichment: matched %d / %d kalshi events "
        "(skipped %d with unmapped series)",
        matched,
        len(kalshi_events),
        len(skipped),
    )
    return enriched_list, matched, unmatched


def is_within_horizon(market: KalshiMarket, hours: int) -> bool:
    """True when the market's expected settlement is within `hours` from now.

    Uses `expected_expiration_time` (shortly after game end), not `close_time` (which
    is the outer market expiry, often weeks out even for tonight's games).
    """
    if market.expected_expiration_time is None:
        return False
    horizon = datetime.now(tz=UTC) + timedelta(hours=hours)
    return market.expected_expiration_time <= horizon


def event_within_horizon(event: KalshiEvent, hours: int) -> bool:
    """An event is in-horizon if ANY of its markets is."""
    return any(is_within_horizon(m, hours) for m in event.markets)


async def _run_specialists(
    xai: XAIAsyncClient,
    enriched: EnrichedEvent,
    sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> dict[str, SpecialistReport] | None:
    """Run all specialists for one event. Return None if any specialist failed
    (the event then skips director)."""

    async def _one(specialist: str) -> tuple[str, "SpecialistReport | Exception"]:
        async with sem:
            try:
                return specialist, await SPECIALISTS[specialist](xai, enriched)
            except Exception as e:  # noqa: BLE001
                return specialist, e

    results = await asyncio.gather(*(_one(n) for n in SPECIALISTS))
    reports: dict[str, SpecialistReport] = {}
    failed = False
    for name, result in results:
        if isinstance(result, Exception):
            errors.append(
                ErrorRecord(
                    event_ticker=enriched.kalshi.event_ticker,
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
    enriched: EnrichedEvent,
    specialist_sem: asyncio.Semaphore,
    director_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> SizedMarketPrediction | None:
    event = enriched.kalshi
    log.info(
        "processing event %s (%s) polymarket=%s",
        event.event_ticker,
        event.title,
        "matched" if enriched.has_polymarket else "none",
    )
    reports = await _run_specialists(xai, enriched, specialist_sem, errors)
    if reports is None:
        return None

    async with director_sem:
        try:
            return await synthesize_prediction(anthropic, enriched, reports)
        except Exception as e:  # noqa: BLE001
            errors.append(
                ErrorRecord(
                    event_ticker=event.event_ticker,
                    stage="director",
                    error=f"{type(e).__name__}: {e}",
                )
            )
            return None


async def run_pipeline(
    *,
    series_filter: str | None = None,
    dry_run: bool = False,
    horizon_hours: int = cfg.MAX_HOURS_UNTIL_EXPIRATION,
    polymarket_enabled: bool | None = None,
) -> RunResult:
    """End-to-end: fetch live sports (Kalshi + optional Polymarket overlay), run 4
    specialists + director per event.

    `polymarket_enabled=None` defers to the env-derived `cfg.polymarket_enabled()`.
    Pass False explicitly from the CLI's --no-polymarket flag.
    """
    config = cfg.Config.from_env()
    run_id = uuid.uuid4().hex[:8]
    result = RunResult(run_id=run_id)

    specialist_sem = asyncio.Semaphore(cfg.SPECIALIST_SEM)
    director_sem = asyncio.Semaphore(cfg.DIRECTOR_SEM)
    poly_sem = asyncio.Semaphore(cfg.POLYMARKET_FETCH_SEM)

    use_polymarket = (
        polymarket_enabled
        if polymarket_enabled is not None
        else cfg.polymarket_enabled()
    )

    async with KalshiClient() as kalshi:
        # Polymarket client lifetime is scoped to the kalshi block so both clients
        # are active for the whole fetch + enrichment phase. The exit stack nests
        # cleanly since both are plain async context managers.
        pm_client: PolymarketClient | None = (
            PolymarketClient() if use_polymarket else None
        )
        if pm_client is not None:
            await pm_client.__aenter__()

        try:
            kalshi_events = await fetch_live_sports(kalshi, series_filter)
            result.fetched_events = len(kalshi_events)
            log.info("fetched %d live kalshi events", len(kalshi_events))

            enriched_all, matched, unmatched = await enrich_with_polymarket(
                pm_client,
                kalshi_events,
                poly_sem=poly_sem,
            )
            result.polymarket_matched = matched
            result.polymarket_unmatched = unmatched

            in_horizon = [
                e for e in enriched_all if event_within_horizon(e.kalshi, horizon_hours)
            ]
            if dry_run:
                in_horizon = in_horizon[:1]
            result.considered_events = len(in_horizon)
            log.info(
                "considering %d events (horizon=%sh, dry_run=%s)",
                len(in_horizon),
                horizon_hours,
                dry_run,
            )

            if not in_horizon:
                return result

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
                        for e in in_horizon
                    )
                )
            finally:
                await xai.close()

            for p in predictions:
                if p is not None:
                    result.predictions.append(p)
        finally:
            if pm_client is not None:
                await pm_client.__aexit__(None, None, None)

    return result
