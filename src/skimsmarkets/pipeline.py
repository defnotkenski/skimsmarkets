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
from skimsmarkets.kalshi import KalshiClient
from skimsmarkets.kalshi.models import KalshiEvent, KalshiMarket

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
    event: KalshiEvent,
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
                    event_ticker=event.event_ticker,
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
    event: KalshiEvent,
    specialist_sem: asyncio.Semaphore,
    director_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> SizedMarketPrediction | None:
    log.info("processing event %s (%s)", event.event_ticker, event.title)
    reports = await _run_specialists(xai, event, specialist_sem, errors)
    if reports is None:
        return None

    async with director_sem:
        try:
            return await synthesize_prediction(anthropic, event, reports)
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
) -> RunResult:
    """End-to-end: fetch live sports, run 4 specialists + director per event."""
    config = cfg.Config.from_env()
    run_id = uuid.uuid4().hex[:8]
    result = RunResult(run_id=run_id)

    specialist_sem = asyncio.Semaphore(cfg.SPECIALIST_SEM)
    director_sem = asyncio.Semaphore(cfg.DIRECTOR_SEM)

    async with KalshiClient() as kalshi:
        events = await fetch_live_sports(kalshi, series_filter)
        result.fetched_events = len(events)
        log.info("fetched %d live events", len(events))

        in_horizon = [e for e in events if event_within_horizon(e, horizon_hours)]
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

    return result
