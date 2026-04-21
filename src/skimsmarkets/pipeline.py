from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

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
    market_ticker: str
    event_ticker: str
    stage: str  # "specialist:<name>" or "director"
    error: str


@dataclass
class RunResult:
    run_id: str
    predictions: list[SizedMarketPrediction] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    fetched_events: int = 0
    considered_markets: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "fetched_events": self.fetched_events,
            "considered_markets": self.considered_markets,
            "predictions": [p.model_dump() for p in self.predictions],
            "errors": [vars(e) for e in self.errors],
        }


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


async def _run_specialists(
    xai: XAIAsyncClient,
    event: KalshiEvent,
    market: KalshiMarket,
    sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> dict[str, SpecialistReport] | None:
    """Run all specialists for one market. Return None if any specialist failed
    (the market then skips director)."""

    async def _one(specialist: str) -> tuple[str, "SpecialistReport | Exception"]:
        async with sem:
            try:
                return specialist, await SPECIALISTS[specialist](xai, event, market)
            except Exception as e:  # noqa: BLE001
                return specialist, e

    results = await asyncio.gather(*(_one(n) for n in SPECIALISTS))
    reports: dict[str, SpecialistReport] = {}
    failed = False
    for name, result in results:
        if isinstance(result, Exception):
            errors.append(
                ErrorRecord(
                    market_ticker=market.ticker,
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


async def process_market(
    xai: XAIAsyncClient,
    anthropic: AsyncAnthropic,
    event: KalshiEvent,
    market: KalshiMarket,
    specialist_sem: asyncio.Semaphore,
    director_sem: asyncio.Semaphore,
    errors: list[ErrorRecord],
) -> SizedMarketPrediction | None:
    log.info("processing market %s (%s)", market.ticker, market.title)
    reports = await _run_specialists(xai, event, market, specialist_sem, errors)
    if reports is None:
        return None

    async with director_sem:
        try:
            return await synthesize_prediction(anthropic, event, market, reports)
        except Exception as e:  # noqa: BLE001
            errors.append(
                ErrorRecord(
                    market_ticker=market.ticker,
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
    """End-to-end: fetch live sports, fan out 4 specialists per market, synthesize with Opus."""
    config = cfg.Config.from_env()
    run_id = uuid.uuid4().hex[:8]
    result = RunResult(run_id=run_id)

    specialist_sem = asyncio.Semaphore(cfg.SPECIALIST_SEM)
    director_sem = asyncio.Semaphore(cfg.DIRECTOR_SEM)

    async with KalshiClient() as kalshi:
        events = await fetch_live_sports(kalshi, series_filter)
        result.fetched_events = len(events)
        log.info("fetched %d live events", len(events))

        # Flatten into (event, market) pairs filtered by close_time horizon.
        pairs: list[tuple[KalshiEvent, KalshiMarket]] = []
        for e in events:
            for m in e.markets:
                if not is_within_horizon(m, horizon_hours):
                    continue
                pairs.append((e, m))
        if dry_run:
            pairs = pairs[:1]
        result.considered_markets = len(pairs)
        log.info(
            "considering %d markets (horizon=%sh, dry_run=%s)",
            len(pairs),
            horizon_hours,
            dry_run,
        )

        if not pairs:
            return result

        xai = XAIAsyncClient(api_key=config.xai_api_key)
        anthropic = AsyncAnthropic(api_key=config.anthropic_api_key)
        try:
            predictions = await asyncio.gather(
                *(
                    process_market(
                        xai,
                        anthropic,
                        e,
                        m,
                        specialist_sem,
                        director_sem,
                        result.errors,
                    )
                    for e, m in pairs
                )
            )
        finally:
            await xai.close()

        for p in predictions:
            if p is not None:
                result.predictions.append(p)

    return result
