from __future__ import annotations

import logging
from typing import Awaitable, Callable, TypeVar

from pydantic import BaseModel
from xai_sdk import AsyncClient as XAIAsyncClient
from xai_sdk.chat import system, user
from xai_sdk.tools import code_execution, web_search, x_search

from skimsmarkets.agents.prompts import (
    INJURY_SYSTEM,
    MARKET_PRICING_SYSTEM,
    NARRATIVE_SYSTEM,
    STATISTICS_SYSTEM,
)
from skimsmarkets.agents.schemas import (
    InjuryReport,
    MarketReport,
    NarrativeReport,
    SpecialistReport,
    StatisticsReport,
)
from skimsmarkets.kalshi.models import KalshiEvent, KalshiMarket

log = logging.getLogger(__name__)

GROK_MODEL = "grok-4.20-multi-agent-0309"

_ReportT = TypeVar("_ReportT", bound=BaseModel)

SpecialistFn = Callable[
    [XAIAsyncClient, KalshiEvent, KalshiMarket],
    Awaitable[SpecialistReport],
]


def render_context(event: KalshiEvent, market: KalshiMarket) -> str:
    """Shared user-message body handed to each specialist."""
    yes_mid = market.yes_implied_probability
    return (
        f"Kalshi event: {event.event_ticker} — {event.title or '(no title)'}\n"
        f"Series: {event.series_ticker or '(unknown)'}\n"
        f"Event sub-title: {event.sub_title or '(none)'}\n\n"
        f"Market ticker: {market.ticker}\n"
        f"Market title: {market.title or '(no title)'}\n"
        f"'Yes' resolves if: {market.yes_sub_title or '(see rules)'}\n"
        f"'No' resolves if: {market.no_sub_title or '(see rules)'}\n"
        f"Current yes bid/ask (dollars): {market.yes_bid_dollars} / {market.yes_ask_dollars}\n"
        f"Kalshi-implied probability (midpoint): "
        f"{f'{yes_mid:.3f}' if yes_mid is not None else 'unknown'}\n"
        f"24h volume: {market.volume_24h_fp}, total volume: {market.volume_fp}\n"
        f"Market closes: {market.close_time.isoformat() if market.close_time else '(unknown)'}\n\n"
        f"Rules (primary): {market.rules_primary or '(none)'}\n"
        f"Rules (secondary): {market.rules_secondary or '(none)'}\n\n"
        "Produce your report now, per the schema."
    )


def _tools() -> list:
    """Fresh per-call list of server-side tools. Every specialist gets the full loadout."""
    return [web_search(), x_search(), code_execution()]


async def _run_specialist(
    xai: XAIAsyncClient,
    event: KalshiEvent,
    market: KalshiMarket,
    system_prompt: str,
    shape: type[_ReportT],
) -> _ReportT:
    chat = xai.chat.create(
        model=GROK_MODEL,
        agent_count=4,
        messages=[system(system_prompt)],
        tools=_tools(),
    )
    chat.append(user(render_context(event, market)))
    response, parsed = await chat.parse(shape)
    log.debug(
        "specialist=%s market=%s tokens in/out=%s/%s",
        shape.__name__,
        market.ticker,
        getattr(response.usage, "prompt_tokens", None),
        getattr(response.usage, "completion_tokens", None),
    )
    return parsed


async def run_statistics(
    xai: XAIAsyncClient,
    event: KalshiEvent,
    market: KalshiMarket,
) -> StatisticsReport:
    return await _run_specialist(
        xai, event, market, STATISTICS_SYSTEM, StatisticsReport
    )


async def run_injury(
    xai: XAIAsyncClient,
    event: KalshiEvent,
    market: KalshiMarket,
) -> InjuryReport:
    return await _run_specialist(xai, event, market, INJURY_SYSTEM, InjuryReport)


async def run_narrative(
    xai: XAIAsyncClient,
    event: KalshiEvent,
    market: KalshiMarket,
) -> NarrativeReport:
    return await _run_specialist(xai, event, market, NARRATIVE_SYSTEM, NarrativeReport)


async def run_market_pricing(
    xai: XAIAsyncClient,
    event: KalshiEvent,
    market: KalshiMarket,
) -> MarketReport:
    return await _run_specialist(
        xai, event, market, MARKET_PRICING_SYSTEM, MarketReport
    )


SPECIALISTS: dict[str, SpecialistFn] = {
    "statistics": run_statistics,
    "injury": run_injury,
    "narrative": run_narrative,
    "market_pricing": run_market_pricing,
}
