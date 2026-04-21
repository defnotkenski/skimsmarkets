from __future__ import annotations

import logging
from typing import cast

from anthropic import AsyncAnthropic
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlockParam,
)

from skimsmarkets.agents.prompts import DIRECTOR_SYSTEM
from skimsmarkets.agents.schemas import (
    InjuryReport,
    MarketPrediction,
    MarketReport,
    NarrativeReport,
    SizedMarketPrediction,
    SpecialistReport,
    StatisticsReport,
)
from skimsmarkets.agents.sizing import wrap_with_sizing
from skimsmarkets.kalshi.models import KalshiEvent, KalshiMarket

log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-7"
MAX_OUTPUT_TOKENS = 2048


def _render_user_message(
    event: KalshiEvent,
    market: KalshiMarket,
    statistics: StatisticsReport,
    injury: InjuryReport,
    narrative: NarrativeReport,
    pricing: MarketReport,
) -> str:
    implied = market.yes_implied_probability
    return (
        f"Kalshi market: {market.ticker}\n"
        f"Market title: {market.title or '(untitled)'}\n"
        f"Event: {event.title or event.event_ticker} ({event.series_ticker or '?'})\n"
        f"Yes bid/ask (dollars): {market.yes_bid_dollars} / {market.yes_ask_dollars}\n"
        f"Kalshi implied probability: "
        f"{f'{implied:.4f}' if implied is not None else 'unknown'}\n"
        f"Closes: {market.close_time.isoformat() if market.close_time else '(unknown)'}\n\n"
        f"--- StatisticsReport ---\n{statistics.model_dump_json(indent=2)}\n\n"
        f"--- InjuryReport ---\n{injury.model_dump_json(indent=2)}\n\n"
        f"--- NarrativeReport ---\n{narrative.model_dump_json(indent=2)}\n\n"
        f"--- MarketReport ---\n{pricing.model_dump_json(indent=2)}\n\n"
        "Return a MarketPrediction per the schema."
    )


async def synthesize_prediction(
    anthropic: AsyncAnthropic,
    event: KalshiEvent,
    market: KalshiMarket,
    reports: dict[str, SpecialistReport],
) -> SizedMarketPrediction:
    """Synthesize four specialist reports into a MarketPrediction + Kelly sizing."""
    user_msg = _render_user_message(
        event=event,
        market=market,
        statistics=cast(StatisticsReport, reports["statistics"]),
        injury=cast(InjuryReport, reports["injury"]),
        narrative=cast(NarrativeReport, reports["narrative"]),
        pricing=cast(MarketReport, reports["market_pricing"]),
    )

    system_block = TextBlockParam(
        type="text",
        text=DIRECTOR_SYSTEM,
        cache_control=CacheControlEphemeralParam(type="ephemeral"),
    )
    user_message = MessageParam(role="user", content=user_msg)

    parsed = await anthropic.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=[system_block],
        messages=[user_message],
        output_format=MarketPrediction,
    )
    prediction = parsed.parsed_output
    if prediction is None:
        raise RuntimeError(
            f"Director returned no parsed output for market {market.ticker}; "
            f"stop_reason={parsed.stop_reason}"
        )
    log.debug(
        "director market=%s tokens in/out=%s/%s",
        market.ticker, parsed.usage.input_tokens, parsed.usage.output_tokens,
    )
    return wrap_with_sizing(prediction, market)
