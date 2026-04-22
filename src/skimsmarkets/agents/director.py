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
    EventPrediction,
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


def _render_event_context_block(event: KalshiEvent) -> str:
    lines = [
        f"Event: {event.event_ticker} — {event.title or '(untitled)'}",
        f"Series: {event.series_ticker or '?'}",
        f"Sub-title: {event.sub_title or '(none)'}",
        "Markets in this event:",
    ]
    for m in event.markets:
        implied = m.yes_implied_probability
        lines.append(
            f"  - {m.ticker}: yes='{m.yes_sub_title or '(no label)'}' "
            f"bid/ask=${m.yes_bid_dollars}/${m.yes_ask_dollars} "
            f"implied={f'{implied:.3f}' if implied is not None else 'unknown'}"
        )
    return "\n".join(lines)


def _render_user_message(
    event: KalshiEvent,
    statistics: StatisticsReport,
    injury: InjuryReport,
    narrative: NarrativeReport,
    pricing: MarketReport,
) -> str:
    return (
        _render_event_context_block(event)
        + "\n\n"
        + f"--- StatisticsReport ---\n{statistics.model_dump_json(indent=2)}\n\n"
        + f"--- InjuryReport ---\n{injury.model_dump_json(indent=2)}\n\n"
        + f"--- NarrativeReport ---\n{narrative.model_dump_json(indent=2)}\n\n"
        + f"--- MarketReport ---\n{pricing.model_dump_json(indent=2)}\n\n"
        + "Return an EventPrediction per the schema. "
        "Set predicted_winner to the exact yes_sub_title string of the expected-winning market."
    )


def _find_market_for_winner(event: KalshiEvent, winner_name: str) -> KalshiMarket | None:
    """Find the market whose yes_sub_title matches the director's predicted winner."""
    target = winner_name.strip().lower()
    for m in event.markets:
        if m.yes_sub_title and m.yes_sub_title.strip().lower() == target:
            return m
    return None


def _project_to_market_prediction(
    event: KalshiEvent,
    winner_market: KalshiMarket,
    event_pred: EventPrediction,
) -> MarketPrediction:
    """Build a per-market MarketPrediction by projecting the event prediction onto the
    winner's market (yes = predicted winner)."""
    kalshi_implied = winner_market.yes_implied_probability or 0.0
    edge_bps = int(round((event_pred.predicted_winner_probability - kalshi_implied) * 10000))
    recommendation = "buy_yes" if event_pred.recommendation == "buy_winner" else "pass"
    return MarketPrediction(
        market_ticker=winner_market.ticker,
        event_ticker=event.event_ticker,
        event_title=event.title,
        predicted_winner=event_pred.predicted_winner,
        predicted_yes_probability=event_pred.predicted_winner_probability,
        kalshi_implied_probability=kalshi_implied,
        edge_bps=edge_bps,
        recommendation=recommendation,  # type: ignore[arg-type]
        confidence=event_pred.confidence,
        reasoning=event_pred.reasoning,
        specialist_weights=event_pred.specialist_weights,
        disagreements_flagged=event_pred.disagreements_flagged,
    )


async def synthesize_prediction(
    anthropic: AsyncAnthropic,
    event: KalshiEvent,
    reports: dict[str, SpecialistReport],
) -> SizedMarketPrediction:
    """Synthesize four specialist reports into an event-level EventPrediction, then
    project onto the predicted winner's market and attach Kelly sizing."""
    user_msg = _render_user_message(
        event=event,
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
        output_format=EventPrediction,
    )
    event_pred = parsed.parsed_output
    if event_pred is None:
        raise RuntimeError(
            f"Director returned no parsed output for event {event.event_ticker}; "
            f"stop_reason={parsed.stop_reason}"
        )
    log.debug(
        "director event=%s tokens in/out=%s/%s",
        event.event_ticker, parsed.usage.input_tokens, parsed.usage.output_tokens,
    )

    winner_market = _find_market_for_winner(event, event_pred.predicted_winner)
    if winner_market is None:
        raise RuntimeError(
            f"Director's predicted_winner={event_pred.predicted_winner!r} did not match "
            f"any yes_sub_title in event {event.event_ticker}. "
            f"Known sides: {[m.yes_sub_title for m in event.markets]}"
        )

    prediction = _project_to_market_prediction(event, winner_market, event_pred)
    return wrap_with_sizing(prediction, winner_market)
