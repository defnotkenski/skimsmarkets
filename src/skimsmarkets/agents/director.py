from __future__ import annotations

import logging
from typing import cast

from anthropic import AsyncAnthropic
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ThinkingConfigAdaptiveParam,
)

from skimsmarkets.agents.prompts import DIRECTOR_SYSTEM
from skimsmarkets.agents.schemas import (
    EventPrediction,
    InjuryReport,
    MarketContextReport,
    MarketPrediction,
    NarrativeReport,
    SpecialistReport,
    StatisticsReport,
)
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket
from skimsmarkets.unusual_whales import render_uw_block

log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-7"
# max_tokens is required by the Messages API and must cover thinking + response
# combined. The director emits a small Pydantic object, so 16k gives adaptive
# thinking plenty of room without tripping the SDK's 10-minute non-streaming
# guardrail (which fires when max_tokens × effort implies a longer call).
CLAUDE_MAX_OUTPUT_TOKENS = 16_000


def _render_event_context_block(event: PolymarketEvent) -> str:
    # Venue marker — important when prices come from gamma-api (offshore) vs
    # polymarket-us (US). Different liquidity pools, different participants;
    # the director should know where the numbers came from. Lives in the
    # per-event user message (NOT the cached system prompt) so the prompt
    # cache hit on DIRECTOR_SYSTEM is preserved.
    venue_line = (
        "Polymarket venue: OFFSHORE (gamma-api) — different liquidity pool from "
        "polymarket-us; offshore-only signals like UW flow apply directly here."
        if event.venue == "offshore"
        else "Polymarket venue: US (polymarket-us)"
    )
    lines = [
        f"Event: {event.id} — {event.title or '(untitled)'}",
        f"Series: {event.series_slug or '?'}",
        venue_line,
        event.game_state_line(),
        f"Tradable sides ({len(event.markets)}):",
    ]
    for m in event.markets:
        implied = m.yes_implied_probability
        bid = f"${m.yes_bid_dollars:.3f}" if m.yes_bid_dollars is not None else "?"
        ask = f"${m.yes_ask_dollars:.3f}" if m.yes_ask_dollars is not None else "?"
        implied_str = f"{implied:.3f}" if implied is not None else "unknown"
        side_tag = " [NO side, inverted]" if m.is_no_side else ""
        lines.append(
            f"  - slug={m.slug}{side_tag} yes='{m.yes_sub_title or '(no label)'}' "
            f"bid/ask={bid}/{ask} implied={implied_str}"
        )
    # Unusual Whales flow signals reach the director as raw background data —
    # alongside bid/ask — rather than through any specialist's opinion. The
    # block is only appended when UW had coverage for this event's slug; its
    # absence is normal (most non-NBA/NFL/big-soccer events won't match).
    if event.uw_context is not None:
        lines.append("")
        lines.append(render_uw_block(event.uw_context))
    return "\n".join(lines)


def _render_user_message(
    event: PolymarketEvent,
    statistics: StatisticsReport,
    injury: InjuryReport,
    narrative: NarrativeReport,
    market_context: MarketContextReport,
) -> str:
    return (
        _render_event_context_block(event)
        + "\n\n"
        + f"--- StatisticsReport ---\n{statistics.model_dump_json(indent=2)}\n\n"
        + f"--- InjuryReport ---\n{injury.model_dump_json(indent=2)}\n\n"
        + f"--- NarrativeReport ---\n{narrative.model_dump_json(indent=2)}\n\n"
        + f"--- MarketContextReport ---\n{market_context.model_dump_json(indent=2)}\n\n"
        + "Return an EventPrediction per the schema. "
        "Set predicted_winner to the exact yes_sub_title string of the side you expect to win."
    )


def _find_market_for_winner(
    event: PolymarketEvent, winner_name: str
) -> PolymarketMarket | None:
    """Find the Polymarket side whose yes_sub_title matches the director's predicted winner."""
    target = winner_name.strip().lower()
    for m in event.markets:
        if m.yes_sub_title and m.yes_sub_title.strip().lower() == target:
            return m
    return None


def _project_to_market_prediction(
    event: PolymarketEvent,
    winner_market: PolymarketMarket,
    event_pred: EventPrediction,
) -> MarketPrediction:
    """Project the event-level prediction onto the winning side's Polymarket
    market so reporting has a single self-contained record.
    """
    return MarketPrediction(
        market_slug=winner_market.slug,
        event_id=event.id,
        event_title=event.title,
        venue=event.venue,
        predicted_winner=event_pred.predicted_winner,
        predicted_yes_probability=event_pred.predicted_winner_probability,
        polymarket_implied_probability=winner_market.yes_implied_probability,
        confidence=event_pred.confidence,
        headline=event_pred.headline,
        reasoning=event_pred.reasoning,
        specialist_weights=event_pred.specialist_weights,
        disagreements_flagged=event_pred.disagreements_flagged,
        uw_flow_note=event_pred.uw_flow_note,
    )


async def synthesize_prediction(
    anthropic: AsyncAnthropic,
    event: PolymarketEvent,
    reports: dict[str, SpecialistReport],
) -> MarketPrediction:
    """Synthesize four specialist reports into an event-level EventPrediction,
    then project onto the predicted winner's market.
    """
    user_msg = _render_user_message(
        event=event,
        statistics=cast(StatisticsReport, reports["statistics"]),
        injury=cast(InjuryReport, reports["injury"]),
        narrative=cast(NarrativeReport, reports["narrative"]),
        market_context=cast(MarketContextReport, reports["market_context"]),
    )

    system_block = TextBlockParam(
        type="text",
        text=DIRECTOR_SYSTEM,
        cache_control=CacheControlEphemeralParam(type="ephemeral"),
    )
    user_message = MessageParam(role="user", content=user_msg)

    parsed = await anthropic.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
        system=[system_block],
        messages=[user_message],
        output_format=EventPrediction,
        # Opus 4.7 only supports adaptive thinking. `effort` is NOT inside the
        # thinking dict — it's a sibling field under `output_config`. "max" lets
        # the model spend unconstrained reasoning budget per event.
        thinking=ThinkingConfigAdaptiveParam(type="adaptive"),
        output_config=OutputConfigParam(effort="max"),
    )
    event_pred = parsed.parsed_output
    if event_pred is None:
        raise RuntimeError(
            f"Director returned no parsed output for event {event.id}; "
            f"stop_reason={parsed.stop_reason}"
        )
    log.debug(
        "director event=%s tokens in/out=%s/%s",
        event.id,
        parsed.usage.input_tokens,
        parsed.usage.output_tokens,
    )

    winner_market = _find_market_for_winner(event, event_pred.predicted_winner)
    if winner_market is None:
        raise RuntimeError(
            f"Director's predicted_winner={event_pred.predicted_winner!r} did not match "
            f"any yes_sub_title in event {event.id}. "
            f"Known sides: {[m.yes_sub_title for m in event.markets]}"
        )

    return _project_to_market_prediction(event, winner_market, event_pred)
