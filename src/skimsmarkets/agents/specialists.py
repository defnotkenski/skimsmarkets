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
from skimsmarkets.enriched import EnrichedEvent
from skimsmarkets.kalshi.models import KalshiEvent, KalshiMarket

log = logging.getLogger(__name__)

GROK_MODEL = "grok-4.20-multi-agent-0309"

_ReportT = TypeVar("_ReportT", bound=BaseModel)

SpecialistFn = Callable[
    [XAIAsyncClient, EnrichedEvent],
    Awaitable[SpecialistReport],
]


def pick_team_a_market(event: KalshiEvent) -> KalshiMarket | None:
    """team_a = the Kalshi favorite (highest yes_implied_probability) among the event's markets.

    Returns None if the event has no markets (shouldn't happen after fetch filter) or no
    market has a valid implied probability (all illiquid).
    """
    scored = [
        (m.yes_implied_probability or -1.0, m) for m in event.markets if m.yes_sub_title
    ]
    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    top_prob, top_market = scored[0]
    if top_prob < 0:
        return None
    return top_market


def render_context(enriched: EnrichedEvent) -> str:
    """Event-level user message handed to every specialist.

    Names team_a (Kalshi favorite) and team_b (the first non-team_a side) using the exact
    yes_sub_title strings so specialists echo them back verbatim. When a Polymarket
    counterpart is matched for a side, its bid/ask is printed as a sub-line beneath the
    Kalshi market so the market-pricing specialist (and any other specialist that looks)
    sees both venues' prices.
    """
    event = enriched.kalshi
    team_a_market = pick_team_a_market(event)
    team_b_market = next(
        (m for m in event.markets if m is not team_a_market and m.yes_sub_title),
        None,
    )

    team_a_name = team_a_market.yes_sub_title if team_a_market else "(unknown)"
    team_b_name = team_b_market.yes_sub_title if team_b_market else "(unknown)"

    # Render every market so the market-pricing specialist sees the whole event board.
    market_lines: list[str] = []
    for m in event.markets:
        implied = m.yes_implied_probability
        market_lines.append(
            f"  - {m.ticker}: yes='{m.yes_sub_title or '(no label)'}' "
            f"bid/ask=${m.yes_bid_dollars}/${m.yes_ask_dollars} "
            f"implied={f'{implied:.3f}' if implied is not None else 'unknown'} "
            f"vol24h={m.volume_24h_fp}"
        )
        market_lines.append(_polymarket_sub_line(enriched, m))

    settles = (
        team_a_market.expected_expiration_time.isoformat()
        if team_a_market and team_a_market.expected_expiration_time
        else "(unknown)"
    )

    # Game-state line is rendered whenever Polymarket is matched, regardless
    # of phase (PRE-MATCH / LIVE / ENDED) — making the state explicit beats
    # having the LLM infer "pre-match" from the line's absence. When no
    # Polymarket counterpart exists, the per-market "polymarket: (not matched)"
    # sub-lines already tell the LLM there's no cross-venue data.
    state_line = (
        enriched.polymarket.game_state_line() if enriched.polymarket else None
    )
    live_block = f"{state_line}\n\n" if state_line else ""

    return (
        f"Event: {event.event_ticker} — {event.title or '(no title)'}\n"
        f"Series: {event.series_ticker or '(unknown)'}\n"
        f"Sub-title: {event.sub_title or '(none)'}\n"
        f"Settles: {settles}\n\n"
        f"{live_block}"
        f"team_a_name = {team_a_name}   (the Kalshi favorite going into this event)\n"
        f"team_b_name = {team_b_name}\n\n"
        f"Markets in this event ({len(event.markets)}):\n"
        + "\n".join(market_lines)
        + "\n\n"
        + "Produce your report now, per the schema. "
        "Use the exact team_a_name / team_b_name strings above in your output."
    )


def _polymarket_sub_line(enriched: EnrichedEvent, k_market: KalshiMarket) -> str:
    """Render the `polymarket:` sub-line beneath a Kalshi market line.

    Explicit absence beats silent omission — when no counterpart matched, print
    "(not matched)" so the LLM sees why Polymarket data isn't there rather than
    wondering whether it's a render bug.

    When a counterpart exists, also surface derived signals the specialists
    would otherwise compute themselves:
      - spread width in bps (tight = trustworthy; wide = thin/stale quote)
      - cross-venue delta in bps vs the Kalshi side implied probability
      - volume / liquidity when BBO populated them (per-venue depth)
    All of these are defensive — if the underlying field is None they're
    simply omitted, no placeholder clutter.
    """
    if not k_market.yes_sub_title:
        return "      polymarket: (no kalshi side label)"
    pm = enriched.poly_market_for(k_market.yes_sub_title)
    if pm is None:
        return "      polymarket: (not matched)"
    implied = pm.yes_implied_probability
    bid = f"${pm.yes_bid_dollars:.3f}" if pm.yes_bid_dollars is not None else "?"
    ask = f"${pm.yes_ask_dollars:.3f}" if pm.yes_ask_dollars is not None else "?"
    implied_str = f"{implied:.3f}" if implied is not None else "unknown"

    extras: list[str] = []
    # Spread width in bps. Kept integer-rounded; decimal precision isn't useful
    # at this granularity and noise in the display hurts scanability.
    if pm.yes_bid_dollars is not None and pm.yes_ask_dollars is not None:
        spread_bps = int(round((pm.yes_ask_dollars - pm.yes_bid_dollars) * 10000))
        extras.append(f"spread={spread_bps}bps")
    # Cross-venue delta: +N bps means PM is pricing this side HIGHER than
    # Kalshi. Only rendered when both venues have a live mid — otherwise the
    # subtraction is meaningless.
    k_implied = k_market.yes_implied_probability
    if implied is not None and k_implied is not None:
        delta_bps = int(round((implied - k_implied) * 10000))
        extras.append(f"Δ_vs_kalshi={delta_bps:+d}bps")
    if pm.volume_dollars is not None:
        extras.append(f"vol=${pm.volume_dollars:,.0f}")
    if pm.liquidity_dollars is not None:
        extras.append(f"liq=${pm.liquidity_dollars:,.0f}")

    extras_str = f" {' '.join(extras)}" if extras else ""
    # [NO side] flags head-to-head markets where this Kalshi side pairs to the
    # inverted direction of the same PM slug (common for NBA/NHL moneylines).
    side_tag = " [NO side, inverted]" if pm.is_no_side else ""
    return (
        f"      polymarket: slug={pm.slug}{side_tag} "
        f"bid/ask={bid}/{ask} implied={implied_str}{extras_str}"
    )


def _tools() -> list:
    """Fresh per-call list of server-side tools. Every specialist gets the full loadout."""
    return [web_search(), x_search(), code_execution()]


async def _run_specialist(
    xai: XAIAsyncClient,
    enriched: EnrichedEvent,
    system_prompt: str,
    shape: type[_ReportT],
) -> _ReportT:
    chat = xai.chat.create(
        model=GROK_MODEL,
        agent_count=4,
        messages=[system(system_prompt)],
        tools=_tools(),
    )
    chat.append(user(render_context(enriched)))
    response, parsed = await chat.parse(shape)
    log.debug(
        "specialist=%s event=%s tokens in/out=%s/%s",
        shape.__name__,
        enriched.kalshi.event_ticker,
        getattr(response.usage, "prompt_tokens", None),
        getattr(response.usage, "completion_tokens", None),
    )
    return parsed


async def run_statistics(
    xai: XAIAsyncClient, enriched: EnrichedEvent
) -> StatisticsReport:
    return await _run_specialist(xai, enriched, STATISTICS_SYSTEM, StatisticsReport)


async def run_injury(xai: XAIAsyncClient, enriched: EnrichedEvent) -> InjuryReport:
    return await _run_specialist(xai, enriched, INJURY_SYSTEM, InjuryReport)


async def run_narrative(
    xai: XAIAsyncClient, enriched: EnrichedEvent
) -> NarrativeReport:
    return await _run_specialist(xai, enriched, NARRATIVE_SYSTEM, NarrativeReport)


async def run_market_pricing(
    xai: XAIAsyncClient, enriched: EnrichedEvent
) -> MarketReport:
    return await _run_specialist(xai, enriched, MARKET_PRICING_SYSTEM, MarketReport)


SPECIALISTS: dict[str, SpecialistFn] = {
    "statistics": run_statistics,
    "injury": run_injury,
    "narrative": run_narrative,
    "market_pricing": run_market_pricing,
}
