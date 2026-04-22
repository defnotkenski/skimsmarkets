from __future__ import annotations

import logging
from datetime import UTC, datetime
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
    [XAIAsyncClient, KalshiEvent],
    Awaitable[SpecialistReport],
]


def pick_team_a_market(event: KalshiEvent) -> KalshiMarket | None:
    """team_a = the Kalshi favorite (highest yes_implied_probability) among the event's markets.

    Returns None if the event has no markets (shouldn't happen after fetch filter) or no
    market has a valid implied probability (all illiquid).
    """
    scored = [
        (m.yes_implied_probability or -1.0, m)
        for m in event.markets
        if m.yes_sub_title
    ]
    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    top_prob, top_market = scored[0]
    if top_prob < 0:
        return None
    return top_market


def _game_state_hint(event: KalshiEvent) -> str:
    """Heuristic label for the event's timing relative to now.

    Uses expected_expiration_time (~shortly after game end). Windows are intentionally
    generous since sport durations vary (NBA ~2.5h, NFL ~3.5h, MLB ~3h, tennis up to ~5h).
    Specialists are told to VERIFY via live search when the hint says live.
    """
    m = next((mk for mk in event.markets if mk.expected_expiration_time), None)
    if m is None or m.expected_expiration_time is None:
        return "UNKNOWN (no expected_expiration_time on any market)"
    hours = (m.expected_expiration_time - datetime.now(tz=UTC)).total_seconds() / 3600
    if hours > 5:
        return f"PRE-GAME (settles in {hours:.1f}h; game has not started yet)"
    if hours > -0.5:
        return (
            f"LIKELY LIVE (settles in {hours:.1f}h — game is probably in progress OR "
            "recently tipped off; current score and time remaining dominate pre-game factors)"
        )
    return (
        f"RECENTLY ENDED (settled {-hours:.1f}h ago — market may be resolving or in OT; "
        "verify via live search)"
    )


def render_context(event: KalshiEvent) -> str:
    """Event-level user message handed to every specialist.

    Names team_a (Kalshi favorite) and team_b (the first non-team_a side) using the exact
    yes_sub_title strings so specialists echo them back verbatim.
    """
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

    settles = (
        team_a_market.expected_expiration_time.isoformat()
        if team_a_market and team_a_market.expected_expiration_time
        else "(unknown)"
    )

    return (
        f"Event: {event.event_ticker} — {event.title or '(no title)'}\n"
        f"Series: {event.series_ticker or '(unknown)'}\n"
        f"Sub-title: {event.sub_title or '(none)'}\n"
        f"Settles: {settles}\n"
        f"Game state hint: {_game_state_hint(event)}\n\n"
        f"team_a_name = {team_a_name}   (Kalshi's current favorite — if game is live "
        "this may reflect the scoreboard, not pre-game odds)\n"
        f"team_b_name = {team_b_name}\n\n"
        f"Markets in this event ({len(event.markets)}):\n"
        + "\n".join(market_lines)
        + "\n\n"
        + "Produce your report now, per the schema. "
        "Use the exact team_a_name / team_b_name strings above in your output."
    )


def _tools() -> list:
    """Fresh per-call list of server-side tools. Every specialist gets the full loadout."""
    return [web_search(), x_search(), code_execution()]


async def _run_specialist(
    xai: XAIAsyncClient,
    event: KalshiEvent,
    system_prompt: str,
    shape: type[_ReportT],
) -> _ReportT:
    chat = xai.chat.create(
        model=GROK_MODEL,
        agent_count=4,
        messages=[system(system_prompt)],
        tools=_tools(),
    )
    chat.append(user(render_context(event)))
    response, parsed = await chat.parse(shape)
    log.debug(
        "specialist=%s event=%s tokens in/out=%s/%s",
        shape.__name__,
        event.event_ticker,
        getattr(response.usage, "prompt_tokens", None),
        getattr(response.usage, "completion_tokens", None),
    )
    return parsed


async def run_statistics(xai: XAIAsyncClient, event: KalshiEvent) -> StatisticsReport:
    return await _run_specialist(xai, event, STATISTICS_SYSTEM, StatisticsReport)


async def run_injury(xai: XAIAsyncClient, event: KalshiEvent) -> InjuryReport:
    return await _run_specialist(xai, event, INJURY_SYSTEM, InjuryReport)


async def run_narrative(xai: XAIAsyncClient, event: KalshiEvent) -> NarrativeReport:
    return await _run_specialist(xai, event, NARRATIVE_SYSTEM, NarrativeReport)


async def run_market_pricing(xai: XAIAsyncClient, event: KalshiEvent) -> MarketReport:
    return await _run_specialist(xai, event, MARKET_PRICING_SYSTEM, MarketReport)


SPECIALISTS: dict[str, SpecialistFn] = {
    "statistics": run_statistics,
    "injury": run_injury,
    "narrative": run_narrative,
    "market_pricing": run_market_pricing,
}
