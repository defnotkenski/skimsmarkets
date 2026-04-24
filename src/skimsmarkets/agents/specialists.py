from __future__ import annotations

import logging
from typing import Awaitable, Callable, TypeVar

from pydantic import BaseModel
from xai_sdk import AsyncClient as XAIAsyncClient
from xai_sdk.chat import system, user
from xai_sdk.tools import code_execution, web_search, x_search

from skimsmarkets.agents.prompts import (
    INJURY_SYSTEM,
    MARKET_CONTEXT_SYSTEM,
    NARRATIVE_SYSTEM,
    STATISTICS_SYSTEM,
)
from skimsmarkets.agents.schemas import (
    InjuryReport,
    MarketContextReport,
    NarrativeReport,
    SpecialistReport,
    StatisticsReport,
)
from skimsmarkets.polymarket.models import PolymarketEvent, PolymarketMarket

log = logging.getLogger(__name__)

GROK_MODEL = "grok-4.20-multi-agent-0309"

_ReportT = TypeVar("_ReportT", bound=BaseModel)

SpecialistFn = Callable[
    [XAIAsyncClient, PolymarketEvent],
    Awaitable[SpecialistReport],
]


def pick_team_a_market(event: PolymarketEvent) -> PolymarketMarket | None:
    """team_a = the Polymarket favorite (highest yes_implied_probability) among the
    event's tradable sides.

    Returns None when no market has a label and a valid implied probability. Head-to-
    head events have two sides after Polymarket's NO-side expansion; 3-way soccer
    events have three. Either way, the favorite bubbles to the top of a simple sort.
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


def render_context(event: PolymarketEvent) -> str:
    """Event-level user message handed to every specialist.

    Names team_a (Polymarket favorite) and team_b (the first non-team_a side) using
    the exact yes_sub_title strings so specialists echo them back verbatim. Every
    tradable side is rendered with its bid/ask, implied, volume and liquidity so the
    market-context specialist sees the whole event board without another fetch.
    """
    team_a_market = pick_team_a_market(event)
    team_b_market = next(
        (m for m in event.markets if m is not team_a_market and m.yes_sub_title),
        None,
    )

    team_a_name = team_a_market.yes_sub_title if team_a_market else "(unknown)"
    team_b_name = team_b_market.yes_sub_title if team_b_market else "(unknown)"

    market_lines: list[str] = []
    for m in event.markets:
        implied = m.yes_implied_probability
        bid = f"${m.yes_bid_dollars:.3f}" if m.yes_bid_dollars is not None else "?"
        ask = f"${m.yes_ask_dollars:.3f}" if m.yes_ask_dollars is not None else "?"
        implied_str = f"{implied:.3f}" if implied is not None else "unknown"
        extras: list[str] = []
        if m.yes_bid_dollars is not None and m.yes_ask_dollars is not None:
            spread_bps = int(round((m.yes_ask_dollars - m.yes_bid_dollars) * 10000))
            extras.append(f"spread={spread_bps}bps")
        if m.volume_dollars is not None:
            extras.append(f"vol=${m.volume_dollars:,.0f}")
        if m.liquidity_dollars is not None:
            # Labeled "oi" (open interest) because polymarket-us doesn't
            # publish order-book liquidity as a dollar figure — what we have
            # is dollar open interest, not "how much can I trade right now."
            extras.append(f"oi=${m.liquidity_dollars:,.0f}")
        # [NO side, inverted] flags head-to-head markets where this side's prices
        # were derived by inverting the slug's YES book. Keep readers aware that
        # the numbers came from that flip, not a directly-quoted second market.
        side_tag = " [NO side, inverted]" if m.is_no_side else ""
        extras_str = f" {' '.join(extras)}" if extras else ""
        market_lines.append(
            f"  - slug={m.slug}{side_tag} yes='{m.yes_sub_title or '(no label)'}' "
            f"bid/ask={bid}/{ask} implied={implied_str}{extras_str}"
        )

    # Walrus-bind so the `is not None` check narrows `t` to `datetime` inside
    # the comprehension — without it, static checkers keep the element type as
    # `datetime | None` and flag `min(...)` / `.isoformat()` downstream.
    start_times = [
        t for m in event.markets if (t := m.game_start_time) is not None
    ]
    tipoff = min(start_times).isoformat() if start_times else "(unknown)"

    # Game-state line (PRE-MATCH / LIVE / ENDED) is rendered whenever Polymarket
    # provides it — making the state explicit beats having the LLM infer phase
    # from absent fields.
    state_line = event.game_state_line()

    return (
        f"Event: {event.id} — {event.title or '(no title)'}\n"
        f"Series: {event.series_slug or '(unknown)'}\n"
        f"Tipoff: {tipoff}\n\n"
        f"{state_line}\n\n"
        f"team_a_name = {team_a_name}   (the Polymarket favorite going into this event)\n"
        f"team_b_name = {team_b_name}\n\n"
        f"Tradable sides on Polymarket ({len(event.markets)}):\n"
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
    event: PolymarketEvent,
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
        event.id,
        getattr(response.usage, "prompt_tokens", None),
        getattr(response.usage, "completion_tokens", None),
    )
    return parsed


async def run_statistics(
    xai: XAIAsyncClient, event: PolymarketEvent
) -> StatisticsReport:
    return await _run_specialist(xai, event, STATISTICS_SYSTEM, StatisticsReport)


async def run_injury(xai: XAIAsyncClient, event: PolymarketEvent) -> InjuryReport:
    return await _run_specialist(xai, event, INJURY_SYSTEM, InjuryReport)


async def run_narrative(
    xai: XAIAsyncClient, event: PolymarketEvent
) -> NarrativeReport:
    return await _run_specialist(xai, event, NARRATIVE_SYSTEM, NarrativeReport)


async def run_market_context(
    xai: XAIAsyncClient, event: PolymarketEvent
) -> MarketContextReport:
    return await _run_specialist(xai, event, MARKET_CONTEXT_SYSTEM, MarketContextReport)


SPECIALISTS: dict[str, SpecialistFn] = {
    "statistics": run_statistics,
    "injury": run_injury,
    "narrative": run_narrative,
    "market_context": run_market_context,
}
