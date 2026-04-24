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
from skimsmarkets.unusual_whales.models import (
    UnusualWhalesContext,
    UWInsider,
    UWTrade,
)

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
            extras.append(f"liq=${m.liquidity_dollars:,.0f}")
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


def _fmt_money(v: float | None, prec: int = 0) -> str:
    if v is None:
        return "?"
    if prec == 0:
        return f"${v:,.0f}"
    return f"${v:,.{prec}f}"


def _trade_shares_and_usdc(t: UWTrade) -> tuple[float | None, float | None]:
    """Map a Polymarket fill to (shares, usdc_notional).

    A Polymarket trade pairs a share quantity with a USDC quantity; which leg
    landed on maker vs. taker depends on the maker's side (maker=seller means
    maker gave shares and received USDC; maker=buyer means maker gave USDC
    and received shares).
    """
    if t.maker_side == "seller":
        return t.maker_amount_filled, t.taker_amount_filled
    if t.maker_side == "buyer":
        return t.taker_amount_filled, t.maker_amount_filled
    return None, None


def _fmt_trade(t: UWTrade) -> str:
    shares, usdc = _trade_shares_and_usdc(t)
    if shares and usdc and shares > 0:
        implied = usdc / shares
        implied_s = f"implied=${implied:.3f}"
    else:
        implied_s = "implied=?"
    when = t.executed_at.isoformat() if t.executed_at else "?"
    # The active side (taker) is what reveals directional pressure: taker=buyer
    # means someone hit the ask; taker=seller means someone hit the bid.
    side = t.taker_side or "?"
    notional = _fmt_money(usdc, prec=2) if usdc is not None else "?"
    share_s = f"{shares:,.0f}" if shares is not None else "?"
    return (
        f"    {when}  taker={side}  shares={share_s}  notional={notional}  {implied_s}"
    )


def _fmt_insider(i: UWInsider) -> str:
    addr = i.user_address or "?"
    short = f"{addr[:8]}…{addr[-4:]}" if len(addr) >= 12 else addr
    price = f"${i.avg_price:.3f}" if i.avg_price is not None else "?"
    inv = _fmt_money(i.total_invested_usd)
    return f"    {short}  avg_price={price}  invested={inv}"


def _render_uw_block(ctx: UnusualWhalesContext) -> str:
    """Compact render of Unusual Whales flow signals for the market_context prompt.

    YES-side only — the NO-side flow is the mirror (inverted price, same trades)
    so we don't double-render it. The market_context specialist already reasons
    about the event from the YES lens; all bid/ask/implied fields in the main
    context block follow the same convention.
    """
    tags = ctx.tag_scores

    def _fmt_tag(name: str, val: float | None) -> str:
        return f"{name}=?" if val is None else f"{name}={val:.2f}"

    tag_line = " ".join(
        _fmt_tag(n, getattr(tags, n))
        for n in (
            "smart_money",
            "contrarian_whales",
            "insider_trades",
            "momentum",
            "closing_soon",
        )
    )

    lines: list[str] = []
    header = "Flow signals (Unusual Whales, YES side"
    if ctx.question:
        header += f" — {ctx.question!r}"
    header += "):"
    lines.append(header)

    score_parts: list[str] = []
    if ctx.unusual_score is not None:
        score_parts.append(f"unusual_score={ctx.unusual_score:.2f}")
    if ctx.volume is not None:
        score_parts.append(f"volume={_fmt_money(ctx.volume)}")
    if score_parts:
        lines.append("  " + "  ".join(score_parts))

    lines.append(f"  tag weights: {tag_line}")

    if ctx.mci is not None and (ctx.mci.value is not None or ctx.mci.delta is not None):
        mci_parts: list[str] = []
        if ctx.mci.value is not None:
            mci_parts.append(f"value={ctx.mci.value:.3f}")
        if ctx.mci.delta is not None:
            mci_parts.append(f"delta={ctx.mci.delta:+.3f}")
        lines.append(f"  MCI: {' '.join(mci_parts)}")

    liq = ctx.liquidity
    if liq is not None:
        liq_parts: list[str] = []
        if liq.best_bid is not None and liq.best_ask is not None:
            liq_parts.append(f"best_bid/ask=${liq.best_bid:.3f}/${liq.best_ask:.3f}")
            if liq.spread is not None:
                liq_parts.append(f"spread={int(round(liq.spread * 10000))}bps")
        if liq.total_liquidity is not None:
            liq_parts.append(f"total_liq={_fmt_money(liq.total_liquidity)}")
        if liq_parts:
            lines.append(f"  liquidity: {'  '.join(liq_parts)}")

    if ctx.smart_trades:
        lines.append(f"  recent smart-money trades ({len(ctx.smart_trades)}):")
        for t in ctx.smart_trades:
            lines.append(_fmt_trade(t))
    if ctx.contrarian_whale_trades:
        lines.append(f"  top contrarian whales ({len(ctx.contrarian_whale_trades)}):")
        for t in ctx.contrarian_whale_trades:
            lines.append(_fmt_trade(t))
    if ctx.insiders:
        lines.append(f"  top insiders ({len(ctx.insiders)}):")
        for i in ctx.insiders:
            lines.append(_fmt_insider(i))

    return "\n".join(lines)


async def _run_specialist(
    xai: XAIAsyncClient,
    event: PolymarketEvent,
    system_prompt: str,
    shape: type[_ReportT],
    *,
    extra_context: str | None = None,
) -> _ReportT:
    chat = xai.chat.create(
        model=GROK_MODEL,
        agent_count=4,
        messages=[system(system_prompt)],
        tools=_tools(),
    )
    user_msg = render_context(event)
    if extra_context:
        user_msg = f"{user_msg}\n\n{extra_context}"
    chat.append(user(user_msg))
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
    # UW flow data is market-context's domain only — statistics/injury/narrative
    # stay narrow to their own lens, and the director receives UW indirectly via
    # this specialist's report.
    extra = _render_uw_block(event.uw_context) if event.uw_context is not None else None
    return await _run_specialist(
        xai,
        event,
        MARKET_CONTEXT_SYSTEM,
        MarketContextReport,
        extra_context=extra,
    )


SPECIALISTS: dict[str, SpecialistFn] = {
    "statistics": run_statistics,
    "injury": run_injury,
    "narrative": run_narrative,
    "market_context": run_market_context,
}
