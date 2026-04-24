"""Render `UnusualWhalesContext` as a compact prompt-friendly text block.

Kept out of the LLM agent modules so any future consumer (director, logs,
reports) can reuse the same rendering without reaching into `agents/`.
"""

from __future__ import annotations

from skimsmarkets.unusual_whales.models import (
    UnusualWhalesContext,
    UWInsider,
    UWTrade,
)


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


def render_uw_block(ctx: UnusualWhalesContext) -> str:
    """Compact render of Unusual Whales flow signals for LLM prompts.

    YES-side only — the NO-side flow is the mirror (inverted price, same trades)
    so we don't double-render it. All consumers (currently just the director)
    reason about the event from the YES lens; all bid/ask/implied fields in the
    main context block follow the same convention.
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
