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


def _fmt_trade(t: UWTrade) -> str:
    """Render one fill as a single line.

    Hashdive UWTrade carries `size` (shares) and `price` (per-share USDC)
    directly; we derive USDC notional via `t.usdc_notional` (the model
    property does the size × price math). The active side (taker)
    reveals directional pressure: taker=buyer means someone hit the
    ask; taker=seller means someone hit the bid. The per-fill price
    is deliberately omitted from the rendered line — the director is
    blind to market price; flow direction and size are the signal,
    not the level.
    """
    when = t.executed_at.isoformat() if t.executed_at else "?"
    side = t.taker_side or "?"
    usdc = t.usdc_notional
    notional = _fmt_money(usdc, prec=2) if usdc is not None else "?"
    share_s = f"{t.size:,.0f}" if t.size is not None else "?"
    return f"    {when}  taker={side}  shares={share_s}  notional={notional}"


def _fmt_insider(i: UWInsider) -> str:
    addr = i.user_address or "?"
    short = f"{addr[:8]}…{addr[-4:]}" if len(addr) >= 12 else addr
    inv = _fmt_money(i.total_invested_usd)
    # avg_price omitted — the director is blind to the market price; the
    # signal here is that a known-insider wallet holds a position and its size.
    return f"    {short}  invested={inv}"


def render_uw_block(ctx: UnusualWhalesContext) -> str:
    """Compact render of Unusual Whales flow signals for LLM prompts.

    YES-side only — the NO-side flow is the mirror (same trades) so we don't
    double-render it. Market-price fields (per-fill implied price, insider
    avg price, best bid/ask, spread) are deliberately omitted: the director
    is blind to the market price, so UW reaches it as a pure flow signal —
    direction, size, and reputation tags only.
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
    # Header explicitly names the team this flow data is about — `outcome_label`
    # is the exact `outcomes[outcome_index]` value from the UW API, so the
    # director / reader doesn't have to infer which side the flow is on.
    side = ctx.outcome_label or "YES side"
    header = f"Flow signals (Unusual Whales, side='{side}'"
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
    if liq is not None and liq.total_liquidity is not None:
        # Only total resting liquidity is rendered — best_bid/ask and spread
        # are market-price microstructure and the director is blind to them.
        lines.append(f"  liquidity: total_liq={_fmt_money(liq.total_liquidity)}")

    if ctx.smart_trades:
        lines.append(f"  recent smart-money trades ({len(ctx.smart_trades)}):")
        for t in ctx.smart_trades:
            lines.append(_fmt_trade(t))
    if ctx.whale_trades:
        # Hashdive (2026-05) returns ALL whale-size fills, not just the
        # tag-classified "contrarian" subset the old API exposed. The
        # contrarian-direction reading lives in `tag_scores.contrarian_whales`
        # if a downstream reader wants to call out the directional split.
        lines.append(f"  recent whale trades ({len(ctx.whale_trades)}):")
        for t in ctx.whale_trades:
            lines.append(_fmt_trade(t))
    if ctx.insiders:
        lines.append(f"  top insiders ({len(ctx.insiders)}):")
        for i in ctx.insiders:
            lines.append(_fmt_insider(i))

    return "\n".join(lines)
