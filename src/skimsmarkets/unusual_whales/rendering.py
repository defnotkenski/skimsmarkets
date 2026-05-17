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
    """One-line insider position summary for the director prompt.

    avg_price is deliberately omitted — the director is blind to market
    price; what reaches them is position size (`invested`), the Hashdive
    "outsized commitment" z-score, the wallet's running PnL on this
    market (`pnl`), the wallet's concurrent-positions count (`n_pos`,
    a diversification proxy), and recency-of-first-fill (`days_in`).
    Notable wallets (`is_notable()` → invested_zscore ≥ 2) get an arrow
    suffix so the director can spot them on the rendered slate without
    re-parsing the number.

    All new fields are conditionally included — older records or wallets
    without enough history to compute z / PnL omit those sub-fields
    rather than padding with "?" placeholders.
    """
    addr = i.user_address or "?"
    short = f"{addr[:8]}…{addr[-4:]}" if len(addr) >= 12 else addr
    parts = [f"invested={_fmt_money(i.total_invested_usd)}"]
    # Z-score: "outsized commitment vs own baseline". 2 decimals so the
    # director can compare across wallets without truncation noise.
    if i.invested_zscore is not None:
        parts.append(f"zscore={i.invested_zscore:+.2f}")
    # PnL %: positive = wallet is winning on this market so far. Format
    # as a signed percentage with one decimal to keep the line compact.
    if i.pnl_percent is not None:
        parts.append(f"pnl={i.pnl_percent * 100:+.1f}%")
    # Concurrent positions across all markets — a diversification
    # proxy. A wallet with n_pos=1 has all-in conviction; n_pos=20 is
    # spreading risk.
    if i.n_positions is not None:
        parts.append(f"n_pos={i.n_positions}")
    # Days since this wallet's first trade ON THIS MARKET. Recency
    # signal — fresh entries (≤7d) are more directional than old
    # accumulated positions.
    if i.days_since_first_trade is not None:
        parts.append(f"days_in={i.days_since_first_trade}")
    marker = "  ⚑ NOTABLE" if i.is_notable() else ""
    return f"    {short}  " + "  ".join(parts) + marker


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
        # Sort notable insiders to the top so the director sees the
        # outsized-commitment wallets first. Within each group keep the
        # API's native ordering (UW already ranks by invested USD
        # descending). Header tags how many of the surfaced insiders
        # are notable so the director can prioritize their reading.
        sorted_insiders = sorted(
            ctx.insiders, key=lambda x: (not x.is_notable(), 0),
        )
        n_notable = sum(1 for i in ctx.insiders if i.is_notable())
        if n_notable > 0:
            header = (
                f"  top insiders ({len(ctx.insiders)}, "
                f"{n_notable} notable ⚑):"
            )
        else:
            header = f"  top insiders ({len(ctx.insiders)}):"
        lines.append(header)
        for i in sorted_insiders:
            lines.append(_fmt_insider(i))

    return "\n".join(lines)
