"""Compact Pydantic schema for Unusual Whales per-asset context.

The Hashdive `/api/assets/{asset_id}/detail_agg` endpoint (2026-05 API,
served from `phx.unusualwhales.com/hashdive/`) returns a large payload
(50 trades × 3 arrays, full outcome pair with daily price series, MCI,
liquidity snapshot, insider positions). We squash it into a small,
render-friendly shape that:

- keeps only the signals the director can actually use (tag weights, MCI,
  liquidity, handful of top trades, top insiders);
- is JSON-serialisable and Pydantic-validated so we never render garbage into
  a prompt;
- is cheap to attach to every `PolymarketEvent` in memory.

Raw numeric fields from the UW API come back as JSON strings, so every
float-valued field parses via `_coerce_float` for tolerance.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_dt(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # UW uses trailing 'Z' for UTC; Python <3.11 doesn't accept it.
        return s.replace("Z", "+00:00")
    return v


class UWTagScores(BaseModel):
    """Weighted signal scores produced by UW's tag engine for this asset.

    Every field is the `weighted` value from `tag_scores[]` in the UW detail
    response, coerced to float. `None` means the tag was absent from the
    response (not "zero") so downstream rendering can hide empty rows.
    """

    model_config = ConfigDict(extra="ignore")

    smart_money: float | None = None
    contrarian_whales: float | None = None
    insider_trades: float | None = None
    momentum: float | None = None
    closing_soon: float | None = None


class UWLiquidity(BaseModel):
    """UW's snapshot of the asset's order book — complementary to our BBO."""

    model_config = ConfigDict(extra="ignore")

    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None
    spread: float | None = None
    total_liquidity: float | None = None

    @field_validator(
        "best_bid", "best_ask", "mid_price", "spread", "total_liquidity",
        mode="before",
    )
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _coerce_float(v)


class UWMci(BaseModel):
    """Market Confidence Index — proprietary UW signal. `delta` is recent change."""

    model_config = ConfigDict(extra="ignore")

    value: float | None = None
    delta: float | None = None

    @field_validator("value", "delta", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _coerce_float(v)


class UWTrade(BaseModel):
    """One on-chain fill from UW's trades / smart_trades / whale_trades.

    Hashdive shape (2026-05 API): `size` is the share quantity, `price`
    is the per-share USDC price; USDC notional is the product. Earlier
    `api.unusualwhales.com` shape paired maker/taker amounts in two
    legs (`maker_amount_filled`, `taker_amount_filled`) which we used
    to disambiguate via `maker_side`; the new shape is simpler.

    `taker_address` and `transaction_hash` carried for completeness
    (renderer doesn't surface them but downstream auditors might want
    chain-level traceability without re-fetching).
    """

    model_config = ConfigDict(extra="ignore")

    executed_at: datetime | None = None
    maker_side: str | None = None  # "buyer" or "seller" (from maker POV)
    taker_side: str | None = None
    # Hashdive trade shape: size (shares) + price (per-share USDC).
    size: float | None = None
    price: float | None = None
    fee: float | None = None
    taker_address: str | None = None
    transaction_hash: str | None = None

    @field_validator("executed_at", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> Any:
        return _coerce_dt(v)

    @field_validator("size", "price", "fee", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _coerce_float(v)

    @property
    def usdc_notional(self) -> float | None:
        """USDC value of the fill = size × price. None when either
        leg is missing.
        """
        if self.size is None or self.price is None:
            return None
        return self.size * self.price


class UWInsider(BaseModel):
    """One insider position snapshot from `/assets/{id}/detail_agg.insiders[]`.

    Hashdive adds four signal fields the prior API didn't ship:
    `pnl_percent` (running PnL on the position), `invested_zscore`
    (how outsized this wallet's investment is vs its own baseline),
    `n_positions` (how many markets the wallet has open right now),
    `days_since_first_trade` (recency of first fill on this market).
    `invested_zscore` is the most directly actionable — a z-score
    above ~2.0 means the wallet sized this bet unusually large
    relative to its own trading history.
    """

    model_config = ConfigDict(extra="ignore")

    user_address: str | None = None
    avg_price: float | None = None
    total_invested_usd: float | None = None
    first_trade_at: datetime | None = None
    # Hashdive-only signal fields. All optional; older responses or
    # markets without enough history will lack them.
    pnl_percent: float | None = None
    invested_zscore: float | None = None
    n_positions: int | None = None
    days_since_first_trade: int | None = None

    @field_validator("first_trade_at", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> Any:
        return _coerce_dt(v)

    @field_validator(
        "avg_price", "total_invested_usd", "pnl_percent", "invested_zscore",
        mode="before",
    )
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _coerce_float(v)

    # Notable-insider z-score threshold. The `invested_zscore` field
    # measures how outsized THIS wallet's investment is vs its OWN
    # trading-history baseline (not vs the whole population). A z of 2
    # is the standard "2 sigma outlier" cutoff — empirically this
    # catches wallets going materially out of their usual range while
    # filtering out routine "I always bet ~$1k on tennis matches"
    # wallets. Tuned to match the prompt's "notable" framing on tag
    # weights (`unusual_score >= 5.0`), with the same property: small
    # enough that real signal fires, large enough that baseline noise
    # doesn't.
    NOTABLE_ZSCORE_THRESHOLD: ClassVar[float] = 2.0

    def is_notable(self) -> bool:
        """True iff this insider sized their position unusually large
        vs their own trading-history baseline (invested_zscore >= 2.0).

        Captures the "outsized commitment" signal new in the Hashdive
        API — old `api.unusualwhales.com` couldn't compute this because
        the per-wallet baseline wasn't surfaced. Returns False when
        invested_zscore is missing (older records or wallets without
        enough history to compute a meaningful z).
        """
        return (
            self.invested_zscore is not None
            and self.invested_zscore >= self.NOTABLE_ZSCORE_THRESHOLD
        )


class UnusualWhalesContext(BaseModel):
    """Compact per-asset UW blob attached to a `PolymarketEvent`.

    Built from the YES-side asset_id. The NO-side context is mostly the
    mirror image (inverted price, same flow) so we don't duplicate it —
    the director reasons about the game from the YES lens, and the
    `is_no_side` rendering convention already handles direction.
    """

    model_config = ConfigDict(extra="ignore")

    asset_id: str
    question: str | None = None
    # The team / outcome name this asset_id represents — taken directly from
    # `outcomes[outcome_index]` in the UW detail response. Lets renderers and
    # the director identify which side flow is on without inferring from price.
    outcome_label: str | None = None
    unusual_score: float | None = None
    volume: float | None = None
    tag_scores: UWTagScores = Field(default_factory=UWTagScores)
    mci: UWMci | None = None
    liquidity: UWLiquidity | None = None
    smart_trades: list[UWTrade] = Field(default_factory=list)
    # Hashdive (2026-05) ships `whale_trades` — ALL whale-size fills,
    # not just the tag-classified "contrarian" subset the old API
    # exposed under `contrarian_whale_trades`. The contrarian angle
    # is still available in `tag_scores.contrarian_whales` (a weighted
    # score across recent whale flow); the raw trade list is now
    # broader so the renderer + director see whale fills going WITH
    # consensus too.
    whale_trades: list[UWTrade] = Field(default_factory=list)
    insiders: list[UWInsider] = Field(default_factory=list)

    @field_validator("unusual_score", "volume", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _coerce_float(v)

    def has_actionable_signal(self) -> bool:
        """True iff this context carries flow signal worth surfacing.

        Threshold logic matches the director prompt's "notable / material"
        bar: `unusual_score >= 5.0` is the prompt-defined cutoff for a
        notable composite signal. Below that, individual tag values are
        baseline noise (UW returns ~0.30 for tags that haven't fired
        rather than nulls), and pulling the UW block into the director's
        context just earns a verbose "effectively no flow" `uw_flow_note`
        while burning tokens.

        Three things make a context actionable:
          - Real fills (smart_trades / whale_trades) or top insiders.
            UW filters these server-side to wallets that actually
            triggered a tag pipeline, so presence alone is a signal
            regardless of composite score.
          - `unusual_score >= 5.0` — the prompt's "notable" threshold.
          - MCI `delta` of meaningful magnitude (`abs(delta) >= 5.0`).
            Static MCI values without a delta carry no directional info.

        Liquidity / volume / individual tag values / static MCI do NOT
        count: they either duplicate the per-market microstructure block
        we already render (liquidity, volume) or are sub-threshold noise
        (a momentum tag at 0.30 means "the price moved at all").
        """
        if self.smart_trades or self.whale_trades or self.insiders:
            return True
        if self.unusual_score is not None and self.unusual_score >= 5.0:
            return True
        if (
            self.mci is not None
            and self.mci.delta is not None
            and abs(self.mci.delta) >= 5.0
        ):
            return True
        return False


def tag_scores_from_list(tag_scores_raw: Any) -> UWTagScores:
    """Build `UWTagScores` from UW's `tag_scores: list[{tag, weighted, ...}]` shape.

    UW returns tag scores as a list of dicts (one entry per tag). We pluck
    the `weighted` value for each known tag name; unknown tags are ignored.
    """
    if not isinstance(tag_scores_raw, list):
        return UWTagScores()
    by_tag: dict[str, float | None] = {}
    for entry in tag_scores_raw:
        if not isinstance(entry, dict):
            continue
        tag = entry.get("tag")
        if not isinstance(tag, str):
            continue
        by_tag[tag] = _coerce_float(entry.get("weighted"))
    return UWTagScores(
        smart_money=by_tag.get("smart_money"),
        contrarian_whales=by_tag.get("contrarian_whales"),
        insider_trades=by_tag.get("insider_trades"),
        momentum=by_tag.get("momentum"),
        closing_soon=by_tag.get("closing_soon"),
    )
