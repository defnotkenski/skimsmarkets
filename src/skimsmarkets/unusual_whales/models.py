"""Compact Pydantic schema for Unusual Whales per-asset context.

The UW `/api/predictions/market/{asset_id}` detail endpoint returns a large
payload (50 trades × 3 arrays, full outcome pair with daily price series,
etc). We squash it into a small, render-friendly shape that:

- keeps only the signals the market_context specialist can actually use
  (tag weights, MCI, liquidity, handful of top trades, top insiders);
- is JSON-serialisable and Pydantic-validated so we never render garbage into
  a prompt;
- is cheap to attach to every `PolymarketEvent` in memory.

Raw numeric fields from the UW API come back as JSON strings, so every
float-valued field parses via `_coerce_float` for tolerance.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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
    """One on-chain fill from UW's trades / smart_trades / contrarian_whale_trades.

    A Polymarket fill comes in two legs: shares quantity and USDC quantity.
    Which leg landed on maker vs taker depends on the side — we keep them
    both raw and let the renderer surface whichever is more useful.
    """

    model_config = ConfigDict(extra="ignore")

    executed_at: datetime | None = None
    maker_side: str | None = None  # "buyer" or "seller"
    taker_side: str | None = None
    maker_amount_filled: float | None = None
    taker_amount_filled: float | None = None

    @field_validator("executed_at", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> Any:
        return _coerce_dt(v)

    @field_validator("maker_amount_filled", "taker_amount_filled", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _coerce_float(v)


class UWInsider(BaseModel):
    """One insider position snapshot (from /market.insiders[] or top_insiders[])."""

    model_config = ConfigDict(extra="ignore")

    user_address: str | None = None
    avg_price: float | None = None
    total_invested_usd: float | None = None
    first_trade_at: datetime | None = None

    @field_validator("first_trade_at", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> Any:
        return _coerce_dt(v)

    @field_validator("avg_price", "total_invested_usd", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _coerce_float(v)


class UnusualWhalesContext(BaseModel):
    """Compact per-asset UW blob attached to a `PolymarketEvent`.

    Built from the YES-side asset_id. The NO-side context is mostly the
    mirror image (inverted price, same flow) so we don't duplicate it —
    the market_context specialist reasons about the game from the YES lens,
    and the `is_no_side` rendering convention already handles direction.
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
    contrarian_whale_trades: list[UWTrade] = Field(default_factory=list)
    insiders: list[UWInsider] = Field(default_factory=list)

    @field_validator("unusual_score", "volume", mode="before")
    @classmethod
    def _f(cls, v: Any) -> Any:
        return _coerce_float(v)

    def has_actionable_signal(self) -> bool:
        """True iff this context carries any flow signal beyond raw liquidity.

        UW's index covers more markets than its smart-money / insider / tag
        pipelines do — offshore (gamma) events in particular often resolve on
        the detail endpoint with a 200 but no enrichment: empty trade arrays,
        all tag weights null, volume=0, MCI absent. The only field UW still
        returns in that case is `liquidity`, which is just the gamma book
        and duplicates our main bid/ask block.

        Caller (pipeline) drops these signal-less contexts so the director
        doesn't see a UW block whose every field reads `?`. `liquidity`
        alone is intentionally NOT enough to count as a signal — it adds no
        information beyond what the per-market microstructure block carries.
        """
        if self.smart_trades or self.contrarian_whale_trades or self.insiders:
            return True
        tags = self.tag_scores
        if any(
            getattr(tags, n) is not None
            for n in (
                "smart_money",
                "contrarian_whales",
                "insider_trades",
                "momentum",
                "closing_soon",
            )
        ):
            return True
        if self.mci is not None and (
            self.mci.value is not None or self.mci.delta is not None
        ):
            return True
        if self.unusual_score is not None and self.unusual_score > 0:
            return True
        if self.volume is not None and self.volume > 0:
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
