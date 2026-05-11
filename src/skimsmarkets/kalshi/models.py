"""Pydantic models for the Kalshi v2 trade API.

Mirrors the read-only fields we consume from `/events` + nested
`/markets`, plus request/response shapes for `/portfolio/orders`.
Kalshi rotates field names without a versioning policy; `extra="ignore"`
and permissive coercion keeps us forward-compatible.

Money fields arrive as strings in the public read paths (`"0.1700"`)
but as integer cents in the order paths. We preserve both conventions
verbatim — the matcher reads dollar floats; the order code writes cents.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class KalshiMarket(BaseModel):
    """One Kalshi binary market — the YES side of a two-market event.

    For tennis match events (`KX{ATP|WTA}MATCH`), each event holds two
    mutually-exclusive markets, one per player as the YES side.
    `yes_sub_title` is the full player name (first + last); the
    matcher resolves the predicted winner to one of these by
    last-token substring.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str
    event_ticker: str | None = None
    yes_sub_title: str | None = None
    no_sub_title: str | None = None
    # Public-read price fields ship as strings ("0.1700"); coerce to
    # float so callers can do arithmetic without unwrapping. Missing /
    # unparseable → None, which the trader treats as "no ask, skip".
    yes_ask_dollars: float | None = None
    yes_bid_dollars: float | None = None
    status: str | None = None

    @field_validator("yes_ask_dollars", "yes_bid_dollars", mode="before")
    @classmethod
    def _coerce_dollars(cls, v: Any) -> float | None:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


class KalshiEvent(BaseModel):
    """One Kalshi event — a wrapper around mutually-exclusive markets.

    For tennis match-level series, `title` is `"{LastA} vs {LastB}"` —
    no tournament prefix, no first names. `sub_title` appends the date
    (e.g. `"(May 11)"`). The matcher hits `title` for both last names
    and uses `sub_title` to disambiguate same-day repeats.
    """

    model_config = ConfigDict(extra="ignore")

    event_ticker: str
    series_ticker: str | None = None
    title: str | None = None
    sub_title: str | None = None
    mutually_exclusive: bool | None = None
    markets: list[KalshiMarket] = Field(default_factory=list)


class OrderRequest(BaseModel):
    """Wire-format payload for `POST /portfolio/orders`.

    v1 sends market buys with `time_in_force="immediate_or_cancel"` so
    thin tennis books partial-fill rather than rejecting outright (the
    FOK default would only fully fill or kill, which fails on most
    tennis matches). Worst-case spend is `count × yes_price` cents,
    so we size `count = bet_size_cents // yes_price` to keep that
    product under the budget without needing the separate
    `buy_max_cost` knob (which would silently force FOK behaviour
    per Kalshi's docs).

    `client_order_id` is the audit row's idempotency token; Kalshi
    dedupes retries by this when set, so a retry storm can't double-
    fill.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str
    action: Literal["buy"] = "buy"
    side: Literal["yes", "no"] = "yes"
    # `type` is intentionally NOT in the request — per Kalshi's spec
    # it's a response-only field. The server auto-classifies as
    # "limit" because we supply `yes_price`. IOC time-in-force gets
    # us partial-fill semantics without needing the "market" type.
    time_in_force: Literal[
        "immediate_or_cancel", "fill_or_kill", "good_till_canceled",
    ] = "immediate_or_cancel"
    count: int
    # Per-contract price ceiling — Kalshi requires either a price-cents
    # field (yes_price/no_price) or a price-dollars field. Integer
    # cents (1-99). Order fills any contracts at this price or better.
    yes_price: int
    client_order_id: str | None = None


class OrderResponse(BaseModel):
    """Kalshi's reply to `/portfolio/orders`.

    Mirrors the actual wire schema observed on a successful market buy
    (2026-05-10). Notable conventions:
      - Counts use `_fp` (floating-point) suffixes and arrive as
        strings like `"49.00"` — integer in practice today but the wire
        format allows fractional.
      - Money is in **dollars as strings** (`"22.540000"`), not cents.
        The trader converts to cents for the audit row.
      - `status="executed"` means fully filled synchronously;
        `"resting"` means waiting in the book; `"canceled"` means killed
        by IOC/FOK or manual cancel.
    """

    model_config = ConfigDict(extra="ignore")

    order_id: str | None = None
    status: str | None = None
    ticker: str | None = None

    # Counts — float so Kalshi's "49.00" string parses cleanly.
    initial_count_fp: float | None = None
    fill_count_fp: float | None = None
    remaining_count_fp: float | None = None

    # Contract cost (dollars). Maker = liquidity provider, taker =
    # liquidity consumer. A market buy on an existing resting offer
    # makes us the taker, so `taker_fill_cost_dollars` is where the
    # spend shows up.
    taker_fill_cost_dollars: float | None = None
    maker_fill_cost_dollars: float | None = None

    # Fees (dollars). Charged on top of the fill cost — same
    # taker/maker split. Kalshi's tennis contracts use a quadratic fee
    # schedule, so the per-contract fee scales with price.
    taker_fees_dollars: float | None = None
    maker_fees_dollars: float | None = None

    # Prices we sent, echoed back.
    yes_price_dollars: float | None = None
    no_price_dollars: float | None = None
