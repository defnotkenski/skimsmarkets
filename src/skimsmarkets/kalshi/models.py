"""Pydantic models for the Kalshi v2 trade API.

Mirrors the read-only fields we consume from `/events` + nested
`/markets`, plus request/response shapes for `/portfolio/orders`.
Kalshi rotates field names without a versioning policy; `extra="ignore"`
and permissive coercion keeps us forward-compatible.

Money fields arrive as strings in the public read paths (`"0.1700"`)
but as integer cents in the order paths. We preserve both conventions
verbatim — the matcher reads dollar floats; the order code writes cents.

The slate adapter (`kalshi/slate.py`) consumes the full set of
`/events?with_nested_markets=true` fields for ranker discovery, not
just the matcher's bid/ask subset. Most fields ship as strings on the
wire (`"942.00"`, `"0.4000"`) — `_coerce_dollars` handles them
identically since both qty and price strings parse to floats.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class KalshiProductMetadata(BaseModel):
    """Nested `/events.markets[].product_metadata` block.

    `competition` is the human-readable tournament label (e.g.
    `"ATP Rome"`) — the slate adapter slugifies it for
    `series_slug` so `tennis/identity.py` surface/tier lookups can
    fire. Other fields (e.g. `competition_scope`) are forward-compat
    only.
    """

    model_config = ConfigDict(extra="ignore")

    competition: str | None = None


class KalshiCustomStrike(BaseModel):
    """Nested `/events.markets[].custom_strike` block.

    `tennis_competitor` is Kalshi's opaque player UUID. No public
    `/competitors/{uuid}` lookup endpoint exists (verified 2026-05-11),
    so it functions only as a stable within-Kalshi player key — kept
    for forward-compat in case Kalshi ships a competitor lookup later.
    """

    model_config = ConfigDict(extra="ignore")

    tennis_competitor: str | None = None


class KalshiMarket(BaseModel):
    """One Kalshi binary market — the YES side of a two-market event.

    For tennis match events (`KX{ATP|WTA}MATCH`), each event holds two
    mutually-exclusive markets, one per player as the YES side.
    `yes_sub_title` is the full player name (first + last); the
    matcher resolves the predicted winner to one of these by
    last-token substring.

    Both YES sides expose **independent books** (verified 2026-05-11):
    Tirante's `yes_bid=0.40` and Medvedev's `yes_ask=0.61` don't
    perfectly cross — they're separate order books that happen to be
    tightly arbitraged. Read each side's quote and depth directly
    rather than inverting the favorite's book.
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
    no_ask_dollars: float | None = None
    no_bid_dollars: float | None = None
    status: str | None = None
    # Tipoff (`"2026-05-12T21:30:00Z"`) — load-bearing for the slate
    # adapter's horizon filter. Verified 100% populated across ATP,
    # WTA, and Challenger series.
    occurrence_datetime: datetime | None = None
    # Top-of-book size in contracts (string-encoded float).
    yes_bid_size_fp: float | None = None
    yes_ask_size_fp: float | None = None
    # Lifetime + 24h volume in contracts; open interest in contracts.
    volume_fp: float | None = None
    volume_24h_fp: float | None = None
    open_interest_fp: float | None = None
    # Kalshi's resting-book liquidity number (dollars).
    liquidity_dollars: float | None = None
    # Last trade + previous-tick references for derived 1d move.
    last_price_dollars: float | None = None
    previous_yes_bid_dollars: float | None = None
    previous_yes_ask_dollars: float | None = None
    previous_price_dollars: float | None = None
    # Settlement rules (natural language) — director context fallback
    # in lieu of Polymarket's AI-generated `context_description`.
    rules_primary: str | None = None
    custom_strike: KalshiCustomStrike | None = None

    @field_validator(
        "yes_ask_dollars",
        "yes_bid_dollars",
        "no_ask_dollars",
        "no_bid_dollars",
        "yes_bid_size_fp",
        "yes_ask_size_fp",
        "volume_fp",
        "volume_24h_fp",
        "open_interest_fp",
        "liquidity_dollars",
        "last_price_dollars",
        "previous_yes_bid_dollars",
        "previous_yes_ask_dollars",
        "previous_price_dollars",
        mode="before",
    )
    @classmethod
    def _coerce_dollars(cls, v: Any) -> float | None:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("occurrence_datetime", mode="before")
    @classmethod
    def _coerce_datetime(cls, v: Any) -> datetime | None:
        if v is None or v == "":
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                # Kalshi ships `"2026-05-12T21:30:00Z"` — fromisoformat
                # accepts the `+00:00` form, so swap the trailing `Z`.
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                return None
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
    # Tournament metadata — `product_metadata.competition` is the
    # human-readable label (e.g. `"ATP Rome"`) the slate adapter
    # slugifies for `series_slug` so `tennis/identity.py` surface and
    # tier lookups can fire.
    product_metadata: KalshiProductMetadata | None = None
    markets: list[KalshiMarket] = Field(default_factory=list)


class KalshiOrderbook(BaseModel):
    """`/markets/{ticker}/orderbook` response.

    Wire shape: `{"orderbook_fp": {"yes_dollars": [[price_str,
    size_str], ...], "no_dollars": [...]}}`. Each side is
    `list[(price_dollars, size_contracts)]`. Levels arrive low-to-high
    by price; the slate adapter reverses bids so top-of-book is `[0]`
    on both sides (mirrors `clob/__init__.py:fetch_book` convention).
    """

    model_config = ConfigDict(extra="ignore")

    yes_levels: list[tuple[float, float]] = Field(default_factory=list)
    no_levels: list[tuple[float, float]] = Field(default_factory=list)


class KalshiCandle(BaseModel):
    """One bucket from `/series/{s}/markets/{t}/candlesticks`.

    Each bucket carries OHLC for `yes_bid`, `yes_ask`, and `price`
    (last trade) separately, plus per-bucket `volume_fp` and
    `open_interest_fp`. Only `period_interval=1` and `=60` are legal
    (verified 2026-05-11; 5/15/30 all return HTTP 400).

    The slate enrichment uses `yes_bid.close_dollars` for the
    sparkline + recency moves so the rendered series matches what
    Polymarket's CLOB price-history exposed (a single mid-price
    stream per market, indexed at the bid).
    """

    model_config = ConfigDict(extra="ignore")

    end_period_ts: int | None = None
    yes_bid: dict[str, Any] = Field(default_factory=dict)
    yes_ask: dict[str, Any] = Field(default_factory=dict)
    price: dict[str, Any] = Field(default_factory=dict)
    volume_fp: float | None = None
    open_interest_fp: float | None = None

    @field_validator("volume_fp", "open_interest_fp", mode="before")
    @classmethod
    def _coerce_dollars(cls, v: Any) -> float | None:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


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
