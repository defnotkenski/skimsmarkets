"""Polymarket US data shapes.

Polymarket's JSON shapes are not fully nailed down in their public docs yet —
notably the settlement-time field name — so we parse defensively: a
`model_validator(mode="before")` tries a prioritized list of candidate field
names and leaves the value as `None` if none hit. Slate filtering runs on
`game_start_time` (tipoff); a missing `expected_expiration_time` is fine,
it's captured for completeness only.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skimsmarkets.clob import invert_sparkline as _invert_sparkline
from skimsmarkets.tennis.models import (
    TennisGbtContext,
    TennisSimulationContext,
    TennisStatsContext,
)
from skimsmarkets.unusual_whales.models import UnusualWhalesContext


def _coerce_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_str(value: Any) -> str | None:
    """Stringify gamma scalars (id fields arrive as either int or str depending
    on endpoint). Returns None for absent / non-scalar inputs so callers don't
    end up with `<memory at 0x…>` from accidental Buffer / object stringify.
    """
    if value is None:
        return None
    if isinstance(value, (str, int)):
        return str(value)
    return None


# Candidate field names, in preference order, for the Polymarket settlement
# timestamp. Probed in order during model validation; first non-None wins.
# Expand as real data reveals more.
_SETTLEMENT_TIME_CANDIDATES: tuple[str, ...] = (
    "expected_expiration_time",
    "end_date",
    "endDate",
    "resolution_time",
    "resolutionTime",
    "close_time",
    "closeTime",
    "event_end_time",
    "eventEndTime",
    "end_time",
    "endTime",
)


def _invert_price(outcome_prices: Any, *, want: str) -> float | None:
    """Derive NO-side bid/ask from the YES-side `outcomePrices` snapshot.

    `outcomePrices` is a 2-element list where the smaller value is YES bid and
    the larger is YES ask, regardless of the order-within-list that matches
    `outcomes`. `no_bid = 1 - yes_ask`, `no_ask = 1 - yes_bid`. Returns None
    when the shape is unexpected so the caller can leave the price unset.
    """
    if not isinstance(outcome_prices, list) or len(outcome_prices) != 2:
        return None
    try:
        p0, p1 = float(outcome_prices[0]), float(outcome_prices[1])
    except (TypeError, ValueError):
        return None
    yes_bid, yes_ask = (p0, p1) if p0 <= p1 else (p1, p0)
    if want == "no_bid":
        return 1.0 - yes_ask
    if want == "no_ask":
        return 1.0 - yes_bid
    return None


def _team_aliases(team: dict[str, Any] | None) -> list[str]:
    """Collect every label variant Polymarket gives us for a team.

    The team record carries `name` (mascot, e.g. 'Cavaliers'), `safeName`
    (city, e.g. 'Cleveland'), `abbreviation` ('cle'), and `alias` (often
    mascot again). We collect all forms so future cross-venue matching or
    display layers can pick whichever shape fits. De-duplicated, order preserved.
    """
    if not isinstance(team, dict):
        return []
    seen: list[str] = []
    for key in ("name", "safeName", "alias", "abbreviation"):
        v = team.get(key)
        if isinstance(v, str) and v and v not in seen:
            seen.append(v)
    return seen


def _team_record(team: dict[str, Any] | None) -> str | None:
    """Pull the W/L record string from a team dict, normalizing empty to None.

    Polymarket populates `record` (e.g. "28-6") for real sports teams; for
    futures-style "team" entries (player MVP candidates) the field is the
    empty string — treat that as absence.
    """
    if not isinstance(team, dict):
        return None
    raw = team.get("record")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _team_provider_ids(team: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the team's external-provider ID list verbatim, or [] if absent.

    Sportradar / DraftKings / etc. IDs land here as `[{"provider": "...",
    "id": "..."}, ...]`. Kept as raw dicts for forward compatibility — we
    don't currently consume them, but they unlock cross-vendor joins later.
    """
    if not isinstance(team, dict):
        return []
    raw = team.get("providerIds")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _extract_team_side(
    market_sides: Any,
    *,
    want_long: bool,
) -> tuple[str | None, list[str], str | None, list[dict[str, Any]]]:
    """Pull the team label + alias list + record + providerIds for one side.

    `want_long=True` → YES side, `False` → NO side. Returns
    `(display_label, aliases, record, provider_ids)`. Display label is the
    `name` field (mascot for team sports, person for individual sports);
    aliases carry every form we know for matching. Record is the team's
    W/L string ("28-6") or None for futures placeholders. provider_ids is
    the raw external-ID list. All default to absent values when the
    requested side isn't represented or the team record is missing.
    """
    if not isinstance(market_sides, list):
        return None, [], None, []
    # Prefer explicit long=True/False flag; fall back to description=Yes/No.
    for side in market_sides:
        if not isinstance(side, dict):
            continue
        if side.get("long") is want_long:
            team = side.get("team")
            aliases = _team_aliases(team)
            display = aliases[0] if aliases else None
            return display, aliases, _team_record(team), _team_provider_ids(team)
    for side in market_sides:
        if not isinstance(side, dict):
            continue
        if side.get("description") == ("Yes" if want_long else "No"):
            team = side.get("team")
            aliases = _team_aliases(team)
            display = aliases[0] if aliases else None
            return display, aliases, _team_record(team), _team_provider_ids(team)
    return None, [], None, []


# Candidate field names for Polymarket's "when does the game START" timestamp.
# This is the load-bearing signal for slate filtering: gameStartTime / startTime
# / startDate all sit right around tipoff, while `endDate` is a ~2-week
# settlement window — don't use `endDate` for "is this game in the horizon".
_GAME_START_CANDIDATES: tuple[str, ...] = (
    "game_start_time",
    "gameStartTime",
    "startTime",
    "startDate",
    "start_time",
    "start_date",
)


class PolymarketMarket(BaseModel):
    """A single Polymarket binary side.

    Head-to-head games produce ONE underlying market on Polymarket but carry
    two `marketSides` (e.g. YES=Cavaliers, NO=Raptors). To keep the rest of
    the pipeline "one record per tradable side," the event's `model_validator`
    expands such markets into two PolymarketMarket instances: one for YES,
    one for NO (prices inverted). Both share the same `slug`; `is_no_side`
    flags the inverted one so BBO fetching can dedupe by slug and consumers
    can render the distinction.

    `team_aliases` carries every label Polymarket gives for this side's team
    (name/safeName/abbreviation/alias) so display and comparison code can
    pick whichever form fits.

    Initial prices are parsed from the events.list snapshot (`outcomePrices` +
    `marketSides`) when present; the pipeline refreshes authoritative bid/ask
    + book depth + intraday stats from `markets.book(slug)` for the sides it
    actually needs. Fields default to None so an unfetched market is still
    representable (used while building the side map before the book is
    resolved).

    `game_start_time` and `expected_expiration_time` are both captured because
    they serve different purposes: the former is the actual game time (used
    for slate filtering — "today's games"), the latter is the settlement
    window (~2 weeks past the game). Don't conflate them.
    """

    model_config = ConfigDict(extra="ignore")

    slug: str
    id: str | None = None
    title: str | None = None
    yes_sub_title: str | None = Field(
        default=None,
        description="Display label for this side (team mascot or player name).",
    )
    team_aliases: list[str] = Field(
        default_factory=list,
        description="All labels (name, safeName, abbreviation, alias) for matching.",
    )
    sports_market_type: str | None = Field(
        default=None,
        description=(
            "Polymarket's sportsMarketType ('moneyline', 'drawable_outcome', "
            "'spreads', 'totals', 'futures'). The pipeline filters to moneyline "
            "and drawable_outcome only — futures/spreads/totals are skipped."
        ),
    )
    is_no_side: bool = Field(
        default=False,
        description="True when this record represents the NO direction of the "
        "underlying slug — prices are inverted vs Polymarket's YES book.",
    )
    yes_bid_dollars: float | None = None
    yes_ask_dollars: float | None = None
    last_trade_price_dollars: float | None = None
    # Number of distinct PRICE LEVELS on each side of the book (not contract
    # counts at top — that lives in `yes_bid_size_top` / `yes_ask_size_top`).
    # Empirical: BBO's `bidDepth`/`askDepth` exactly equal `len(bids)` /
    # `len(offers)` from the book response. A 1-level book vs. a 7-level
    # book tells a real story about how spread out the resting interest is.
    # Inverted on the NO clone — depth is symmetric, just the side label flips.
    yes_bid_depth: int | None = None
    yes_ask_depth: int | None = None
    # Top-of-book SIZE in contracts — the qty available to trade at exactly
    # the best bid / best ask. From `bids[0].qty` / `offers[0].qty` on the
    # book response. A `5/4` here vs `500/400` is the difference between
    # "would move on a single fill" and "can absorb real flow."
    yes_bid_size_top: float | None = None
    yes_ask_size_top: float | None = None
    # Total dollars resting across the entire visible book on each side
    # (sum of `qty × px` across all levels). Includes far-out rest orders
    # (e.g. $0.001 panic-buy bids) so it's a "what's everyone willing to
    # do at any price" number, not a tight-spread depth measure.
    yes_bid_book_dollars: float | None = None
    yes_ask_book_dollars: float | None = None
    # `MARKET_STATE_OPEN` / `…SUSPENDED` / `…HALTED` / `…MATCH_AND_CLOSE_AUCTION`
    # / `…PREOPEN` / `…EXPIRED` / `…TERMINATED`. Authoritative tradability
    # flag from the book response. `None` means we never fetched the book
    # (e.g. offshore events go through `from_gamma()`, which has no analog).
    market_state: str | None = None
    # Intraday price stats from `marketData.stats` on the book response.
    # `notional_traded_dollars` is the TRUE USD volume — Polymarket
    # computes it as Σ(price_at_fill × qty), which captures price drift
    # during the session that our derived `volume_dollars` (sharesTraded ×
    # current ref price) does not. Prefer this over `volume_dollars` in
    # renderers when present.
    notional_traded_dollars: float | None = None
    high_px_dollars: float | None = None
    low_px_dollars: float | None = None
    open_px_dollars: float | None = None
    close_px_dollars: float | None = None
    # Size on the most recent print, in contracts. A 5-share dust trade
    # vs a 500-share rip carry very different information about that
    # last_trade_price.
    last_trade_qty: float | None = None
    # Dollar volume is derived in `PolymarketClient.get_book` as
    # `sharesTraded × reference_price` when `notional_traded_dollars` is
    # absent. When present, `notional_traded_dollars` is the truth and
    # this number falls back to it for backwards compat.
    volume_dollars: float | None = None
    # `open_interest_dollars` is the canonical name for "outstanding shares ×
    # price" — what user-facing renderers should read. `liquidity_dollars` is
    # populated identically and kept as a backwards-compat alias for any
    # external consumer (logs, downstream tools) that already reads the old
    # field name. The misleading "liquidity" framing is what we're moving
    # away from: this number is open interest, NOT order-book depth.
    # Real CLOB liquidity arrives via `gamma_liquidity_dollars` below when
    # the gamma piggyback fires.
    open_interest_dollars: float | None = None
    liquidity_dollars: float | None = None
    # Per-side W/L record string (e.g. "28-6"), pulled from
    # `marketSides[].team.record`. Empty for futures-style "team" entries
    # (player MVP candidates). Surfaced in the leaderboard side label so
    # the user sees season form at a glance and specialists don't have to
    # web-search for it.
    team_record: str | None = None
    # Raw external-provider ID list (sportradar / draftkings / etc.) for
    # this side's team. Currently unconsumed; carried forward so future
    # cross-vendor joins (injuries, lineups, advanced stats) don't need
    # slug-fuzzing.
    team_provider_ids: list[dict[str, Any]] = Field(default_factory=list)
    # Gamma piggyback fields — populated by the pipeline only when the
    # Unusual Whales gamma resolver runs (UW key set). Source:
    # `gamma /markets?slug=` response, the same call the resolver already
    # makes for clobTokenIds. All optional; None means "UW disabled" or
    # "gamma had no matching market." Naming uses a `gamma_` prefix so the
    # source is obvious in the JSONL log and so we don't shadow our own
    # derived `volume_dollars` / `open_interest_dollars`.
    gamma_spread: float | None = None
    gamma_one_day_price_change: float | None = None
    gamma_one_month_price_change: float | None = None
    gamma_competitive: float | None = None
    # `gamma_liquidity_dollars` is gamma's `liquidityClob` — the *real*
    # CLOB order-book liquidity in dollars, distinct from our derived
    # `open_interest_dollars`. Renderers can show both side-by-side so
    # the LLM sees "how much sits on the book" and "how much is held" as
    # separate signals.
    gamma_liquidity_dollars: float | None = None
    gamma_volume_dollars: float | None = None
    gamma_accepting_orders: bool | None = None
    # CLOB price-history enrichment fields — populated only when the
    # pipeline's `enrich_price_history` stage runs (gated by
    # `CLOB_HISTORY_ENABLED`). Source: `clob.polymarket.com/prices-history`.
    # Naming uses a `clob_` prefix so the source is unambiguous in JSONL
    # logs and so we don't shadow gamma's `gamma_one_day_price_change`
    # (different windowing — gamma's value is publisher-defined; CLOB
    # values are computed by us from a fixed sample-and-window).
    # Scalars are signed price moves over the named window (positive =
    # YES side moved up). On NO clones they're sign-flipped via
    # `inverted_no_side`.
    clob_price_change_30m: float | None = None
    clob_price_change_1h: float | None = None
    clob_price_change_4h: float | None = None
    clob_price_change_24h: float | None = None
    # Pre-formatted N-point sparkline string (e.g. `"0.520→0.554→0.601"`)
    # ready for direct insertion into LLM context. NO clone carries the
    # `1 - p` inverted version.
    clob_price_path_sparkline: str | None = None
    # Raw `(epoch_seconds, mid_price)` points kept for backtest / debug
    # consumers. Never rendered to LLM context — too verbose. NO clone
    # carries the `(t, 1 - p)` inverted version.
    clob_price_history: list[tuple[int, float]] | None = None
    game_start_time: datetime | None = None
    expected_expiration_time: datetime | None = None

    @field_validator(
        "yes_bid_dollars",
        "yes_ask_dollars",
        "last_trade_price_dollars",
        "yes_bid_size_top",
        "yes_ask_size_top",
        "yes_bid_book_dollars",
        "yes_ask_book_dollars",
        "notional_traded_dollars",
        "high_px_dollars",
        "low_px_dollars",
        "open_px_dollars",
        "close_px_dollars",
        "last_trade_qty",
        "volume_dollars",
        "open_interest_dollars",
        "liquidity_dollars",
        "gamma_spread",
        "gamma_one_day_price_change",
        "gamma_one_month_price_change",
        "gamma_competitive",
        "gamma_liquidity_dollars",
        "gamma_volume_dollars",
        "clob_price_change_30m",
        "clob_price_change_1h",
        "clob_price_change_4h",
        "clob_price_change_24h",
        mode="before",
    )
    @classmethod
    def _parse_float(cls, v: Any) -> Any:
        return _coerce_float(v)

    @field_validator("expected_expiration_time", "game_start_time", mode="before")
    @classmethod
    def _parse_time(cls, v: Any) -> Any:
        return _coerce_time(v)

    @model_validator(mode="before")
    @classmethod
    def _extract_from_raw_shape(cls, data: Any) -> Any:
        """Normalize Polymarket's raw event-listing shape into our flat fields.

        - yes_sub_title <- marketSides long=True team.name (or 'Yes' description)
        - team_aliases <- name / safeName / abbreviation / alias for that team
        - yes_bid_dollars / yes_ask_dollars <- outcomePrices[0]/[1] if not already set
          (snapshot only; BBO is the source of truth post-match).
        - expected_expiration_time <- first non-null of settlement-time candidates.
        - game_start_time <- first non-null of game-start candidates.

        NO-side expansion (for head-to-head markets) happens at the event
        level — see `PolymarketEvent._expand_head_to_head_markets`.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)  # shallow copy; don't mutate caller's dict

        if (
            data.get("yes_sub_title") is None
            or not data.get("team_aliases")
            or data.get("team_record") is None
            or not data.get("team_provider_ids")
        ):
            display, aliases, record, provider_ids = _extract_team_side(
                data.get("marketSides"), want_long=True
            )
            if data.get("yes_sub_title") is None and display:
                data["yes_sub_title"] = display
            if not data.get("team_aliases") and aliases:
                data["team_aliases"] = aliases
            if data.get("team_record") is None and record:
                data["team_record"] = record
            if not data.get("team_provider_ids") and provider_ids:
                data["team_provider_ids"] = provider_ids

        if data.get("sports_market_type") is None and data.get("sportsMarketType"):
            data["sports_market_type"] = data["sportsMarketType"]

        # Snapshot bid/ask from outcomePrices when the model wasn't handed
        # explicit values. Heuristic: outcomePrices is a 2-element list of
        # strings in bid-then-ask order (observed empirically across NBA/NFL/MLS).
        if data.get("yes_bid_dollars") is None or data.get("yes_ask_dollars") is None:
            prices = data.get("outcomePrices")
            if isinstance(prices, list) and len(prices) == 2:
                try:
                    p0 = float(prices[0])
                    p1 = float(prices[1])
                    # Order bid/ask so bid <= ask regardless of list order.
                    bid, ask = (p0, p1) if p0 <= p1 else (p1, p0)
                    data.setdefault("yes_bid_dollars", bid)
                    data.setdefault("yes_ask_dollars", ask)
                except (TypeError, ValueError):
                    pass

        if data.get("expected_expiration_time") is None:
            for candidate in _SETTLEMENT_TIME_CANDIDATES[1:]:  # skip our own name
                if candidate in data and data[candidate] is not None:
                    data["expected_expiration_time"] = data[candidate]
                    break

        if data.get("game_start_time") is None:
            for candidate in _GAME_START_CANDIDATES[1:]:  # skip our own name
                if candidate in data and data[candidate] is not None:
                    data["game_start_time"] = data[candidate]
                    break
        return data

    def inverted_no_side(
        self,
        no_display: str,
        no_aliases: list[str],
        *,
        no_record: str | None = None,
        no_provider_ids: list[dict[str, Any]] | None = None,
    ) -> "PolymarketMarket":
        """Return a sibling market record representing the NO direction.

        Same slug, team aliases swapped to the NO-side team, bid/ask inverted
        so the consumer sees "price to buy this team" directly. Top-of-book
        depth is also swapped (the bid stack of the YES book is the ask stack
        of the implied NO book and vice versa — depth is symmetric, no
        `1 - x` flip). Gamma fields (`spread`, `1d`, `competitive`,
        `liquidityClob`) are market-level not side-directional, so they
        carry through unchanged. Kept as a method so the PolymarketEvent
        validator can derive NO-side records without rebuilding the full
        model state by hand.
        """
        inv_bid = (
            1.0 - self.yes_ask_dollars if self.yes_ask_dollars is not None else None
        )
        inv_ask = (
            1.0 - self.yes_bid_dollars if self.yes_bid_dollars is not None else None
        )
        return self.model_copy(
            update={
                "is_no_side": True,
                "yes_sub_title": no_display,
                "team_aliases": no_aliases,
                "team_record": no_record,
                "team_provider_ids": no_provider_ids or [],
                "yes_bid_dollars": inv_bid,
                "yes_ask_dollars": inv_ask,
                # All depth-style fields swap sides — the YES bid book IS the
                # implied NO ask book and vice versa. None of them need a
                # `1 - x` flip; only prices do.
                "yes_bid_depth": self.yes_ask_depth,
                "yes_ask_depth": self.yes_bid_depth,
                "yes_bid_size_top": self.yes_ask_size_top,
                "yes_ask_size_top": self.yes_bid_size_top,
                "yes_bid_book_dollars": self.yes_ask_book_dollars,
                "yes_ask_book_dollars": self.yes_bid_book_dollars,
                # Intraday range fields are session-level (not side-directional)
                # but they describe the YES price trajectory — drop on the NO
                # clone rather than try to invert (`1 - high_px` is meaningless
                # when high and low were set at different timestamps).
                "high_px_dollars": None,
                "low_px_dollars": None,
                "open_px_dollars": None,
                "close_px_dollars": None,
                # last_trade is directional — drop it on the NO clone to avoid misleading display.
                "last_trade_price_dollars": None,
                "last_trade_qty": None,
                # CLOB scalars are signed price moves of the YES side over
                # a window — the implied NO move is the negation. Sparkline
                # and raw history get value-inverted (`1 - p`) so the NO
                # series tells the same story from the NO side.
                "clob_price_change_30m": (
                    -self.clob_price_change_30m
                    if self.clob_price_change_30m is not None
                    else None
                ),
                "clob_price_change_1h": (
                    -self.clob_price_change_1h
                    if self.clob_price_change_1h is not None
                    else None
                ),
                "clob_price_change_4h": (
                    -self.clob_price_change_4h
                    if self.clob_price_change_4h is not None
                    else None
                ),
                "clob_price_change_24h": (
                    -self.clob_price_change_24h
                    if self.clob_price_change_24h is not None
                    else None
                ),
                "clob_price_path_sparkline": _invert_sparkline(
                    self.clob_price_path_sparkline
                ),
                "clob_price_history": (
                    [(t, 1.0 - p) for t, p in self.clob_price_history]
                    if self.clob_price_history is not None
                    else None
                ),
            }
        )

    @property
    def yes_implied_probability(self) -> float | None:
        """Midpoint of yes bid/ask as an implied probability (0-1)."""
        if self.yes_bid_dollars is None or self.yes_ask_dollars is None:
            return None
        return (self.yes_bid_dollars + self.yes_ask_dollars) / 2


class PolymarketEvent(BaseModel):
    """A Polymarket event — container for one or more binary markets.

    `id` + `slug` are the identifiers, `markets` are the binary yes/no markets
    attached to this event. `series_slug` is used for league filtering (e.g.
    'nba-2025', 'mlb-2026'); the SDK's events.list doesn't accept a league
    query param, so we filter client-side by slug prefix. `teams` is kept as
    a light list of `{name, abbreviation, league}` records — useful for debug
    logging, not for identity (the authoritative side label is on
    PolymarketMarket.yes_sub_title).

    Live-game fields (`live`, `ended`, `score`, `period`, `elapsed`,
    `main_spread_line`, `main_total_line`, `sport_type`) come from the event's
    top-level shape when present and fall back to the nested `eventState` dict
    — Polymarket populates both but the nested copy is the source of truth for
    mid-game deltas. Score is a bare string like "24-30" (team order matches
    the event title).
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    slug: str
    title: str | None = None
    category: str | None = None
    series_slug: str | None = None
    active: bool | None = None
    closed: bool | None = None
    live: bool | None = None
    ended: bool | None = None
    score: str | None = None
    period: str | None = None
    elapsed: str | None = None
    main_spread_line: float | None = None
    main_total_line: float | None = None
    sport_type: str | None = None
    teams: list[dict[str, Any]] = Field(default_factory=list)
    markets: list[PolymarketMarket] = Field(default_factory=list)
    # AI-generated pre-match prose summary that gamma ships in
    # `eventMetadata.context_description`. ~150 tokens of narrative context
    # (form, recent H2H, line motivation) — fed verbatim to the director's
    # per-event user message. None when gamma hasn't generated one yet
    # (common for niche tennis events).
    context_description: str | None = None
    # Attached post-validation by `resolve_unusual_whales()` when UW is enabled
    # and this event's YES-side asset_id resolved to an UW-tracked market.
    # Always None when the event comes straight off the SDK response.
    uw_context: UnusualWhalesContext | None = None
    # Attached post-validation by `enrich_tennis_stats()` for ATP/WTA singles
    # head-to-heads when a `TennisStatsProvider` returns an actionable
    # context. Always None for non-tennis events, doubles markets, and
    # whenever the provider had no record / failed. Consumed by the
    # `tennis_form_and_surface` fetcher/reasoner (full block) and the
    # `tennis_conditions_and_context` fetcher/reasoner (narrow fatigue
    # slice). Director sees neither — see CLAUDE.md and the tennis
    # package docstring for the silo posture.
    tennis_stats: TennisStatsContext | None = None
    # Attached post-validation by `enrich_tennis_simulation()` after
    # `enrich_tennis_stats` populates the player career serve/return
    # primitives this depends on. Director-only (same posture as
    # `uw_context`) — a long-run statistical baseline lenses shouldn't
    # second-guess at the lens layer. Always None for non-tennis events,
    # tennis events without populated career serve/return % on both
    # players, and whenever the gate failed.
    tennis_simulation: TennisSimulationContext | None = None
    # Attached post-validation by `enrich_tennis_gbt()` after
    # `enrich_tennis_stats` populates the player MatchStat ids the
    # GBT predictor uses to look up historical aggregates. Director-
    # only (same posture as `tennis_simulation`) — a finite-window
    # historical prior the lenses shouldn't second-guess. Always None
    # for non-tennis events, tennis events whose players miss the
    # cold-start gate, and runs where the GBT artefact / parquet
    # are absent (no spike training has occurred yet).
    tennis_gbt: TennisGbtContext | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _stringify_id(cls, v: Any) -> Any:
        # Some Polymarket endpoints return numeric IDs; normalize to str so the
        # field is comparable across call sites without type juggling.
        return str(v) if v is not None else v

    @model_validator(mode="before")
    @classmethod
    def _pull_event_aliases(cls, data: Any) -> Any:
        """Flatten camelCase + nested `eventState` into our snake_case fields.

        Precedence: explicit flat field → top-level camelCase → `eventState`
        nested value. The nested `eventState` is where live-game deltas land
        first, so it's the authoritative source for score/period/elapsed; the
        top-level flat fields are convenient but sometimes stale by a tick.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        if data.get("series_slug") is None and data.get("seriesSlug") is not None:
            data["series_slug"] = data["seriesSlug"]

        state = data.get("eventState")
        state_d = state if isinstance(state, dict) else {}

        # (flat_key, top_level_camel, event_state_key)
        _aliases: tuple[tuple[str, str | None, str | None], ...] = (
            ("live", "live", "live"),
            ("ended", "ended", "ended"),
            ("score", "score", "score"),
            ("period", "period", "period"),
            ("elapsed", "elapsed", "elapsed"),
            ("sport_type", None, "type"),
            ("main_spread_line", None, "mainSpreadLine"),
            ("main_total_line", None, "mainTotalLine"),
        )
        for flat, camel, state_key in _aliases:
            if data.get(flat) is not None:
                continue
            if camel and data.get(camel) is not None and flat != camel:
                data[flat] = data[camel]
                continue
            if state_key and state_d.get(state_key) is not None:
                data[flat] = state_d[state_key]

        # Head-to-head expansion: for each raw market with two distinct team
        # sides in `marketSides`, append a synthesized NO-side dict so the
        # markets list contains one record per tradable side. The NO record
        # carries the opposing team's aliases and pre-inverted bid/ask. This
        # runs on the raw data (before PolymarketMarket strips marketSides),
        # which is why it lives here rather than on the market validator.
        raw_markets = data.get("markets")
        if isinstance(raw_markets, list):
            expanded: list[Any] = []
            for raw in raw_markets:
                expanded.append(raw)
                if not isinstance(raw, dict):
                    continue
                sides = raw.get("marketSides")
                yes_display, yes_aliases, _yes_record, _yes_provider_ids = (
                    _extract_team_side(sides, want_long=True)
                )
                no_display, no_aliases, no_record, no_provider_ids = (
                    _extract_team_side(sides, want_long=False)
                )
                # Only expand when both sides carry a DIFFERENT team (head-to-
                # head). MVP-style futures have the same team on both sides —
                # those stay as a single record.
                if not yes_display or not no_display:
                    continue
                if yes_display == no_display:
                    continue
                no_side_raw: dict[str, Any] = {
                    "slug": raw.get("slug"),
                    "id": raw.get("id"),
                    "title": raw.get("title"),
                    "yes_sub_title": no_display,
                    "team_aliases": no_aliases,
                    "team_record": no_record,
                    "team_provider_ids": no_provider_ids,
                    "sports_market_type": raw.get("sportsMarketType"),
                    "is_no_side": True,
                    # Invert prices when we have both halves; otherwise leave None.
                    "yes_bid_dollars": _invert_price(
                        raw.get("outcomePrices"), want="no_bid"
                    ),
                    "yes_ask_dollars": _invert_price(
                        raw.get("outcomePrices"), want="no_ask"
                    ),
                    # Carry forward time + volume signals; last_trade is YES-directional so skip.
                    "volume_dollars": raw.get("volume"),
                    # Populate both naming variants from the same source so
                    # the legacy field stays valid for older readers.
                    "open_interest_dollars": raw.get("liquidity"),
                    "liquidity_dollars": raw.get("liquidity"),
                    "gameStartTime": raw.get("gameStartTime")
                    or raw.get("startTime")
                    or raw.get("startDate"),
                    "endDate": raw.get("endDate"),
                }
                expanded.append(no_side_raw)
            data["markets"] = expanded
        return data

    @property
    def is_live(self) -> bool:
        """True when the event is currently in progress (not pre-game, not finished)."""
        return bool(self.live) and not bool(self.ended)

    @property
    def is_pre_game(self) -> bool:
        """True when we have live-state plumbing but the game hasn't tipped off yet."""
        return not bool(self.live) and not bool(self.ended) and self.period == "NS"

    def game_state_line(self) -> str:
        """Format a single-line game-state context string.

        Always returns one of PRE-MATCH / LIVE / ENDED so the LLM never has to
        infer game state from absence. Consumed by specialist and director
        context rendering — kept here so both render sites emit the same
        format. Score is team-attributed when we have ≥2 teams and a simple
        `A-B` string; falls back to the raw form (e.g. tennis compound scores)
        otherwise.
        """
        sport = f", {self.sport_type}" if self.sport_type else ""
        prefix = f"Game state (Polymarket{sport}):"

        if self.ended:
            score = f" — score={self._format_score()}" if self.score else ""
            return f"{prefix} ENDED{score}"

        if self.live:
            parts: list[str] = []
            if self.period and self.period != "NS":
                parts.append(self.period)
            if self.elapsed:
                parts.append(self.elapsed)
            if self.score:
                parts.append(f"score={self._format_score()}")
            if self.main_spread_line is not None:
                parts.append(f"spread={self.main_spread_line}")
            if self.main_total_line is not None:
                parts.append(f"total={self.main_total_line}")
            body = " | ".join(parts) if parts else "in progress"
            return f"{prefix} LIVE — {body}"

        # Pre-match: anchor with the game's scheduled start time (from one of
        # the event's markets) and surface Polymarket's consensus spread/total
        # when present — those lines carry sharp-money positioning from the
        # opening bell and are useful to market_pricing even before tipoff.
        parts: list[str] = []
        # Walrus-bind so `is not None` narrows `t` to `datetime` inside the
        # comprehension; without it static checkers keep the element type as
        # `datetime | None` and flag `min(...)` / `.isoformat()` downstream.
        start_times = [t for m in self.markets if (t := m.game_start_time) is not None]
        if start_times:
            parts.append(f"starts {min(start_times).isoformat()}")
        if self.main_spread_line is not None:
            parts.append(f"spread={self.main_spread_line}")
        if self.main_total_line is not None:
            parts.append(f"total={self.main_total_line}")
        if parts:
            return f"{prefix} PRE-MATCH — {' | '.join(parts)}"
        return f"{prefix} PRE-MATCH"

    def _format_score(self) -> str:
        """Attribute the score to each team when we have ≥2 teams AND a simple
        `A-B` score. For tennis' compound `sets:current-game` format or any
        score we can't cleanly split, return it verbatim — the LLM will still
        parse it, just without team labels.
        """
        raw = (self.score or "").strip()
        if not raw:
            return ""
        # Compound scores (tennis) use ':' between sets and current game; the
        # attribution isn't straightforward, so leave them as-is.
        if ":" in raw:
            return raw
        halves = raw.split("-")
        if len(halves) != 2:
            return raw
        if len(self.teams) < 2:
            return raw
        a_name = (self.teams[0] or {}).get("name")
        b_name = (self.teams[1] or {}).get("name")
        if not a_name or not b_name:
            return raw
        return f"{a_name} {halves[0].strip()}, {b_name} {halves[1].strip()}"

    @classmethod
    def from_gamma(cls, payload: dict[str, Any]) -> "PolymarketEvent | None":
        """Build a PolymarketEvent from gamma-api's `/events?slug=…` shape.

        Gamma's payload is structurally different from the polymarket-us SDK:
        - Head-to-head games are split into N separate single-side markets
          (one per outcome — team-A, team-B, draw), each with its own slug,
          its own book, and `groupItemTitle` carrying the side label.
        - There's no `marketSides`, no `sportsMarketType`, no `eventState`.
        - Bid/ask/last/volume/liquidity are already populated on the payload —
          no separate BBO refresh exists on gamma.

        So this bypasses `_expand_head_to_head_markets` entirely and constructs
        the event manually. Returns None when no moneyline-style markets
        survive the filter (alternate-bets event variants, all-resolved
        markets, etc.) so the caller can drop the event the same way the US
        tradability filter does.

        Filtering rules:
        - Skip the event entirely if its slug ends in any of
          `_GAMMA_VARIANT_EVENT_SUFFIXES` — these are gamma's alternate-bets
          bundle slugs (spreads/totals/BTTS, halftime result, exact score,
          total corners, player props). They host markets that look like
          moneylines on the surface (`groupItemTitle="Seattle Sounders FC"`
          for halftime-result-home) but resolve on a different question
          than full-match winner.
        - Within each event, skip markets whose slug matches
          `_is_non_moneyline_gamma_slug` (spreads, totals, BTTS, set
          handicaps, partial-period winners). These can leak into the base
          moneyline event (e.g. tennis events ship a set-handicap market
          alongside the moneyline).
        - Skip markets without bid/ask (unfunded) so the tradability
          invariant holds: every kept market has live prices.
        - Skip settled events (`closed=True` at event level) and settled
          markets (`closed=True` at market level). Stale bid/ask survives
          match end, so bid/ask presence alone doesn't gate completed
          matches out — explicit `closed` check is required.
        """
        slug = payload.get("slug")
        ev_id = payload.get("id")
        if not isinstance(slug, str) or not slug or ev_id is None:
            return None
        if slug.endswith(_GAMMA_VARIANT_EVENT_SUFFIXES):
            return None
        # Settled events: drop the whole event so completed matches don't
        # show up in the slate. bid/ask presence isn't a sufficient proxy —
        # gamma keeps stale prices on the book for a window after match end.
        if payload.get("closed") is True:
            return None

        # Tipoff resolution. Mirrors the US-side precedence at the NO-side
        # synthesis below — `gameStartTime` || `startTime` || (last resort)
        # `endDate`. Empirical shape across gamma:
        # - Per-market `gameStartTime` is the load-bearing field. It's
        #   rescheduling-aware (updates when fixtures move) and populated
        #   for every match-shaped event observed (soccer, tennis, UFC).
        # - Event-level `startTime` carries the same value and is the right
        #   fallback when a market lacks `gameStartTime`.
        # - Event-level `endDate` is a *trap*: it's frozen at market
        #   creation, so on rescheduled fixtures it lags by days/months
        #   (e.g. `arg-gye-def-2026-03-05` has endDate=2026-03-05 but the
        #   actual game is 2026-05-04). It also represents tournament-end
        #   for ATP-shaped events. Kept only as a defensive last resort —
        #   real game events should never hit it in practice.
        event_start_fallback = _coerce_time(
            payload.get("startTime")
        ) or _coerce_time(payload.get("endDate"))

        # Build a name→team-record lookup from the event's `teams[]`. Used
        # below to populate `team_record` on each per-side market by matching
        # the YES/NO label against gamma's team `name`. Falls back to None
        # if the matchup is futures-style (no team objects) or the side
        # label doesn't match any team — render path already handles None.
        team_record_by_name: dict[str, str] = {}
        for t in payload.get("teams") or []:
            if not isinstance(t, dict):
                continue
            n = t.get("name")
            r = t.get("record")
            if isinstance(n, str) and isinstance(r, str) and r.strip():
                team_record_by_name[n.lower()] = r.strip()

        markets: list[PolymarketMarket] = []
        for raw in payload.get("markets") or []:
            if not isinstance(raw, dict):
                continue
            m_slug = raw.get("slug")
            if not isinstance(m_slug, str) or not m_slug:
                continue
            if _is_non_moneyline_gamma_slug(m_slug):
                continue
            bid = _coerce_float(raw.get("bestBid"))
            ask = _coerce_float(raw.get("bestAsk"))
            if bid is None or ask is None:
                # Mirror the US tradability filter — drop sides without a
                # live two-sided book.
                continue
            # Settled markets carry stale bid/ask after game end, so the
            # tradability check above isn't sufficient. Defensive belt to
            # the event-level `closed` check above for the case where one
            # market in a multi-market event has settled while siblings
            # haven't (e.g. tournament brackets).
            if raw.get("closed") is True:
                continue
            # Side-label resolution. Soccer-style gamma events split each
            # outcome into its own market with `groupItemTitle` set per side
            # (e.g. "Real Madrid" / "Barcelona" / "Draw"). Non-soccer
            # head-to-heads (ATP, UFC, etc.) instead expose a SINGLE binary
            # market with `groupItemTitle: None` and the question carrying
            # both names ("Tournament: Player A vs Player B"). When we see
            # the latter shape, parse player names from the question and
            # synthesize a NO clone — same shape the US head-to-head path
            # produces via `_pull_event_aliases`.
            group_item = raw.get("groupItemTitle")
            h2h: tuple[str, str] | None = None
            if not (isinstance(group_item, str) and group_item):
                h2h = _parse_h2h_question(raw.get("question"))
            if h2h is not None:
                yes_label, no_label = h2h
            else:
                fallback = group_item or raw.get("question")
                if not isinstance(fallback, str) or not fallback:
                    continue
                yes_label = fallback
                no_label = None
            # Per-market `gameStartTime` first — see the
            # `event_start_fallback` comment above for the precedence
            # rationale and why event-level `endDate` is a trap.
            game_time = (
                _coerce_time(raw.get("gameStartTime")) or event_start_fallback
            )
            # Gamma payloads carry the same supplementary fields the US-path
            # piggyback pulls from `gamma /markets?slug=` — we already have
            # them on this dict, so populate the gamma_* fields directly
            # instead of forcing offshore events through a separate fetch.
            oi_dollars = _coerce_float(raw.get("liquidity"))
            accepting = raw.get("acceptingOrders")
            m_closed = raw.get("closed")
            # Collapse gamma's `acceptingOrders` + `closed` booleans into the
            # same string `market_state` field that downstream renderers read.
            # Lossy vs. the US enum (US had OPEN/HALTED/CLOSED/PAUSED), but
            # the actionable distinctions for ranking are preserved: open
            # books are tradable, halted/closed books are not.
            market_state = _gamma_market_state(accepting, m_closed)
            # Pull market-level volume24hr first; this is the gamma global-
            # window volume that replaces US's `notional_traded_dollars`.
            # Falls back to lifetime `volume` only when 24h is missing.
            volume24 = _coerce_float(raw.get("volume24hr"))
            yes_record = team_record_by_name.get(yes_label.lower())
            yes_market = PolymarketMarket(
                slug=m_slug,
                id=_coerce_str(raw.get("id")),
                title=raw.get("question"),
                yes_sub_title=yes_label,
                team_aliases=[yes_label],
                team_record=yes_record,
                # Synthesized: gamma omits sportsMarketType, but the slug
                # filter above keeps only moneyline-style outcomes.
                sports_market_type="moneyline",
                is_no_side=False,
                yes_bid_dollars=bid,
                yes_ask_dollars=ask,
                last_trade_price_dollars=_coerce_float(raw.get("lastTradePrice")),
                market_state=market_state,
                # `notional_traded_dollars` is gamma's 24h CLOB volume — same
                # semantic role as US's `notionalTraded` (today's trading)
                # but scoped globally instead of US-only.
                notional_traded_dollars=volume24,
                volume_dollars=_coerce_float(raw.get("volume")),
                open_interest_dollars=oi_dollars,
                liquidity_dollars=oi_dollars,
                gamma_spread=_coerce_float(raw.get("spread")),
                gamma_one_day_price_change=_coerce_float(
                    raw.get("oneDayPriceChange")
                ),
                gamma_one_month_price_change=_coerce_float(
                    raw.get("oneMonthPriceChange")
                ),
                gamma_competitive=_coerce_float(raw.get("competitive")),
                gamma_liquidity_dollars=_coerce_float(raw.get("liquidityClob")),
                gamma_volume_dollars=_coerce_float(raw.get("volumeClob")),
                gamma_accepting_orders=(
                    bool(accepting) if isinstance(accepting, bool) else None
                ),
                game_start_time=game_time,
                expected_expiration_time=_coerce_time(raw.get("endDate")),
            )
            markets.append(yes_market)
            if no_label is not None:
                # `inverted_no_side` flips prices, swaps depth, drops
                # intraday/last-trade, and pre-emptively negates CLOB scalars
                # (all None at this stage — CLOB enrichment runs later in
                # the pipeline and applies its own sign-flip when it sees
                # `is_no_side=True`).
                no_record = team_record_by_name.get(no_label.lower())
                markets.append(
                    yes_market.inverted_no_side(
                        no_label, [no_label], no_record=no_record
                    )
                )

        if not markets:
            return None

        # Pick the most specific sport tag as a series_slug stand-in. Gamma
        # tags don't carry league granularity (just `soccer`, `tennis`, etc.),
        # but a series with explicit `seriesSlug` (e.g. `serie-a-2025`) takes
        # precedence when present.
        series_slug = _gamma_series_slug(payload) or _gamma_series_from_tags(
            payload.get("tags")
        )
        # `sport_type` is the broad sport class (soccer / tennis / basketball)
        # — comes from the same tag list but stripped of league specificity
        # so the renderer's `game_state_line()` can prefix `Game state
        # (Polymarket, soccer):` regardless of which league is showing.
        sport_type = _gamma_sport_from_tags(payload.get("tags"))
        # Live game state — gamma exposes these at event top-level for
        # in-progress matches; verified live-update parity with the
        # legacy US `eventState` block.
        live = payload.get("live")
        ended = payload.get("ended")
        score = payload.get("score") if isinstance(payload.get("score"), str) else None
        period = (
            payload.get("period") if isinstance(payload.get("period"), str) else None
        )
        elapsed = (
            payload.get("elapsed") if isinstance(payload.get("elapsed"), str) else None
        )
        # `eventMetadata.context_description` is gamma's AI-generated
        # pre-match prose — pass through verbatim so the director can fold
        # it into per-event user context (no caching impact).
        ctx_meta = payload.get("eventMetadata")
        context_description: str | None = None
        if isinstance(ctx_meta, dict):
            cd = ctx_meta.get("context_description")
            if isinstance(cd, str) and cd.strip():
                context_description = cd
        teams_payload = payload.get("teams")
        teams_list: list[dict[str, Any]] = (
            [t for t in teams_payload if isinstance(t, dict)]
            if isinstance(teams_payload, list)
            else []
        )

        return cls(
            id=str(ev_id),
            slug=slug,
            title=payload.get("title"),
            category=payload.get("category"),
            series_slug=series_slug,
            active=payload.get("active"),
            closed=payload.get("closed"),
            live=live if isinstance(live, bool) else None,
            ended=ended if isinstance(ended, bool) else None,
            score=score,
            period=period,
            elapsed=elapsed,
            sport_type=sport_type,
            teams=teams_list,
            markets=markets,
            context_description=context_description,
        )


# Partial-period winners — tennis "first set winner", soccer "first/second
# half winner", basketball "Q1 winner", hockey "period 1 winner", baseball
# "1st inning winner". These resolve before the match ends and aren't full
# match-moneyline markets, so they shouldn't be ranked alongside them.
# Anchored to a dash on both sides so we don't accidentally match a team
# slug that contains "-set-" or similar mid-token. Centralized regex
# rather than a long substring-or chain because the ordinal × period
# product is too broad to enumerate.
_PARTIAL_PERIOD_WINNER_RE = re.compile(
    r"-(?:set|half|quarter|period|inning)-\d+-winner(?:-|$)"
    r"|-(?:first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)-"
    r"(?:set|half|quarter|period|inning)-winner(?:-|$)"
)

# Whole-event slug suffixes that gamma uses to host alternate-bets bundles
# (one event per market type). Each event-slug below is a sibling of a base
# moneyline event sharing the same `<league>-<teamA>-<teamB>-<date>` prefix,
# so dropping by suffix doesn't risk hiding the moneyline. Probed empirically
# 2026-04-29; expand if gamma ships new alternate-bundle variants.
#
# `-halftime-result` is the most dangerous to leave in: its markets carry
# `groupItemTitle="Seattle Sounders FC"` / `"Draw"` / `"Real Salt Lake"`,
# indistinguishable from the full-match moneyline on the side label, but
# they resolve on the half-time score rather than the final.
_GAMMA_VARIANT_EVENT_SUFFIXES: tuple[str, ...] = (
    "-more-markets",
    "-halftime-result",
    "-exact-score",
    "-total-corners",
    "-player-props",
)


def _is_non_moneyline_gamma_slug(slug: str) -> bool:
    """True for gamma market slugs that aren't full match-winner moneylines —
    spreads, totals, BTTS, over/under, set handicaps, and partial-period
    winners (e.g. tennis first-set winner, soccer first-half winner).

    Match on slug suffix tokens because gamma omits `sportsMarketType`. False
    positives here would silently drop a legit moneyline side, so the patterns
    are anchored to the dash-separated tail of the slug — e.g. team slugs end
    in the team abbreviation (`...-syd`, `...-auc`, `...-draw`) which never
    look like the suffixes below.

    `-set-handicap-` lives inline next to the moneyline in tennis events
    (e.g. `wta-blinkov-jovic-2026-02-23-set-handicap-away-1pt5` ships in the
    same `wta-blinkov-jovic-2026-02-23` event as the binary moneyline) so we
    have to filter at the market level — the event-level variant filter
    can't catch it.

    NBA events ship player-props markets (`-points-`, `-rebounds-`,
    `-assists-`) and a first-half moneyline (`-1h-moneyline`) inline next
    to the full-match moneyline — same pattern as set-handicap, also
    filtered here.
    """
    s = slug.lower()
    return (
        s.endswith("-btts")
        or "-spread-" in s
        or s.endswith("-spread")
        or "-total-" in s
        or "-totals-" in s
        or s.endswith("-total")
        or s.endswith("-totals")
        or "-ou-" in s
        or "-set-handicap-" in s
        or "-points-" in s
        or "-rebounds-" in s
        or "-assists-" in s
        or "-1h-" in s
        or s.endswith("-1h-moneyline")
        or "-nrfi" in s
        or _PARTIAL_PERIOD_WINNER_RE.search(s) is not None
    )


def _parse_h2h_question(question: Any) -> tuple[str, str] | None:
    """Parse a gamma single-binary head-to-head question into (yes, no) names.

    Used by `PolymarketEvent.from_gamma` to detect non-soccer head-to-heads
    (ATP/UFC/etc.) where gamma exposes the moneyline as ONE binary market with
    `groupItemTitle: None` and the question carrying both names — distinct
    from soccer events which carry one market per outcome with
    `groupItemTitle` set per side.

    Pattern: `"Tournament: Player A vs Player B"`. The tournament prefix
    (everything before the first `:`) is stripped if present; the body is
    then split on " vs. " (preferred) or " vs ". Returns None when the
    string isn't a recognizable head-to-head — caller falls back to using
    the question as a single-side label.

    Examples:
        "Shymkent 2: Mathys Erhard vs Andrej Nedic" → ("Mathys Erhard", "Andrej Nedic")
        "Real Madrid vs Barcelona" → ("Real Madrid", "Barcelona")
        "Will Erhard win?" → None  (no separator)
        "A vs B vs C" → None  (more than two halves; ambiguous)
    """
    if not isinstance(question, str) or not question.strip():
        return None
    body = question.split(":", 1)[-1].strip()
    for sep in (" vs. ", " vs "):
        if sep in body:
            parts = body.split(sep)
            if len(parts) != 2:
                return None
            a, b = parts[0].strip(), parts[1].strip()
            if a and b:
                return a, b
            return None
    return None


def _gamma_series_from_tags(tags: Any) -> str | None:
    """Pick the most informative tag slug from gamma's `tags` array.

    Gamma tags look like `[{slug:'soccer', label:'Soccer'}, {slug:'sports', ...}]`.
    Prefer specific sport tags over the generic `sports` umbrella; fall back
    to None when neither is present.
    """
    if not isinstance(tags, list):
        return None
    sport_tags: list[str] = []
    for t in tags:
        if not isinstance(t, dict):
            continue
        s = t.get("slug")
        if isinstance(s, str) and s and s != "sports":
            sport_tags.append(s)
    return sport_tags[0] if sport_tags else None


# Tag slugs gamma uses for the broad sport class (vs. league specificity like
# `serie-a-2025`). Used to populate `sport_type` so the renderer's game-state
# line can prefix `Game state (Polymarket, soccer):` regardless of which
# league within that sport is showing.
_GAMMA_SPORT_TAGS: frozenset[str] = frozenset(
    {
        "soccer",
        "tennis",
        "basketball",
        "baseball",
        "hockey",
        "mma",
        "ufc",
        "cricket",
        "golf",
        "esports",
        "boxing",
        "rugby",
        "football",
    }
)


def _gamma_sport_from_tags(tags: Any) -> str | None:
    """Pull the broad sport class from gamma's `tags` array (e.g. 'soccer',
    'tennis'). Returns None when no recognized sport tag is present — the
    renderer handles None by omitting the sport prefix.
    """
    if not isinstance(tags, list):
        return None
    for t in tags:
        if not isinstance(t, dict):
            continue
        s = t.get("slug")
        if isinstance(s, str) and s in _GAMMA_SPORT_TAGS:
            return s
    return None


def _gamma_series_slug(payload: dict[str, Any]) -> str | None:
    """Pull the league slug from gamma's `series[]` array when present.

    Gamma populates this for major-league events (`serie-a-2025`,
    `epl-2025`, etc.) but leaves it empty for niche tennis/UFC. Falls back
    to the generic sport tag via `_gamma_series_from_tags` when absent.
    """
    series = payload.get("series")
    if not isinstance(series, list):
        return None
    for s in series:
        if not isinstance(s, dict):
            continue
        slug = s.get("slug")
        if isinstance(slug, str) and slug:
            return slug
    return None


def _gamma_market_state(accepting: Any, closed: Any) -> str:
    """Collapse gamma's `acceptingOrders` + `closed` booleans into the same
    `MARKET_STATE_*` string shape downstream renderers already read.

    Lossy vs. the US enum (US had OPEN/HALTED/CLOSED/PAUSED), but the
    actionable distinctions for ranking are preserved:
    - `closed=True`           → MARKET_STATE_CLOSED  (settled / ended)
    - `acceptingOrders=False` → MARKET_STATE_HALTED  (not tradable)
    - otherwise               → MARKET_STATE_OPEN    (live tradable book)

    The renderer at `agents/fetchers.py` strips the `MARKET_STATE_`
    prefix before display, and only surfaces the field when it isn't OPEN —
    so the "OPEN" string is effectively a no-op tag that downstream code
    already gates against.
    """
    if closed is True:
        return "MARKET_STATE_CLOSED"
    if accepting is False:
        return "MARKET_STATE_HALTED"
    return "MARKET_STATE_OPEN"
