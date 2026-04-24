"""Polymarket US data shapes.

Polymarket's JSON shapes are not fully nailed down in their public docs yet —
notably the settlement-time field name — so we parse defensively: a
`model_validator(mode="before")` tries a prioritized list of candidate field
names and leaves the value as `None` if none hit. Slate filtering runs on
`game_start_time` (tipoff); a missing `expected_expiration_time` is fine,
it's captured for completeness only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


def _extract_team_side(
    market_sides: Any,
    *,
    want_long: bool,
) -> tuple[str | None, list[str]]:
    """Pull the team label + alias list for one side (long=True → YES, False → NO).

    Returns (display_label, aliases). Display label is the `name` field (mascot
    for team sports, person for individual sports); aliases carry every form
    we know for matching. Both default to ``(None, [])`` when the requested
    side isn't represented or the team record is missing.
    """
    if not isinstance(market_sides, list):
        return None, []
    # Prefer explicit long=True/False flag; fall back to description=Yes/No.
    for side in market_sides:
        if not isinstance(side, dict):
            continue
        if side.get("long") is want_long:
            team = side.get("team")
            aliases = _team_aliases(team)
            display = aliases[0] if aliases else None
            return display, aliases
    for side in market_sides:
        if not isinstance(side, dict):
            continue
        if side.get("description") == ("Yes" if want_long else "No"):
            team = side.get("team")
            aliases = _team_aliases(team)
            display = aliases[0] if aliases else None
            return display, aliases
    return None, []


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
    from `markets.bbo(slug)` for the sides it actually needs. Fields default to
    None so an unfetched market is still representable (used while building the
    side map before BBO is resolved).

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
    volume_dollars: float | None = None
    liquidity_dollars: float | None = None
    game_start_time: datetime | None = None
    expected_expiration_time: datetime | None = None

    @field_validator(
        "yes_bid_dollars",
        "yes_ask_dollars",
        "last_trade_price_dollars",
        "volume_dollars",
        "liquidity_dollars",
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

        if data.get("yes_sub_title") is None or not data.get("team_aliases"):
            display, aliases = _extract_team_side(
                data.get("marketSides"), want_long=True
            )
            if data.get("yes_sub_title") is None and display:
                data["yes_sub_title"] = display
            if not data.get("team_aliases") and aliases:
                data["team_aliases"] = aliases

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
        self, no_display: str, no_aliases: list[str]
    ) -> "PolymarketMarket":
        """Return a sibling market record representing the NO direction.

        Same slug, team aliases swapped to the NO-side team, bid/ask inverted
        so the consumer sees "price to buy this team" directly. Kept as a
        method so the PolymarketEvent validator can derive NO-side records
        without rebuilding the full model state by hand.
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
                "yes_bid_dollars": inv_bid,
                "yes_ask_dollars": inv_ask,
                # last_trade is directional — drop it on the NO clone to avoid misleading display.
                "last_trade_price_dollars": None,
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
    # Attached post-validation by `resolve_unusual_whales()` when UW is enabled
    # and this event's YES-side asset_id resolved to an UW-tracked market.
    # Always None when the event comes straight off the SDK response.
    uw_context: UnusualWhalesContext | None = None

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
                yes_display, yes_aliases = _extract_team_side(sides, want_long=True)
                no_display, no_aliases = _extract_team_side(sides, want_long=False)
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
