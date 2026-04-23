"""Polymarket US data shapes, deliberately parallel to the Kalshi models.

Field names are aligned with Kalshi where they carry the same meaning
(`yes_bid_dollars`, `yes_ask_dollars`, `yes_implied_probability`,
`expected_expiration_time`) so downstream code can treat the two venues
symmetrically without branching on vendor.

Polymarket's JSON shapes are not fully nailed down in their public docs yet —
notably the settlement-time field name — so we parse defensively: a
`model_validator(mode="before")` tries a prioritized list of candidate field
names and leaves the value as `None` if none hit. Time filtering continues to
run on Kalshi's `expected_expiration_time` (see CLAUDE.md rule), so a missing
Polymarket time is fine; it only affects the optional proximity tiebreaker in
the matcher.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


# Candidate field names, in preference order, for the Polymarket settlement-time
# equivalent of Kalshi's `expected_expiration_time`. Probed in order during
# model validation; first non-None wins. Expand as real data reveals more.
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


def _extract_yes_team_name(market_sides: Any) -> str | None:
    """Polymarket's events.list payload encodes the YES-side team name inside
    `marketSides`: each side has `description` ('Yes'/'No'), `long` (bool), and
    `team: {name: ...}`. We want the team attached to the Yes/long side — that
    string is what we compare against Kalshi's `yes_sub_title`.
    """
    if not isinstance(market_sides, list):
        return None
    for side in market_sides:
        if not isinstance(side, dict):
            continue
        if side.get("description") == "Yes" and side.get("long") is True:
            team = side.get("team") or {}
            name = team.get("name") if isinstance(team, dict) else None
            if name:
                return name
    # Fallback: first side with a team name (rare — some markets don't label
    # description explicitly).
    for side in market_sides:
        if isinstance(side, dict):
            team = side.get("team") or {}
            name = team.get("name") if isinstance(team, dict) else None
            if name:
                return name
    return None


# Candidate field names for Polymarket's "when does the game START" timestamp.
# This is the load-bearing signal for cross-venue matching: Polymarket's
# endDate is a settlement window (~2 weeks after the game) which produces
# inverted rankings vs Kalshi's game-end-adjacent expected_expiration_time,
# while gameStartTime / startTime / startDate all sit right around tipoff.
_GAME_START_CANDIDATES: tuple[str, ...] = (
    "game_start_time",
    "gameStartTime",
    "startTime",
    "startDate",
    "start_time",
    "start_date",
)


class PolymarketMarket(BaseModel):
    """A single Polymarket binary yes/no market.

    Initial prices are parsed from the events.list snapshot (`outcomePrices` +
    `marketSides`) when present; the pipeline refreshes authoritative bid/ask
    from `markets.bbo(slug)` for the sides it actually needs. Fields default to
    None so an unfetched market is still representable (used while building the
    side map before BBO is resolved).

    `game_start_time` and `expected_expiration_time` are both captured because
    they serve different purposes: the former is the actual game time (used by
    the matcher for time proximity), the latter is the settlement window.
    Don't conflate them — Polymarket's settlement window sits ~2 weeks past
    the game, so comparing it to Kalshi's shortly-after-game expiration
    produces misleading "close in time" scores.
    """

    model_config = ConfigDict(extra="ignore")

    slug: str
    id: str | None = None
    title: str | None = None
    yes_sub_title: str | None = Field(
        default=None,
        description="Label of the YES outcome (team/player name from marketSides[0].team.name).",
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

        - yes_sub_title <- marketSides[0].team.name where description=Yes, long=True
        - yes_bid_dollars / yes_ask_dollars <- outcomePrices[0]/[1] if not already set
          (snapshot only; BBO is the source of truth post-match).
        - expected_expiration_time <- first non-null of settlement-time candidates.
        - game_start_time <- first non-null of game-start candidates.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)  # shallow copy; don't mutate caller's dict

        if data.get("yes_sub_title") is None:
            yes_name = _extract_yes_team_name(data.get("marketSides"))
            if yes_name:
                data["yes_sub_title"] = yes_name

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

    @property
    def yes_implied_probability(self) -> float | None:
        """Midpoint of yes bid/ask as an implied probability (0-1)."""
        if self.yes_bid_dollars is None or self.yes_ask_dollars is None:
            return None
        return (self.yes_bid_dollars + self.yes_ask_dollars) / 2


class PolymarketEvent(BaseModel):
    """A Polymarket event — container for one or more binary markets.

    Mirrors KalshiEvent in spirit: `id` + `slug` are the identifiers, `markets`
    are the binary yes/no markets attached to this event. `series_slug` is used
    for league filtering (e.g. 'nba-2025', 'mlb-2026'); the SDK's events.list
    doesn't accept a league query param, so we filter client-side by slug prefix.
    `teams` is kept as a light list of `{name, abbreviation, league}` records —
    useful for matcher debug logging, not for identity (the authoritative side
    label is on PolymarketMarket.yes_sub_title).
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    slug: str
    title: str | None = None
    category: str | None = None
    series_slug: str | None = None
    active: bool | None = None
    closed: bool | None = None
    ended: bool | None = None
    teams: list[dict[str, Any]] = Field(default_factory=list)
    markets: list[PolymarketMarket] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def _stringify_id(cls, v: Any) -> Any:
        # Some Polymarket endpoints return numeric IDs; normalize to str so the
        # field is comparable across call sites without type juggling.
        return str(v) if v is not None else v

    @model_validator(mode="before")
    @classmethod
    def _pull_series_slug(cls, data: Any) -> Any:
        """Polymarket uses camelCase (`seriesSlug`) on the wire; alias it to snake_case."""
        if isinstance(data, dict) and data.get("series_slug") is None:
            if data.get("seriesSlug") is not None:
                data = {**data, "series_slug": data["seriesSlug"]}
        return data
