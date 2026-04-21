from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_kalshi_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise TypeError(f"Unsupported timestamp value: {value!r}")


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


class KalshiSeries(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticker: str
    title: str | None = None
    category: str | None = None
    frequency: str | None = None


class KalshiMarket(BaseModel):
    """Kalshi market. Price fields are in dollars (e.g. 0.73 = 73¢ = 73% implied)."""

    model_config = ConfigDict(extra="ignore")

    ticker: str
    event_ticker: str
    market_type: str | None = None
    title: str | None = None
    yes_sub_title: str | None = None
    no_sub_title: str | None = None
    status: str | None = None
    yes_bid_dollars: float | None = None
    yes_ask_dollars: float | None = None
    no_bid_dollars: float | None = None
    no_ask_dollars: float | None = None
    last_price_dollars: float | None = None
    volume_fp: float | None = None
    volume_24h_fp: float | None = None
    open_interest_fp: float | None = None
    liquidity_dollars: float | None = None
    notional_value_dollars: float | None = None
    open_time: datetime | None = None
    close_time: datetime | None = None
    expected_expiration_time: datetime | None = None
    rules_primary: str | None = None
    rules_secondary: str | None = None
    result: str | None = None

    @field_validator(
        "open_time", "close_time", "expected_expiration_time", mode="before"
    )
    @classmethod
    def _parse_time(cls, v: Any) -> Any:
        return _coerce_kalshi_time(v)

    @field_validator(
        "yes_bid_dollars",
        "yes_ask_dollars",
        "no_bid_dollars",
        "no_ask_dollars",
        "last_price_dollars",
        "volume_fp",
        "volume_24h_fp",
        "open_interest_fp",
        "liquidity_dollars",
        "notional_value_dollars",
        mode="before",
    )
    @classmethod
    def _parse_float(cls, v: Any) -> Any:
        return _coerce_float(v)

    @property
    def yes_implied_probability(self) -> float | None:
        """Midpoint of yes bid/ask as an implied probability (0-1)."""
        if self.yes_bid_dollars is None or self.yes_ask_dollars is None:
            return None
        return (self.yes_bid_dollars + self.yes_ask_dollars) / 2


class KalshiEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_ticker: str
    series_ticker: str | None = None
    title: str | None = None
    sub_title: str | None = None
    category: str | None = None
    mutually_exclusive: bool | None = None
    strike_date: datetime | None = None
    strike_period: str | None = None
    markets: list[KalshiMarket] = Field(default_factory=list)

    @field_validator("strike_date", mode="before")
    @classmethod
    def _parse_strike(cls, v: Any) -> Any:
        return _coerce_kalshi_time(v)
