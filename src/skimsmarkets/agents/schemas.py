from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["low", "medium", "high"]

# Conventional safety cap on half-Kelly sizing — never stake more than 25% of bankroll
# on a single contract, even if Kelly math would recommend more.
KELLY_BANKROLL_CAP = 0.25


class StatisticsReport(BaseModel):
    """Quantitative lens: form, head-to-head, splits, base rates."""

    team_a_name: str = Field(description="Exact yes_sub_title of the team you call 'team_a'.")
    team_b_name: str = Field(description="Exact yes_sub_title of the team you call 'team_b'.")
    team_a_win_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability team_a wins the event, 0-1.",
    )
    confidence: Confidence = Field(description="low if data was thin or stale.")
    key_stats: list[str] = Field(
        default_factory=list,
        description="Concrete stat lines supporting the prediction.",
    )
    head_to_head_summary: str = Field(description="Recent head-to-head history.")
    form_delta: str = Field(description="Recent-form comparison between the two sides.")
    caveats: list[str] = Field(default_factory=list)


class PlayerStatus(BaseModel):
    name: str
    team: str
    status: str = Field(
        description="e.g. 'out', 'questionable', 'probable', 'suspended'."
    )
    impact_note: str = Field(description="How this affects the matchup.")


class InjuryReport(BaseModel):
    """Availability lens: injuries, suspensions, rest, lineup uncertainty."""

    team_a_name: str
    team_b_name: str
    team_a_availability_impact: float = Field(
        ge=-0.2,
        le=0.2,
        description="Signed probability shift for team_a from availability, -0.2 to +0.2.",
    )
    team_b_availability_impact: float = Field(ge=-0.2, le=0.2)
    key_absences: list[PlayerStatus] = Field(default_factory=list)
    lineup_confidence: Literal["confirmed", "probable", "uncertain"]
    sources_checked: list[str] = Field(
        default_factory=list,
        description="Real URLs of injury sources consulted.",
    )


class NarrativeFactor(BaseModel):
    factor: str
    direction: Literal["team_a", "team_b", "neutral"]
    strength: Literal["weak", "moderate", "strong"]


class NarrativeReport(BaseModel):
    """Storyline lens: motivation, coaching, locker-room, weather for outdoor sports."""

    team_a_name: str
    team_b_name: str
    dominant_storyline: str
    motivation_edge: Literal["team_a", "team_b", "neutral"]
    narrative_factors: list[NarrativeFactor] = Field(default_factory=list)
    public_perception_bias: str = Field(
        description="e.g. 'public heavy on favorite', 'contrarian value on underdog'.",
    )
    sentiment_sources: list[str] = Field(default_factory=list)


class MarketReport(BaseModel):
    """Pricing lens: compare Kalshi to consensus, spot line movement and sharp signals."""

    team_a_name: str
    team_b_name: str
    kalshi_implied_team_a_probability: float = Field(
        ge=0.0, le=1.0,
        description="Kalshi's implied probability for team_a winning (midpoint of yes bid/ask on team_a's market).",
    )
    consensus_team_a_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Fair probability team_a wins, implied by consensus sportsbooks / betting exchanges.",
    )
    edge_bps: int = Field(
        description="Signed basis points for team_a: (consensus - kalshi) * 10000. Positive = team_a undervalued on Kalshi.",
    )
    line_movement_note: str
    sharp_money_signal: Literal["on_team_a", "on_team_b", "unclear", "no_data"]
    comparable_markets: list[str] = Field(
        default_factory=list,
        description="URLs or identifiers of the comparable markets consulted.",
    )


SpecialistReport = StatisticsReport | InjuryReport | NarrativeReport | MarketReport


class EventPrediction(BaseModel):
    """Event-level synthesis emitted by the director (LLM output)."""

    event_ticker: str
    predicted_winner: str = Field(
        description="Exact yes_sub_title of the team expected to win. Must match one of the event's markets.",
    )
    predicted_winner_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probability the predicted winner actually wins, 0-1.",
    )
    recommendation: Literal["buy_winner", "pass"] = Field(
        description="buy_winner = back the predicted winner on their market. pass = no edge worth trading.",
    )
    confidence: Confidence
    reasoning: str = Field(
        description="3-6 sentences explaining the synthesis and how specialists were weighted.",
    )
    specialist_weights: dict[str, float] = Field(
        description="Per-specialist weight in [0,1]; should roughly sum to 1.",
    )
    disagreements_flagged: list[str] = Field(
        default_factory=list,
        description="Empty when specialists aligned.",
    )


class MarketPrediction(BaseModel):
    """Per-market projection of an EventPrediction (built deterministically)."""

    market_ticker: str
    event_ticker: str
    event_title: str | None = None
    predicted_winner: str
    predicted_yes_probability: float = Field(ge=0.0, le=1.0)
    kalshi_implied_probability: float = Field(ge=0.0, le=1.0)
    edge_bps: int
    recommendation: Literal["buy_yes", "buy_no", "pass"]
    confidence: Confidence
    reasoning: str = Field(
        description="3-6 sentences explaining the synthesis and how specialists were weighted.",
    )
    specialist_weights: dict[str, float] = Field(
        description="Per-specialist weight in [0,1]; should roughly sum to 1.",
    )
    disagreements_flagged: list[str] = Field(
        default_factory=list,
        description="Empty when specialists aligned.",
    )


class PositionSizing(BaseModel):
    """Kelly-based position sizing as a fraction of bankroll.

    All fractions are dimensionless (multiply by your actual bankroll to get a dollar stake).
    `full_kelly_fraction` is the pure Kelly-optimal stake; `half_kelly_fraction` is the common
    variance-reduction variant; `capped_half_kelly_fraction` additionally caps at
    KELLY_BANKROLL_CAP (0.25) for safety.
    """

    side: Literal["yes", "no", "none"] = Field(
        description="Which contract to buy. 'none' = do not size a position.",
    )
    entry_price_dollars: float | None = Field(
        description="Ask price used for the sizing calc (yes_ask or no_ask). None if unavailable.",
    )
    win_probability: float | None = Field(
        description="Probability of the chosen side winning (q if yes, 1-q if no). None if no side.",
    )
    edge: float = Field(
        description="Signed edge: win_probability - entry_price_dollars. 0 when side='none'.",
    )
    full_kelly_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description="Kelly-optimal fraction of bankroll. 0 when no +EV side.",
    )
    half_kelly_fraction: float = Field(ge=0.0, le=1.0)
    capped_half_kelly_fraction: float = Field(
        ge=0.0,
        le=KELLY_BANKROLL_CAP,
        description=f"Half-Kelly capped at {KELLY_BANKROLL_CAP:.0%} of bankroll — recommended stake.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Warnings such as 'director recommended buy but Kelly is -EV against ask'.",
    )


class SizedMarketPrediction(BaseModel):
    """Director's prediction + deterministic Kelly sizing."""

    prediction: MarketPrediction
    sizing: PositionSizing
