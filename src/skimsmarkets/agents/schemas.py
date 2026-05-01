from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["low", "medium", "high"]

LensName = Literal["statistics", "injury", "narrative", "market_context"]


class Citation(BaseModel):
    """A single primary-source pull from web_search / x_search.

    `retrieved_value` is the concrete fact lifted from the page (a stat line, a
    line move, an injury status), not a paraphrase — the reasoner should be
    able to use it without re-fetching. URLs MUST be ones the fetcher actually
    retrieved; never fabricated.
    """

    url: str
    claim: str = Field(description="One-line summary of what this source supports.")
    retrieved_value: str | None = Field(
        default=None,
        description="Concrete value pulled (e.g. '28-6 record', 'Out — knee').",
    )


class ComputedNumber(BaseModel):
    """A number derived via the fetcher's `code_execution` tool.

    Fetchers MUST surface every numeric derivation here (de-vig, log5, on/off
    splits, rating differentials) so the reasoner can use the value as-is
    rather than recomputing. `method` is a one-line note on the math; the
    reasoner trusts it without re-deriving.
    """

    label: str = Field(
        description="What was computed (e.g. 'devig_pinnacle_team_a', 'log5_team_a_baseline').",
    )
    value: float
    method: str = Field(
        description="One-line method note (e.g. 'log5 from team_a 0.62 vs team_b 0.55, neutral 0.50').",
    )


class LensNotebook(BaseModel):
    """Free-form research notebook emitted by a Grok fetcher for one lens.

    The fetcher's job is evidence capture, not judgment — so this schema has
    NO probability, NO signed shift, NO directional verdict. Those land in the
    typed report a downstream Claude reasoner emits from this notebook plus
    the same event context the fetcher saw.

    `research_notes` is intentionally free-form prose (sectioned by the
    fetcher) so Grok's adaptive search loop ("found X, now look up Y") can
    capture whatever it stumbles on without the schema predicting structure
    upfront. Structured citation + computed-number lists ride alongside so
    URLs and numbers stay machine-extractable.

    `coverage` is the fetcher's self-assessment of how thin/rich the evidence
    is — the reasoner downgrades `confidence` to `low` when this is `thin`.
    """

    model_config = ConfigDict(extra="ignore")

    lens: LensName
    team_a_name: str = Field(
        description="Echoed verbatim from the event context's team_a_name.",
    )
    team_b_name: str = Field(
        description="Echoed verbatim from the event context's team_b_name.",
    )
    research_notes: str = Field(
        description=(
            "Free-form prose. Bullet what was found and what's missing. "
            "Do NOT include a probability, signed shift, or directional verdict."
        ),
    )
    citations: list[Citation] = Field(default_factory=list)
    computed_numbers: list[ComputedNumber] = Field(default_factory=list)
    coverage: Literal["thin", "adequate", "rich"] = Field(
        description="'thin' when primary sources were unavailable.",
    )


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


class MarketContextReport(BaseModel):
    """Market-context lens: what Polymarket and sportsbook consensus are pricing,
    plus sharp-money signals. NOT an edge-hunting report — just a read of where
    the market stands so the director has context alongside the other specialists.
    """

    team_a_name: str
    team_b_name: str
    polymarket_implied_team_a_probability: float = Field(
        ge=0.0, le=1.0,
        description="Polymarket's implied probability for team_a winning (midpoint of yes bid/ask on team_a's market).",
    )
    consensus_team_a_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Fair probability team_a wins, implied by consensus sportsbooks / betting "
            "exchanges (de-vigged). Null when no comparable sportsbook market was found."
        ),
    )
    line_movement_note: str = Field(
        description="Short note on recent line movement, open-vs-current, or notable steam moves.",
    )
    sharp_money_signal: Literal["on_team_a", "on_team_b", "unclear", "no_data"]
    comparable_markets: list[str] = Field(
        default_factory=list,
        description="URLs or identifiers of the comparable markets consulted.",
    )


SpecialistReport = StatisticsReport | InjuryReport | NarrativeReport | MarketContextReport


class EventPrediction(BaseModel):
    """Event-level synthesis emitted by the director (LLM output).

    Confidence-ranker framing: the director names the likely winner and how sure
    it is. No buy/pass gate — ranking happens downstream by
    `predicted_winner_probability`.
    """

    event_id: str = Field(
        description="Polymarket event id this prediction applies to.",
    )
    predicted_winner: str = Field(
        description="Exact yes_sub_title of the team expected to win. Must match one of the event's markets.",
    )
    predicted_winner_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probability the predicted winner actually wins, 0-1.",
    )
    confidence: Confidence = Field(
        description=(
            "Robustness of the prediction to any single input being wrong — NOT a "
            "measure of how lopsided the matchup is. high = multiple specialists "
            "(and UW flow when present) independently support predicted_winner, so "
            "removing any one input leaves the winner unchanged; medium = most agree "
            "but one is load-bearing; low = predicted_winner flips if a single input "
            "is wrong, or specialists themselves mostly reported confidence='low'. "
            "A 52-48 call with all four lenses agreeing directionally IS high "
            "confidence; an 80-20 call resting entirely on one late injury report "
            "is low confidence."
        ),
    )
    headline: str = Field(
        description=(
            "ONE sentence, max ~20 words, plain English. The single most decisive "
            "reason this side wins — readable at a glance with no jargon. Example: "
            "'Lakers win behind a fully-healthy LeBron and a 7-game home win streak.' "
            "This is what shows up in the at-a-glance leaderboard; the long-form "
            "synthesis lives in `reasoning`."
        ),
    )
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
    uw_flow_note: str | None = Field(
        default=None,
        description=(
            "2-4 sentence observation of the Unusual Whales flow signals when "
            "a 'Flow signals (Unusual Whales...)' block was present in the event "
            "context. Cover: which tags fired and their magnitude, the direction "
            "of recent smart-money / contrarian-whale trades (buyers hitting ask "
            "vs sellers hitting bid), notable insider positions, MCI value/delta "
            "if meaningful, and whether flow agreed with or diverged from the "
            "sportsbook consensus. Null when UW had no coverage for this event."
        ),
    )


class MarketPrediction(BaseModel):
    """Per-market projection of an EventPrediction (built deterministically).

    Attaches the winning side's Polymarket slug + implied probability alongside
    the director's prediction so downstream sizing and reporting have a single
    self-contained record.
    """

    market_slug: str = Field(
        description="Polymarket slug of the market that represents the predicted winner's side.",
    )
    event_id: str
    event_title: str | None = None
    predicted_winner: str
    predicted_yes_probability: float = Field(ge=0.0, le=1.0)
    polymarket_implied_probability: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Polymarket's implied probability for the predicted winner side (midpoint of yes bid/ask).",
    )
    confidence: Confidence
    headline: str = Field(
        description="One-sentence glanceable summary — projected from EventPrediction.",
    )
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
    uw_flow_note: str | None = None
