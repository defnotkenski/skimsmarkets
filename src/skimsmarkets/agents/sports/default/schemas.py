"""Pydantic report schemas for the default lens set.

These three classes (`StatisticsReport`, `InjuryReport`, `NarrativeReport`)
were previously in `agents/schemas.py` as the only specialist reports
the system knew about. They moved here as part of the per-sport lens-set
refactor so each sport's bespoke schemas live alongside that sport's
prompts. The field semantics are unchanged from the pre-refactor versions
— no behavior change for any sport that opts into `DEFAULT_LENS_SET`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from skimsmarkets.agents.schemas import Confidence, PlayerStatus


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
