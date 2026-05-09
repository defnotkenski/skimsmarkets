"""Pydantic report schemas for the tennis lens set.

Three classes, one per tennis lens. The director composes its
`predicted_winner_probability` from the baseline on
`TennisFormSurfaceReport.team_a_win_probability` plus six signed shifts
(two per lens), all clipped to [0, 1] at the end. The stacking math is
spelled out in `agents/sports/tennis/prompts.py:DIRECTOR_SYSTEM_TENNIS_TAIL`
so reasoners and the director use the shifts without double-counting.

Signed-shift sign convention: positive values push the synthesized
probability TOWARD `team_a` (the Polymarket favorite); negative values
push it toward `team_b`. The reasoners are told this explicitly in their
system prompts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from skimsmarkets.agents.schemas import ComputedNumber, Confidence, PlayerStatus

TennisFormGrade = Literal["poor", "below_avg", "average", "strong", "elite"]


class TennisFormSurfaceReport(BaseModel):
    """Tennis lens 1 of 3 — recent quality + surface fit.

    Carries the BASELINE probability for the tennis director's stacking
    math (`team_a_win_probability`). The other two tennis lenses emit
    only signed shifts; this lens is the lone source of the
    "absent-matchup-and-conditions" baseline.
    """

    team_a_name: str = Field(description="Exact yes_sub_title of team_a (the favorite).")
    team_b_name: str = Field(description="Exact yes_sub_title of team_b.")
    team_a_win_probability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Baseline probability team_a wins this match assuming neutral matchup "
            "and neutral conditions. The director uses this as the stacking "
            "baseline for the six signed shifts across the three tennis lenses."
        ),
    )
    form_signed_shift: float = Field(
        ge=-0.15,
        le=0.15,
        description=(
            "Signed shift in [-0.15, +0.15] toward team_a. Positive = recent "
            "form skews toward team_a, negative = toward team_b. Drives off "
            "last_10_form, recent_matches quality, ytd_win_loss."
        ),
    )
    surface_signed_shift: float = Field(
        ge=-0.10,
        le=0.10,
        description=(
            "Signed shift in [-0.10, +0.10] toward team_a from THIS-surface fit. "
            "Drives off surface_win_loss[surface] split and recent on-surface "
            "trajectory. Owns the surface effect entirely — the matchup lens's "
            "h2h_signed_shift should NOT also adjust for surface."
        ),
    )
    team_a_form_grade: TennisFormGrade = Field(
        description="Qualitative grade of team_a's recent form."
    )
    team_b_form_grade: TennisFormGrade = Field(
        description="Qualitative grade of team_b's recent form."
    )
    key_form_facts: list[str] = Field(
        default_factory=list,
        description="3-7 short bullets — most decisive form/surface evidence.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Thin samples, surface debutants, missing splits, etc.",
    )
    confidence: Confidence = Field(
        description="low if data was thin or stale; high when multiple form signals converge."
    )


class TennisMatchupClutchReport(BaseModel):
    """Tennis lens 2 of 3 — tactical fit + pressure handling for THIS matchup.

    Owns handedness (`plays`) and career BP-save/convert percentages as
    matchup-style signals. Reads in-matchup-conditioned clutch records
    from the structured tennis_stats payload.
    """

    team_a_name: str
    team_b_name: str
    h2h_signed_shift: float = Field(
        ge=-0.15,
        le=0.15,
        description=(
            "Signed shift in [-0.15, +0.15] toward team_a from H2H + style fit. "
            "Drives off head_to_head counts, recent_meetings, handedness "
            "matchup, big-server-vs-returner dynamics. Do NOT adjust for "
            "surface here — the form_and_surface lens's surface_signed_shift "
            "owns that effect. Surface H2H informs reasoning qualitatively "
            "but not the numeric shift."
        ),
    )
    clutch_signed_shift: float = Field(
        ge=-0.10,
        le=0.10,
        description=(
            "Signed shift in [-0.10, +0.10] toward team_a from clutch / pressure "
            "handling. Drives off in-matchup decider/tiebreak records, comeback "
            "and closeout rates, career BP-save/convert percentages."
        ),
    )
    style_advantage: Literal["team_a", "team_b", "neutral"] = Field(
        description=(
            "Game-style fit: lefty advantages, baseliner-vs-net-rusher, "
            "big-server-vs-returner. 'neutral' when no clear stylistic edge."
        ),
    )
    pressure_handler: Literal["team_a", "team_b", "neutral"] = Field(
        description=(
            "Who handles late-set / decider pressure better in THIS matchup. "
            "'neutral' when clutch records are comparable."
        ),
    )
    key_matchup_facts: list[str] = Field(
        default_factory=list,
        description="3-7 short bullets — most decisive matchup/clutch evidence.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Small H2H sample, first-time meeting, etc.",
    )
    confidence: Confidence = Field(
        description="low if H2H is sparse or matchup-conditioned data is thin."
    )


class TennisConditionsContextReport(BaseModel):
    """Tennis lens 3 of 3 — physical match-day reality + stakes.

    Mostly fetcher-search territory (court conditions, weather, fatigue,
    current niggling injuries, stakes/motivation, coaching, narrative).
    Uses `last_match_date` from the structured tennis_stats payload as
    the fatigue baseline.
    """

    team_a_name: str
    team_b_name: str
    physical_signed_shift: float = Field(
        ge=-0.15,
        le=0.15,
        description=(
            "Signed shift in [-0.15, +0.15] toward team_a combining FITNESS "
            "and COURT CONDITIONS. Drives off current niggling injuries, "
            "fatigue from prior rounds, weather forecast for the match window, "
            "court speed, ball brand, altitude, indoor/outdoor, time of day. "
            "Withdrawal-class signals (confirmed pre-match retirement) can "
            "reach the cap."
        ),
    )
    stakes_signed_shift: float = Field(
        ge=-0.10,
        le=0.10,
        description=(
            "Signed shift in [-0.10, +0.10] toward team_a from STAKES + "
            "MOTIVATION. Drives off ranking-points pressure, defending-title "
            "context, first-time-finalist nerves, draw-management heuristics, "
            "post-Slam letdown."
        ),
    )
    court_conditions_summary: str = Field(
        description=(
            "1-3 sentences on court speed, ball brand, weather forecast for the "
            "match window, altitude, indoor/outdoor, time-of-day scheduling. "
            "What the LLM searched for, in plain English."
        ),
    )
    fatigue_summary: str = Field(
        description=(
            "1-3 sentences on each player's tournament path so far — sets/games/minutes "
            "played, time since last match, time-zone shift if any."
        ),
    )
    stakes_summary: str = Field(
        description=(
            "1-3 sentences on what's on the line for each player — ranking-points "
            "pressure, defending title, first-time-finalist nerves, etc."
        ),
    )
    injury_concerns: list[PlayerStatus] = Field(
        default_factory=list,
        description=(
            "Current niggling injury / withdrawal-risk concerns. Reuses the existing "
            "PlayerStatus shape; `team` should be `team_a_name` or `team_b_name`."
        ),
    )
    lineup_confidence: Literal["confirmed", "probable", "uncertain"] = Field(
        description=(
            "'confirmed' once both players are on entry list AND have practiced "
            "same-day; 'probable' when entry list is final but warm-up issues "
            "reported; 'uncertain' otherwise."
        ),
    )
    confidence: Confidence = Field(
        description=(
            "Reasoning confidence in the signed shifts (distinct from "
            "`lineup_confidence`, which gauges entry-list certainty). 'low' "
            "when court/weather/injury evidence is thin or speculative, when "
            "no fatigue primitives are present and the player-load picture is "
            "unknown, or when stakes are uncertain. 'high' when fatigue "
            "primitives are present, weather and court conditions are well "
            "characterised, and either both players are confirmed healthy "
            "or one has a credible withdrawal-class flag. The director "
            "down-weights low-confidence shifts in `specialist_weights`."
        ),
    )
    computed_numbers: list[ComputedNumber] = Field(
        default_factory=list,
        description=(
            "Deterministic numeric scalars you derived for THIS event so "
            "retro grading can score the conditions lens against outcomes "
            "(the other two lenses inherit numbers from their fetchers' "
            "code_execution; conditions historically had none). Aim for "
            "3-6 entries when the picture supports them — see the reasoner "
            "sport hint for the suggested label conventions "
            "(`fatigue_index_a`, `weather_serve_drag_a`, "
            "`stakes_pressure_a`, etc., each in [0.0, 1.0] unless noted). "
            "Empty when the notebook is too thin to anchor any of them."
        ),
    )
