"""Tennis-specific lens set — three bespoke specialists tailored to the
signals tennis matches actually carry.

Lens roster:
- `tennis_form_and_surface` — recent quality + this-surface fit. Carries
  the baseline `team_a_win_probability` plus form/surface signed shifts.
- `tennis_matchup_and_clutch` — H2H + tactical fit + pressure handling.
  Owns handedness and career BP-save/convert percentages as the
  sport-specific edges. Emits H2H + clutch signed shifts.
- `tennis_conditions_and_context` — physical match-day reality + stakes.
  Court conditions, weather, fatigue from prior round, current niggling
  injuries, ranking-points/title stakes, narrative. Emits physical +
  stakes signed shifts.

Each lens is two-stage (provider fetcher → Claude reasoner) and emits a
sport-specific Pydantic report. The director composes:

    baseline = tennis_form_and_surface.team_a_win_probability
    + form_signed_shift + surface_signed_shift
    + h2h_signed_shift + clutch_signed_shift
    + physical_signed_shift + stakes_signed_shift
    → clip to [0, 1]

…then committed to as the verdict — the director is blind to the market
price, so there is no market anchoring. The stacking math lives in
`DIRECTOR_SYSTEM_TENNIS_TAIL` explicitly so the reasoners' shifts don't
get double-counted.
"""

from skimsmarkets.agents.sports.tennis.lens_set import TENNIS_LENS_SET
from skimsmarkets.agents.sports.tennis.schemas import (
    TennisConditionsContextReport,
    TennisFormSurfaceReport,
    TennisMatchupClutchReport,
)

__all__ = [
    "TENNIS_LENS_SET",
    "TennisConditionsContextReport",
    "TennisFormSurfaceReport",
    "TennisMatchupClutchReport",
]
