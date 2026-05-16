"""Per-lens GBT for the tennis matchup_and_clutch lens.

Feature subset for the second tennis lens — the one that owns
H2H + style (handedness, big-server-vs-returner) + pressure handling
(tiebreaks, deciders, comebacks, close matches, BP save/convert).

Strictly excluded (per the lens schema in
`agents/sports/tennis/schemas.py:TennisMatchupClutchReport`):
  - surface, surface_record_diff, surface_first_serve_win_pct_diff
    (form_and_surface owns the surface effect)
  - last_n_winrate_diff, days_since_diff, age_diff, rank_diff,
    rank_points_diff (form_and_surface owns recent form + rank)
  - tier (general match-context, not matchup-specific)

The trainer is shared with `lens_gbt_form_surface.train_lens_gbt` —
this module only defines the feature-subset constants. To train, call
`train_lens_gbt(matches_df, profiles_df, feature_columns=LENS_MATCHUP_CLUTCH_ALL_COLUMNS)`.

Why this lens is the right next pivot after `form_and_surface` came
back "don't ship": its features overlap LESS with the rank+surface
columns that dominate `tennis_gbt_spike`'s importance table — the
top 2 features there (rank_points_diff=14.4%, rank_diff=9.7%) are
form_and_surface territory. matchup_and_clutch's feature subset
covers ~50% of the full-feature importance but is concentrated in a
distinct subspace (career rates + h2h + clutch), giving the
structural argument less force here. The structural argument STILL
applies in principle (these features ARE in ALL_FEATURE_COLUMNS),
but the "different subspace" might surface enough orthogonal signal
for PRIMARY to pass.
"""

from __future__ import annotations

# Numerics (21): career serve/return rates + BP + career clutch +
# H2H counts + H2H clutch sub-counts. The schema's "big-server-vs-
# returner dynamics" maps to the 5 career serve/return columns.
LENS_MATCHUP_CLUTCH_NUMERIC_COLUMNS: tuple[str, ...] = (
    # Career serve/return — style (big-server vs returner)
    "career_first_serve_in_pct_diff",
    "career_first_serve_win_pct_diff",
    "career_second_serve_win_pct_diff",
    "career_first_serve_return_win_pct_diff",
    "career_second_serve_return_win_pct_diff",
    # Career BP — pressure handling
    "career_bp_save_pct_diff",
    "career_bp_convert_pct_diff",
    # Career clutch — pressure handling (general)
    "career_tiebreak_winrate_diff",
    "career_decider_winrate_diff",
    "career_comeback_winrate_diff",
    "career_close_match_winrate_diff",
    # H2H — matchup record
    "h2h_advantage",
    "n_h2h_priors",
    # H2H clutch — matchup-conditioned pressure
    "h2h_decider_advantage",
    "n_h2h_decider_priors",
    "h2h_tiebreak_advantage",
    "n_h2h_tiebreak_priors",
    "h2h_comeback_advantage",
    "n_h2h_comeback_priors",
    "h2h_close_match_advantage",
    "n_h2h_close_match_priors",
)

# Categoricals (1): best_of. Decider opportunities differ in bo3 vs
# bo5 — clutch matters more at slam-stage. `surface` and `tier`
# excluded per the lens-ownership boundary.
LENS_MATCHUP_CLUTCH_CATEGORICAL_COLUMNS: tuple[str, ...] = ("best_of",)

LENS_MATCHUP_CLUTCH_ALL_COLUMNS: tuple[str, ...] = (
    LENS_MATCHUP_CLUTCH_NUMERIC_COLUMNS
    + LENS_MATCHUP_CLUTCH_CATEGORICAL_COLUMNS
)

__all__ = [
    "LENS_MATCHUP_CLUTCH_ALL_COLUMNS",
    "LENS_MATCHUP_CLUTCH_CATEGORICAL_COLUMNS",
    "LENS_MATCHUP_CLUTCH_NUMERIC_COLUMNS",
]
