"""Tennis player-stats enrichment.

Wraps the optional third-party tennis stats vendor that feeds two of
the three tennis lenses with structured player data (rankings, surface
splits, recent form, head-to-head, fatigue primitives).
Provider-agnostic by design: the `TennisStatsProvider` Protocol
mirrors `agents/fetchers/base.py:FetcherProvider` so a real adapter
can be dropped in next to the stub without touching the pipeline
wiring.

Scope is narrow on purpose:
- Only ATP/WTA singles head-to-heads. Doubles, qualifiers, and mixed-tour
  novelty markets fall through `tennis_match_identity` to None.
- The `tennis_form_and_surface` lens consumes a form-scoped block —
  rankings, surface splits, recent form, career serve/return, tier
  records, career titles. Routed via `render_tennis_form_block`. It
  EXCLUDES H2H and clutch primitives — those are matchup-owned.
- The `tennis_matchup_and_clutch` lens consumes a matchup-scoped
  block — H2H counts + per-surface H2H + recent meetings +
  matchup-conditioned per-player records (deciders, tiebreaks,
  set-1 conversions, in-matchup serve/BP) + career BP-save / BP-
  convert + handedness. Routed via `render_tennis_matchup_block`.
  Returns None when nothing clutch-relevant is populated.
- The `tennis_conditions_and_context` lens consumes a NARROW
  fatigue-only slice — `days_since_last_match`,
  `match_count_last_14d` per player — derived from
  `last_match_date` + `recent_matches` (same source data, different
  scoped view). Routed via `render_tennis_fatigue_block`. Web-search
  owns the fatigue inputs not on MatchStat's surface
  (travel/timezone, retirement frequency, medical timeouts).
- The director receives the simulator + GBT priors but no raw
  tennis-stats render — the lens reports synthesise that data.

Import shape — only models and rendering are re-exported here.
`identity` and `provider` import `PolymarketEvent` from
`skimsmarkets.polymarket.models`, which itself imports
`TennisStatsContext` from `tennis.models`. Re-exporting
identity/provider from this `__init__` would close the cycle and
break package import. Callers that need them go directly to the
submodule:
`from skimsmarkets.tennis.identity import tennis_match_identity` /
`from skimsmarkets.tennis.provider import build_tennis_provider`.
"""

from skimsmarkets.tennis.models import (
    TennisGbtContext,
    TennisGbtFeatureContribution,
    TennisH2HMeeting,
    TennisHeadToHead,
    TennisInMatchupStats,
    TennisPlayerStats,
    TennisRecentMatch,
    TennisSimulationContext,
    TennisStatsContext,
)
from skimsmarkets.tennis.rendering import (
    render_tennis_fatigue_block,
    render_tennis_form_block,
    render_tennis_gbt_block,
    render_tennis_matchup_block,
    render_tennis_simulation_block,
)
from skimsmarkets.tennis.simulation import (
    detect_best_of,
    simulate_for_event,
    simulate_match,
)

__all__ = [
    "TennisGbtContext",
    "TennisGbtFeatureContribution",
    "TennisH2HMeeting",
    "TennisHeadToHead",
    "TennisInMatchupStats",
    "TennisPlayerStats",
    "TennisRecentMatch",
    "TennisSimulationContext",
    "TennisStatsContext",
    "detect_best_of",
    "render_tennis_fatigue_block",
    "render_tennis_form_block",
    "render_tennis_gbt_block",
    "render_tennis_matchup_block",
    "render_tennis_simulation_block",
    "simulate_for_event",
    "simulate_match",
]
