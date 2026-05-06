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
- The `tennis_form_and_surface` lens consumes the FULL stats block —
  rankings, surface splits, career serve/return, tier records, titles,
  H2H — because that lens owns the plurality of the fields and the
  full block IS its structured payload. Routed via
  `render_tennis_stats_block`.
- The `tennis_conditions_and_context` lens consumes a NARROW
  fatigue-only slice — `days_since_last_match`,
  `match_count_last_14d` per player — derived from
  `last_match_date` + `recent_matches` (same source data, different
  scoped view). Routed via `render_tennis_fatigue_block`. The
  conditions lens needs the fatigue primitives but not the full
  payload; web-search owns the fatigue inputs not on MatchStat's
  surface (travel/timezone, retirement frequency, medical timeouts).
- The `tennis_matchup_and_clutch` lens and the director don't
  receive any structured tennis-stats render. The matchup lens
  web-searches H2H/style-fit primitives via its own fetcher.

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
    render_tennis_simulation_block,
    render_tennis_stats_block,
)
from skimsmarkets.tennis.simulation import (
    detect_best_of,
    simulate_for_event,
    simulate_match,
)

__all__ = [
    "TennisH2HMeeting",
    "TennisHeadToHead",
    "TennisInMatchupStats",
    "TennisPlayerStats",
    "TennisRecentMatch",
    "TennisSimulationContext",
    "TennisStatsContext",
    "detect_best_of",
    "render_tennis_fatigue_block",
    "render_tennis_simulation_block",
    "render_tennis_stats_block",
    "simulate_for_event",
    "simulate_match",
]
