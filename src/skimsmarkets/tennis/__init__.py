"""Tennis player-stats enrichment.

Wraps the optional third-party tennis stats vendor that feeds the
statistics specialist with structured player data (rankings, surface
splits, recent form, head-to-head). Provider-agnostic by design: the
`TennisStatsProvider` Protocol mirrors `agents/fetchers/base.py:FetcherProvider`
so a real adapter can be dropped in next to the stub without touching the
pipeline wiring.

Scope is narrow on purpose:
- Only ATP/WTA singles head-to-heads. Doubles, qualifiers, and mixed-tour
  novelty markets fall through `tennis_match_identity` to None.
- Only the statistics lens consumes the data. The other three lenses and
  the director don't see it — these stats *are* what the statistics
  specialist exists to compute, so feeding them anywhere else conflates
  "data from Polymarket" with "data from a third-party vendor."

Import shape — only models and rendering are re-exported here. `identity`
and `provider` import `PolymarketEvent` from `skimsmarkets.polymarket.models`,
which itself imports `TennisStatsContext` from `tennis.models`. Re-exporting
identity/provider from this `__init__` would close the cycle and break
package import. Callers that need them go directly to the submodule:
`from skimsmarkets.tennis.identity import tennis_match_identity` /
`from skimsmarkets.tennis.provider import build_tennis_provider`.
"""

from skimsmarkets.tennis.models import (
    TennisHeadToHead,
    TennisPlayerStats,
    TennisStatsContext,
)
from skimsmarkets.tennis.rendering import render_tennis_stats_block

__all__ = [
    "TennisHeadToHead",
    "TennisPlayerStats",
    "TennisStatsContext",
    "render_tennis_stats_block",
]
