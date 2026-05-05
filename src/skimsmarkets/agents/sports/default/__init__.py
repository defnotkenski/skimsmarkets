"""Default lens set — the legacy three-lens trio (statistics / injury /
narrative) preserved as an architectural anchor and test fixture.

NOT registered in `SPORT_LENS_SETS` by default. Strict-declaration
posture: events whose `sport_type` has no bespoke registration drop
with `ErrorRecord(stage="lens_dispatch")`. The default set exists so:

- Tests can prove the registry's plumbing is sport-set-shape-agnostic by
  registering it under a synthetic sport.
- A future soft-rollout escape hatch can register it for specific sports
  (e.g. `SPORT_LENS_SETS["soccer"] = DEFAULT_LENS_SET`) if the
  tennis-only window proves too disruptive — one line per sport.
"""

from skimsmarkets.agents.sports.default.lens_set import DEFAULT_LENS_SET
from skimsmarkets.agents.sports.default.schemas import (
    InjuryReport,
    NarrativeFactor,
    NarrativeReport,
    StatisticsReport,
)

__all__ = [
    "DEFAULT_LENS_SET",
    "InjuryReport",
    "NarrativeFactor",
    "NarrativeReport",
    "StatisticsReport",
]
