"""Default lens set assembly — bundles the three legacy specs into one
`LensSet` keyed `"default"`. Not registered in `SPORT_LENS_SETS` by
default; available to import for tests or for the soft-rollout escape
hatch documented in the plan.
"""

from __future__ import annotations

from skimsmarkets.agents.sports.base import LensSet, LensSpec
from skimsmarkets.agents.sports.default.prompts import (
    DIRECTOR_SYSTEM_DEFAULT_TAIL,
    INJURY_REASONER_SYSTEM,
    NARRATIVE_REASONER_SYSTEM,
    STATISTICS_REASONER_SYSTEM,
    injury_notebook_system,
    narrative_notebook_system,
    statistics_notebook_system,
)
from skimsmarkets.agents.sports.default.schemas import (
    InjuryReport,
    NarrativeReport,
    StatisticsReport,
)

DEFAULT_LENS_SET = LensSet(
    sport="default",
    lenses=(
        LensSpec(
            name="statistics",
            fetcher_system_builder=statistics_notebook_system,
            reasoner_system=STATISTICS_REASONER_SYSTEM,
            report_schema=StatisticsReport,
        ),
        LensSpec(
            name="injury",
            fetcher_system_builder=injury_notebook_system,
            reasoner_system=INJURY_REASONER_SYSTEM,
            report_schema=InjuryReport,
        ),
        LensSpec(
            name="narrative",
            fetcher_system_builder=narrative_notebook_system,
            reasoner_system=NARRATIVE_REASONER_SYSTEM,
            report_schema=NarrativeReport,
        ),
    ),
    director_system_tail=DIRECTOR_SYSTEM_DEFAULT_TAIL,
)
