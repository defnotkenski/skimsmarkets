"""Algorithmic tennis lenses — deterministic stand-ins for the LLM
fetcher+reasoner pair on the two pure-stats lenses
(`tennis_form_and_surface`, `tennis_matchup_and_clutch`). Lives alongside
`tennis/simulation.py` and `tennis/gbt.py` for the same reason:
deterministic compute over `TennisStatsContext` returning a Pydantic
object the downstream director consumes unchanged. No LLM calls; no
market-price inputs (the LLM-blindness invariant carries over).

STUB ONLY for now — both functions emit neutral signal (baseline 0.5,
zero shifts, low confidence) with a caveat string. Wired behind
`ALGORITHMIC_LENSES_ENABLED` in `config.py` so default behaviour is
unchanged until the real scoring lands in a follow-up session.
"""

from __future__ import annotations

from skimsmarkets.agents.sports.tennis.schemas import (
    TennisFormSurfaceReport,
    TennisMatchupClutchReport,
)
from skimsmarkets.polymarket.models import PolymarketEvent

_PLACEHOLDER_NOTE = (
    "Algorithmic lens stub — deterministic scoring not yet implemented; "
    "do not weight this report's signal."
)


def _resolve_team_names(event: PolymarketEvent) -> tuple[str, str] | None:
    # Mirror of agents/fetchers/base.py:pick_team_a_market favorite-pick.
    # Kept local so this module doesn't reach into agents/ for one helper.
    candidates = [
        (m.yes_implied_probability or -1.0, m)
        for m in event.markets
        if m.yes_sub_title
    ]
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    team_a = candidates[0][1].yes_sub_title
    team_b = next(
        (m.yes_sub_title for _, m in candidates[1:] if m.yes_sub_title),
        None,
    )
    if team_a is None or team_b is None:
        return None
    return team_a, team_b


def compute_form_surface_report(
    event: PolymarketEvent,
) -> TennisFormSurfaceReport | None:
    if event.tennis_stats is None:
        return None
    names = _resolve_team_names(event)
    if names is None:
        return None
    team_a, team_b = names
    return TennisFormSurfaceReport(
        team_a_name=team_a,
        team_b_name=team_b,
        team_a_win_probability=0.5,
        form_signed_shift=0.0,
        surface_signed_shift=0.0,
        team_a_form_grade="average",
        team_b_form_grade="average",
        key_form_facts=[_PLACEHOLDER_NOTE],
        caveats=[_PLACEHOLDER_NOTE],
        confidence="low",
    )


def compute_matchup_clutch_report(
    event: PolymarketEvent,
) -> TennisMatchupClutchReport | None:
    if event.tennis_stats is None:
        return None
    names = _resolve_team_names(event)
    if names is None:
        return None
    team_a, team_b = names
    return TennisMatchupClutchReport(
        team_a_name=team_a,
        team_b_name=team_b,
        h2h_signed_shift=0.0,
        clutch_signed_shift=0.0,
        style_advantage="neutral",
        pressure_handler="neutral",
        key_matchup_facts=[_PLACEHOLDER_NOTE],
        caveats=[_PLACEHOLDER_NOTE],
        confidence="low",
    )
