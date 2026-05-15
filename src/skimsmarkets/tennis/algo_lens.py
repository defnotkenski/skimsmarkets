"""Algorithmic tennis lenses — deterministic stand-ins for the LLM
fetcher+reasoner pair on the two pure-stats lenses
(`tennis_form_and_surface`, `tennis_matchup_and_clutch`).

Lives alongside `tennis/simulation.py` and `tennis/gbt.py` for the same
reason: deterministic compute over `TennisStatsContext` returning a
Pydantic object the downstream director consumes unchanged. No LLM
calls; no market-price inputs (the LLM-blindness invariant carries
over).

The public entrypoints (`compute_form_surface_report`,
`compute_matchup_clutch_report`) take a `PolymarketEvent` so production
wiring stays uniform with the LLM lenses. Each one delegates to a
pure-`TennisStatsContext` scoring function (`_score_form_surface`,
`_score_matchup_clutch`) so the backtest harness in
`tennis/algo_backtest.py` can call the scoring directly against
synthetic snapshots without faking a `PolymarketEvent`.

Form-and-surface is implemented (V1, see `_score_form_surface`).
Matchup-and-clutch is still a placeholder pending the next iteration
session.
"""

from __future__ import annotations

import math

from skimsmarkets.agents.sports.tennis.schemas import (
    TennisFormGrade,
    TennisFormSurfaceReport,
    TennisMatchupClutchReport,
)
from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.tennis.models import TennisPlayerStats, TennisStatsContext
from skimsmarkets.tennis.simulation import simulate_match

# Shift bounds from the schemas — re-asserted here so the algorithm
# self-documents what it's clipping against.
_FORM_SHIFT_CAP = 0.15
_SURFACE_SHIFT_CAP = 0.10

# Cold-start gates. The algo emits a useful baseline as long as both
# players have SOME priors; below these denominators the corresponding
# shift falls back to zero with the form-grade dropped to 'average'.
_MIN_PRIORS_BASELINE = 5      # any career evidence at all
_MIN_PRIORS_SURFACE = 8       # on-surface matches per player
_MIN_PRIORS_FORM = 5          # recent-form sample size


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
    """Production entrypoint for tennis_form_and_surface."""
    if event.tennis_stats is None:
        return None
    names = _resolve_team_names(event)
    if names is None:
        return None
    team_a, team_b = names
    return _score_form_surface(team_a, team_b, event.tennis_stats)


def compute_matchup_clutch_report(
    event: PolymarketEvent,
) -> TennisMatchupClutchReport | None:
    """Production entrypoint for tennis_matchup_and_clutch. STUB —
    deterministic scoring not yet implemented; emits neutral signal."""
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
        key_matchup_facts=[
            "Algorithmic matchup-and-clutch lens stub — deterministic "
            "scoring not yet implemented; do not weight this report's signal."
        ],
        caveats=[
            "Algorithmic lens stub; do not weight this report's signal."
        ],
        confidence="low",
    )


# ---------------------------------------------------------------------------
# Pure scoring functions — take a `TennisStatsContext` directly so the
# backtest harness can call them without faking a `PolymarketEvent`.
# Production callers go through the `compute_*_report` entrypoints
# above, which resolve names from the event then delegate here.
# ---------------------------------------------------------------------------


def _score_form_surface(
    team_a_name: str,
    team_b_name: str,
    ts: TennisStatsContext,
    *,
    best_of: int = 3,
) -> TennisFormSurfaceReport:
    """Deterministic form-and-surface scoring (V1).

    Baseline: Log5 over a serve-vs-return per-point win probability
    differential. This mirrors what the GBT's highest-importance
    features capture (career_first_serve_win_pct_diff +
    career_second_serve_return_win_pct_diff dominate the importance
    table at ~27% combined) without needing a learned model.

    Form shift: recent-form winrate delta (last-10 from each player's
    recent ring buffer), clipped to ±0.15.

    Surface shift: on-surface winrate delta scaled by min-sample
    confidence, clipped to ±0.10.

    Confidence: 'high' when both players have ≥8 on-surface matches +
    serve/return rates present + recent form populated; 'medium' when
    most are present but one player is thin on surface or rates;
    'low' otherwise.
    """
    a, b = ts.player_a, ts.player_b
    surface = ts.surface  # may be None

    # --- Baseline: per-point Klaassen-Magnus + Monte Carlo match sim -----
    # Delegate the per-point → match conversion to the existing iid
    # simulator. It uses the same symmetric serve/return blend a closed
    # form would, but walks the actual game/set/match grammar so the
    # extremes don't saturate. N_sims=300 trades ~1pp precision for
    # ~1ms per match — workable on a 17k-match holdout walk.
    # Seed via player ids for cross-run determinism.
    seed = (a.api_player_id or "") + (b.api_player_id or "")
    seed_int = abs(hash(seed)) & 0xFFFFFFFF
    sim = simulate_match(ts, best_of=best_of, n_sims=300, seed=seed_int)
    if sim is not None:
        baseline = sim.p_team_a_wins
    else:
        # Serve/return rates missing on at least one side. Fall back to
        # career win-rate Log5 so the lens still emits a baseline.
        baseline = _log5(_career_winrate(a), _career_winrate(b))

    # --- Form shift: last-10 winrate delta --------------------------------
    # Calibrated against the GBT's feature importance: `last_n_winrate_diff`
    # is only ~3% of total signal, so the shift magnitude should be small
    # relative to the schema's [-0.15, +0.15] cap. Multiplier 0.10 +
    # internal cap 0.04 keep this lens's recency contribution proportional
    # to its actual information content.
    a_recent = _last_n_winrate(a.last_10_form)
    b_recent = _last_n_winrate(b.last_10_form)
    a_n_recent = _last_n_count(a.last_10_form)
    b_n_recent = _last_n_count(b.last_10_form)
    if (
        a_recent is not None and b_recent is not None
        and a_n_recent >= _MIN_PRIORS_FORM
        and b_n_recent >= _MIN_PRIORS_FORM
    ):
        form_shift = _clip(0.10 * (a_recent - b_recent), -0.04, 0.04)
    else:
        form_shift = 0.0

    # --- Surface shift: on-surface winrate delta --------------------------
    # Surface importance in the GBT is ~11% (`surface_record_diff` +
    # `surface_first_serve_win_pct_diff` combined), the highest of the
    # non-rate features. The sim baseline already implicitly captures
    # surface via career rates (which are dominated by hard-court matches
    # since most pro matches are on hard), so the on-surface delta is
    # adding ATTRIBUTABLE-TO-SURFACE signal on top of the iid prior.
    # Multiplier 0.15 + cap 0.06 calibrated against the importance share.
    a_surf = _surface_winrate(a, surface)
    b_surf = _surface_winrate(b, surface)
    a_surf_n = _surface_n(a, surface)
    b_surf_n = _surface_n(b, surface)
    if (
        a_surf is not None and b_surf is not None
        and a_surf_n >= _MIN_PRIORS_SURFACE
        and b_surf_n >= _MIN_PRIORS_SURFACE
    ):
        # Sample-size confidence scaling: shifts from thin on-surface
        # samples get downweighted. min(n)/30 saturates at 1.0 once
        # both players have ≥30 priors on this surface, the natural
        # point where sample noise stops dominating.
        conf_scale = min(1.0, math.sqrt(min(a_surf_n, b_surf_n) / 30.0))
        surface_shift = _clip(
            0.15 * (a_surf - b_surf) * conf_scale, -0.06, 0.06
        )
    else:
        surface_shift = 0.0

    # --- Confidence -------------------------------------------------------
    rich = (
        _has_serve_rates(a) and _has_serve_rates(b)
        and a_surf_n >= _MIN_PRIORS_SURFACE and b_surf_n >= _MIN_PRIORS_SURFACE
        and a_n_recent >= _MIN_PRIORS_FORM and b_n_recent >= _MIN_PRIORS_FORM
    )
    thin = not (_has_serve_rates(a) or _has_serve_rates(b)) or (
        a_surf_n < _MIN_PRIORS_BASELINE or b_surf_n < _MIN_PRIORS_BASELINE
    )
    confidence: str = "high" if rich else ("low" if thin else "medium")

    # --- Form grades ------------------------------------------------------
    a_grade = _grade_form(a_recent, a_n_recent)
    b_grade = _grade_form(b_recent, b_n_recent)

    # --- Transparency ledger ----------------------------------------------
    facts: list[str] = []
    caveats: list[str] = []
    if sim is not None:
        facts.append(
            f"sim baseline: P(a)={sim.p_team_a_wins:.3f} "
            f"(point-win a={sim.point_win_pct_a_serving:.3f} / "
            f"b={sim.point_win_pct_b_serving:.3f})"
        )
    else:
        facts.append(
            f"serve/return rates thin → falling back to win-rate Log5 baseline {baseline:.3f}"
        )
    if a_recent is not None and b_recent is not None:
        facts.append(
            f"recent form: a={a_recent:.2f} (n={a_n_recent}) vs b={b_recent:.2f} (n={b_n_recent}) → form_shift {form_shift:+.3f}"
        )
    if a_surf is not None and b_surf is not None and surface:
        facts.append(
            f"{surface} winrate: a={a_surf:.2f} (n={a_surf_n}) vs b={b_surf:.2f} (n={b_surf_n}) → surface_shift {surface_shift:+.3f}"
        )
    elif surface is None:
        caveats.append("surface unknown — surface_shift defaulted to 0")
    if confidence == "low":
        caveats.append("thin priors on serve/return or surface — baseline used cautiously")

    return TennisFormSurfaceReport(
        team_a_name=team_a_name,
        team_b_name=team_b_name,
        team_a_win_probability=_clip(baseline, 0.001, 0.999),
        form_signed_shift=form_shift,
        surface_signed_shift=surface_shift,
        team_a_form_grade=a_grade,
        team_b_form_grade=b_grade,
        key_form_facts=facts,
        caveats=caveats,
        confidence=confidence,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Feature extraction helpers — pure functions over `TennisPlayerStats`.
# ---------------------------------------------------------------------------


def _career_winrate(p: TennisPlayerStats) -> float | None:
    """Career win-rate fallback when serve/return rates are missing.

    Uses ytd_win_loss (in the backtest projection this is set to the
    player's full career counters; in production it's vendor-defined
    YTD). When unavailable, returns None — the caller falls back to
    0.5 via Log5 symmetry.
    """
    if p.ytd_win_loss is None:
        return None
    w, losses = p.ytd_win_loss
    total = w + losses
    if total == 0:
        return None
    return w / total


def _log5(a_wr: float | None, b_wr: float | None) -> float:
    """Bill James's Log5 formula. Symmetric, well-behaved at the
    extremes, neutral when both inputs are None or 0.5."""
    if a_wr is None and b_wr is None:
        return 0.5
    a = a_wr if a_wr is not None else 0.5
    b = b_wr if b_wr is not None else 0.5
    a = max(0.001, min(0.999, a))
    b = max(0.001, min(0.999, b))
    num = a * (1.0 - b)
    den = a * (1.0 - b) + b * (1.0 - a)
    if den < 1e-9:
        return 0.5
    return num / den


def _last_n_winrate(form: str | None) -> float | None:
    if not form:
        return None
    n = len(form)
    if n == 0:
        return None
    wins = sum(1 for ch in form if ch.upper() == "W")
    return wins / n


def _last_n_count(form: str | None) -> int:
    return len(form) if form else 0


def _surface_winrate(p: TennisPlayerStats, surface: str | None) -> float | None:
    if surface is None or p.surface_win_loss is None:
        return None
    rec = p.surface_win_loss.get(surface)
    if rec is None:
        return None
    w, losses = rec
    total = w + losses
    if total == 0:
        return None
    return w / total


def _surface_n(p: TennisPlayerStats, surface: str | None) -> int:
    if surface is None or p.surface_win_loss is None:
        return 0
    rec = p.surface_win_loss.get(surface)
    if rec is None:
        return 0
    return rec[0] + rec[1]


def _has_serve_rates(p: TennisPlayerStats) -> bool:
    return (
        p.first_serve_in_pct is not None
        and p.first_serve_win_pct is not None
        and p.second_serve_win_pct is not None
        and p.first_serve_return_win_pct is not None
    )


def _grade_form(recent_wr: float | None, n: int) -> TennisFormGrade:
    """Map recent winrate + sample size to a 5-tier form grade.

    Thresholds calibrated against typical tour win-rate distributions
    (top-10: 70%+; top-30: 60-70%; mid-tour: 45-60%; lower: <45%).
    """
    if recent_wr is None or n < _MIN_PRIORS_FORM:
        return "average"
    if recent_wr >= 0.80:
        return "elite"
    if recent_wr >= 0.65:
        return "strong"
    if recent_wr >= 0.50:
        return "average"
    if recent_wr >= 0.35:
        return "below_avg"
    return "poor"


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
