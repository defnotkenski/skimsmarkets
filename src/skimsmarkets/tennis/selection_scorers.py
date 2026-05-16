"""Candidate selector scoring functions for the pre-LLM tennis
selection-algorithm overhaul.

Each `score_*` function is a complete scorer the backtest harness can
run. The composite scorers (`make_composite_scorer`) layer tiers on
top of the rank-points base — the same shape the production algorithm
in `selection.py:_tennis_imbalance` uses (additive signed
contributions, with an optional multiplicative tier applied last,
clipped to `[0, 1]`).

The tier functions (`_tier_*`) return `TierContribution` so a
composite can be inspected (which tier contributed what) when
debugging an iteration. Each tier carries its own cap and degrades
gracefully when its underlying data is missing (returns a 0
contribution rather than excluding the whole event from scoring) —
matches the "any miss → 0, rest of calculation continues" posture of
the production algorithm.

Tier coverage as of v1.0 → v1.13:

  T-form              last-10 W/L alignment with points-favorite
  T-surface-wp        per-surface winrate diff (absolute)
  T-surface-relative  (surface_wp - global_wp) diff (relative specialism)
  T-h2h               H2H sample-size + surface-conditioned bonus (bonus-only)
  T-best-of           multiplicative ×1.15 on bo5
  T-serve             career first-serve-win-pct diff (NOVEL)
  T-return            career first-serve-return-win-pct diff (NOVEL)
  T-clutch-decider    career decider-winrate diff (NOVEL)
  T-clutch-tiebreak   career tiebreak-winrate diff (NOVEL)
  T-age               prime-distance asymmetry
  T-layoff            worst-side days-since-last-match penalty
  T-info-density      min-side match-count + surface-count info bonus
  T-round             round_id adjustment — early rounds boost, late rounds penalize

NOVEL = not in production `selection.py` today.

Each tier is tested in isolation against the rank-points base in
`tests_isolated.py` so the user can see per-tier lift before composing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from skimsmarkets.tennis.gbt_features import PlayerHistory
from skimsmarkets.tennis.models import TennisPlayerStats
from skimsmarkets.tennis.selection_backtest import (
    ScoreFn,
    score_rank_points_ratio,
)


# ---------------------------------------------------------------------------
# Caps — first-guess values mirroring the production algorithm where
# applicable. Tuned downstream via ablation.
# ---------------------------------------------------------------------------

CAP_FORM = 0.20
CAP_SURFACE_WP = 0.10
CAP_SURFACE_RELATIVE = 0.10
CAP_H2H_TOTAL = 0.175  # tier-bonus 0.05/0.10/0.15 + surface boost 0.025
CAP_SERVE = 0.10
CAP_RETURN = 0.10
CAP_CLUTCH_DECIDER = 0.10
CAP_CLUTCH_TIEBREAK = 0.05
CAP_AGE = 0.05
CAP_LAYOFF = 0.05
CAP_INFO_DENSITY = 0.10
CAP_ROUND = 0.05

# Sample-size floors before a tier fires.
MIN_FORM_SAMPLES = 5
MIN_SURFACE_SAMPLES = 5
MIN_RECENT_FOR_INFO = 8  # info-density saturates at 8 recent matches per side
MIN_SURFACE_FOR_INFO = 4  # info-density surface saturates at 4 surface matches
MIN_PRIME_AGE_LOW = 24
MIN_PRIME_AGE_HIGH = 29

# Multipliers.
MULT_BO5 = 1.15

# Layoff thresholds (days).
LAYOFF_FRESH_DAYS = 14
LAYOFF_STALE_DAYS = 42

# H2H tier bonuses.
H2H_THIN_THRESHOLD = 1
H2H_MEDIUM_THRESHOLD = 3
H2H_RICH_THRESHOLD = 6
H2H_THIN_BONUS = 0.05
H2H_MEDIUM_BONUS = 0.10
H2H_RICH_BONUS = 0.15

# Round adjustment — late-round matches (semis/finals) are between
# higher-quality opponents, hence more peer-vs-peer (less predictable).
# Early rounds skew toward favorite-vs-qualifier (more predictable).
# round_id 1 = early; higher = later (per MatchStat convention).
# Direction: positive on early rounds, negative on late.
ROUND_EARLY_BOOST = 0.05
ROUND_LATE_PENALTY = -0.05
# Per MatchStat convention: rounds 1-2 = R128/R64 (early), 3-4 = R32/R16
# (mid), 5-6 = QF/SF (late), 7 = F (latest). These cut points apply
# to the round_id integer in the parquet.
ROUND_EARLY_MAX = 2
ROUND_LATE_MIN = 5


# ---------------------------------------------------------------------------
# Tier contribution dataclass + composer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierContribution:
    """Signed contribution from one tier. The `cap` is for debugging
    only — `value` is already capped to `[-cap, +cap]` (or `[0, cap]`
    for bonus-only tiers).
    """

    name: str
    value: float
    cap: float


def _favorite_is_a(a_stats: TennisPlayerStats, b_stats: TennisPlayerStats) -> bool | None:
    """Return True if A is the points-favorite, False if B, None if
    points data is missing on either side (no favorite determinable;
    signed tiers should contribute 0).
    """
    a_pts = a_stats.rank_points
    b_pts = b_stats.rank_points
    if a_pts is None or b_pts is None or a_pts <= 0 or b_pts <= 0:
        return None
    return a_pts >= b_pts


def _clip(value: float, cap: float) -> float:
    """Clip to [-cap, +cap]."""
    return max(-cap, min(cap, value))


# ---------------------------------------------------------------------------
# Individual tier functions. Each consumes the harness kwargs via
# **kwargs and returns a TierContribution. Stateless / pure.
# ---------------------------------------------------------------------------


def _tier_form(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    **_: Any,
) -> TierContribution:
    """Last-10 form W/L alignment with the points-favorite.

    Positive when the points-favorite ALSO has the higher recent W/L
    rate. Negative when the underdog is on a better recent run than
    the favorite — matchup tighter than rank suggests.
    """
    a_form = a_stats.last_10_form
    b_form = b_stats.last_10_form
    if a_form is None or b_form is None:
        return TierContribution("form", 0.0, CAP_FORM)
    if len(a_form) < MIN_FORM_SAMPLES or len(b_form) < MIN_FORM_SAMPLES:
        return TierContribution("form", 0.0, CAP_FORM)
    a_wp = a_form.count("W") / len(a_form)
    b_wp = b_form.count("W") / len(b_form)
    fav = _favorite_is_a(a_stats, b_stats)
    if fav is None:
        return TierContribution("form", 0.0, CAP_FORM)
    high_wp, low_wp = (a_wp, b_wp) if fav else (b_wp, a_wp)
    return TierContribution("form", CAP_FORM * (high_wp - low_wp), CAP_FORM)


def _tier_surface_winrate(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    surface: str | None,
    **_: Any,
) -> TierContribution:
    """Absolute per-surface winrate diff, signed by points-favorite.

    Positive when the favorite has the higher absolute winrate ON
    this surface specifically. Reads from `surface_win_loss`, which
    the backtest projects from HistoryStore.by_surface aggregates.
    """
    if surface is None:
        return TierContribution("surface_wp", 0.0, CAP_SURFACE_WP)
    fav = _favorite_is_a(a_stats, b_stats)
    if fav is None:
        return TierContribution("surface_wp", 0.0, CAP_SURFACE_WP)
    a_surf = (a_stats.surface_win_loss or {}).get(surface)
    b_surf = (b_stats.surface_win_loss or {}).get(surface)
    if a_surf is None or b_surf is None:
        return TierContribution("surface_wp", 0.0, CAP_SURFACE_WP)
    a_w, a_l = a_surf
    b_w, b_l = b_surf
    if a_w + a_l < MIN_SURFACE_SAMPLES or b_w + b_l < MIN_SURFACE_SAMPLES:
        return TierContribution("surface_wp", 0.0, CAP_SURFACE_WP)
    a_wp = a_w / (a_w + a_l)
    b_wp = b_w / (b_w + b_l)
    high_wp, low_wp = (a_wp, b_wp) if fav else (b_wp, a_wp)
    return TierContribution(
        "surface_wp", CAP_SURFACE_WP * (high_wp - low_wp), CAP_SURFACE_WP
    )


def _tier_surface_relative(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    surface: str | None,
    **_: Any,
) -> TierContribution:
    """Relative-specialism diff: `(surface_wp - global_wp)` per side,
    signed by points-favorite.

    Positive when the favorite is the relative specialist on this
    surface (their surface bonus is bigger than the opponent's).
    Captures specialism that's masked by absolute differences — e.g.
    two top-10 players where one is +5pp better on clay relative to
    their hard, and the other is +0pp.
    """
    if surface is None:
        return TierContribution("surface_relative", 0.0, CAP_SURFACE_RELATIVE)
    fav = _favorite_is_a(a_stats, b_stats)
    if fav is None:
        return TierContribution("surface_relative", 0.0, CAP_SURFACE_RELATIVE)
    a_ytd = a_stats.ytd_win_loss
    b_ytd = b_stats.ytd_win_loss
    a_surf = (a_stats.surface_win_loss or {}).get(surface)
    b_surf = (b_stats.surface_win_loss or {}).get(surface)
    if a_ytd is None or b_ytd is None or a_surf is None or b_surf is None:
        return TierContribution("surface_relative", 0.0, CAP_SURFACE_RELATIVE)
    a_g_w, a_g_l = a_ytd
    b_g_w, b_g_l = b_ytd
    a_s_w, a_s_l = a_surf
    b_s_w, b_s_l = b_surf
    if a_s_w + a_s_l < MIN_SURFACE_SAMPLES or b_s_w + b_s_l < MIN_SURFACE_SAMPLES:
        return TierContribution("surface_relative", 0.0, CAP_SURFACE_RELATIVE)
    if a_g_w + a_g_l == 0 or b_g_w + b_g_l == 0:
        return TierContribution("surface_relative", 0.0, CAP_SURFACE_RELATIVE)
    a_specialism = a_s_w / (a_s_w + a_s_l) - a_g_w / (a_g_w + a_g_l)
    b_specialism = b_s_w / (b_s_w + b_s_l) - b_g_w / (b_g_w + b_g_l)
    diff = a_specialism - b_specialism if fav else b_specialism - a_specialism
    return TierContribution(
        "surface_relative",
        _clip(CAP_SURFACE_RELATIVE * diff, CAP_SURFACE_RELATIVE),
        CAP_SURFACE_RELATIVE,
    )


def _tier_h2h(
    *,
    h2h_total_meetings: int = 0,
    surface: str | None = None,
    **_: Any,
) -> TierContribution:
    """H2H sample-size bonus + surface-conditioned boost. Bonus-only.

    Takes `h2h_total_meetings` as a primitive (computed upstream from
    PlayerHistory in backtest, from TennisHeadToHead in production).
    Surface-conditioned boost approximates: if surface is known AND
    the pair has ≥3 total meetings, add the surface bonus (assumes
    the pair likely played on this surface at least twice).
    """
    if h2h_total_meetings <= 0:
        return TierContribution("h2h", 0.0, CAP_H2H_TOTAL)
    n = h2h_total_meetings
    if n >= H2H_RICH_THRESHOLD:
        bonus = H2H_RICH_BONUS
    elif n >= H2H_MEDIUM_THRESHOLD:
        bonus = H2H_MEDIUM_BONUS
    elif n >= H2H_THIN_THRESHOLD:
        bonus = H2H_THIN_BONUS
    else:
        bonus = 0.0
    if surface is not None and n >= 3:
        bonus += 0.025
    return TierContribution("h2h", min(CAP_H2H_TOTAL, bonus), CAP_H2H_TOTAL)


def _tier_best_of_multiplier(
    *, best_of: int, **_: Any,
) -> float:
    """Multiplicative tier: bo5 amplifies imbalance by 15%."""
    return MULT_BO5 if best_of == 5 else 1.0


def _tier_serve_dominance(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    **_: Any,
) -> TierContribution:
    """Career first-serve-win-pct diff, signed by points-favorite.

    Positive when the favorite is the more dominant server (a structural
    advantage that the rank-points base doesn't directly capture —
    rank-points reflect wins, not the QUALITY of those wins). NOVEL —
    not in production `selection.py`.

    Reads from `TennisPlayerStats.first_serve_win_pct` — populated in
    both backtest (via `_project_player` from PlayerHistory career rate)
    and production (via `_player_match_stats`'s career aggregate). No
    `PlayerHistory` dependency so the tier ports cleanly.
    """
    fav = _favorite_is_a(a_stats, b_stats)
    if fav is None:
        return TierContribution("serve", 0.0, CAP_SERVE)
    a_sv = a_stats.first_serve_win_pct
    b_sv = b_stats.first_serve_win_pct
    if a_sv is None or b_sv is None:
        return TierContribution("serve", 0.0, CAP_SERVE)
    diff = (a_sv - b_sv) if fav else (b_sv - a_sv)
    # First-serve-win-pct diffs are typically ±0.05 (tour players cluster
    # in 0.65-0.75). Scale by 5 so a 0.05 diff = max contribution.
    return TierContribution(
        "serve", _clip(CAP_SERVE * diff * 5.0, CAP_SERVE), CAP_SERVE
    )


def _tier_return_dominance(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    a_history: PlayerHistory | None,
    b_history: PlayerHistory | None,
    **_: Any,
) -> TierContribution:
    """Career first-serve-return-win-pct diff, signed by points-favorite.

    Positive when the favorite is the more dominant returner. NOVEL.
    Return-win-pct typically clusters 0.27-0.35 — same scale factor
    as serve.
    """
    if a_history is None or b_history is None:
        return TierContribution("return", 0.0, CAP_RETURN)
    fav = _favorite_is_a(a_stats, b_stats)
    if fav is None:
        return TierContribution("return", 0.0, CAP_RETURN)
    a_rt = a_history.career_first_serve_return_win_pct()
    b_rt = b_history.career_first_serve_return_win_pct()
    if a_rt is None or b_rt is None:
        return TierContribution("return", 0.0, CAP_RETURN)
    diff = (a_rt - b_rt) if fav else (b_rt - a_rt)
    return TierContribution(
        "return", _clip(CAP_RETURN * diff * 5.0, CAP_RETURN), CAP_RETURN
    )


def _tier_clutch_decider(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    a_history: PlayerHistory | None,
    b_history: PlayerHistory | None,
    **_: Any,
) -> TierContribution:
    """Career decider-set winrate diff, signed by points-favorite. NOVEL.

    Decider winrate typically clusters 0.40-0.65; a 0.10 diff is
    meaningful. Scale by 2 so a 0.05 diff ≈ half the cap.
    """
    if a_history is None or b_history is None:
        return TierContribution("clutch_decider", 0.0, CAP_CLUTCH_DECIDER)
    fav = _favorite_is_a(a_stats, b_stats)
    if fav is None:
        return TierContribution("clutch_decider", 0.0, CAP_CLUTCH_DECIDER)
    a_d = a_history.career_decider_winrate()
    b_d = b_history.career_decider_winrate()
    if a_d is None or b_d is None:
        return TierContribution("clutch_decider", 0.0, CAP_CLUTCH_DECIDER)
    diff = (a_d - b_d) if fav else (b_d - a_d)
    return TierContribution(
        "clutch_decider",
        _clip(CAP_CLUTCH_DECIDER * diff * 2.0, CAP_CLUTCH_DECIDER),
        CAP_CLUTCH_DECIDER,
    )


def _tier_clutch_tiebreak(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    a_history: PlayerHistory | None,
    b_history: PlayerHistory | None,
    **_: Any,
) -> TierContribution:
    """Career tiebreak winrate diff, signed by points-favorite. NOVEL.

    Tiebreak winrate clusters 0.40-0.60; smaller cap than decider
    because tiebreaks are higher variance.
    """
    if a_history is None or b_history is None:
        return TierContribution("clutch_tiebreak", 0.0, CAP_CLUTCH_TIEBREAK)
    fav = _favorite_is_a(a_stats, b_stats)
    if fav is None:
        return TierContribution("clutch_tiebreak", 0.0, CAP_CLUTCH_TIEBREAK)
    a_tb = a_history.career_tiebreak_winrate()
    b_tb = b_history.career_tiebreak_winrate()
    if a_tb is None or b_tb is None:
        return TierContribution("clutch_tiebreak", 0.0, CAP_CLUTCH_TIEBREAK)
    diff = (a_tb - b_tb) if fav else (b_tb - a_tb)
    return TierContribution(
        "clutch_tiebreak",
        _clip(CAP_CLUTCH_TIEBREAK * diff * 2.0, CAP_CLUTCH_TIEBREAK),
        CAP_CLUTCH_TIEBREAK,
    )


def _tier_age(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    **_: Any,
) -> TierContribution:
    """Prime-distance asymmetry — penalize favorites whose rank-points
    are over- or under-accumulated relative to age.

    Negative when the favorite is FURTHER from prime years (24-29)
    than the underdog. Older favorites with inflated points get a
    points haircut; very young favorites likewise (career still
    accumulating). Tanh envelope keeps a 35-vs-22 from dominating
    a 30-vs-25.
    """
    if a_stats.age_years is None or b_stats.age_years is None:
        return TierContribution("age", 0.0, CAP_AGE)
    fav = _favorite_is_a(a_stats, b_stats)
    if fav is None:
        return TierContribution("age", 0.0, CAP_AGE)
    center = (MIN_PRIME_AGE_LOW + MIN_PRIME_AGE_HIGH) / 2
    half_width = (MIN_PRIME_AGE_HIGH - MIN_PRIME_AGE_LOW) / 2
    def prime_distance(age: int) -> float:
        return max(0.0, abs(age - center) - half_width)
    fav_dist = prime_distance(a_stats.age_years) if fav else prime_distance(b_stats.age_years)
    und_dist = prime_distance(b_stats.age_years) if fav else prime_distance(a_stats.age_years)
    fav_disadv = fav_dist - und_dist
    return TierContribution(
        "age", -CAP_AGE * math.tanh(fav_disadv / 2.0), CAP_AGE,
    )


def _tier_layoff(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    match_date,
    **_: Any,
) -> TierContribution:
    """Worst-side days-since-last-match penalty. Direction-AGNOSTIC —
    a long layoff on EITHER side weakens the lens chains' recent-form
    read (proxy: predictability dips).

    Ramps linearly from 0 at `LAYOFF_FRESH_DAYS` to `-CAP_LAYOFF` at
    `LAYOFF_STALE_DAYS`.
    """
    a_last = a_stats.last_match_date
    b_last = b_stats.last_match_date
    if a_last is None or b_last is None:
        return TierContribution("layoff", 0.0, CAP_LAYOFF)
    a_days = (match_date - a_last).days
    b_days = (match_date - b_last).days
    worst = max(a_days, b_days)
    if worst <= LAYOFF_FRESH_DAYS:
        return TierContribution("layoff", 0.0, CAP_LAYOFF)
    if worst >= LAYOFF_STALE_DAYS:
        return TierContribution("layoff", -CAP_LAYOFF, CAP_LAYOFF)
    span = LAYOFF_STALE_DAYS - LAYOFF_FRESH_DAYS
    progress = (worst - LAYOFF_FRESH_DAYS) / span
    return TierContribution("layoff", -CAP_LAYOFF * progress, CAP_LAYOFF)


def _tier_info_density(
    *,
    a_total_matches: int = 0,
    b_total_matches: int = 0,
    a_surface_matches: int = 0,
    b_surface_matches: int = 0,
    surface: str | None = None,
    **_: Any,
) -> TierContribution:
    """Direction-AGNOSTIC info-density bonus/penalty. Captures whether
    the lens chains have material to reason with — bottlenecked by
    the WORSE-served side (min, not avg).

    Takes match counts as primitives (computed upstream from
    PlayerHistory in backtest, from MatchHistoryRow lists in
    production). Two components combined:
      - Recent-match depth: min(a, b) / RECENT_HIGH
      - Surface-specific depth (when surface known): min surface
        count / SURFACE_HIGH
    Combined [0, 1] mapped to [-cap, +cap] via (combined - 0.5) * 2.
    Pivot at 0.5 → info-rich matchups get bonus, info-poor penalty.
    """
    if a_total_matches <= 0 or b_total_matches <= 0:
        return TierContribution("info_density", 0.0, CAP_INFO_DENSITY)
    n_min = min(a_total_matches, b_total_matches)
    recent_density = min(1.0, n_min / MIN_RECENT_FOR_INFO)
    if surface is not None:
        s_min = min(a_surface_matches, b_surface_matches)
        surface_density = min(1.0, s_min / MIN_SURFACE_FOR_INFO)
    else:
        surface_density = 0.5  # neutral
    combined = 0.6 * recent_density + 0.4 * surface_density
    return TierContribution(
        "info_density",
        CAP_INFO_DENSITY * (combined - 0.5) * 2.0,
        CAP_INFO_DENSITY,
    )


def _tier_round(
    *, round_id: int | None, **_: Any,
) -> TierContribution:
    """Round-of-tournament adjustment. NOVEL — not in production algo.

    Early rounds (R128/R64) tend to be favorite-vs-qualifier mismatches
    → highly predictable → boost selection score.
    Late rounds (QF/SF/F) tend to be peer-vs-peer → less predictable →
    small penalty.
    Mid rounds (R32/R16) → neutral (0 contribution).
    """
    if round_id is None:
        return TierContribution("round", 0.0, CAP_ROUND)
    if round_id <= ROUND_EARLY_MAX:
        return TierContribution("round", ROUND_EARLY_BOOST, CAP_ROUND)
    if round_id >= ROUND_LATE_MIN:
        return TierContribution("round", ROUND_LATE_PENALTY, CAP_ROUND)
    return TierContribution("round", 0.0, CAP_ROUND)


# ---------------------------------------------------------------------------
# Composite scorer factory.
# ---------------------------------------------------------------------------


TierFn = Callable[..., TierContribution]
MultiplierFn = Callable[..., float]


def make_composite_scorer(
    name: str,
    tier_fns: list[TierFn],
    *,
    multiplier_fns: list[MultiplierFn] | None = None,
    base_fn: ScoreFn = score_rank_points_ratio,
) -> tuple[str, ScoreFn]:
    """Build a (name, scorer) pair where the scorer composes:
        base + sum(tier_contributions)  →  ×product(multipliers)  →  clip[0,1]

    `base_fn` defaults to `score_rank_points_ratio` — the v1.0 floor.
    `multiplier_fns` apply AFTER the additive composition (matches the
    production tier-multiplier behavior).
    """
    multipliers = multiplier_fns or []

    def score(**kwargs: Any) -> float:
        base = base_fn(**kwargs)
        tier_sum = sum(fn(**kwargs).value for fn in tier_fns)
        raw = base + tier_sum
        mult = 1.0
        for m_fn in multipliers:
            mult *= m_fn(**kwargs)
        return max(0.0, min(1.0, raw * mult))

    return name, score


# ---------------------------------------------------------------------------
# Pre-built ablation scorers — one tier each, for measuring isolated lift.
# ---------------------------------------------------------------------------


ABLATION_SCORERS: list[tuple[str, ScoreFn]] = [
    make_composite_scorer("v1.1_form", [_tier_form]),
    make_composite_scorer("v1.2_surface_wp", [_tier_surface_winrate]),
    make_composite_scorer("v1.3_surface_relative", [_tier_surface_relative]),
    make_composite_scorer("v1.4_h2h", [_tier_h2h]),
    make_composite_scorer(
        "v1.5_bo5", [], multiplier_fns=[_tier_best_of_multiplier]
    ),
    make_composite_scorer("v1.6_serve", [_tier_serve_dominance]),
    make_composite_scorer("v1.7_return", [_tier_return_dominance]),
    make_composite_scorer("v1.8_clutch_decider", [_tier_clutch_decider]),
    make_composite_scorer("v1.9_clutch_tiebreak", [_tier_clutch_tiebreak]),
    make_composite_scorer("v1.10_age", [_tier_age]),
    make_composite_scorer("v1.11_layoff", [_tier_layoff]),
    make_composite_scorer("v1.12_info_density", [_tier_info_density]),
    make_composite_scorer("v1.13_round", [_tier_round]),
]


def composite_all_winners(winning_tier_fns: list[TierFn], *, name: str = "v1_composite", multiplier_fns: list[MultiplierFn] | None = None) -> tuple[str, ScoreFn]:
    """Build the final composite from tiers that lifted in ablation."""
    return make_composite_scorer(name, winning_tier_fns, multiplier_fns=multiplier_fns)


# Public names exported for the ablation driver script.
TIER_REGISTRY: dict[str, TierFn] = {
    "form": _tier_form,
    "surface_wp": _tier_surface_winrate,
    "surface_relative": _tier_surface_relative,
    "h2h": _tier_h2h,
    "serve": _tier_serve_dominance,
    "return": _tier_return_dominance,
    "clutch_decider": _tier_clutch_decider,
    "clutch_tiebreak": _tier_clutch_tiebreak,
    "age": _tier_age,
    "layoff": _tier_layoff,
    "info_density": _tier_info_density,
    "round": _tier_round,
}

MULTIPLIER_REGISTRY: dict[str, MultiplierFn] = {
    "bo5": _tier_best_of_multiplier,
}


# ---------------------------------------------------------------------------
# v1_selection — the production-ready scorer.
#
# Empirically derived via 11 iterations of single-tier ablation + composition
# + cap fine-tuning + base-scaling sweep on the 127k-match parquet (2025+
# holdout, 895 evaluable per-day-per-tour slates at K=5). Picks landed in
# 0.2487 Lock precision / 0.6212 Lock-or-Lean precision — +29% / +15% lift
# over the rank-points-ratio baseline.
#
# Recipe (each piece earned its place; dropping any one drops precision):
#   - Base = 0.3 × rank_points_ratio.  The rank-points tier dominates if
#     left at 1.0; scaling it down lets the differential tiers below
#     control the ordering.  Sweep across [0.0, 1.3] in iteration 7
#     identified 0.3 as the optimum.
#   - + form alignment (cap 0.20 from production).
#   - + serve dominance × 2.5  (cap effectively 0.25).  Iteration 10
#     identified 2.5 as the sweet spot — 2.0 and 3.0 are within 0.002.
#     NOVEL — not in the production 10-tier algorithm.
#   - + h2h sample-size bonus (cap 0.175).
#   - + surface absolute winrate diff (cap 0.10).
#   - + info-density (cap 0.10).  Lifts LoL more than Lock; kept because
#     LoL was below target without it.
#
# Tiers tested and DROPPED (no lift at the v1_selection composition):
# surface_relative, clutch_decider, clutch_tiebreak, return, age, layoff,
# round, bo5_multiplier.  Some lifted in isolation but didn't survive
# composition with the chosen base/serve scaling.
#
# K-robust: beats rank_points at K=3 (+28%), K=5 (+29%), K=10 (+14%).
# No overfit: holdout > train at every K (recent years easier to predict).
# ---------------------------------------------------------------------------


def score_v1_selection(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    surface: str | None,
    # H2H and info-density primitives — callers (backtest harness or
    # production wrapper) extract these from PlayerHistory /
    # TennisHeadToHead / MatchHistoryRow lists upstream.
    h2h_total_meetings: int = 0,
    a_total_matches: int = 0,
    b_total_matches: int = 0,
    a_surface_matches: int = 0,
    b_surface_matches: int = 0,
    **_: Any,
) -> float:
    """The v1 tennis selection scorer.

    Returns a `[0, 1]` selection score (higher = better candidate).

    Composition (additive, clipped at the end):
        score = clip(
            0.3 * rank_points_ratio
            + tier_form
            + 2.5 * tier_serve
            + tier_h2h
            + tier_surface_wp
            + tier_info_density,
            0, 1
        )

    Each tier degrades to 0 contribution when its underlying data is
    missing — never aborts scoring. Missing primitives default to
    `0`/empty, which makes the corresponding tier neutral.

    See module docstring for the ablation history that produced this
    composition.
    """
    base = 0.3 * score_rank_points_ratio(a_stats=a_stats, b_stats=b_stats)
    contributions = (
        _tier_form(a_stats=a_stats, b_stats=b_stats).value
        + 2.5 * _tier_serve_dominance(a_stats=a_stats, b_stats=b_stats).value
        + _tier_h2h(
            h2h_total_meetings=h2h_total_meetings, surface=surface
        ).value
        + _tier_surface_winrate(
            a_stats=a_stats, b_stats=b_stats, surface=surface
        ).value
        + _tier_info_density(
            a_total_matches=a_total_matches,
            b_total_matches=b_total_matches,
            a_surface_matches=a_surface_matches,
            b_surface_matches=b_surface_matches,
            surface=surface,
        ).value
    )
    raw = base + contributions
    return max(0.0, min(1.0, raw))
