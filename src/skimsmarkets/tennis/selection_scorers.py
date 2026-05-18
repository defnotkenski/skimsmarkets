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

Each tier is tested in isolation against the rank-points base via the
`ABLATION_SCORERS` list below (one composite per tier) so the user can
see per-tier lift before composing.
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
    # EV-mode tiers — see section below for definitions; registered via
    # `_register_ev_tiers()` at module-load to keep the dict declaration
    # readable without forward-defining all the tier functions.
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


# ---------------------------------------------------------------------------
# EV-mode scorers — new family (2026-05-17).
#
# Mathematical premise (from `memory/project_ev_selector.md` four-archetype
# table):
#   - Confidence-mode picks high-fundamental-imbalance events. Synthetic-EV
#     analogue: GBT picks the rank-favorite. EV ≈ 0 by construction.
#   - EV-mode wants events where GBT disagrees with rank-implied price.
#     Equivalent: events where TRUE skill (Elo, surface specialism, form,
#     serve quality) says something DIFFERENT from rank-implied probability.
#
# Pre-LLM signals that predict GBT-vs-rank divergence (every one of these
# is something the GBT itself uses as a feature — making them natural pre-
# LLM proxies for GBT's eventual disagreement with rank):
#
#   T-elo-rank-gap        |elo_implied(a) − rank_implied(a)|. The literature
#                         (Gorgi 2019, Yue 2022, Kovalchik 2020) calls Elo
#                         the single best market-beating signal in tennis;
#                         the gap with rank-implied is the headline EV
#                         signal.
#   T-underdog-form       hot form on the lower-ranked side. Captures
#                         "rank lags reality" — a classic upset profile.
#   T-underdog-serve      serve-dominance asymmetry where the underdog
#                         out-serves the favorite. Markets price wins, not
#                         serve quality; sharp tennis quants exploit this.
#   T-underdog-surface    surface specialism on the lower-ranked side.
#   T-underdog-elo        absolute Elo of the underdog. High Elo + low
#                         rank = "true skill not yet reflected in ranking
#                         points" (often: returning player, young breakout,
#                         player who skipped tournaments).
#   T-h2h-underdog        h2h favoring the underdog. When the lower-ranked
#                         player has actually beaten the favorite before,
#                         it's strong evidence of mispricing.
#   T-competitive-floor   penalize events with extreme rank ratios. Even
#                         when GBT disagrees with rank on a 10x-favorite,
#                         the absolute EV is capped by the tight market_p.
#
# Bridge between backtest signals and production signals:
#   - All `_tier_ev_*` tiers READ from PlayerHistory (for Elo + form +
#     career rates) OR TennisPlayerStats (for rank + form proxy). In the
#     backtest harness, both are populated from the parquet. In production
#     the `_tennis_imbalance_ev_v1` wrapper (selection.py) projects the
#     equivalent fields from the provider's warmed caches + a `_get_bundle()`
#     call to pull the live HistoryStore for Elo. Elo lookup is free
#     (in-memory; the bundle is loaded for GBT prediction anyway).
# ---------------------------------------------------------------------------


# Caps — first-guess values; tuned via ablation.
CAP_EV_ELO_RANK_GAP = 0.30
CAP_EV_UNDERDOG_FORM = 0.20
CAP_EV_UNDERDOG_SERVE = 0.20
CAP_EV_UNDERDOG_SURFACE = 0.20
CAP_EV_UNDERDOG_ELO = 0.20
CAP_EV_H2H_UNDERDOG = 0.15
CAP_EV_COMPETITIVE_FLOOR = 0.30

# Rank-points-ratio above which the event's EV ceiling is tight. The
# rank-implied market_p for a 10x-favorite is ~0.91, leaving payoff ≈
# 0.10 — so even a 10pp GBT disagreement (model_p = 0.81) is +EV but
# small. We bias toward more competitive events. log10(ratio) scale.
COMPETITIVE_RANK_LOG_RATIO_CUTOFF = 0.6  # ratio ≈ 4× → market_p ≈ 0.80


def _elo_implied_p(elo_a: float, elo_b: float) -> float:
    """Standard logistic Elo: P(A wins) = 1 / (1 + 10^((Bb − Ba)/400))."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _rank_implied_p(pts_a: int | None, pts_b: int | None) -> float | None:
    """Bradley-Terry rank-points-implied prob (matches the backtest
    harness's `_market_proxy_p_anchor`). Returns None on missing or
    non-positive points.
    """
    if pts_a is None or pts_b is None or pts_a <= 0 or pts_b <= 0:
        return None
    return pts_a / (pts_a + pts_b)


def _underdog_is_a(a_stats: TennisPlayerStats, b_stats: TennisPlayerStats) -> bool | None:
    """Inverse of `_favorite_is_a` — returns True if A is the rank-
    underdog. None when points are missing on either side."""
    fav = _favorite_is_a(a_stats, b_stats)
    return None if fav is None else (not fav)


def _tier_ev_elo_rank_gap(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    a_history: PlayerHistory | None,
    b_history: PlayerHistory | None,
    surface: str | None = None,
    **_: Any,
) -> TierContribution:
    """Magnitude of Elo-vs-rank disagreement. NOT signed by favorite —
    the gap itself is the EV signal regardless of direction.

    Uses surface-specific Elo when surface is known (Elo by surface is
    a feature on `PlayerHistory.elo_by_surface`; falls back to global
    Elo for the surface bucket the player hasn't played).

    Value: `|elo_implied(a) − rank_implied(a)| * scale`, capped.
    """
    if a_history is None or b_history is None:
        return TierContribution("ev_elo_rank_gap", 0.0, CAP_EV_ELO_RANK_GAP)
    if surface is not None:
        elo_a = a_history.elo_by_surface.get(surface, a_history.elo_global)
        elo_b = b_history.elo_by_surface.get(surface, b_history.elo_global)
    else:
        elo_a = a_history.elo_global
        elo_b = b_history.elo_global
    rank_p = _rank_implied_p(a_stats.rank_points, b_stats.rank_points)
    if rank_p is None:
        return TierContribution("ev_elo_rank_gap", 0.0, CAP_EV_ELO_RANK_GAP)
    elo_p = _elo_implied_p(elo_a, elo_b)
    gap = abs(elo_p - rank_p)
    # gap of 0.15 (e.g. Elo says 0.55, rank says 0.40) is a meaningful
    # disagreement. Scale so gap=0.15 → ~half cap.
    return TierContribution(
        "ev_elo_rank_gap",
        min(CAP_EV_ELO_RANK_GAP, gap * 2.0 * CAP_EV_ELO_RANK_GAP / 0.30),
        CAP_EV_ELO_RANK_GAP,
    )


def _tier_ev_underdog_form(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    **_: Any,
) -> TierContribution:
    """Hot form on the rank-UNDERDOG side. Signed positive when the
    underdog has a better recent W/L rate than the favorite — captures
    "rank lags reality" upset profile.

    Bonus-only: a cold underdog is just confirming rank, no extra EV.
    Form-winrate diff capped at 0.50 (e.g. underdog 9/10 vs favorite
    4/10) → full cap.
    """
    a_form = a_stats.last_10_form
    b_form = b_stats.last_10_form
    if a_form is None or b_form is None:
        return TierContribution("ev_underdog_form", 0.0, CAP_EV_UNDERDOG_FORM)
    if len(a_form) < MIN_FORM_SAMPLES or len(b_form) < MIN_FORM_SAMPLES:
        return TierContribution("ev_underdog_form", 0.0, CAP_EV_UNDERDOG_FORM)
    under = _underdog_is_a(a_stats, b_stats)
    if under is None:
        return TierContribution("ev_underdog_form", 0.0, CAP_EV_UNDERDOG_FORM)
    a_wp = a_form.count("W") / len(a_form)
    b_wp = b_form.count("W") / len(b_form)
    under_wp, fav_wp = (a_wp, b_wp) if under else (b_wp, a_wp)
    diff = under_wp - fav_wp
    if diff <= 0:
        return TierContribution("ev_underdog_form", 0.0, CAP_EV_UNDERDOG_FORM)
    return TierContribution(
        "ev_underdog_form",
        min(CAP_EV_UNDERDOG_FORM, diff * 2.0 * CAP_EV_UNDERDOG_FORM / 0.50),
        CAP_EV_UNDERDOG_FORM,
    )


def _tier_ev_underdog_serve(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    **_: Any,
) -> TierContribution:
    """Serve dominance on the rank-UNDERDOG side. Bonus-only.

    Markets price wins, not serve quality; a big-serving underdog is a
    classic mispricing archetype (literature: Wilkens 2021 *J. Sports
    Analytics*). First-serve-win-pct diff > 0 favoring underdog → bonus.
    """
    a_sv = a_stats.first_serve_win_pct
    b_sv = b_stats.first_serve_win_pct
    if a_sv is None or b_sv is None:
        return TierContribution("ev_underdog_serve", 0.0, CAP_EV_UNDERDOG_SERVE)
    under = _underdog_is_a(a_stats, b_stats)
    if under is None:
        return TierContribution("ev_underdog_serve", 0.0, CAP_EV_UNDERDOG_SERVE)
    under_sv, fav_sv = (a_sv, b_sv) if under else (b_sv, a_sv)
    diff = under_sv - fav_sv
    if diff <= 0:
        return TierContribution("ev_underdog_serve", 0.0, CAP_EV_UNDERDOG_SERVE)
    # First-serve-win-pct diffs cluster ±0.05; scale so 0.05 = full cap.
    return TierContribution(
        "ev_underdog_serve",
        min(CAP_EV_UNDERDOG_SERVE, diff * CAP_EV_UNDERDOG_SERVE / 0.05),
        CAP_EV_UNDERDOG_SERVE,
    )


def _tier_ev_underdog_surface(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    surface: str | None,
    **_: Any,
) -> TierContribution:
    """Surface winrate advantage on the rank-UNDERDOG side. Bonus-only.

    Surface specialism is a documented value signal (Gorgi 2019, Sipko
    2015). Captures "this lower-ranked player is actually the favorite
    on THIS surface" archetype.
    """
    if surface is None:
        return TierContribution("ev_underdog_surface", 0.0, CAP_EV_UNDERDOG_SURFACE)
    under = _underdog_is_a(a_stats, b_stats)
    if under is None:
        return TierContribution("ev_underdog_surface", 0.0, CAP_EV_UNDERDOG_SURFACE)
    a_surf = (a_stats.surface_win_loss or {}).get(surface)
    b_surf = (b_stats.surface_win_loss or {}).get(surface)
    if a_surf is None or b_surf is None:
        return TierContribution("ev_underdog_surface", 0.0, CAP_EV_UNDERDOG_SURFACE)
    a_w, a_l = a_surf
    b_w, b_l = b_surf
    if a_w + a_l < MIN_SURFACE_SAMPLES or b_w + b_l < MIN_SURFACE_SAMPLES:
        return TierContribution("ev_underdog_surface", 0.0, CAP_EV_UNDERDOG_SURFACE)
    a_wp = a_w / (a_w + a_l)
    b_wp = b_w / (b_w + b_l)
    under_wp, fav_wp = (a_wp, b_wp) if under else (b_wp, a_wp)
    diff = under_wp - fav_wp
    if diff <= 0:
        return TierContribution("ev_underdog_surface", 0.0, CAP_EV_UNDERDOG_SURFACE)
    return TierContribution(
        "ev_underdog_surface",
        min(CAP_EV_UNDERDOG_SURFACE, diff * 2.0 * CAP_EV_UNDERDOG_SURFACE / 0.40),
        CAP_EV_UNDERDOG_SURFACE,
    )


def _tier_ev_underdog_elo(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    a_history: PlayerHistory | None,
    b_history: PlayerHistory | None,
    **_: Any,
) -> TierContribution:
    """Absolute Elo on the rank-UNDERDOG side. Bonus when underdog's
    Elo is HIGH relative to typical tour pool — "true skill not yet
    reflected in rank points" archetype (returning player, young
    breakout, player who skipped tournaments).

    Tier-pool Elo on tour is ~1450-1700 with peaks ~2000. Bonus scales
    with how much the underdog's Elo exceeds 1500 (the ELO_INITIAL
    baseline; ~tour-median for active players).
    """
    if a_history is None or b_history is None:
        return TierContribution("ev_underdog_elo", 0.0, CAP_EV_UNDERDOG_ELO)
    under = _underdog_is_a(a_stats, b_stats)
    if under is None:
        return TierContribution("ev_underdog_elo", 0.0, CAP_EV_UNDERDOG_ELO)
    under_elo = a_history.elo_global if under else b_history.elo_global
    excess = max(0.0, under_elo - 1500.0)
    # Scale so 300 Elo points above 1500 (a top-50 caliber player) = full cap.
    return TierContribution(
        "ev_underdog_elo",
        min(CAP_EV_UNDERDOG_ELO, excess * CAP_EV_UNDERDOG_ELO / 300.0),
        CAP_EV_UNDERDOG_ELO,
    )


def _tier_ev_h2h_underdog(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    a_history: PlayerHistory | None,
    b_history: PlayerHistory | None,
    **_: Any,
) -> TierContribution:
    """H2H favoring the UNDERDOG. The lower-ranked player having
    actually beaten the favorite before is the strongest single piece
    of mispricing evidence (it's a literal counter-example to the
    rank-based prior).

    Bonus-only. Scaled by the underdog's share of past meetings:
    underdog wins / total meetings.
    """
    if a_history is None or b_history is None:
        return TierContribution("ev_h2h_underdog", 0.0, CAP_EV_H2H_UNDERDOG)
    under = _underdog_is_a(a_stats, b_stats)
    if under is None:
        return TierContribution("ev_h2h_underdog", 0.0, CAP_EV_H2H_UNDERDOG)
    # a_history.h2h_against returns (wins, total)
    a_wins, total = a_history.h2h_against(b_history.player_id)
    if total < 2:
        return TierContribution("ev_h2h_underdog", 0.0, CAP_EV_H2H_UNDERDOG)
    under_wins = a_wins if under else (total - a_wins)
    under_share = under_wins / total
    if under_share <= 0.5:
        return TierContribution("ev_h2h_underdog", 0.0, CAP_EV_H2H_UNDERDOG)
    # Scale: 0.5 → 0, 1.0 → full cap. (under_share - 0.5) * 2 = [0, 1]
    bonus = (under_share - 0.5) * 2.0 * CAP_EV_H2H_UNDERDOG
    return TierContribution("ev_h2h_underdog", bonus, CAP_EV_H2H_UNDERDOG)


def _tier_ev_competitive_floor(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    **_: Any,
) -> TierContribution:
    """Penalty on extreme-imbalance events. NEGATIVE contribution only.

    Even when GBT disagrees with rank on a 10x-rank-favorite, the
    absolute EV is capped by the tight market_p (market_p ≈ 0.91 →
    payoff ≈ 0.10). EV-mode should avoid these structurally-low-ceiling
    events. Penalty ramps in for ratios > 4× (log10 ratio > 0.6, where
    market_p > 0.80).
    """
    a_pts = a_stats.rank_points
    b_pts = b_stats.rank_points
    if a_pts is None or b_pts is None or a_pts <= 0 or b_pts <= 0:
        return TierContribution(
            "ev_competitive_floor", 0.0, CAP_EV_COMPETITIVE_FLOOR
        )
    ratio = max(a_pts, b_pts) / min(a_pts, b_pts)
    log_ratio = math.log10(ratio)
    if log_ratio <= COMPETITIVE_RANK_LOG_RATIO_CUTOFF:
        return TierContribution(
            "ev_competitive_floor", 0.0, CAP_EV_COMPETITIVE_FLOOR
        )
    # Ramps from 0 at log_ratio=0.6 to −CAP at log_ratio=1.0 (10× → market_p ≈ 0.91).
    overage = min(1.0, (log_ratio - COMPETITIVE_RANK_LOG_RATIO_CUTOFF) / 0.4)
    return TierContribution(
        "ev_competitive_floor",
        -CAP_EV_COMPETITIVE_FLOOR * overage,
        CAP_EV_COMPETITIVE_FLOOR,
    )


# Register EV tiers in the public registry so the ablation driver
# can compose them via `make()` the same way confidence-mode tiers
# are composed.
TIER_REGISTRY.update({
    "ev_elo_rank_gap": _tier_ev_elo_rank_gap,
    "ev_underdog_form": _tier_ev_underdog_form,
    "ev_underdog_serve": _tier_ev_underdog_serve,
    "ev_underdog_surface": _tier_ev_underdog_surface,
    "ev_underdog_elo": _tier_ev_underdog_elo,
    "ev_h2h_underdog": _tier_ev_h2h_underdog,
    "ev_competitive_floor": _tier_ev_competitive_floor,
})


# EV-mode default base — 0.4 (tuned via base sweep at v4.x ablation).
# Confidence mode bases its score on rank-points-ratio (favors lopsided
# matchups). EV mode skips the rank base entirely and rides ONLY on the
# EV tiers, with a small positive bias so events with zero positive tier
# signal still sit above the negative-only floor tier's pure penalty.
# Base in [0.0, 0.4] all measure within $0.001 realized return; 0.4 was
# the marginal best in the sweep. Above 0.4, clipping at score=1.0 starts
# losing differentiation between high-signal events.
EV_BASE = 0.4


def score_ev_v1_selection(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    surface: str | None,
    a_history: PlayerHistory | None = None,
    b_history: PlayerHistory | None = None,
    **_: Any,
) -> float:
    """v1 EV-mode tennis selection scorer.

    Returns a `[0, 1]` selection score targeting events likely to be
    high-EV bets in production (GBT-vs-market-price disagreement
    archetype). See module-level "EV-mode scorers" header for the
    design rationale and `memory/project_ev_selector.md` for the
    decision history.

    Composition (additive, clipped):
        score = clip(
            0.4  (small positive base — no rank-points anchor)
            + ev_elo_rank_gap           (LOAD-BEARING; cap 0.30)
            + ev_underdog_form          (cap 0.20)
            + ev_underdog_serve         (cap 0.20)
            + ev_underdog_surface       (cap 0.20)
            + ev_h2h_underdog           (cap 0.15)
            + ev_competitive_floor      (NEGATIVE-only; cap 0.30)
            , 0, 1)

    Tiers tested and DROPPED at this composition:
      - `ev_underdog_elo` (absolute underdog Elo) — REDUNDANT with
        `ev_elo_rank_gap` (both encode "Elo good for underdog"). Drop
        lifted realized return by $0.005/$1 on the v2 ablation.
      - `ev_surface_elo_gap` (surface-Elo vs rank-implied) — redundant
        with `ev_elo_rank_gap` (which already falls back to surface
        Elo when surface is known).
      - `ev_elo_momentum` (recent-form proxy on top of Elo) — redundant
        with `ev_underdog_form`.

    Backtest (2025+, K=5, 4378 EV-labelable picks):
        realized return per \\$1 = +\\$0.1475 vs base rate +\\$0.0747
        (+97% lift). v1_confidence selector: +\\$0.0478 (under base
        rate, i.e. ACTIVELY hurts EV mode — the design hypothesis
        from `project_ev_selector.md`). K-robust (K=3 +65%, K=10
        +39% vs baseline rank). Stronger on ATP (+$0.19) than WTA
        (+$0.10); per-tour tuning is a future lever.

    Production note: the `a_history`/`b_history` PlayerHistory args
    are the source of Elo. The backtest harness passes them directly;
    `selection._tennis_imbalance_ev_v1` pulls them from the GBT bundle
    (loaded for the prediction stage anyway, so free).
    """
    contributions = (
        _tier_ev_elo_rank_gap(
            a_stats=a_stats, b_stats=b_stats,
            a_history=a_history, b_history=b_history, surface=surface,
        ).value
        + _tier_ev_underdog_form(a_stats=a_stats, b_stats=b_stats).value
        + _tier_ev_underdog_serve(a_stats=a_stats, b_stats=b_stats).value
        + _tier_ev_underdog_surface(
            a_stats=a_stats, b_stats=b_stats, surface=surface
        ).value
        + _tier_ev_h2h_underdog(
            a_stats=a_stats, b_stats=b_stats,
            a_history=a_history, b_history=b_history,
        ).value
        + _tier_ev_competitive_floor(a_stats=a_stats, b_stats=b_stats).value
    )
    raw = EV_BASE + contributions
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Tail-mode scorer — sibling of `score_ev_v1_selection` for the deep-
# underdog asymmetric-payoff strategy.
#
# Architectural diff from EV scorer: DROPS `ev_competitive_floor` (the
# negative-only tier that penalizes extreme rank-imbalance events).
# Rationale: at deep underdog prices (market_p ≤ 0.15), the payoff
# ratio explodes (≥ 5.7×), so the +EV math works even with a tiny
# model_p − market_p gap (e.g. 0.13 vs 0.10 model lift = +$0.30 EV).
# The EV scorer's `ev_competitive_floor` was designed assuming bets
# get sized on absolute EV; in tail mode the per-bet variance is huge
# but the asymmetric-payoff thesis is exactly the strategy.
#
# All other tiers (elo-rank-gap, underdog-form/serve/surface/h2h) are
# unchanged — they're all signed by the rank-underdog, so they're
# already aligned with tail-bet hunting.
#
# Recommended CLI flag combination for tail mode (selector picks the
# right events; trader needs aggressive defaults to actually fire):
#     skims rank    --mode tail --max-prob 0.95 --min-favorite-prob 0.75 \
#                              --sport tennis
#     skims execute --mode tail --min-market-implied 0.0
#                              --min-ev 0.30
#                              --bet-size-cents 100
#
# `--min-favorite-prob 0.75` is the slate-level floor: drops events where
# the favorite is priced below 0.75 on the YES mid (no deep-underdog side
# exists). Without it, tail-mode wastes LLM tokens on competitive 0.55/
# 0.45 matchups that can't produce Prime EV through the asymmetric-
# payoff path even when the selector correctly identifies them as
# mispricing-likely.
#
# Critical pre-trade calibration check (not done by the selector):
# validate GBT tail calibration before live betting. Pull settled rows
# with `predicted_winner_probability < 0.20` from `logs/runs/*.jsonl`,
# bin by decile, compare to realized win rate. If model_p > realized
# rate at the tails (the suspected "bottom decile over-predicts"
# pattern, inferred from the documented top-decile under-prediction),
# tail bets bleed money regardless of selector quality.
# ---------------------------------------------------------------------------


def score_tail_v1_selection(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    surface: str | None,
    a_history: PlayerHistory | None = None,
    b_history: PlayerHistory | None = None,
    **_: Any,
) -> float:
    """v1 tail-mode tennis selection scorer.

    Returns a `[0, 1]` selection score targeting deep-underdog asymmetric-
    payoff candidates (market_p ≤ ~0.15 territory). Identical composition
    to `score_ev_v1_selection` EXCEPT drops `ev_competitive_floor` so
    extreme-imbalance events aren't suppressed.

    Composition (additive, clipped):
        score = clip(
            0.4  (same base as EV scorer)
            + ev_elo_rank_gap           (LOAD-BEARING; cap 0.30)
            + ev_underdog_form          (cap 0.20)
            + ev_underdog_serve         (cap 0.20)
            + ev_underdog_surface       (cap 0.20)
            + ev_h2h_underdog           (cap 0.15)
            , 0, 1)

    See module-level "Tail-mode scorer" header for the rationale and
    recommended CLI flag combination.
    """
    contributions = (
        _tier_ev_elo_rank_gap(
            a_stats=a_stats, b_stats=b_stats,
            a_history=a_history, b_history=b_history, surface=surface,
        ).value
        + _tier_ev_underdog_form(a_stats=a_stats, b_stats=b_stats).value
        + _tier_ev_underdog_serve(a_stats=a_stats, b_stats=b_stats).value
        + _tier_ev_underdog_surface(
            a_stats=a_stats, b_stats=b_stats, surface=surface
        ).value
        + _tier_ev_h2h_underdog(
            a_stats=a_stats, b_stats=b_stats,
            a_history=a_history, b_history=b_history,
        ).value
    )
    raw = EV_BASE + contributions
    return max(0.0, min(1.0, raw))
