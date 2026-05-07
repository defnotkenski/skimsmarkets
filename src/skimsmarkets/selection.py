"""Pre-LLM event selection — pick the top-N matchups by *fundamental
imbalance* rather than by tipoff time.

Why this stage exists:
- Slates often arrive bigger than `MAX_SLATE_EVENTS`, and we cap before
  spending LLM tokens on lens chains and director synthesis.
- The historical cap was tipoff-sorted, which optimised for "soonest
  games" — fine when latency matters, wrong when the goal is "most
  defensible picks." Tipoff carries no signal about which event will
  produce a confident, well-aligned lens read.
- The slate judge scores `defensibility_score` on reasoning coherence +
  lens alignment + UW agreement. Empirically the events that score
  highest are the ones with **clear quality differentials** — lopsided
  matchups where specialists agree on direction. Coin-flip matchups
  where fundamentals are balanced can't produce confident reads no
  matter how many tokens the lenses get.

What "imbalance" means per sport, with cheap signals available pre-LLM:

- **Tennis**: rank-points ratio between the two players, sourced from
  the cached MatchStat rankings index. The provider warms the index
  once at startup (5 HTTP calls per tour) and `lookup_player_rank` is
  O(1) thereafter, so pre-cap scoring costs ~zero per event regardless
  of slate size. Points (not position) is the load-bearing field —
  ATP/WTA points spread is non-linear in rank, so points capture skill
  gap better. Sinner (14k pts) vs Alcaraz (13k pts) reads as nearly
  even (ratio 1.1), Sinner vs a rank-300 player reads as a blowout
  (ratio ~30x).

- **Team sports** (soccer, NBA, NFL, MLB, NHL, etc.): win-pct delta
  between the two sides, parsed from the `team_record` string
  ("28-6", "10-3", "12-2-3") gamma exposes on every market. Free —
  already on the bulk gamma `/events` payload at cap time.

- **Sports with no record** (futures-style, niche events): score 0 and
  fall through to the tipoff tiebreaker. We don't try to guess
  fundamental imbalance for sports we have no data on.

Cross-sport scaling: tennis points-ratio uses log10 normalisation
clipped to [0, 1] (a 10× points ratio caps the score at 1.0); team
win-pct delta is naturally [0, 1]. Both share a unit so a
mixed-sport slate sorts coherently.

Tipoff is the explicit tiebreaker for events sharing the same
imbalance score (typically: events without stat-based signal all
score 0.0). That preserves the "soonest first" intuition for the
fallback while letting genuine high-imbalance events override it.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import UTC, datetime

from skimsmarkets.polymarket.models import PolymarketEvent
from skimsmarkets.tennis.identity import (
    TennisMatchIdentity,
    _slug_surface,
    _slug_tier,
    tennis_match_identity,
)
from skimsmarkets.tennis.provider import MatchHistoryRow, TennisStatsProvider

log = logging.getLogger(__name__)

# Tennis imbalance normaliser. A points ratio of `_TENNIS_POINTS_RATIO_CAP`
# saturates the score at 1.0 — beyond that, the matchup is already as
# lopsided as our scoring needs to know. 10× chosen because most ATP/WTA
# tour-level matchups land in 1× to 5× ratio territory; only true
# qualifier-vs-top-10 matchups exceed 10×, and they all deserve the
# ceiling.
_TENNIS_POINTS_RATIO_CAP = 10.0

# Form-alignment adjustment cap. When both players have a populated form
# string from the matchstat profile cache, we add a signed bump to the
# points-ratio base score equal to `cap × (high_points_form_wp -
# low_points_form_wp)`. The `_FORM_ADJUSTMENT_CAP` value bounds the
# adjustment in `[-cap, +cap]`. 0.2 is the starting value: it keeps
# points-ratio as the dominant signal (a 10× points gap saturates at
# 1.0; a perfect form contradiction can only pull that down to 0.8) but
# lets form move the needle on close calls (events scoring 0.20-0.35 in
# points-only mode are exactly the ones where form should sway things).
_FORM_ADJUSTMENT_CAP = 0.2

# Minimum form sample size before the adjustment fires. A 3-of-3 form
# string is too noisy to read as confirmation/contradiction; we want at
# least a half-decent sample. 5 matches is the floor — most active
# tour players ship 10+ entries on a profile call, so this only filters
# out genuinely thin records (recent debutants, returning-from-injury
# players with one match logged).
_MIN_FORM_SAMPLES = 5

# Consistency-alignment adjustment cap. Stacks on top of base + form.
# Half the form cap (0.2) because the form signal already captures
# direction-of-recent-play; consistency is a finer signal that should
# fine-tune, not dominate. With both adjustments at the cap and base
# saturated, the score still clamps to [0, 1] (the final clip in
# `_tennis_imbalance` enforces it). A 10× points gap can still be
# pulled down by 0.3 (form -0.2 + consistency -0.1) → 0.7, which is
# a meaningful but bounded contradiction signal.
_CONSISTENCY_ADJUSTMENT_CAP = 0.1

# Minimum recent-match samples (with usable score / stat data) before
# the consistency adjustment fires. Below this we don't have enough
# signal to estimate dispersion robustly, and asymmetric data on the
# two sides would be misleading. 5 matches matches the form floor —
# the warmup pulls 10 rows per player, so this only filters out
# returning-from-injury or debutant players with thin histories.
_MIN_CONSISTENCY_SAMPLES = 5

# Std-dev → consistency-score mapping bounds. Empirically the dispersion
# we see across recent tour-level matches falls in narrow ranges:
#   - completeness std-dev: ~0.05 for steady players, ~0.20 for
#     volatile ones (mostly because nailbiters cluster around 0.51 and
#     blowouts around 0.85).
#   - first-serve-win % std-dev: ~0.04 for steady servers, ~0.15 for
#     erratic ones.
# Mapping linearly to [1.0, 0.0] over [0, max] gives "consistency
# score" where 1.0 = perfectly steady, 0.0 = max dispersion. Values
# beyond max clamp to 0.0 — anything that volatile is just "very
# erratic" regardless of how much.
_COMPLETENESS_STD_MAX = 0.25
_FIRST_SERVE_STD_MAX = 0.20

# Surface-specialism adjustment cap. Same magnitude as consistency —
# tour-level specialism swings serve-hold rates 10-15pp, so a clear
# specialism delta is a major signal, but rank-points already
# partially encodes it (a clay specialist accumulates points on clay,
# inflating their global rank during clay swing). The 0.1 cap reflects
# that partial overlap.
_SURFACE_ADJUSTMENT_CAP = 0.1

# Minimum matches per side ON THE EVENT SURFACE before the surface
# specialism adjustment fires. Below this, the surface-specific
# win-pct is dominated by sample noise and we'd be reading dispersion
# as signal. 5 is the same floor used for consistency / form sample
# requirements — most tour-active players ship 10+ surface-specific
# matches per year on at least one surface.
_MIN_SURFACE_SAMPLES = 5

# Modal-recent-surface inference threshold (Step B of the surface
# cascade). When `_slug_surface` doesn't recognise the tournament,
# we fall back to "what surface have the favorite's recent matches
# been on". A surface counts as inferred only when at least
# `_MIN_MODAL_SURFACE_ROWS` of the favorite's cached past-match rows
# have a non-None surface AND at least
# `_MODAL_SURFACE_DOMINANCE` of those agree on a single surface. The
# dual gate prevents false-positives during transition weeks
# (e.g. last week of clay → first week of grass: half the recent
# matches are clay, half grass, modal would flip a coin without
# the dominance threshold).
_MIN_MODAL_SURFACE_ROWS = 3
_MODAL_SURFACE_DOMINANCE = 0.6

# Age + career-trajectory adjustment caps. Both signal "rank-points
# lag" — a 35yo journeyman's points are inflated by career
# accumulation, a 19yo's points haven't caught up to the trajectory.
# Each component caps at 0.05; their combined contribution caps at
# 0.075 (less than the sum) because they correlate.
_AGE_ADJUSTMENT_CAP = 0.05
_TRAJECTORY_ADJUSTMENT_CAP = 0.05
_AGE_TRAJECTORY_TOTAL_CAP = 0.075

# Tennis prime years. The 24-29 window is where ATP/WTA point
# accumulation tracks current play-strength most tightly.
# `prime_distance(age)` is 0 inside the window, ramping up linearly
# as the player ages out either side. The `tanh` envelope on the
# adjustment ensures a 35-vs-22 matchup doesn't dominate a
# 30-vs-25 one by 5×.
_PRIME_AGE_CENTER = 26.5
_PRIME_AGE_HALF_WIDTH = 2.5

# Tournament-tier multipliers. Applied AFTER all additive
# adjustments. Slams (best-of-5) amplify imbalance: depth and
# fitness advantages compound across more sets, so the same matchup
# is more lopsided at a Slam than at a 250. Masters (best-of-3 but
# deeper draws) get a smaller bump. Multiplier shape is intentional
# — at saturated 1.0 scores the clamp absorbs the excess (no
# over-1.0 inflation), at mid-range scores the bump moves the needle
# in exactly the cap-vs-no-cap decision band.
_TIER_MULTIPLIER_SLAM = 1.15
_TIER_MULTIPLIER_MASTERS = 1.05


def _parse_team_record(record: str) -> tuple[int, int, int] | None:
    """Parse `"W-L"` or `"W-L-T"` into `(wins, losses, ties)`.

    Returns None for unparseable input (futures-style empty records,
    non-numeric content, wrong field count). NHL-style "W-L-OTL" looks
    like a 3-tuple but the third number is overtime losses, not ties —
    which is a *minor* over-counting in win-pct (treating OTL as a tie
    bumps up the win-pct slightly). We don't try to disambiguate by
    sport because the small mis-weighting is dwarfed by the real
    imbalance signal between e.g. a 35-15 team and a 12-38 team.
    """
    parts = record.split("-")
    if len(parts) < 2 or len(parts) > 3:
        return None
    try:
        nums = [int(p.strip()) for p in parts]
    except ValueError:
        return None
    if any(n < 0 for n in nums):
        return None
    if len(nums) == 2:
        return (nums[0], nums[1], 0)
    return (nums[0], nums[1], nums[2])


def _win_pct_from_record(record: str) -> float | None:
    """Convert a `team_record` string to a `[0, 1]` win percentage.

    Ties / draws / OTLs count as half-wins (standard sports
    convention). Returns None when the record has zero games (e.g.
    pre-season "0-0") so the caller doesn't see misleading 0.5 win-pct
    on no-sample teams.
    """
    parsed = _parse_team_record(record)
    if parsed is None:
        return None
    wins, losses, ties = parsed
    games = wins + losses + ties
    if games <= 0:
        return None
    return (wins + 0.5 * ties) / games


def _team_record_imbalance(event: PolymarketEvent) -> float | None:
    """Win-pct spread across all markets in the event.

    Two market shapes need different handling and a single iteration
    works for both:
      - **Binary head-to-heads** (MLB / NBA / NFL / UFC etc., the
        `_parse_h2h_question` shape): one YES market + a synthesized
        NO clone via `inverted_no_side`. The clone's `team_record`
        carries the *opposite* team's record (passed in as `no_record=`
        at synthesis time), so iterating both YES and NO surfaces
        gives the two distinct team records we need.
      - **3-way multi-outcome** (soccer with home/draw/away): each
        outcome is its own YES market with its own `team_record`;
        no NO clones synthesized, so iteration just reads each YES
        market once.

    `max - min` captures the gap between the strongest and weakest
    sides. For 3-way soccer with a clear favourite + clear underdog,
    that's the biggest imbalance regardless of where the draw row
    sits. Duplicate records (if any) collapse harmlessly because both
    `max` and `min` are idempotent.

    Returns None when fewer than two markets carry parseable records —
    typically futures, niche events, or H2H markets where gamma
    omitted one team's record entry.
    """
    win_pcts: list[float] = []
    for m in event.markets:
        if m.team_record is None:
            continue
        wp = _win_pct_from_record(m.team_record)
        if wp is not None:
            win_pcts.append(wp)
    if len(win_pcts) < 2:
        return None
    return max(win_pcts) - min(win_pcts)


def _std_dev(values: list[float]) -> float:
    """Population std dev. `statistics.pstdev` would do, but we already
    have the mean implicitly so a one-pass two-step is just as cheap
    and avoids the import for one helper."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return var ** 0.5


def _player_consistency_score(rows: list[MatchHistoryRow]) -> float | None:
    """Combine completeness and first-serve dispersion into one
    `[0, 1]` consistency score (higher = steadier).

    Two component dispersions, mapped linearly to [0, 1] consistency:
      - **Margin steadiness**: std-dev of `match_completeness` across
        rows where the score parsed cleanly. Aborted matches (RET, W/O)
        and unparseable scores are skipped — small denominator is the
        cost of clean data. Below `_MIN_CONSISTENCY_SAMPLES` usable
        completeness rows → component returns None.
      - **Serve steadiness**: std-dev of `first_serve_win_pct` across
        rows where the stat block was populated. Same `_MIN_*` floor
        applies. The warmup path includes `stat`, so this is populated
        for warmed players; the live fallback path does not, so events
        whose enrichment ran without a warmup will see this component
        skip. That's fine — those events don't reach this helper
        (selector skips them via the cache-miss path in
        `_tennis_imbalance`).

    Returns the simple mean of present components, or just the present
    one when only one is computable. Returns None when neither
    component crosses the sample floor — caller skips the consistency
    adjustment.
    """
    completeness_vals = [
        r.match_completeness for r in rows if r.match_completeness is not None
    ]
    first_serve_vals = [
        r.first_serve_win_pct for r in rows if r.first_serve_win_pct is not None
    ]

    parts: list[float] = []
    if len(completeness_vals) >= _MIN_CONSISTENCY_SAMPLES:
        std = _std_dev(completeness_vals)
        normalised = max(0.0, 1.0 - std / _COMPLETENESS_STD_MAX)
        parts.append(min(1.0, normalised))
    if len(first_serve_vals) >= _MIN_CONSISTENCY_SAMPLES:
        std = _std_dev(first_serve_vals)
        normalised = max(0.0, 1.0 - std / _FIRST_SERVE_STD_MAX)
        parts.append(min(1.0, normalised))

    if not parts:
        return None
    return sum(parts) / len(parts)


def _resolve_event_surface(
    identity: TennisMatchIdentity,
    slug: str,
    provider: TennisStatsProvider,
) -> str | None:
    """Two-step deterministic cascade for the event's surface.

    Step A: slug parse. Hardcoded mapping in
    `tennis/identity._slug_surface` covers the 4 Slams + ATP/WTA
    Masters/1000 swing — the events that dominate Polymarket tennis
    volume.

    Step B: modal recent surface from the FAVORITE's cached past-
    matches. When the favorite has played ≥`_MIN_MODAL_SURFACE_ROWS`
    matches on a single surface that comprises ≥`_MODAL_SURFACE_DOMINANCE`
    of their non-None recent surfaces, infer that. Captures
    mid-tournament events (favorite already played at THIS tournament
    on THIS surface) and seasonal swings (clay April-May, grass
    June-July, hard the rest of the year).

    Returns None when both steps fail. Selection callers treat None
    as "no surface signal" and skip the surface tier.
    """
    surface = _slug_surface(slug)
    if surface is not None:
        return surface
    history = provider.lookup_player_match_history(identity.tour, identity.player_a)
    if history is None:
        return None
    surfaces = [r.surface for r in history if r.surface is not None]
    if len(surfaces) < _MIN_MODAL_SURFACE_ROWS:
        return None
    most_common, count = Counter(surfaces).most_common(1)[0]
    if count / len(surfaces) < _MODAL_SURFACE_DOMINANCE:
        return None
    return most_common


def _resolve_event_tier(
    identity: TennisMatchIdentity,
    slug: str,
    provider: TennisStatsProvider,
) -> str | None:
    """Two-step deterministic cascade for the event's tournament tier.

    Step A: slug parse via `_slug_tier`. Returns "grand_slam" or
    "masters" for the recognized tier-1/tier-2 events.

    Step B: cached recent-match tier — find a row in the favorite's
    history whose `tournament_name` matches `identity.tournament_hint`
    case-insensitively, and read its `tournament_tier`. Useful when
    Polymarket uses a slug we haven't enumerated but the favorite
    happens to be playing this tournament (i.e. has recent rows
    against the same tournament).

    Returns None when both steps fail. Selection callers treat None
    as "main_tour" → multiplier ×1.0.
    """
    tier = _slug_tier(slug)
    if tier is not None:
        return tier
    if identity.tournament_hint is None:
        return None
    hint_normalised = identity.tournament_hint.strip().lower()
    if not hint_normalised:
        return None
    history = provider.lookup_player_match_history(identity.tour, identity.player_a)
    if history is None:
        return None
    for row in history:
        if row.tournament_name is None or row.tournament_tier is None:
            continue
        if row.tournament_name.strip().lower() == hint_normalised:
            return row.tournament_tier
    return None


def _surface_specialism_adjustment(
    event_surface: str,
    favorite_record: tuple[
        tuple[int, int] | None, dict[str, tuple[int, int]] | None
    ] | None,
    underdog_record: tuple[
        tuple[int, int] | None, dict[str, tuple[int, int]] | None
    ] | None,
) -> float:
    """Signed surface-specialism adjustment in `[-cap, +cap]`.

    For each player: compute `surface_wp - global_wp`. The diff
    `favorite_specialism - underdog_specialism` is positive when the
    favorite is the relative specialist on this surface (lopsidedness
    reinforced) and negative when the underdog is. Returns 0 when
    either side is missing the surface entry, missing the YTD
    aggregate, or has fewer than `_MIN_SURFACE_SAMPLES` matches on
    the event surface.

    Capped magnitude is `_SURFACE_ADJUSTMENT_CAP`; the natural range
    of `surface_diff` is roughly [-0.4, +0.4] (specialism deltas of
    ±20pp are realistic ceilings), so the cap binds for most
    high-confidence specialism signals.
    """
    if favorite_record is None or underdog_record is None:
        return 0.0
    fav_ytd, fav_surfaces = favorite_record
    und_ytd, und_surfaces = underdog_record
    if fav_ytd is None or und_ytd is None:
        return 0.0
    if fav_surfaces is None or und_surfaces is None:
        return 0.0
    fav_surface = fav_surfaces.get(event_surface)
    und_surface = und_surfaces.get(event_surface)
    if fav_surface is None or und_surface is None:
        return 0.0
    fav_w, fav_l = fav_surface
    und_w, und_l = und_surface
    if fav_w + fav_l < _MIN_SURFACE_SAMPLES:
        return 0.0
    if und_w + und_l < _MIN_SURFACE_SAMPLES:
        return 0.0
    fav_global_w, fav_global_l = fav_ytd
    und_global_w, und_global_l = und_ytd
    if fav_global_w + fav_global_l <= 0 or und_global_w + und_global_l <= 0:
        return 0.0
    fav_specialism = fav_w / (fav_w + fav_l) - fav_global_w / (
        fav_global_w + fav_global_l
    )
    und_specialism = und_w / (und_w + und_l) - und_global_w / (
        und_global_w + und_global_l
    )
    surface_diff = fav_specialism - und_specialism
    return max(
        -_SURFACE_ADJUSTMENT_CAP,
        min(_SURFACE_ADJUSTMENT_CAP, _SURFACE_ADJUSTMENT_CAP * surface_diff),
    )


def _prime_distance(age: int) -> float:
    """Distance from the tennis prime window (24-29).

    Returns 0 inside the window, ramps linearly outside. Used by the
    age component of the trajectory adjustment to penalise rank-points
    that have over-accumulated (older players) or under-accumulated
    (very young players) relative to the player's actual current
    play-strength.
    """
    return max(0.0, abs(age - _PRIME_AGE_CENTER) - _PRIME_AGE_HALF_WIDTH)


def _peak_decay(current_rank: int, best_rank: int) -> float:
    """Log-rank decay from career-best to current.

    Returns ~0 for a player at peak (current ≈ best), ~1 for a
    player whose current rank is 10× lower than their career-high.
    Uses log10 so the decay is comparable across rank tiers — a
    top-10 player going from #2 to #20 is "as decayed" as a
    journeyman going from #50 to #500.
    """
    return math.log10(max(current_rank, 1) / max(best_rank, 1))


def _age_trajectory_adjustment(
    favorite_age: int | None,
    favorite_best_rank: int | None,
    favorite_current_rank: int | None,
    underdog_age: int | None,
    underdog_best_rank: int | None,
    underdog_current_rank: int | None,
) -> float:
    """Combined age + career-trajectory adjustment, capped at
    `_AGE_TRAJECTORY_TOTAL_CAP`.

    Two components, both signing NEGATIVE when the favorite's
    rank-points are stale relative to the underdog's:

      - **Age**: prime-distance asymmetry. `tanh` envelope so a huge
        age gap doesn't 5× a moderate one.
      - **Trajectory**: peak-decay asymmetry. Same `tanh` envelope.

    Either component contributes 0 when its underlying data is
    missing on either side. Combined capped at
    `_AGE_TRAJECTORY_TOTAL_CAP` (less than the sum of the per-
    component caps) because both components signal "rank-points lag"
    and we don't want them to double-count.
    """
    age_adjustment = 0.0
    if favorite_age is not None and underdog_age is not None:
        favorite_disadvantage = _prime_distance(favorite_age) - _prime_distance(
            underdog_age
        )
        age_adjustment = -_AGE_ADJUSTMENT_CAP * math.tanh(
            favorite_disadvantage / 2.0
        )

    trajectory_adjustment = 0.0
    if (
        favorite_best_rank is not None
        and favorite_current_rank is not None
        and underdog_best_rank is not None
        and underdog_current_rank is not None
    ):
        fav_decay = _peak_decay(favorite_current_rank, favorite_best_rank)
        und_decay = _peak_decay(underdog_current_rank, underdog_best_rank)
        trajectory_adjustment = -_TRAJECTORY_ADJUSTMENT_CAP * math.tanh(
            (fav_decay - und_decay) / 0.5
        )

    combined = age_adjustment + trajectory_adjustment
    return max(
        -_AGE_TRAJECTORY_TOTAL_CAP,
        min(_AGE_TRAJECTORY_TOTAL_CAP, combined),
    )


def _tier_multiplier(tier: str | None) -> float:
    """Map tier label to the score multiplier.

    Unknown / None / sub-tour-level tiers default to ×1.0 (no-op).
    """
    if tier == "grand_slam":
        return _TIER_MULTIPLIER_SLAM
    if tier == "masters":
        return _TIER_MULTIPLIER_MASTERS
    return 1.0


def _tennis_imbalance(
    event: PolymarketEvent, provider: TennisStatsProvider
) -> float | None:
    """Tennis imbalance score: composes six tiers, cheapest-skip-first.

    Decision tree mirrors the enrichment gate (`tennis_match_identity`):
      - Sport must be tennis with an ATP/WTA slug prefix.
      - Both players must look like singles names.
      - Both must resolve in the warm rankings index with non-None
        `(position, points)`.
    Any miss returns None and the caller falls back to other signals.

    Tier composition (additive except where noted):
      1. **Base** (rank-points ratio). `log10(max_points / min_points)
         / log10(cap)` clipped to [0, 1]. Points-ratio is the right
         scale because ATP/WTA points are roughly proportional to
         tour-level wins weighted by tier.
      2. **Form** (±0.2 cap). Last-10 W/L alignment with the
         points-favorite. Positive when form confirms the points lead;
         negative when form contradicts.
      3. **Consistency** (±0.1 cap). Variance of recent-match
         completeness + first-serve %. Captures the variance
         dimension form misses — a `WLWLWLWL` and a `WWWWLLLL` both
         score 50% form but the matchups they contribute to are not
         equally predictable.
      4. **Surface specialism** (±0.1 cap). Difference of
         (per-surface wp − global wp) across players, signed by who's
         the favorite. Positive when the favorite is the relative
         specialist on this event's surface.
      5. **Age + career trajectory** (±0.075 combined cap). Two
         components signing the same direction: prime-distance
         asymmetry (penalises rank-points that have over- or
         under-accumulated relative to age) and peak-decay asymmetry
         (penalises stale rank-points on declining favorites).
      6. **Tournament-tier multiplier** (×1.0 / ×1.05 / ×1.15).
         Applied AFTER all additive adjustments. Slams amplify
         imbalance via best-of-5 + depth compounding; Masters get a
         smaller bump.

    All six tiers degrade gracefully and independently: any missing
    cache or below-floor sample → that tier's contribution is 0
    (or ×1.0) and the rest of the calculation continues. Final clamp
    to [0, 1] absorbs both directions of stacked adjustment.
    """
    identity = tennis_match_identity(event)
    if identity is None:
        return None
    a_hit = provider.lookup_player_rank(identity.tour, identity.player_a)
    b_hit = provider.lookup_player_rank(identity.tour, identity.player_b)
    if a_hit is None or b_hit is None:
        return None
    a_position, points_a = a_hit
    b_position, points_b = b_hit
    if points_a <= 0 or points_b <= 0:
        return None
    ratio = max(points_a, points_b) / min(points_a, points_b)
    base_score = min(
        1.0, math.log10(ratio) / math.log10(_TENNIS_POINTS_RATIO_CAP)
    )

    # Form-alignment adjustment. Skip silently when either side's form
    # data is unavailable — base_score is already a defensible signal
    # and asymmetric form data would be misleading.
    a_form = provider.lookup_player_form(identity.tour, identity.player_a)
    b_form = provider.lookup_player_form(identity.tour, identity.player_b)
    if a_form is None or b_form is None:
        return base_score
    a_form_str, _a_best = a_form
    b_form_str, _b_best = b_form
    if (
        len(a_form_str) < _MIN_FORM_SAMPLES
        or len(b_form_str) < _MIN_FORM_SAMPLES
    ):
        return base_score
    a_wp = a_form_str.count("W") / len(a_form_str)
    b_wp = b_form_str.count("W") / len(b_form_str)
    # Sign by which player has more points: alignment is positive when
    # the higher-points player also has the better form.
    a_is_favorite = points_a >= points_b
    if a_is_favorite:
        high_wp, low_wp = a_wp, b_wp
    else:
        high_wp, low_wp = b_wp, a_wp
    form_adjustment = _FORM_ADJUSTMENT_CAP * (high_wp - low_wp)

    # Consistency-alignment adjustment. Stacks on form. Skip silently
    # when either side's match-history cache is empty (warmup didn't
    # run for this slate — small slates fit under the cap and bypass
    # the warmup) or the consistency score can't be computed (thin
    # histories, mostly aborted matches). Same graceful-degrade
    # posture as form: never abort scoring on a vendor hiccup.
    a_history = provider.lookup_player_match_history(
        identity.tour, identity.player_a
    )
    b_history = provider.lookup_player_match_history(
        identity.tour, identity.player_b
    )
    consistency_adjustment = 0.0
    if a_history is not None and b_history is not None:
        a_consistency = _player_consistency_score(a_history)
        b_consistency = _player_consistency_score(b_history)
        if a_consistency is not None and b_consistency is not None:
            if a_is_favorite:
                high_c, low_c = a_consistency, b_consistency
            else:
                high_c, low_c = b_consistency, a_consistency
            # Positive when the points-favorite is the steadier player
            # (lopsidedness reinforced — predictable favorite vs erratic
            # underdog); negative when the underdog is steadier than
            # the favorite (matchup is closer than points + form alone
            # suggest — a hot streaky favorite against a steady-eddie
            # underdog is exactly the kind of upset risk we want to
            # de-rank in the selector).
            consistency_adjustment = _CONSISTENCY_ADJUSTMENT_CAP * (high_c - low_c)

    # Surface-specialism adjustment. Stacks on form + consistency.
    # Skip silently when the event surface can't be resolved (slug
    # parse miss + favorite history too thin for modal inference) or
    # when either player's surface record is missing / below the
    # sample floor — graceful degrade to whatever upstream tiers
    # already produced.
    surface_adjustment = 0.0
    event_surface = _resolve_event_surface(identity, event.slug or "", provider)
    if event_surface is not None:
        fav_surface_record = provider.lookup_player_surface_record(
            identity.tour, identity.player_a
        )
        und_surface_record = provider.lookup_player_surface_record(
            identity.tour, identity.player_b
        )
        if a_is_favorite:
            surface_adjustment = _surface_specialism_adjustment(
                event_surface, fav_surface_record, und_surface_record
            )
        else:
            surface_adjustment = _surface_specialism_adjustment(
                event_surface, und_surface_record, fav_surface_record
            )

    # Age + career-trajectory adjustment. Free reads from the warmed
    # profile cache (same slot `lookup_player_form` reads). Skips per-
    # component when underlying data is missing — see helper docstring.
    a_extras = provider.lookup_player_profile_extras(
        identity.tour, identity.player_a
    )
    b_extras = provider.lookup_player_profile_extras(
        identity.tour, identity.player_b
    )
    a_age = a_extras[0] if a_extras is not None else None
    a_best = a_extras[1] if a_extras is not None else None
    b_age = b_extras[0] if b_extras is not None else None
    b_best = b_extras[1] if b_extras is not None else None
    # `a_position` / `b_position` already destructured at the top of
    # this function from `a_hit` / `b_hit` (next to `points_a` /
    # `points_b`) — no extra read.
    if a_is_favorite:
        age_traj_adjustment = _age_trajectory_adjustment(
            a_age, a_best, a_position,
            b_age, b_best, b_position,
        )
    else:
        age_traj_adjustment = _age_trajectory_adjustment(
            b_age, b_best, b_position,
            a_age, a_best, a_position,
        )

    # Tier multiplier — applied AFTER all additive adjustments. Slams
    # amplify imbalance (best-of-5 + depth compounding); Masters get a
    # smaller bump. Default ×1.0 for everything else (250s, 500s,
    # qualifiers, futures, exhibitions). The multiplicative shape
    # combined with the final clamp means saturated 1.0 scores don't
    # over-inflate at Slams.
    tier = _resolve_event_tier(identity, event.slug or "", provider)
    multiplier = _tier_multiplier(tier)

    score_pre_tier = (
        base_score
        + form_adjustment
        + consistency_adjustment
        + surface_adjustment
        + age_traj_adjustment
    )
    return max(0.0, min(1.0, score_pre_tier * multiplier))


def imbalance_score(
    event: PolymarketEvent, tennis_provider: TennisStatsProvider
) -> float:
    """Composite per-event imbalance in `[0, 1]` (higher = more lopsided).

    Sport detection cascade: tennis first (cheapest to compute, highest
    confidence in signal because we have explicit ranking points), then
    team-record-based for everything else, then 0 as fallback. The
    cascade short-circuits at the first sport that produces a non-None
    signal — we don't try to combine tennis + team-record on the same
    event because they're never both populated.

    `tennis_provider` is the same provider used for full enrichment;
    the rank lookup against its warm index is free (no HTTP). The stub
    provider's `lookup_player_rank` always returns None, so under the
    no-key configuration tennis events fall through to score 0 and
    the tipoff tiebreaker decides.
    """
    s = _tennis_imbalance(event, tennis_provider)
    if s is not None:
        return s
    s = _team_record_imbalance(event)
    if s is not None:
        return s
    return 0.0


_FAR_FUTURE = datetime.max.replace(tzinfo=UTC)


def _earliest_tipoff(event: PolymarketEvent) -> datetime:
    """Earliest `game_start_time` across the event's markets.

    Mirrors the previous tipoff-sort key in `fetch_gamma_slate`. Events
    without any populated game-start time sort last so they don't
    displace tradable events at the head when used as the tipoff
    tiebreaker.
    """
    starts = [t for m in event.markets if (t := m.game_start_time) is not None]
    return min(starts) if starts else _FAR_FUTURE


async def select_top_events(
    events: list[PolymarketEvent],
    *,
    max_events: int,
    tennis_provider: TennisStatsProvider,
) -> list[PolymarketEvent]:
    """Apply imbalance scoring and cap the slate to `max_events`.

    No-op fast path: if the slate already fits under the cap, we skip
    the entire scoring pass and return events sorted by tipoff (the
    pipeline's ambient ordering — preserved so downstream stages that
    log "first event" / "last event" stay deterministic).

    When the slate exceeds the cap:
      1. Pre-warm the tennis rankings index for every tour represented
         in the slate (idempotent; ~5 HTTP calls per tour, one-time
         per process). Non-tennis-only slates skip this step.
      2. Score every event by `imbalance_score`.
      3. Sort by `(score desc, tipoff asc)` so high-imbalance events
         lead and tipoff is the deterministic tiebreaker among
         equally-scored events.
      4. Slice to `max_events`.

    The tipoff tiebreaker matters most for events scoring 0.0 (sports
    with no stat-based signal): they all share the bottom of the
    ranking and tipoff order picks among them, preserving the
    "soonest first" intuition for the fallback layer.
    """
    if max_events <= 0 or len(events) <= max_events:
        return events

    # Warm tennis index for every tour present in the slate. Stub
    # provider no-ops; no key configured = no warmup cost.
    tennis_tours: set[str] = set()
    tennis_identities: list = []
    for ev in events:
        ident = tennis_match_identity(ev)
        if ident is not None:
            tennis_tours.add(ident.tour)
            tennis_identities.append(ident)
    if tennis_tours:
        await tennis_provider.warm_for_selection(tennis_tours)
        # Form + match-history warmups run AFTER the rank index is warm
        # because both resolve names via the index to find player IDs.
        # Both share their respective caches with the downstream
        # `enrich_tennis_stats` pass — every event that survives the
        # cap reuses the warmed profile + past-matches responses for
        # free. Form warmup feeds `lookup_player_form` (last-10 W/L);
        # match-history warmup feeds `lookup_player_match_history`
        # (per-row completeness + first-serve % for the consistency
        # adjustment). One HTTP each per unique player.
        if tennis_identities:
            await tennis_provider.warm_form_for_selection(tennis_identities)
            await tennis_provider.warm_match_history_for_selection(
                tennis_identities
            )
            # Surface-summary warmup feeds `lookup_player_surface_record`
            # (per-surface YTD record) for the surface-specialism tier.
            # Same dedup-by-unique-player pattern as the other two
            # warmups; one HTTP per unique player. The cached payload
            # is reused by `_player_surface_year_record` in the post-
            # cap enrichment path so the warmup HTTP isn't wasted on
            # events that survive selection.
            await tennis_provider.warm_surface_summary_for_selection(
                tennis_identities
            )

    scored = [
        (ev, imbalance_score(ev, tennis_provider), _earliest_tipoff(ev))
        for ev in events
    ]
    # Sort key: descending score, ascending tipoff. Negate score so
    # tuple-sort lands the right direction without `reverse=True`
    # (which would flip the tipoff direction too).
    scored.sort(key=lambda triple: (-triple[1], triple[2]))
    selected = [ev for ev, _, _ in scored[:max_events]]
    cut = len(events) - max_events
    top_score = scored[0][1] if scored else 0.0
    cut_score = scored[max_events - 1][1] if max_events <= len(scored) else 0.0
    log.info(
        "selected %d/%d events by imbalance (top=%.2f, cut@=%.2f); dropped %d",
        max_events, len(events), top_score, cut_score, cut,
    )
    return selected
