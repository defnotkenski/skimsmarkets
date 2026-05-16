"""Point-in-time feature builder for the tennis GBT spike.

Single source of truth for feature definitions used by both
`gbt_train.py` (walk forward over the parquet) and `gbt.py` (slate-
time prediction). Drift between the two would silently mis-calibrate
the live model, so they share this module.

The core abstraction is `PlayerHistory` — an incremental per-player
accumulator. Walking the historical parquet in chronological order,
we snapshot both players' state BEFORE adding the current match's
contribution; this is what gives us perfect point-in-time discipline
without any explicit date-cutoff filtering. After the snapshot we
fold the match's per-side counters into both accumulators and
continue.

For predict time, the entire historical parquet is walked once at
module init (in `gbt.py`), leaving each player's accumulator at its
"as-of-today" state. The slate-time call snapshots both players
against an upcoming match's surface/tier/best_of and produces the
feature row.

Feature naming convention: `<metric>_diff` for anchor − opponent
columns. The anchor is the player with the LOWER MatchStat id; the
target `y = 1` if the anchor won. With this discipline, swapping
sides flips every numeric feature's sign and the prediction's
probability — symmetry by construction. See the plan doc for the
full justification.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from skimsmarkets.tennis.matchstat import (
    _COURT_ID_TO_SURFACE,
    _RANK_ID_TO_TIER,
)
from skimsmarkets.tennis.provider import ScoreDetails, parse_score_details

log = logging.getLogger(__name__)

# Cold-start gate. Matches where either side has < this many priors
# get dropped from training and no GBT prediction is produced at
# slate time. Calibrated to give meaningful career denominators
# without losing the bulk of tour-level matchups.
MIN_PRIORS_PER_SIDE = 20

# Days-since-last-match cap. A player coming back from a 6-month
# layoff and one back from a 4-month layoff are both "rust risk" — the
# sign of the gap is what matters, not the precise magnitude. Capping
# also bounds the feature for catboost's split criterion.
DAYS_SINCE_CAP = 30

# Recent-form window. Last-10 win rate is the standard tennis-stats
# baseline; we mirror it exactly.
RECENT_FORM_WINDOW = 10

# Best-of-3 default when the source row doesn't carry `best_of`. Most
# tour matches are bo3 — consistent with `tennis/simulation.py`'s
# `detect_best_of` fallback.
DEFAULT_BEST_OF = 3

# Recency weighting half-life. Each prior match's contribution to the
# career rate aggregates is decayed by `exp(-elapsed_days /
# HALF_LIFE_DAYS)`. 365d (12 months) chosen so:
#   - Recent form dominates (last 12mo carries ~50%+ of weight)
#   - Matches from 3 years ago contribute ~5% (still informative)
#   - Junior-era / debut-year box scores from 5+ years ago effectively
#     drop out (~1% weight)
# The decay applies to RATE counters (serve %, return %, BP %) and
# surface bucket records — i.e., the metrics where "recent quality"
# differs from "career-long quality" in a load-bearing way.
# It does NOT apply to:
#   - The cold-start gate count `matches` — that measures EVIDENCE
#     COVERAGE, where every prior counts equally.
#   - The last-N form deque — that's already inherently recent.
#   - Elo ratings — Elo decays naturally via the K-factor update; an
#     extra exponential decay would double-discount recent matches.
HALF_LIFE_DAYS = 365.0

# Elo initialisation. Standard tennis-Elo default; the exact value
# cancels in the diff feature so 1500 vs 1000 is identical at predict
# time. Kept at 1500 to match the chess convention so debug eyeballing
# is intuitive.
ELO_INITIAL = 1500.0

# Elo K-factor. Fixed at 24 — between the chess default (32, too
# swingy for tennis match variance) and the conservative end (16, too
# slow to track form changes). 24 matches the Sackmann tennis-Elo
# reference K for a player with ~150 career matches under his rank-
# decayed formula, which is the modal player in our backfill. Fixed K
# is intentionally simpler than the rank-decayed variant — the GBT
# can pick up the "high-K player" signal via the cold-start match
# count if it matters.
ELO_K = 24.0

# Per-surface recent-form window. Mirrors RECENT_FORM_WINDOW but
# bucketed by surface, so a player who's gone 7-3 on hard recently
# but 2-8 on clay over the same span shows a sharp surface-specific
# form signal that the all-surfaces last-N misses.
SURFACE_RECENT_WINDOW = 10

# Fatigue lookback — count matches played in this rolling window
# ending at the match date. 14d picked because it captures both ends
# of the "busy tour stretch" distribution: a player who's played
# 5 matches in 14d is mid-event in a deep tournament run; one who's
# played 0 is fresh off a layoff. Single window (not 7d AND 30d AND
# 90d) to keep feature surface compact — the GBT can learn the
# nonlinear "how much is too much" via the histogram split.
FATIGUE_WINDOW_DAYS = 14

# Recent-deque max length for the rolling history a player carries
# for the opp-quality and set-dominance features. Mirrors
# RECENT_FORM_WINDOW so the three "last N matches" signals (winrate,
# opp rank, set diff) all read off the same N — easier to reason
# about than three different windows.
RECENT_HISTORY_WINDOW = 10

# Recent-match-date deque max length. Sized to comfortably exceed
# the FATIGUE_WINDOW_DAYS at the busiest schedule (a player playing
# every day for 2 weeks = 14 dates); 32 gives headroom for double-
# headers and exhibition stretches without truncation.
RECENT_DATES_WINDOW = 32


def surface_key(court_id: int | None) -> str | None:
    """Map MatchStat `tournament.courtId` to our 4-key surface taxonomy."""
    if court_id is None:
        return None
    return _COURT_ID_TO_SURFACE.get(court_id)


def tier_key(rank_id: int | None) -> str | None:
    """Map MatchStat `tournament.rankId` to a tier label."""
    if rank_id is None:
        return None
    return _RANK_ID_TO_TIER.get(rank_id)


def infer_best_of(n_sets: int | None, tier: str | None) -> int:
    """Derive best-of from observed set count + tournament tier.

    The MatchStat parquet doesn't carry best_of as a column — every
    row's `best_of` is None — so without this inference both the
    `best_of` categorical and any set-arithmetic-derived feature
    (e.g. subject set differential) lose all signal.

    Rules:
    - n_sets in (4, 5): definitively bo5 — the only format that allows
      4+ completed sets. Davis Cup / Laver Cup edge cases that played
      bo3 ending in straights still fall under the n_sets ∈ (2, 3)
      branches and get assigned by tier (correct since they typically
      sit in non-grand-slam tiers).
    - n_sets == 2: bo3 — the only format that completes in 2 sets.
    - n_sets == 3 (ambiguous — bo3 finale or bo5 sweep): Grand Slam
      tier → bo5; everything else → bo3. Matches the live-path
      `detect_best_of` heuristic so training and slate-time priors
      agree on this for slates without explicit best_of.
    - n_sets in (None, 0, 1): default to bo3 (conservative — most
      tour matches are bo3, and a bo3-assumed slam over-counts loser
      sets less harmfully than the reverse).
    """
    if n_sets is None:
        return DEFAULT_BEST_OF
    if n_sets >= 4:
        return 5
    if n_sets == 2:
        return 3
    if n_sets == 3:
        return 5 if tier == "grand_slam" else 3
    return DEFAULT_BEST_OF


def _safe_div(num: float, den: float) -> float | None:
    """Ratio from running counters. None when the denominator is
    effectively zero — catboost handles missing values natively, so
    an unobserved aggregate doesn't propagate as a misleading 0.0.

    Denominators are floats post-decay; the 1e-9 epsilon catches the
    "no priors observed" case without rejecting heavily-decayed but
    real aggregates (a player with 30 priors all from 5y ago decays
    to a denominator around 0.04, comfortably above epsilon).
    """
    if den <= 1e-9:
        return None
    return num / den


@dataclass
class _H2HRecord:
    """Per-opponent head-to-head record with clutch sub-counts.

    Kept un-decayed (raw integer counts) deliberately — H2H samples
    are small (often <5) and recency-decay would dilute them below the
    point where the rate is informative. The career-aggregate clutch
    counters on `PlayerHistory` are recency-weighted; the per-opponent
    cross-product here is not.

    Sub-counts mirror the career-aggregate clutch fields so a future
    algorithmic-lens scoring rule can compare "this player's career
    decider rate" vs "this player's decider rate specifically against
    THIS opponent" — the matchup-conditioned clutch signal that
    `/h2h/stats/{a}/{b}` exposes at live time, derivable here from the
    raw match parquet without per-pair API calls.
    """

    wins: int = 0
    losses: int = 0
    decider_won: int = 0
    decider_played: int = 0
    tiebreak_won: int = 0
    tiebreak_played: int = 0
    comeback_won: int = 0
    comeback_set1_lost: int = 0
    close_match_won: int = 0
    close_match_played: int = 0


@dataclass
class _SurfaceAggregate:
    """Per-surface running counts for one player. Identical schema to
    the all-surfaces aggregate but scoped to a single court type.
    Used for `surface_first_serve_win_pct` and surface-record features.

    Stored as floats because the recency-decay multiplier in
    `PlayerHistory._decay_to` scales these counters in place; integer
    truncation would silently bias the surface aggregates downward
    over many decay applications.
    """

    matches: float = 0.0
    wins: float = 0.0
    won_first_serve_n: float = 0.0
    won_first_serve_d: float = 0.0


@dataclass
class PlayerHistory:
    """Incremental per-player aggregator over MatchStat box-score rows.

    Every counter is a numerator/denominator pair so partial
    aggregation across an arbitrary subset of priors gives a true
    proportion, never a biased average-of-percentages. The vendor
    ships counters per match; we sum them here and divide on snapshot.

    Return-side counters are derived from the OPPONENT'S serve numbers
    in matches where this player returned: when A returns B's first
    serve, A's "return-points-won" numerator picks up
    `(B.first_serve_in - B.won_first_serve)` points and the
    denominator picks up `B.first_serve_in`. Same pattern for BP-save
    (A serving) and BP-convert (A returning).
    """

    player_id: int

    # Serve-side career totals. Stored as floats because the
    # recency-decay multiplier in `_decay_to` scales them in place;
    # integer truncation would systematically bias the aggregates
    # downward over many decay applications. Ratios computed from
    # these (career_first_serve_in_pct etc.) are recency-weighted by
    # construction — see `_decay_to` for the math.
    first_serve_in_n: float = 0.0
    first_serve_in_d: float = 0.0
    won_first_serve_n: float = 0.0
    won_first_serve_d: float = 0.0
    won_second_serve_n: float = 0.0
    won_second_serve_d: float = 0.0

    # Return-side career totals (derived from opponents' serve numbers).
    return_first_n: float = 0.0
    return_first_d: float = 0.0
    return_second_n: float = 0.0
    return_second_d: float = 0.0

    # Break-point save (when this player served) — saved / faced.
    bp_saved: float = 0.0
    bp_faced: float = 0.0
    # Break-point convert (when this player returned) — converted / chances.
    bp_converted: float = 0.0
    bp_chances: float = 0.0

    # Serve-domination counters — aces (pure first-serve dominance) and
    # double faults (second-serve failure under pressure). Both
    # recency-decayed on the same HALF_LIFE_DAYS schedule as the rate
    # counters above; rate = numerator / appropriate denominator on
    # snapshot. Distinct from `career_first_serve_win_pct_diff`: a
    # player can win 70% of first serves via rallies vs another who
    # wins 70% via aces — the ace differential captures raw power
    # that the win-rate aggregate doesn't separate from rally play.
    career_aces: float = 0.0
    career_double_faults: float = 0.0

    # Career-aggregate clutch counters, recency-weighted on the same
    # HALF_LIFE_DAYS schedule as the rate counters above. Derived from
    # the per-row score string via `parse_score_details`. Each pair is
    # numerator / denominator so partial aggregations stay unbiased
    # under the shared decay multiplier.
    # - tiebreak: subject's TBs won / played (across all matches)
    # - decider: subject's deciding-set wins / matches that went the
    #   distance (n_sets == best_of)
    # - comeback: subject's match wins given subject lost set 1 / total
    #   matches where subject lost set 1
    # - close_match: subject's wins / total close matches (close =
    #   final-set margin ≤ 2 OR final set was a tiebreak)
    tiebreak_won: float = 0.0
    tiebreak_played: float = 0.0
    decider_won: float = 0.0
    decider_played: float = 0.0
    comeback_won: float = 0.0
    comeback_set1_lost: float = 0.0
    close_match_won: float = 0.0
    close_match_played: float = 0.0

    # Win/loss totals — UNDECAYED. `matches` drives the cold-start
    # gate (`MIN_PRIORS_PER_SIDE`), where every prior should count
    # equally regardless of age — that's a coverage signal, not a
    # weighted-quality signal. `wins` is unused as a feature directly
    # (we use the per-surface buckets and the recent deque for
    # win-rate features), kept here for symmetry / debug inspection.
    matches: int = 0
    wins: int = 0

    # Per-surface aggregates. Keyed by `surface_key()` value (e.g.
    # "hard"). Surfaces unknown to the taxonomy land under None and
    # are surfaced separately on snapshot (or rather, ignored — only
    # the requested surface matters).
    by_surface: dict[str | None, _SurfaceAggregate] = field(default_factory=dict)

    # H2H by opponent. `_H2HRecord` per opponent MatchStat id with
    # win/loss totals plus matchup-conditioned clutch sub-counts
    # (decider, tiebreak, comeback, close-match). Sparse — most
    # pairings have no priors at slate time. Un-decayed: H2H samples
    # are small and recency-weighting them would over-dilute.
    h2h: dict[int, _H2HRecord] = field(default_factory=dict)

    # Recent-form ring buffer (newest at the right). bool wins.
    recent: deque[bool] = field(default_factory=lambda: deque(maxlen=RECENT_FORM_WINDOW))

    # Per-surface recent-form ring buffers. Keyed by surface_key()
    # value; each holds the last SURFACE_RECENT_WINDOW results on that
    # surface. Captures the form-on-this-surface signal that the
    # all-surfaces `recent` deque misses (a 7-3 hard-court run paired
    # with a 2-8 clay run averages to the same global last-10 as a
    # uniform 4-6 across both, but the surface-specific reads diverge
    # sharply on a clay-court slate).
    surface_recent: dict[str | None, deque[bool]] = field(default_factory=dict)

    # Elo ratings — global (cross-surface) plus per-surface buckets.
    # `elo_global` is the Glicko-flat-K Elo updated after every match
    # regardless of surface. `elo_by_surface` holds per-surface Elos
    # updated only by matches on that surface; warm-started to the
    # player's global Elo at first appearance on a new surface (a
    # 1700-Elo player without grass priors shouldn't reset to 1500 on
    # grass debut). Both are UNDECAYED on the time axis — Elo's
    # natural recency-weighting comes from the K-factor update, and
    # layering exponential decay on top would double-discount recent
    # matches. Read by `compute_features` as differential signals
    # (`elo_global_diff`, `elo_surface_diff`).
    elo_global: float = ELO_INITIAL
    elo_by_surface: dict[str | None, float] = field(default_factory=dict)

    # Rolling deques for load / quality / dominance signals. All three
    # are FIFO ring buffers (newest at the right); they track only the
    # last RECENT_HISTORY_WINDOW (or RECENT_DATES_WINDOW for dates)
    # entries because the corresponding features are explicitly
    # recent-window summaries — older entries would dilute the
    # "current form" signal these features are meant to capture.
    #
    # - recent_match_dates: dates of last N matches; used to count
    #   matches-in-last-K-days (fatigue / scheduling density).
    # - recent_opp_ranks: opponent's point-in-time rank at the match;
    #   used to compute schedule-strength (avg rank of recent
    #   opponents — lower = tougher schedule).
    # - recent_set_diff: subject's set differential per match
    #   (subject sets - opp sets, signed); used for dominance signal
    #   (a winner who goes 6-0 6-1 averages set_diff=+2 vs a winner
    #   who goes 7-6 7-6 with diff=+2 too, BUT a loser tagged 0-6 0-6
    #   averages -2 vs a loser pushed to 7-6 6-7 7-6 with diff=-1).
    recent_match_dates: deque[date] = field(
        default_factory=lambda: deque(maxlen=RECENT_DATES_WINDOW)
    )
    recent_opp_ranks: deque[int] = field(
        default_factory=lambda: deque(maxlen=RECENT_HISTORY_WINDOW)
    )
    recent_set_diff: deque[int] = field(
        default_factory=lambda: deque(maxlen=RECENT_HISTORY_WINDOW)
    )

    # Latest prior match date — used for fatigue.
    last_match_date: date | None = None

    # Internal: tracks the "as-of" date of the decayed counters above.
    # Each `add_match` advances this to the new match's date and
    # multiplies all rate counters by `exp(-elapsed/HALF_LIFE_DAYS)`
    # before folding in the new contributions. Snapshot reads return
    # ratios — the decay factor cancels in numerator/denominator, so
    # the read is the recency-weighted rate at the as-of date.
    _last_decay_date: date | None = None

    def _decay_to(self, target_date: date) -> None:
        """Apply the recency decay to bring all rate counters to
        `target_date`.

        First call on this player initialises `_last_decay_date` to
        the target without decaying (no priors to decay yet). Subsequent
        calls multiply every rate counter by `exp(-(elapsed)/HALF_LIFE_DAYS)`
        where elapsed is the day-count since the last decay. A target
        ≤ the last decay date is a no-op (defensive — callers walk
        chronologically and shouldn't trigger this branch).
        """
        if self._last_decay_date is None:
            self._last_decay_date = target_date
            return
        if target_date <= self._last_decay_date:
            return
        elapsed = (target_date - self._last_decay_date).days
        factor = math.exp(-elapsed / HALF_LIFE_DAYS)
        # Per-rate-counter decay. Listed explicitly rather than via a
        # tuple-of-attr-names loop to keep the inner-loop arithmetic
        # transparent and avoid a getattr/setattr round-trip per attr
        # (this method runs ~25k times during training).
        self.first_serve_in_n *= factor
        self.first_serve_in_d *= factor
        self.won_first_serve_n *= factor
        self.won_first_serve_d *= factor
        self.won_second_serve_n *= factor
        self.won_second_serve_d *= factor
        self.return_first_n *= factor
        self.return_first_d *= factor
        self.return_second_n *= factor
        self.return_second_d *= factor
        self.bp_saved *= factor
        self.bp_faced *= factor
        self.bp_converted *= factor
        self.bp_chances *= factor
        self.tiebreak_won *= factor
        self.tiebreak_played *= factor
        self.decider_won *= factor
        self.decider_played *= factor
        self.comeback_won *= factor
        self.comeback_set1_lost *= factor
        self.close_match_won *= factor
        self.close_match_played *= factor
        self.career_aces *= factor
        self.career_double_faults *= factor
        # Surface-bucket counters too — the surface_winrate /
        # surface_first_serve_win_pct features should be
        # recency-weighted on the same schedule as the global rates.
        for bucket in self.by_surface.values():
            bucket.matches *= factor
            bucket.wins *= factor
            bucket.won_first_serve_n *= factor
            bucket.won_first_serve_d *= factor
        self._last_decay_date = target_date

    def add_match(
        self,
        *,
        match_date: date,
        won: bool,
        surface: str | None,
        opponent_id: int,
        # This player's serve box score from the row.
        my_first_serve: int | None,
        my_first_serve_of: int | None,
        my_won_first_serve: int | None,
        my_won_first_serve_of: int | None,
        my_won_second_serve: int | None,
        my_won_second_serve_of: int | None,
        my_bp_converted: int | None,
        my_bp_converted_of: int | None,
        my_aces: int | None = None,
        my_double_faults: int | None = None,
        # Opponent's serve box score (used to derive my return / BP-save).
        opp_won_first_serve: int | None,
        opp_won_first_serve_of: int | None,
        opp_won_second_serve: int | None,
        opp_won_second_serve_of: int | None,
        opp_bp_converted: int | None,
        opp_bp_converted_of: int | None,
        # Pre-parsed `ScoreDetails` (perspective: match-winner-relative)
        # from the row's score string. Caller calls `parse_score_details`
        # once with the row's `winner_side`. None when the row's score
        # is missing or unparseable (older rows occasionally drop scores
        # — we just skip the clutch counters for that match).
        score_details: ScoreDetails | None = None,
        # Opponent's pre-match Elos. Caller must snapshot these BEFORE
        # calling add_match on either side of the pair (otherwise the
        # second call sees the first's post-update Elo, biasing the
        # update for the second side). None falls back to ELO_INITIAL
        # (i.e. opp had no prior Elo state — treat as average).
        opp_pre_match_elo_global: float | None = None,
        opp_pre_match_elo_surface: float | None = None,
        # Opponent's rank at match time (point-in-time from
        # rankings_history). None when opp had no ranking snapshot at
        # the match date — the recent-opp-rank deque skips the entry
        # rather than imputing a bogus value.
        opp_rank_at_match: int | None = None,
        # Subject's set differential for this match (subject sets won
        # − opp sets won). Caller computes from score_details and
        # best_of (which are both in scope at the parse layer) so
        # add_match stays free of score-parsing concerns. None when
        # the score didn't parse cleanly — the dominance deque skips
        # rather than imputing 0.
        subject_set_diff: int | None = None,
    ) -> None:
        """Fold one prior match's contribution into the running totals.

        Missing counters from the vendor are coerced to 0 — they don't
        bias the proportion, just don't contribute. The cold-start
        filter (`MIN_PRIORS_PER_SIDE`) operates on `matches` count,
        which is incremented unconditionally even when the box score
        was sparse — having played the match still counts toward the
        evidence base, even if box-score quality was thin.
        """

        def n(x: int | None) -> int:
            return int(x) if x is not None else 0

        # Decay all prior rate contributions to today BEFORE folding
        # in this match's counters. The new match enters at full
        # weight; older matches get progressively down-weighted.
        self._decay_to(match_date)

        # Serve-side.
        self.first_serve_in_n += n(my_first_serve)
        self.first_serve_in_d += n(my_first_serve_of)
        self.won_first_serve_n += n(my_won_first_serve)
        self.won_first_serve_d += n(my_won_first_serve_of)
        self.won_second_serve_n += n(my_won_second_serve)
        self.won_second_serve_d += n(my_won_second_serve_of)

        # Return-side: I won the points my opponent did NOT win on
        # their serves. Numerator = (opp_first_in - opp_won_first);
        # denominator = opp_first_in (== opp_won_first_serve_of).
        opp_first_in = n(opp_won_first_serve_of)
        self.return_first_n += opp_first_in - n(opp_won_first_serve)
        self.return_first_d += opp_first_in
        opp_second_attempts = n(opp_won_second_serve_of)
        self.return_second_n += opp_second_attempts - n(opp_won_second_serve)
        self.return_second_d += opp_second_attempts

        # BP save (I served): opponent's BPs converted = my BPs lost.
        opp_bp_chances = n(opp_bp_converted_of)
        self.bp_faced += opp_bp_chances
        self.bp_saved += opp_bp_chances - n(opp_bp_converted)

        # BP convert (I returned): direct from my-side counters.
        self.bp_chances += n(my_bp_converted_of)
        self.bp_converted += n(my_bp_converted)

        # Serve domination — aces and double-faults are vendor-shipped
        # per-match counts. Folded as raw integers; the snapshot
        # divides by an appropriate denominator (first_serve_in_d for
        # ace rate, won_second_serve_d for DF rate) to yield a rate
        # comparable across players regardless of match length.
        self.career_aces += n(my_aces)
        self.career_double_faults += n(my_double_faults)

        # Surface bucket — same fields as overall but scoped.
        bucket = self.by_surface.setdefault(surface, _SurfaceAggregate())
        bucket.matches += 1
        bucket.wins += 1 if won else 0
        bucket.won_first_serve_n += n(my_won_first_serve)
        bucket.won_first_serve_d += n(my_won_first_serve_of)

        # H2H bookkeeping. Direction = "my wins / total meetings"; the
        # snapshot derives the anchor's H2H rate vs the opponent. The
        # per-opponent record also carries decider/tiebreak/comeback/
        # close-match sub-counts so backtesting can read matchup-
        # conditioned clutch without per-pair API calls.
        record = self.h2h.get(opponent_id)
        if record is None:
            record = _H2HRecord()
            self.h2h[opponent_id] = record
        if won:
            record.wins += 1
        else:
            record.losses += 1

        # Clutch counters — only when the score parsed cleanly.
        # `score_details` is match-winner-relative; rotate onto subject
        # using `won`. Career-aggregate counters are recency-weighted
        # via `_decay_to` above; the per-opponent `_H2HRecord` is left
        # un-decayed (small samples; recency-weighting would dilute).
        subject_won_set_one: bool | None = None
        if score_details is not None:
            if score_details.went_to_decider:
                self.decider_played += 1.0
                record.decider_played += 1
                if won:
                    self.decider_won += 1.0
                    record.decider_won += 1
            self.tiebreak_played += score_details.n_tiebreaks_played
            record.tiebreak_played += score_details.n_tiebreaks_played
            subject_tbs_won = (
                score_details.winner_tiebreaks_won
                if won
                else score_details.n_tiebreaks_played
                - score_details.winner_tiebreaks_won
            )
            self.tiebreak_won += subject_tbs_won
            record.tiebreak_won += subject_tbs_won
            if score_details.is_close_match:
                self.close_match_played += 1.0
                record.close_match_played += 1
                if won:
                    self.close_match_won += 1.0
                    record.close_match_won += 1
            subject_won_set_one = (
                score_details.winner_won_set_one
                if won
                else not score_details.winner_won_set_one
            )
            if not subject_won_set_one:
                self.comeback_set1_lost += 1.0
                record.comeback_set1_lost += 1
                if won:
                    self.comeback_won += 1.0
                    record.comeback_won += 1

        # Recent / counts.
        self.recent.append(won)
        # Per-surface recent deque. Set up the bucket on first
        # appearance; the deque is bounded so we don't have to prune.
        surf_deque = self.surface_recent.get(surface)
        if surf_deque is None:
            surf_deque = deque(maxlen=SURFACE_RECENT_WINDOW)
            self.surface_recent[surface] = surf_deque
        surf_deque.append(won)
        self.matches += 1
        if won:
            self.wins += 1

        # Elo updates — global plus the surface bucket. Both use the
        # same K-factor; the surface bucket warm-starts from this
        # player's current global Elo on first appearance on the
        # surface (rather than the chess-default 1500), so a strong
        # player's grass debut doesn't reset their grass Elo to
        # average. Opponent's pre-match Elo is supplied by the caller
        # (which snapshots both players BEFORE calling add_match on
        # either side); None falls back to ELO_INITIAL.
        opp_elo_g = (
            ELO_INITIAL if opp_pre_match_elo_global is None
            else opp_pre_match_elo_global
        )
        expected_g = 1.0 / (1.0 + 10.0 ** ((opp_elo_g - self.elo_global) / 400.0))
        actual = 1.0 if won else 0.0
        self.elo_global = self.elo_global + ELO_K * (actual - expected_g)

        surf_elo_self = self.elo_by_surface.get(surface)
        if surf_elo_self is None:
            # Warm start: use the player's pre-match GLOBAL Elo — i.e.
            # the value before this match's update folded in. The
            # `self.elo_global - ELO_K * (actual - expected_g)` term
            # is the pre-update global; using it here keeps the
            # surface-Elo update strictly first-match-on-surface.
            surf_elo_self = self.elo_global - ELO_K * (actual - expected_g)
        opp_elo_s = (
            ELO_INITIAL if opp_pre_match_elo_surface is None
            else opp_pre_match_elo_surface
        )
        expected_s = 1.0 / (1.0 + 10.0 ** ((opp_elo_s - surf_elo_self) / 400.0))
        self.elo_by_surface[surface] = surf_elo_self + ELO_K * (actual - expected_s)

        # Rolling deques for the Batch B features (load / quality /
        # dominance). All bounded by maxlen — ring-buffer drop happens
        # automatically. Opp rank and subject set diff skip the append
        # when their source is None rather than imputing a bogus value
        # (a 0 set-diff would falsely register as "even match", a 999
        # opp rank would tilt the schedule-strength average).
        self.recent_match_dates.append(match_date)
        if opp_rank_at_match is not None:
            self.recent_opp_ranks.append(opp_rank_at_match)
        if subject_set_diff is not None:
            self.recent_set_diff.append(subject_set_diff)

        # Latest date — used for fatigue. We only ever advance forward
        # because callers walk the table chronologically.
        if self.last_match_date is None or match_date > self.last_match_date:
            self.last_match_date = match_date

    # --- Snapshot helpers (read-only). ----------------------------------

    def career_first_serve_in_pct(self) -> float | None:
        return _safe_div(self.first_serve_in_n, self.first_serve_in_d)

    def career_first_serve_win_pct(self) -> float | None:
        return _safe_div(self.won_first_serve_n, self.won_first_serve_d)

    def career_second_serve_win_pct(self) -> float | None:
        return _safe_div(self.won_second_serve_n, self.won_second_serve_d)

    def career_first_serve_return_win_pct(self) -> float | None:
        """Recency-weighted return-points-won % when facing the
        opponent's FIRST serve. Mirrors the split that
        `tennis/simulation.py` consumes — separating first vs second
        serve return gives the model two distinct signals (a returner
        who eats first serves is a different profile from one who
        only punishes second serves).
        """
        return _safe_div(self.return_first_n, self.return_first_d)

    def career_second_serve_return_win_pct(self) -> float | None:
        """Recency-weighted return-points-won % when facing the
        opponent's SECOND serve. See sibling method's docstring.
        """
        return _safe_div(self.return_second_n, self.return_second_d)

    def career_bp_save_pct(self) -> float | None:
        return _safe_div(self.bp_saved, self.bp_faced)

    def career_bp_convert_pct(self) -> float | None:
        return _safe_div(self.bp_converted, self.bp_chances)

    def career_ace_rate(self) -> float | None:
        """Aces per first-serve point attempted (recency-weighted).
        first_serve_in_d is the count of first serves attempted (in
        + out), which is the natural denominator — every ace by
        definition came from a first serve. Returns None when the
        denominator is below the safe-div epsilon (no observed first
        serves at all).
        """
        return _safe_div(self.career_aces, self.first_serve_in_d)

    def career_df_rate(self) -> float | None:
        """Double faults per second-serve point attempted
        (recency-weighted). won_second_serve_d is the count of
        second serves played — a DF is the special case of "second
        serve attempted, no point won by server". Returns None on
        empty denominator.
        """
        return _safe_div(self.career_double_faults, self.won_second_serve_d)

    def career_tiebreak_winrate(self) -> float | None:
        return _safe_div(self.tiebreak_won, self.tiebreak_played)

    def career_decider_winrate(self) -> float | None:
        return _safe_div(self.decider_won, self.decider_played)

    def career_comeback_winrate(self) -> float | None:
        """Subject's match wins given subject lost set 1, divided by
        the count of matches where subject lost set 1. Differs from
        the H2H lens's matchup-conditioned comeback rate (which is
        opponent-conditioned and lives on `TennisInMatchupStats`)."""
        return _safe_div(self.comeback_won, self.comeback_set1_lost)

    def career_close_match_winrate(self) -> float | None:
        return _safe_div(self.close_match_won, self.close_match_played)

    def surface_first_serve_win_pct(self, surface: str | None) -> float | None:
        bucket = self.by_surface.get(surface)
        if bucket is None:
            return None
        return _safe_div(bucket.won_first_serve_n, bucket.won_first_serve_d)

    def surface_winrate(self, surface: str | None) -> float | None:
        bucket = self.by_surface.get(surface)
        if bucket is None or bucket.matches == 0:
            return None
        return bucket.wins / bucket.matches

    def last_n_winrate(self) -> float | None:
        if not self.recent:
            return None
        return sum(self.recent) / len(self.recent)

    def surface_last_n_winrate(self, surface: str | None) -> float | None:
        """Recent-form winrate on the requested surface, computed off
        the per-surface ring buffer (last SURFACE_RECENT_WINDOW
        results on this surface). None when the player has no prior
        matches on this surface — the caller materialises as NaN so
        catboost reads the cold-start as missing rather than 0.5.
        """
        buf = self.surface_recent.get(surface)
        if not buf:
            return None
        return sum(buf) / len(buf)

    def global_elo(self) -> float:
        """Current global Elo rating. Never None — defaults to
        ELO_INITIAL until the first match folds in.
        """
        return self.elo_global

    def surface_elo(self, surface: str | None) -> float:
        """Per-surface Elo rating. Falls back to the player's current
        global Elo when this surface has no prior contributions — a
        warm start that respects player strength rather than resetting
        to chess-default 1500 on every surface debut.
        """
        return self.elo_by_surface.get(surface, self.elo_global)

    def matches_in_last_n_days(self, on_date: date, n_days: int) -> int:
        """Count of matches played in the n_days ending at on_date.
        Read off `recent_match_dates` (the rolling deque) — bounded by
        deque length but RECENT_DATES_WINDOW=32 comfortably exceeds
        the busiest 14d schedule, so the count is exact within the
        FATIGUE_WINDOW_DAYS range.

        A player with no priors returns 0 (genuinely 0 matches in the
        window — not None). The cold-start gate already prevents
        sub-MIN_PRIORS_PER_SIDE players from training, so this branch
        only fires at the very first match.
        """
        if not self.recent_match_dates:
            return 0
        cutoff = on_date - pd.Timedelta(days=n_days)
        cutoff_date = cutoff.date() if isinstance(cutoff, pd.Timestamp) else cutoff
        return sum(1 for d in self.recent_match_dates if d > cutoff_date)

    def avg_recent_opp_rank(self) -> float | None:
        """Mean rank of the last RECENT_HISTORY_WINDOW opponents (with
        a known rank at match time). Lower = tougher recent schedule.
        Returns None when the deque is empty (no recent opponents had
        a ranking snapshot, which is a real edge case for very-junior
        debut runs).
        """
        if not self.recent_opp_ranks:
            return None
        return sum(self.recent_opp_ranks) / len(self.recent_opp_ranks)

    def avg_recent_set_diff(self) -> float | None:
        """Mean signed set differential over last RECENT_HISTORY_WINDOW
        matches with parseable scores. Positive = dominant recent run
        (winning in straights), negative = grinding (winning in 3 or
        losing decisively). Distinct from winrate: two players at the
        same 5-5 last-10 can have very different set_diff averages.
        Returns None when no scores in the window parsed cleanly.
        """
        if not self.recent_set_diff:
            return None
        return sum(self.recent_set_diff) / len(self.recent_set_diff)

    def days_since(self, on_date: date) -> int | None:
        if self.last_match_date is None:
            return None
        delta = (on_date - self.last_match_date).days
        if delta < 0:
            # Future-dated query against a stale state — clamp to 0
            # rather than emit a negative feature.
            return 0
        return min(delta, DAYS_SINCE_CAP)

    def h2h_against(self, opponent_id: int) -> tuple[float | None, int]:
        """Return `(my_winrate_vs_opp, n_priors)`. None winrate when no
        priors — caller materialises as NaN for catboost.
        """
        record = self.h2h.get(opponent_id)
        if record is None:
            return None, 0
        total = record.wins + record.losses
        if total == 0:
            return None, 0
        return record.wins / total, total

    def h2h_decider_winrate(
        self, opponent_id: int
    ) -> tuple[float | None, int]:
        """Matchup-conditioned decider-set winrate. Returns
        `(winrate, n_deciders)`; None winrate when no deciders played
        vs this opponent. Mirrors what `/h2h/stats/{a}/{b}` returns
        live, derivable from the raw match parquet here.
        """
        record = self.h2h.get(opponent_id)
        if record is None or record.decider_played == 0:
            return None, 0
        return record.decider_won / record.decider_played, record.decider_played

    def h2h_tiebreak_winrate(
        self, opponent_id: int
    ) -> tuple[float | None, int]:
        """Matchup-conditioned tiebreak winrate. Returns `(winrate, n)`;
        None winrate when no tiebreaks played vs this opponent.
        """
        record = self.h2h.get(opponent_id)
        if record is None or record.tiebreak_played == 0:
            return None, 0
        return record.tiebreak_won / record.tiebreak_played, record.tiebreak_played

    def h2h_comeback_winrate(
        self, opponent_id: int
    ) -> tuple[float | None, int]:
        """Matchup-conditioned comeback rate — match wins given subject
        lost set 1, divided by total set-1 losses vs this opponent.
        """
        record = self.h2h.get(opponent_id)
        if record is None or record.comeback_set1_lost == 0:
            return None, 0
        return (
            record.comeback_won / record.comeback_set1_lost,
            record.comeback_set1_lost,
        )

    def h2h_close_match_winrate(
        self, opponent_id: int
    ) -> tuple[float | None, int]:
        """Matchup-conditioned close-match winrate. Close = final-set
        margin ≤ 2 OR final set was a tiebreak. Returns `(winrate, n)`.
        """
        record = self.h2h.get(opponent_id)
        if record is None or record.close_match_played == 0:
            return None, 0
        return (
            record.close_match_won / record.close_match_played,
            record.close_match_played,
        )


@dataclass
class HistoryStore:
    """Container for per-tour player histories. Slate-time predictor
    holds one of these in module-level state, refreshed lazily when
    the parquet's mtime changes.
    """

    by_id: dict[int, PlayerHistory] = field(default_factory=dict)

    def get(self, player_id: int) -> PlayerHistory | None:
        return self.by_id.get(player_id)

    def get_or_create(self, player_id: int) -> PlayerHistory:
        h = self.by_id.get(player_id)
        if h is None:
            h = PlayerHistory(player_id=player_id)
            self.by_id[player_id] = h
        return h


def _row_get_int(row: dict[str, Any] | pd.Series, key: str) -> int | None:
    v = row.get(key) if isinstance(row, dict) else row[key]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _add_match_from_row(
    store: HistoryStore,
    row: pd.Series,
    rank_lookup: dict[tuple[str, int], list[tuple[int, int, int | None]]] | None = None,
) -> None:
    """Fold one chronologically-sorted parquet row into both players'
    accumulators. Caller is responsible for snapshotting BEFORE this
    call when the row is also a training example.

    `rank_lookup` is the pre-built `{(tour, player_id) → [(ts, rank,
    points), ...]}` structure from `_build_rank_lookup`. When provided,
    each side's pre-match rank is looked up via bisect and folded into
    the OPPONENT's `recent_opp_ranks` deque (so the schedule-strength
    feature reflects "the average rank of the players I've recently
    faced"). None → schedule-strength deque accumulates nothing and
    the corresponding feature lands NaN.
    """
    p1 = _row_get_int(row, "p1_id")
    p2 = _row_get_int(row, "p2_id")
    if p1 is None or p2 is None:
        return
    winner_side = _row_get_int(row, "winner_side")
    surface = surface_key(_row_get_int(row, "court_id"))
    on_date = pd.Timestamp(row["match_date"]).date()
    tour = str(row["tour"]) if "tour" in row.index else None

    h1 = store.get_or_create(p1)
    h2 = store.get_or_create(p2)

    p1_won = winner_side == 1
    p2_won = winner_side == 2

    # Parse the score string ONCE per row; pass the result to both
    # add_match calls. The parser returns winner-relative facts; each
    # PlayerHistory rotates onto subject perspective via `won`.
    score_raw = row.get("score") if isinstance(row, dict) else (
        row["score"] if "score" in row.index else None
    )
    if isinstance(score_raw, float) and pd.isna(score_raw):
        score_raw = None
    best_of = _row_get_int(row, "best_of")
    score_details = parse_score_details(score_raw, best_of, winner_side)

    # Pre-snapshot both players' Elos BEFORE either add_match call.
    # Critical for symmetry — if we read h2's Elo after h1.add_match
    # ran, h1's update would have shifted h1's state but h2 would
    # still see the OLD state for the symmetric update, breaking the
    # zero-sum property of Elo.
    h1_pre_elo_g = h1.elo_global
    h2_pre_elo_g = h2.elo_global
    h1_pre_elo_s = h1.elo_by_surface.get(surface, h1_pre_elo_g)
    h2_pre_elo_s = h2.elo_by_surface.get(surface, h2_pre_elo_g)

    # Opponent's rank at match time — folded into each subject's
    # `recent_opp_ranks` deque so the schedule-strength snapshot
    # reflects "ranks of players I've recently faced". rank_lookup
    # missing → both look up None and the feature lands NaN.
    p1_rank_at_match: int | None = None
    p2_rank_at_match: int | None = None
    if rank_lookup and tour:
        p1_rank_at_match, _ = lookup_rank_at(rank_lookup, tour, p1, on_date)
        p2_rank_at_match, _ = lookup_rank_at(rank_lookup, tour, p2, on_date)

    # Subject set differential per match (subject sets won − opp sets
    # won). The MatchStat parquet doesn't carry best_of, so we infer
    # it from the parsed score (n_sets) and the tournament tier — see
    # `infer_best_of`. With inferred best_of in {3, 5} and a parsed
    # score, set arithmetic gives winner_sets_won = (best_of+1)//2,
    # loser = n_sets − winner, and the winner-relative diff rotates
    # onto each player's subject frame via `won`. None only when the
    # score itself didn't parse cleanly.
    p1_set_diff: int | None = None
    p2_set_diff: int | None = None
    if score_details is not None:
        tier = tier_key(_row_get_int(row, "rank_id"))
        inferred_bo = infer_best_of(score_details.n_sets, tier)
        winner_sets_won = inferred_bo // 2 + 1
        loser_sets_won = score_details.n_sets - winner_sets_won
        # Defensive: a malformed score that reports more sets than
        # the inferred best_of allows would give a negative loser
        # count. Clamp to 0 — better to under-count set diff than to
        # emit nonsense into the dominance deque.
        loser_sets_won = max(loser_sets_won, 0)
        winner_diff = winner_sets_won - loser_sets_won
        p1_set_diff = winner_diff if p1_won else -winner_diff
        p2_set_diff = winner_diff if p2_won else -winner_diff

    h1.add_match(
        match_date=on_date,
        won=p1_won,
        surface=surface,
        opponent_id=p2,
        my_first_serve=_row_get_int(row, "p1_first_serve"),
        my_first_serve_of=_row_get_int(row, "p1_first_serve_of"),
        my_won_first_serve=_row_get_int(row, "p1_won_first_serve"),
        my_won_first_serve_of=_row_get_int(row, "p1_won_first_serve_of"),
        my_won_second_serve=_row_get_int(row, "p1_won_second_serve"),
        my_won_second_serve_of=_row_get_int(row, "p1_won_second_serve_of"),
        my_bp_converted=_row_get_int(row, "p1_bp_converted"),
        my_bp_converted_of=_row_get_int(row, "p1_bp_converted_of"),
        my_aces=_row_get_int(row, "p1_aces"),
        my_double_faults=_row_get_int(row, "p1_double_faults"),
        opp_won_first_serve=_row_get_int(row, "p2_won_first_serve"),
        opp_won_first_serve_of=_row_get_int(row, "p2_won_first_serve_of"),
        opp_won_second_serve=_row_get_int(row, "p2_won_second_serve"),
        opp_won_second_serve_of=_row_get_int(row, "p2_won_second_serve_of"),
        opp_bp_converted=_row_get_int(row, "p2_bp_converted"),
        opp_bp_converted_of=_row_get_int(row, "p2_bp_converted_of"),
        score_details=score_details,
        opp_pre_match_elo_global=h2_pre_elo_g,
        opp_pre_match_elo_surface=h2_pre_elo_s,
        opp_rank_at_match=p2_rank_at_match,
        subject_set_diff=p1_set_diff,
    )
    h2.add_match(
        match_date=on_date,
        won=p2_won,
        surface=surface,
        opponent_id=p1,
        my_first_serve=_row_get_int(row, "p2_first_serve"),
        my_first_serve_of=_row_get_int(row, "p2_first_serve_of"),
        my_won_first_serve=_row_get_int(row, "p2_won_first_serve"),
        my_won_first_serve_of=_row_get_int(row, "p2_won_first_serve_of"),
        my_won_second_serve=_row_get_int(row, "p2_won_second_serve"),
        my_won_second_serve_of=_row_get_int(row, "p2_won_second_serve_of"),
        my_bp_converted=_row_get_int(row, "p2_bp_converted"),
        my_bp_converted_of=_row_get_int(row, "p2_bp_converted_of"),
        my_aces=_row_get_int(row, "p2_aces"),
        my_double_faults=_row_get_int(row, "p2_double_faults"),
        opp_won_first_serve=_row_get_int(row, "p1_won_first_serve"),
        opp_won_first_serve_of=_row_get_int(row, "p1_won_first_serve_of"),
        opp_won_second_serve=_row_get_int(row, "p1_won_second_serve"),
        opp_won_second_serve_of=_row_get_int(row, "p1_won_second_serve_of"),
        opp_bp_converted=_row_get_int(row, "p1_bp_converted"),
        opp_bp_converted_of=_row_get_int(row, "p1_bp_converted_of"),
        score_details=score_details,
        opp_pre_match_elo_global=h1_pre_elo_g,
        opp_pre_match_elo_surface=h1_pre_elo_s,
        opp_rank_at_match=p1_rank_at_match,
        subject_set_diff=p2_set_diff,
    )


def build_history_store(
    matches_df: pd.DataFrame,
    rankings_df: pd.DataFrame | None = None,
) -> HistoryStore:
    """Walk the entire match parquet chronologically and return the
    final state. Used by `gbt.py` at slate time — the resulting store
    represents the corpus's "as-of-latest-row" snapshot, which is the
    correct prior for predicting tomorrow's matches.

    `matches_df` is expected to be sorted by `match_date` ascending;
    `gbt_backfill.backfill` already does this at write time, but we
    re-sort defensively in case the caller hands us a filtered view.

    `rankings_df` is the optional rankings_history table. When provided,
    each match's opponent-rank is looked up point-in-time and folded
    into the schedule-strength deque (`recent_opp_ranks`); without it
    that feature lands NaN at predict time and catboost reads as
    missing.
    """
    store = HistoryStore()
    df = matches_df.sort_values("match_date")
    rank_lookup = _build_rank_lookup(rankings_df) if rankings_df is not None else None
    for _, row in df.iterrows():
        _add_match_from_row(store, row, rank_lookup=rank_lookup)
    log.info(
        "built history store: %d players, %d matches walked",
        len(store.by_id), len(df),
    )
    return store


def _diff_or_nan(a: float | None, b: float | None) -> float | None:
    """Return a − b, or None if either operand is missing. Caller turns
    None into NaN for catboost ingestion at materialisation time.
    """
    if a is None or b is None:
        return None
    return a - b


# Schema constants — exposed so train and predict use the same column
# order and the same categorical declarations.
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "career_first_serve_in_pct_diff",
    "career_first_serve_win_pct_diff",
    "career_second_serve_win_pct_diff",
    # Split first/second-serve return — distinct signals (the sim
    # already splits them; v1 collapsed for compactness, v2 restores
    # the split now that recency weighting makes the columns more
    # informative).
    "career_first_serve_return_win_pct_diff",
    "career_second_serve_return_win_pct_diff",
    "career_bp_save_pct_diff",
    "career_bp_convert_pct_diff",
    # Career-aggregate clutch — derived from per-match score strings
    # via `parse_score_details`. All recency-weighted on the same
    # HALF_LIFE_DAYS schedule as the rate counters above. Distinct from
    # the H2H lens's matchup-conditioned clutch (which is per-opponent
    # and lives on `TennisInMatchupStats`); these are career-aggregate
    # signals — "this player wins tiebreaks generally" rather than "vs
    # this opponent".
    "career_tiebreak_winrate_diff",
    "career_decider_winrate_diff",
    "career_comeback_winrate_diff",
    "career_close_match_winrate_diff",
    "surface_first_serve_win_pct_diff",
    "surface_record_diff",
    "last_n_winrate_diff",
    "days_since_diff",
    "age_diff",
    # `h2h_advantage` = (anchor's winrate vs opp) − 0.5, in [−0.5, 0.5].
    # Centered at 0 so the feature is sign-flip symmetric under
    # anchor swap (a swap maps x → −x), which is what every other
    # numeric column does. The earlier `h2h_anchor_winrate` formulation
    # mapped x → 1−x under swap, which broke the model's symmetry
    # (verified empirically: max swap-symmetry error 0.22 → 0 after
    # this change).
    "h2h_advantage",
    "n_h2h_priors",
    # Matchup-conditioned clutch — derived from `_H2HRecord` sub-counts
    # the parquet now carries (added 2026-05-15). Each advantage is
    # centered at 0 like `h2h_advantage` so anchor-swap symmetry holds.
    # Each pairs with its own sample-size count so the GBT can learn
    # to discount thin-sample rates. Distinct from the career-aggregate
    # clutch columns above — these say "deciders specifically vs THIS
    # opponent" rather than "deciders generally."
    "h2h_decider_advantage",
    "n_h2h_decider_priors",
    "h2h_tiebreak_advantage",
    "n_h2h_tiebreak_priors",
    "h2h_comeback_advantage",
    "n_h2h_comeback_priors",
    "h2h_close_match_advantage",
    "n_h2h_close_match_priors",
    # Point-in-time rankings — anchor minus opp under the symmetric
    # form. `rank_diff` uses opp − anchor because lower rank number =
    # stronger player; positive `rank_diff` therefore means anchor is
    # the higher-ranked side. `rank_points_diff` is the natural sign
    # (anchor − opp) since higher points = stronger.
    "rank_diff",
    "rank_points_diff",
    # Elo ratings — global plus surface-specific, both anchor − opp.
    # Higher = anchor stronger by Elo. Surface Elo warm-starts from
    # the player's global Elo on first appearance on a new surface
    # so the diff is meaningful even for surface-debut matches. These
    # are the single highest-signal numeric features in the tennis-
    # prediction literature (Sackmann, Kovalchik); adding them to the
    # full-feature GBT closes most of the gap between rate-aggregate
    # models and the published Elo-only benchmarks.
    "elo_global_diff",
    "elo_surface_diff",
    # Form-on-this-surface — last SURFACE_RECENT_WINDOW results on the
    # requested surface, anchor − opp. Captures the surface-specific
    # momentum signal that the all-surfaces `last_n_winrate_diff`
    # misses (a 7-3 hard run paired with a 2-8 clay run averages to
    # 4.5/10 globally but reads 3-7 on a clay-court slate).
    "surface_last_n_winrate_diff",
    # Load / scheduling density — matches in last FATIGUE_WINDOW_DAYS
    # ending the day before the slate. Positive = anchor more loaded
    # (running long). The GBT learns the nonlinear "how much is too
    # much" via histogram splits — a 4-matches-in-14d run could be
    # mid-run for a deep-tournament push (positive value) or signal
    # accumulated fatigue (negative value), and the split target on
    # the leaf decides which interpretation applies in context.
    "fatigue_matches_14d_diff",
    # Schedule strength — mean rank of the last RECENT_HISTORY_WINDOW
    # opponents (point-in-time, on match date). Sign convention
    # mirrors `rank_diff`: lower rank = stronger opponent, so opp's
    # avg-recent-opp-rank − anchor's puts "positive = anchor faced
    # tougher schedule" — same orientation as the other "positive
    # favours anchor" numerics for downstream interpretability.
    "recent_opp_rank_diff",
    # Dominance — mean signed set differential over last
    # RECENT_HISTORY_WINDOW matches with parseable scores. Positive
    # = anchor more dominant in recent matches (winning in straights
    # vs grinding three-setters). Complements `last_n_winrate_diff`
    # (same N, same source matches, but a magnitude rather than
    # binary signal — two 7-3 players can have very different
    # set-diff averages if one swept their wins and the other
    # ground them out).
    "recent_set_diff_diff",
    # Season-time encoded as a sin/cos pair (cyclical so December →
    # January doesn't appear as a discontinuity to the splitter).
    # Match-level (identical across both anchor orientations under
    # augmentation), so the GBT treats them as match context — letting
    # it learn calendar-effects within a surface (clay season early
    # vs late, end-of-year fatigue, ATP Finals selection effect) that
    # the `surface` categorical alone can't express. Expected
    # marginal contribution is small — included for completeness.
    "season_doy_sin",
    "season_doy_cos",
    # Serve-domination differentials. Aces per first-serve point and
    # double-faults per second-serve point — vendor-shipped per-match
    # box-score figures aggregated with the same recency-weight as
    # the other rate counters. Distinct from the existing
    # `career_first_serve_win_pct_diff` (which collapses ace-driven
    # wins and rally-driven wins into one bucket); a big-server who
    # wins 70% via aces reads differently from one who wins 70% in
    # rallies — relevant for surface and matchup interactions the
    # tree can exploit.
    "career_ace_rate_diff",
    "career_df_rate_diff",
)

CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = ("surface", "tier", "best_of")

ALL_FEATURE_COLUMNS: tuple[str, ...] = (
    NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
)


def compute_features(
    *,
    anchor_history: PlayerHistory,
    opponent_history: PlayerHistory,
    anchor_id: int,
    opponent_id: int,
    on_date: date,
    surface: str | None,
    tier: str | None,
    best_of: int | None,
    anchor_birthdate: date | None,
    opponent_birthdate: date | None,
    anchor_rank: int | None = None,
    opponent_rank: int | None = None,
    anchor_rank_points: int | None = None,
    opponent_rank_points: int | None = None,
) -> dict[str, Any]:
    """Build one feature row from two pre-positioned `PlayerHistory`
    snapshots. Returns a dict suitable for catboost's pandas ingestion
    (None / NaN allowed; catboost handles them natively).

    Anchor convention: the caller has already chosen anchor = lower
    MatchStat id. Differential features are anchor − opponent, so a
    positive value means the anchor is the stronger side on that
    metric. Categoricals are match-level, no anchor-relativity.

    `anchor_rank` / `opponent_rank` (and the corresponding `_points`)
    are optional point-in-time values. The training path looks them up
    from `rankings_history.parquet`; the inference path reads them off
    the live `TennisStatsContext.player_*` blocks. None on either side
    → rank_diff / rank_points_diff land NaN and catboost ignores them.
    """

    # Differential numerics. _diff_or_nan returns None when either
    # operand is unobserved; catboost reads None as missing.
    feats: dict[str, Any] = {
        "career_first_serve_in_pct_diff": _diff_or_nan(
            anchor_history.career_first_serve_in_pct(),
            opponent_history.career_first_serve_in_pct(),
        ),
        "career_first_serve_win_pct_diff": _diff_or_nan(
            anchor_history.career_first_serve_win_pct(),
            opponent_history.career_first_serve_win_pct(),
        ),
        "career_second_serve_win_pct_diff": _diff_or_nan(
            anchor_history.career_second_serve_win_pct(),
            opponent_history.career_second_serve_win_pct(),
        ),
        "career_first_serve_return_win_pct_diff": _diff_or_nan(
            anchor_history.career_first_serve_return_win_pct(),
            opponent_history.career_first_serve_return_win_pct(),
        ),
        "career_second_serve_return_win_pct_diff": _diff_or_nan(
            anchor_history.career_second_serve_return_win_pct(),
            opponent_history.career_second_serve_return_win_pct(),
        ),
        "career_bp_save_pct_diff": _diff_or_nan(
            anchor_history.career_bp_save_pct(),
            opponent_history.career_bp_save_pct(),
        ),
        "career_bp_convert_pct_diff": _diff_or_nan(
            anchor_history.career_bp_convert_pct(),
            opponent_history.career_bp_convert_pct(),
        ),
        "career_tiebreak_winrate_diff": _diff_or_nan(
            anchor_history.career_tiebreak_winrate(),
            opponent_history.career_tiebreak_winrate(),
        ),
        "career_decider_winrate_diff": _diff_or_nan(
            anchor_history.career_decider_winrate(),
            opponent_history.career_decider_winrate(),
        ),
        "career_comeback_winrate_diff": _diff_or_nan(
            anchor_history.career_comeback_winrate(),
            opponent_history.career_comeback_winrate(),
        ),
        "career_close_match_winrate_diff": _diff_or_nan(
            anchor_history.career_close_match_winrate(),
            opponent_history.career_close_match_winrate(),
        ),
        "surface_first_serve_win_pct_diff": _diff_or_nan(
            anchor_history.surface_first_serve_win_pct(surface),
            opponent_history.surface_first_serve_win_pct(surface),
        ),
        "surface_record_diff": _diff_or_nan(
            anchor_history.surface_winrate(surface),
            opponent_history.surface_winrate(surface),
        ),
        "last_n_winrate_diff": _diff_or_nan(
            anchor_history.last_n_winrate(),
            opponent_history.last_n_winrate(),
        ),
    }

    a_days = anchor_history.days_since(on_date)
    o_days = opponent_history.days_since(on_date)
    feats["days_since_diff"] = (
        None if a_days is None or o_days is None else a_days - o_days
    )

    # Age delta (in years, fractional). Computed from birthdates so
    # the GBT can read both signs (older vs younger anchor) on a
    # smooth scale.
    a_age = _years_from(on_date, anchor_birthdate)
    o_age = _years_from(on_date, opponent_birthdate)
    feats["age_diff"] = _diff_or_nan(a_age, o_age)

    # H2H — anchor's advantage (winrate − 0.5) and the meeting count.
    # The centered form is sign-flip symmetric under anchor swap, which
    # is the property every other numeric column has by construction.
    h2h_rate, n_h2h = anchor_history.h2h_against(opponent_id)
    feats["h2h_advantage"] = None if h2h_rate is None else h2h_rate - 0.5
    feats["n_h2h_priors"] = n_h2h

    # Matchup-conditioned clutch sub-counts from `_H2HRecord`. Same
    # `winrate − 0.5` centering as `h2h_advantage` for anchor-swap
    # symmetry. Each rate pairs with its own sample-size column so the
    # GBT can learn that a "1-0 in deciders" rate is less trustworthy
    # than "7-2 in deciders" without us having to bake in a hand-tuned
    # confidence-shrinkage rule.
    h2h_dec, n_dec = anchor_history.h2h_decider_winrate(opponent_id)
    feats["h2h_decider_advantage"] = None if h2h_dec is None else h2h_dec - 0.5
    feats["n_h2h_decider_priors"] = n_dec

    h2h_tb, n_tb = anchor_history.h2h_tiebreak_winrate(opponent_id)
    feats["h2h_tiebreak_advantage"] = None if h2h_tb is None else h2h_tb - 0.5
    feats["n_h2h_tiebreak_priors"] = n_tb

    h2h_cb, n_cb = anchor_history.h2h_comeback_winrate(opponent_id)
    feats["h2h_comeback_advantage"] = None if h2h_cb is None else h2h_cb - 0.5
    feats["n_h2h_comeback_priors"] = n_cb

    h2h_cm, n_cm = anchor_history.h2h_close_match_winrate(opponent_id)
    feats["h2h_close_match_advantage"] = (
        None if h2h_cm is None else h2h_cm - 0.5
    )
    feats["n_h2h_close_match_priors"] = n_cm

    # Point-in-time rank diffs. `rank` is lower=stronger so flip the
    # sign convention (opp − anchor) to land in the "positive = anchor
    # stronger" frame the other numeric columns use. `rank_points` is
    # higher=stronger so keep the natural anchor − opp orientation.
    feats["rank_diff"] = (
        None if (anchor_rank is None or opponent_rank is None)
        else opponent_rank - anchor_rank
    )
    feats["rank_points_diff"] = (
        None if (anchor_rank_points is None or opponent_rank_points is None)
        else anchor_rank_points - opponent_rank_points
    )

    # Elo diffs. global_elo / surface_elo never return None (warm
    # start to ELO_INITIAL or to global on cold start), so no
    # None-guard needed. Surface Elo with no prior on the requested
    # surface falls back to global, so even a cold-surface row gets
    # a meaningful diff.
    feats["elo_global_diff"] = (
        anchor_history.global_elo() - opponent_history.global_elo()
    )
    feats["elo_surface_diff"] = (
        anchor_history.surface_elo(surface) - opponent_history.surface_elo(surface)
    )

    # Surface-form diff. None on either side when that player has no
    # priors on the requested surface — `_diff_or_nan` rolls it up to
    # None, catboost reads as missing.
    feats["surface_last_n_winrate_diff"] = _diff_or_nan(
        anchor_history.surface_last_n_winrate(surface),
        opponent_history.surface_last_n_winrate(surface),
    )

    # Fatigue / load (anchor − opp). matches_in_last_n_days always
    # returns an int (0 when no priors fall in the window), so the
    # diff is always defined post-cold-start — no NaN-handling
    # needed.
    feats["fatigue_matches_14d_diff"] = (
        anchor_history.matches_in_last_n_days(on_date, FATIGUE_WINDOW_DAYS)
        - opponent_history.matches_in_last_n_days(on_date, FATIGUE_WINDOW_DAYS)
    )

    # Schedule strength (opp − anchor on the rank scale so positive
    # means anchor's recent schedule was tougher). _diff_or_nan
    # rolls up None when either side's deque is empty (unlikely
    # post-cold-start but the cold-start gate measures match count,
    # not opp-rank coverage).
    feats["recent_opp_rank_diff"] = _diff_or_nan(
        opponent_history.avg_recent_opp_rank(),
        anchor_history.avg_recent_opp_rank(),
    )

    # Dominance (anchor − opp). Range roughly [-2, +2] per side under
    # the standard tennis set arithmetic (bo3 winner = +2 in
    # straights, +1 in a three-setter; loser flips sign). Mean over
    # the recent window pulls toward 0 for "average tour player",
    # away from 0 for streaky-dominant or streaky-shaky form.
    feats["recent_set_diff_diff"] = _diff_or_nan(
        anchor_history.avg_recent_set_diff(),
        opponent_history.avg_recent_set_diff(),
    )

    # Season-time as a cyclical pair. Match-level (identical across
    # both anchor orientations under augmentation), so adds match
    # context — calendar effects within a surface that the surface
    # categorical alone can't express. 366.0 (not 365.0) avoids the
    # leap-year off-by-one without affecting the cyclical assumption.
    doy = on_date.timetuple().tm_yday
    feats["season_doy_sin"] = math.sin(2.0 * math.pi * doy / 366.0)
    feats["season_doy_cos"] = math.cos(2.0 * math.pi * doy / 366.0)

    # Ace + DF differentials. _diff_or_nan rolls up None when either
    # side's denominator is unobserved (cold-start serve-stats — rare
    # post the MIN_PRIORS_PER_SIDE gate, but defensive).
    feats["career_ace_rate_diff"] = _diff_or_nan(
        anchor_history.career_ace_rate(),
        opponent_history.career_ace_rate(),
    )
    feats["career_df_rate_diff"] = _diff_or_nan(
        anchor_history.career_df_rate(),
        opponent_history.career_df_rate(),
    )

    # Categoricals — never None; catboost can take a literal "unknown"
    # token but mixing None with strings breaks pandas → catboost
    # ingestion. Use string sentinels so the column dtype stays object.
    feats["surface"] = surface or "unknown"
    feats["tier"] = tier or "unknown"
    feats["best_of"] = str(best_of if best_of in (3, 5) else DEFAULT_BEST_OF)

    return feats


def _years_from(on_date: date, birthdate: date | None) -> float | None:
    if birthdate is None:
        return None
    delta = on_date - birthdate
    return delta.days / 365.25


# ---------------------------------------------------------------------------
# Rank-history helpers — load `rankings_history.parquet` into a
# (tour, player_id) → sorted timeline dict, look up the most recent
# rank on or before a given match date via bisect. Used by both the
# training path (build_training_table) and the algo backtest harness.
# ---------------------------------------------------------------------------


def _build_rank_lookup(
    rankings_df: pd.DataFrame,
) -> dict[tuple[str, int], list[tuple[int, int, int | None]]]:
    """Return `{(tour, player_id): [(epoch_seconds, rank, points), ...]}`
    sorted ascending by date. Caller uses bisect for the "most recent
    rank on or before match_date" lookup.
    """
    out: dict[tuple[str, int], list[tuple[int, int, int | None]]] = {}
    ts = pd.to_datetime(rankings_df["ranking_date"]).astype("int64") // 10**9
    tours = rankings_df["tour"].to_numpy()
    pids = rankings_df["player_id"].to_numpy()
    ranks = rankings_df["rank"].to_numpy()
    points = rankings_df["rank_points"].to_numpy()
    for i in range(len(rankings_df)):
        pt = int(points[i]) if pd.notna(points[i]) else None
        key = (str(tours[i]), int(pids[i]))
        out.setdefault(key, []).append(
            (int(ts.iloc[i]), int(ranks[i]), pt)
        )
    for k in out:
        out[k].sort()
    return out


def lookup_rank_at(
    lookup: dict[tuple[str, int], list[tuple[int, int, int | None]]],
    tour: str,
    pid: int,
    on_date: date,
) -> tuple[int | None, int | None]:
    """Return (rank, rank_points) at or before `on_date`, or (None, None)
    when the (tour, player_id) has no snapshot history (e.g. unranked
    player or pre-2008 dates). Bisect on the sorted timestamps; O(log n)
    per lookup.
    """
    entries = lookup.get((tour, pid))
    if not entries:
        return None, None
    target_ts = int(pd.Timestamp(on_date).timestamp())
    lo, hi = 0, len(entries)
    while lo < hi:
        mid = (lo + hi) // 2
        if entries[mid][0] <= target_ts:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if idx < 0:
        return None, None
    return entries[idx][1], entries[idx][2]


@dataclass
class TrainingTable:
    """Output container for `build_training_table`. Carries the feature
    rows alongside the metadata columns trainers care about for cuts
    (date for walk-forward split, tour for per-tour Brier breakdown,
    match_id for retro joins).
    """

    rows: pd.DataFrame
    n_dropped_cold_start: int
    n_dropped_other: int


def build_training_table(
    matches_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    *,
    rankings_df: pd.DataFrame | None = None,
    min_priors: int = MIN_PRIORS_PER_SIDE,
) -> TrainingTable:
    """Walk the parquet chronologically. For each row, snapshot both
    players' state BEFORE folding the match in; that snapshot becomes
    the feature row IF both sides have ≥ min_priors prior matches.

    `rankings_df` is the optional `rankings_history.parquet` table; when
    provided, `compute_features` receives point-in-time `rank` and
    `rank_points` for each player and emits the `rank_diff` /
    `rank_points_diff` features. When None, those features land NaN and
    catboost ignores them (the model degrades gracefully on rank-free
    re-trains).

    Returns a `TrainingTable` whose `rows` DataFrame carries
    `ALL_FEATURE_COLUMNS + ['target', 'match_id', 'match_date',
    'tour', 'anchor_id', 'opponent_id']` for downstream filtering.
    """
    profile_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for _, p in profiles_df.iterrows():
        tour = p.get("tour")
        pid = _row_get_int(p, "player_id")
        if tour is None or pid is None:
            continue
        bd = p.get("birthdate")
        bd_date: date | None = None
        if isinstance(bd, pd.Timestamp) and not pd.isna(bd):
            bd_date = bd.date()
        profile_lookup[(str(tour), int(pid))] = {"birthdate": bd_date}

    rank_lookup = _build_rank_lookup(rankings_df) if rankings_df is not None else {}

    df = matches_df.sort_values("match_date").reset_index(drop=True)
    store = HistoryStore()
    out_rows: list[dict[str, Any]] = []
    n_cold = 0
    n_other = 0

    for _, row in df.iterrows():
        p1 = _row_get_int(row, "p1_id")
        p2 = _row_get_int(row, "p2_id")
        winner_side = _row_get_int(row, "winner_side")
        if p1 is None or p2 is None or winner_side not in (1, 2):
            n_other += 1
            continue
        on_date = pd.Timestamp(row["match_date"]).date()
        h1 = store.get_or_create(p1)
        h2 = store.get_or_create(p2)

        # Cold-start gate. Snapshot is allowed only when BOTH sides
        # have ≥ min_priors matches in their state.
        if h1.matches < min_priors or h2.matches < min_priors:
            n_cold += 1
            _add_match_from_row(store, row, rank_lookup=rank_lookup or None)
            continue

        # Augmentation: emit BOTH anchor directions per match. This
        # forces model-level symmetry (the trained classifier learns
        # f(features) + f(-features) ≈ 1) — without it, the model only
        # ever sees lower-id-anchor and learns a slightly asymmetric
        # function (verified empirically: max swap-symmetry error
        # 0.22 → ~0 after augmentation).
        tour = str(row["tour"])
        p1_prof = profile_lookup.get((tour, p1), {})
        p2_prof = profile_lookup.get((tour, p2), {})
        surface = surface_key(_row_get_int(row, "court_id"))
        tier = tier_key(_row_get_int(row, "rank_id"))
        # Best-of inference. Parquet's best_of column is None for
        # every row in the MatchStat backfill — use parsed score
        # length + tier instead. This restores the `best_of`
        # categorical to a useful 3-vs-5 split (it would otherwise
        # collapse to a constant "3" via DEFAULT_BEST_OF and add no
        # signal). Parse score once per row, here, mirroring what
        # `_add_match_from_row` does for set-diff inference.
        best_of = _row_get_int(row, "best_of")
        if best_of not in (3, 5):
            score_raw_for_bo = (
                row["score"] if "score" in row.index else None
            )
            if isinstance(score_raw_for_bo, float) and pd.isna(score_raw_for_bo):
                score_raw_for_bo = None
            sd_for_bo = parse_score_details(
                score_raw_for_bo, best_of, winner_side,
            )
            best_of = infer_best_of(
                sd_for_bo.n_sets if sd_for_bo is not None else None,
                tier,
            )
        match_id = _row_get_int(row, "match_id")

        # Point-in-time rank lookups (None when rankings_df was not
        # provided OR when this player has no snapshot ≤ match_date).
        p1_rank, p1_pts = (
            lookup_rank_at(rank_lookup, tour, p1, on_date)
            if rank_lookup else (None, None)
        )
        p2_rank, p2_pts = (
            lookup_rank_at(rank_lookup, tour, p2, on_date)
            if rank_lookup else (None, None)
        )

        for (
            anchor_id, opp_id, anchor_h, opp_h, a_prof, o_prof,
            anchor_won, a_rank, a_pts, o_rank, o_pts,
        ) in (
            (p1, p2, h1, h2, p1_prof, p2_prof, winner_side == 1,
             p1_rank, p1_pts, p2_rank, p2_pts),
            (p2, p1, h2, h1, p2_prof, p1_prof, winner_side == 2,
             p2_rank, p2_pts, p1_rank, p1_pts),
        ):
            feats = compute_features(
                anchor_history=anchor_h,
                opponent_history=opp_h,
                anchor_id=anchor_id,
                opponent_id=opp_id,
                on_date=on_date,
                surface=surface,
                tier=tier,
                best_of=best_of,
                anchor_birthdate=a_prof.get("birthdate"),
                opponent_birthdate=o_prof.get("birthdate"),
                anchor_rank=a_rank,
                opponent_rank=o_rank,
                anchor_rank_points=a_pts,
                opponent_rank_points=o_pts,
            )
            feats.update({
                "target": int(anchor_won),
                "match_id": match_id,
                "match_date": on_date,
                "tour": tour,
                "anchor_id": anchor_id,
                "opponent_id": opp_id,
                # Snapshot prior-counts so retro grading and the spike
                # report can see the "evidence base size" per row.
                "n_prior_matches_anchor": anchor_h.matches,
                "n_prior_matches_opponent": opp_h.matches,
            })
            out_rows.append(feats)

        # Now fold the match in for future rows. Once per match — both
        # augmented rows share the same prior-state.
        _add_match_from_row(store, row, rank_lookup=rank_lookup or None)

    rows_df = pd.DataFrame(out_rows)
    log.info(
        "built training table: %d rows kept, %d dropped (cold start), %d dropped (other)",
        len(rows_df), n_cold, n_other,
    )
    return TrainingTable(rows=rows_df, n_dropped_cold_start=n_cold, n_dropped_other=n_other)


__all__ = [
    "ALL_FEATURE_COLUMNS",
    "CATEGORICAL_FEATURE_COLUMNS",
    "DAYS_SINCE_CAP",
    "DEFAULT_BEST_OF",
    "ELO_INITIAL",
    "ELO_K",
    "FATIGUE_WINDOW_DAYS",
    "HALF_LIFE_DAYS",
    "HistoryStore",
    "MIN_PRIORS_PER_SIDE",
    "NUMERIC_FEATURE_COLUMNS",
    "PlayerHistory",
    "RECENT_DATES_WINDOW",
    "RECENT_FORM_WINDOW",
    "RECENT_HISTORY_WINDOW",
    "SURFACE_RECENT_WINDOW",
    "TrainingTable",
    "build_history_store",
    "build_training_table",
    "compute_features",
    "surface_key",
    "tier_key",
]
