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
HALF_LIFE_DAYS = 365.0


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

    # H2H by opponent. (wins, total) pairs keyed by opponent
    # MatchStat id. Sparse — most pairings have no priors at slate
    # time.
    h2h: dict[int, tuple[int, int]] = field(default_factory=dict)

    # Recent-form ring buffer (newest at the right). bool wins.
    recent: deque[bool] = field(default_factory=lambda: deque(maxlen=RECENT_FORM_WINDOW))

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

        # Surface bucket — same fields as overall but scoped.
        bucket = self.by_surface.setdefault(surface, _SurfaceAggregate())
        bucket.matches += 1
        bucket.wins += 1 if won else 0
        bucket.won_first_serve_n += n(my_won_first_serve)
        bucket.won_first_serve_d += n(my_won_first_serve_of)

        # H2H bookkeeping. Direction = "my wins / total meetings"; the
        # snapshot derives the anchor's H2H rate vs the opponent.
        prior_w, prior_t = self.h2h.get(opponent_id, (0, 0))
        self.h2h[opponent_id] = (prior_w + (1 if won else 0), prior_t + 1)

        # Clutch counters — only when the score parsed cleanly.
        # `score_details` is match-winner-relative; rotate onto subject
        # using `won`. Recency-weighted on the same schedule as the rate
        # counters (decay applied via `_decay_to` above).
        if score_details is not None:
            if score_details.went_to_decider:
                self.decider_played += 1.0
                if won:
                    self.decider_won += 1.0
            self.tiebreak_played += score_details.n_tiebreaks_played
            subject_tbs_won = (
                score_details.winner_tiebreaks_won
                if won
                else score_details.n_tiebreaks_played
                - score_details.winner_tiebreaks_won
            )
            self.tiebreak_won += subject_tbs_won
            if score_details.is_close_match:
                self.close_match_played += 1.0
                if won:
                    self.close_match_won += 1.0
            subject_won_set_one = (
                score_details.winner_won_set_one
                if won
                else not score_details.winner_won_set_one
            )
            if not subject_won_set_one:
                self.comeback_set1_lost += 1.0
                if won:
                    self.comeback_won += 1.0

        # Recent / counts.
        self.recent.append(won)
        self.matches += 1
        if won:
            self.wins += 1

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
        wins, total = self.h2h.get(opponent_id, (0, 0))
        if total == 0:
            return None, 0
        return wins / total, total


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


def _add_match_from_row(store: HistoryStore, row: pd.Series) -> None:
    """Fold one chronologically-sorted parquet row into both players'
    accumulators. Caller is responsible for snapshotting BEFORE this
    call when the row is also a training example.
    """
    p1 = _row_get_int(row, "p1_id")
    p2 = _row_get_int(row, "p2_id")
    if p1 is None or p2 is None:
        return
    winner_side = _row_get_int(row, "winner_side")
    surface = surface_key(_row_get_int(row, "court_id"))
    on_date = pd.Timestamp(row["match_date"]).date()

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
        opp_won_first_serve=_row_get_int(row, "p2_won_first_serve"),
        opp_won_first_serve_of=_row_get_int(row, "p2_won_first_serve_of"),
        opp_won_second_serve=_row_get_int(row, "p2_won_second_serve"),
        opp_won_second_serve_of=_row_get_int(row, "p2_won_second_serve_of"),
        opp_bp_converted=_row_get_int(row, "p2_bp_converted"),
        opp_bp_converted_of=_row_get_int(row, "p2_bp_converted_of"),
        score_details=score_details,
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
        opp_won_first_serve=_row_get_int(row, "p1_won_first_serve"),
        opp_won_first_serve_of=_row_get_int(row, "p1_won_first_serve_of"),
        opp_won_second_serve=_row_get_int(row, "p1_won_second_serve"),
        opp_won_second_serve_of=_row_get_int(row, "p1_won_second_serve_of"),
        opp_bp_converted=_row_get_int(row, "p1_bp_converted"),
        opp_bp_converted_of=_row_get_int(row, "p1_bp_converted_of"),
        score_details=score_details,
    )


def build_history_store(matches_df: pd.DataFrame) -> HistoryStore:
    """Walk the entire match parquet chronologically and return the
    final state. Used by `gbt.py` at slate time — the resulting store
    represents the corpus's "as-of-latest-row" snapshot, which is the
    correct prior for predicting tomorrow's matches.

    `matches_df` is expected to be sorted by `match_date` ascending;
    `gbt_backfill.backfill` already does this at write time, but we
    re-sort defensively in case the caller hands us a filtered view.
    """
    store = HistoryStore()
    df = matches_df.sort_values("match_date")
    for _, row in df.iterrows():
        _add_match_from_row(store, row)
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
) -> dict[str, Any]:
    """Build one feature row from two pre-positioned `PlayerHistory`
    snapshots. Returns a dict suitable for catboost's pandas ingestion
    (None / NaN allowed; catboost handles them natively).

    Anchor convention: the caller has already chosen anchor = lower
    MatchStat id. Differential features are anchor − opponent, so a
    positive value means the anchor is the stronger side on that
    metric. Categoricals are match-level, no anchor-relativity.
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
    min_priors: int = MIN_PRIORS_PER_SIDE,
) -> TrainingTable:
    """Walk the parquet chronologically. For each row, snapshot both
    players' state BEFORE folding the match in; that snapshot becomes
    the feature row IF both sides have ≥ min_priors prior matches.

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
            _add_match_from_row(store, row)
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
        best_of = _row_get_int(row, "best_of")
        match_id = _row_get_int(row, "match_id")

        for anchor_id, opp_id, anchor_h, opp_h, a_prof, o_prof, anchor_won in (
            (p1, p2, h1, h2, p1_prof, p2_prof, winner_side == 1),
            (p2, p1, h2, h1, p2_prof, p1_prof, winner_side == 2),
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
        _add_match_from_row(store, row)

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
    "HistoryStore",
    "MIN_PRIORS_PER_SIDE",
    "NUMERIC_FEATURE_COLUMNS",
    "PlayerHistory",
    "RECENT_FORM_WINDOW",
    "TrainingTable",
    "build_history_store",
    "build_training_table",
    "compute_features",
    "surface_key",
    "tier_key",
]
