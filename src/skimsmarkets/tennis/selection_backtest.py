"""Walk-forward backtest harness for the pre-LLM tennis selector.

The selector's job (per `selection.py`) is to pick top-K events from a
slate so the LLM stack spends tokens on matchups likely to bucket as
Lock or Lean downstream. Iterating on the selector requires a way to
score "did this candidate pick well?" without running the full LLM
stack offline (impossible — no historical prompts, lens fetchers,
judge). This harness fills that gap with a **synthetic Lock label**
derived from the trained outcome GBT (`tennis_gbt_spike.cbm`):

  synthetic_lock = (gbt_p_winner >= LOCK_THRESHOLD) AND (gbt_won)

This proxies the two downstream Lock ingredients that ARE computable
offline (magnitude + correctness). The third — judge defensibility +
market convergence — is not. The relative ordering of selector
algorithms on this metric transfers to production even if the absolute
Lock rate shifts; if algorithm X beats algorithm Y on synthetic-Lock
precision-at-K, X will tend to beat Y on production Lock rate too.

Why not use the live retro corpus (`logs/runs/*.jsonl`) for labels?
N≈25 prediction rows across 4 runs at the time of writing — far too
few for meaningful per-tier iteration. The GBT parquet (127k matches,
2009-2026) gives the iteration loop the sample size it needs; live
retro stays as the final-validation surface (phase 3+).

Walk structure mirrors `algo_backtest.py`:
  - Sort the parquet by match_date.
  - For each calendar day, snapshot every match BEFORE folding any of
    the day's matches into HistoryStore (point-in-time discipline,
    same as the GBT training pipeline).
  - Partition day-records by tour to form synthetic slates (production
    slates are per-tour: ATP and WTA never share a Polymarket event).
  - Slates with size ≤ K are skipped — no selection happens, so
    measuring "selection quality" is meaningless.
  - On selectable slates: score every match with the candidate
    `score_fn`, take top-K by score, count synthetic-Locks in the
    top-K. precision_at_k = locks / k.
  - Aggregate across slates: mean precision-at-K, weighted by slate
    size (large slates exercise the selector more than small ones).

Cold-start matches (either player < MIN_PRIORS_PER_SIDE) get no GBT
prediction and therefore no synthetic Lock label. They CAN still
appear in slates and be picked by the selector — in that case they
count as "not Lock" in the precision computation. This is intentional:
picking a cold-start matchup wastes a cap slot regardless of how the
algorithm justifies it, since downstream the lens chains have thin
material to reason with.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from catboost import CatBoostClassifier, Pool

from skimsmarkets.tennis.algo_backtest import (
    _age_at,
    _build_profile_lookup,
    _build_rank_lookup,
    _lookup_rank_at,
    _project_player,
)
from skimsmarkets.tennis.gbt_backfill import (
    PLAYER_PROFILES_PATH,
    RAW_MATCHES_PATH,
)
from skimsmarkets.tennis.gbt_features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    MIN_PRIORS_PER_SIDE,
    HistoryStore,
    PlayerHistory,
    _add_match_from_row,
    _row_get_int,
    compute_features,
    surface_key,
)
from skimsmarkets.classify import (
    EV_BUCKET_EDGE,
    EV_BUCKET_PRIME,
    classify_ev,
)
from skimsmarkets.ev import compute_ev_per_dollar
from skimsmarkets.tennis.gbt_rankings_backfill import RANKINGS_HISTORY_PATH
from skimsmarkets.tennis.gbt_train import MODEL_PATH
from skimsmarkets.tennis.models import TennisPlayerStats

log = logging.getLogger(__name__)

# Selection cap matches production (`config.MAX_SLATE_EVENTS`). Held
# locally rather than imported so the harness stays runnable without
# touching the live config import chain.
DEFAULT_SLATE_CAP = 5

# Synthetic Lock thresholds on the GBT's winner-side probability. 0.75
# mirrors `classify.THRESHOLD_LOCK` on the production risk_score; 0.60
# mirrors `classify.THRESHOLD_LEAN`. Both are reported in the result so
# the user can pick which to optimize.
LOCK_THRESHOLD = 0.75
LEAN_THRESHOLD = 0.60

# Same train/holdout split as `algo_backtest.py` and `gbt_train.py`.
# Reported baseline numbers should always be the HOLDOUT precision —
# train-fold precision is shown for diagnostics but is in-sample.
TRAIN_CUTOFF = date(2024, 12, 31)


# ---------------------------------------------------------------------------
# Type interface for candidate scorers.
# ---------------------------------------------------------------------------


# A scoring function takes per-match data and returns a [0, 1] score.
# Higher = better candidate for selection. Concrete kwargs passed by
# the harness (see `_process_match` below):
#   a_stats: TennisPlayerStats        (anchor side projection)
#   b_stats: TennisPlayerStats        (opponent side projection)
#   a_history: PlayerHistory | None   (raw HistoryStore entry — richer than
#                                       a_stats; gives clutch rates, H2H,
#                                       surface-specific serve %, etc.)
#   b_history: PlayerHistory | None
#   surface: str | None               (resolved from court_id)
#   best_of: int                       (3 or 5)
#   tour: str                          ("atp" | "wta")
#   match_date: date                   (point-in-time anchor)
#   round_id: int | None               (parquet round_id — late rounds
#                                       are higher-quality opponents)
#   rank_id: int | None                (parquet rank_id — tournament tier
#                                       category)
#
# Scorers ignore kwargs they don't need via `**_`. PlayerHistory exposes
# more than TennisPlayerStats does (the latter is the production
# pydantic schema). Backtest scorers can use either; the production
# wire-up will need to surface additional fields when those tiers ship.
ScoreFn = Callable[..., float]


# ---------------------------------------------------------------------------
# Result dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlateMetrics:
    """Aggregated metrics for one fold (train or holdout).

    Two parallel metric families:

      - Lock-side  (confidence-mode KPI): the synthetic Lock label that
        the v1 selector was tuned against (+29% Lock precision over rank
        baseline). Picks land in `precision_at_k_lock`.

      - EV-side    (ev-mode KPI):   synthesised from GBT-vs-rank-implied
        market_p. `mean_realized_return_at_k` is THE headline number —
        average $ return per $1 staked on GBT's pick at rank-implied
        market_p across all picks. A random selector should yield ~0;
        a perfect EV selector should yield ~mean(model_p − market_p).
        `precision_at_k_ev_*` are bucketing-coverage diagnostics.

    Both families are computed for every fold so a single backtest run
    serves both modes; the operator picks which to optimise.
    """

    n_slates: int
    n_picks: int  # total events picked across all slates (n_slates * cap, modulo small-slate skips)
    n_labelable: int  # picks where the GBT was able to assign a label
    precision_at_k_lock: float  # picks that are synthetic Locks / n_picks
    precision_at_k_lock_or_lean: float
    base_rate_lock: float  # base rate across the full labeled pool
    base_rate_lock_or_lean: float
    # EV-side metrics (NaN when no EV-labelable picks in the fold).
    n_ev_labelable: int = 0
    precision_at_k_ev_prime: float = float("nan")
    precision_at_k_ev_edge_or_better: float = float("nan")
    mean_ev_at_k: float = float("nan")
    mean_realized_return_at_k: float = float("nan")  # THE EV-mode headline
    base_rate_ev_prime: float = float("nan")
    base_rate_ev_edge_or_better: float = float("nan")
    base_mean_realized_return: float = float("nan")
    per_tour: dict[str, dict[str, float | int]] = field(default_factory=dict)
    per_slate_size: dict[str, dict[str, float | int]] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionBacktestResult:
    train: SlateMetrics
    holdout: SlateMetrics
    n_dropped_cold_start: int
    n_dropped_other: int
    n_total_matches: int
    n_slates_seen: int
    n_slates_evaluable: int  # slate size > cap
    train_cutoff: str
    slate_cap: int
    scorer_name: str


# ---------------------------------------------------------------------------
# Catboost loader — backtest-local, decoupled from the production
# singleton in `tennis/gbt.py` so we can run without touching live state.
# ---------------------------------------------------------------------------


def _load_gbt_model() -> tuple[CatBoostClassifier, list[int]]:
    """Load the trained outcome GBT + return its categorical feature
    column indices for use in `Pool(... cat_features=...)`.

    Same `tennis_gbt_spike.cbm` artefact the production predictor uses.
    Raises if the file is missing — the backtest has no fallback (no
    GBT → no synthetic Lock labels → no metric).
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GBT model not found at {MODEL_PATH}. Train it with "
            "`uv run python -m skimsmarkets.tennis.gbt_train` first."
        )
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    cat_idx = [
        list(ALL_FEATURE_COLUMNS).index(c) for c in CATEGORICAL_FEATURE_COLUMNS
    ]
    return model, cat_idx


def _features_dataframe(features: dict[str, Any]) -> pd.DataFrame:
    """One-row DataFrame in the column order catboost was trained on.
    Mirrors `tennis/gbt._features_dataframe` but inlined to avoid the
    production import.
    """
    return pd.DataFrame([{col: features.get(col) for col in ALL_FEATURE_COLUMNS}])


def _predict_p_anchor_won(
    model: CatBoostClassifier,
    cat_idx: list[int],
    *,
    anchor_history: PlayerHistory,
    opp_history: PlayerHistory,
    anchor_id: int,
    opp_id: int,
    on_date: date,
    surface: str | None,
    best_of: int,
    anchor_birthdate: date | None,
    opp_birthdate: date | None,
    anchor_rank: int | None,
    opp_rank: int | None,
    anchor_rank_points: int | None,
    opp_rank_points: int | None,
) -> float:
    """Predict P(anchor wins) using both orientations and averaging for
    symmetry — same trick the production predictor uses to absorb
    catboost's residual anchor asymmetry (drift up to ~0.07 on random
    pairs without the average).
    """
    feats_a = compute_features(
        anchor_history=anchor_history, opponent_history=opp_history,
        anchor_id=anchor_id, opponent_id=opp_id,
        on_date=on_date, surface=surface, tier=None, best_of=best_of,
        anchor_birthdate=anchor_birthdate, opponent_birthdate=opp_birthdate,
        anchor_rank=anchor_rank, opponent_rank=opp_rank,
        anchor_rank_points=anchor_rank_points, opponent_rank_points=opp_rank_points,
    )
    feats_b = compute_features(
        anchor_history=opp_history, opponent_history=anchor_history,
        anchor_id=opp_id, opponent_id=anchor_id,
        on_date=on_date, surface=surface, tier=None, best_of=best_of,
        anchor_birthdate=opp_birthdate, opponent_birthdate=anchor_birthdate,
        anchor_rank=opp_rank, opponent_rank=anchor_rank,
        anchor_rank_points=opp_rank_points, opponent_rank_points=anchor_rank_points,
    )
    pool_a = Pool(data=_features_dataframe(feats_a), cat_features=cat_idx)
    pool_b = Pool(data=_features_dataframe(feats_b), cat_features=cat_idx)
    p_a_as_anchor = float(model.predict_proba(pool_a)[0, 1])
    p_b_as_anchor = float(model.predict_proba(pool_b)[0, 1])
    # Symmetric estimate of P(anchor wins).
    return (p_a_as_anchor + (1.0 - p_b_as_anchor)) / 2.0


# ---------------------------------------------------------------------------
# Synthetic Lock label.
# ---------------------------------------------------------------------------


def _synthetic_lock_label(
    p_anchor_won: float, anchor_won: int, *, threshold: float
) -> bool:
    """A match is a "synthetic Lock at threshold T" if the GBT's
    winner-side probability is >= T AND the prediction was correct.

    "Winner-side probability" = max(p, 1-p): we don't penalize the
    selector for picking events where the GBT was very confident
    against the anchor — the algorithm doesn't see anchor convention,
    only "is one player clearly favored". The orientation only matters
    for the correctness check.
    """
    p_winner_side = max(p_anchor_won, 1.0 - p_anchor_won)
    if p_winner_side < threshold:
        return False
    gbt_picks_anchor = p_anchor_won >= 0.5
    actual_anchor_won = anchor_won == 1
    return gbt_picks_anchor == actual_anchor_won


# ---------------------------------------------------------------------------
# Synthetic market_p + EV labels.
#
# Polymarket — the venue we trade against — has no point-in-time history we
# can join to the GBT parquet, so the EV-precision metric needs a synthesized
# market_p. We use rank-points-implied probability via Bradley-Terry:
#
#     market_proxy_p_anchor = pts_anchor / (pts_anchor + pts_opp)
#
# Rank-implied is the closest free proxy to RETAIL prediction markets like
# Polymarket. The literature (Gorgi 2019, Kovalchik 2020, Yue 2022, plus the
# favorite-longshot-bias work in Sport Finance 2017) consistently shows:
#   - Retail markets anchor heavily on visible signals (rank, seeding) and
#     under-price true-skill features (Elo, surface specialism, serve quality).
#   - Sharp markets (Pinnacle) close that gap; Polymarket does NOT.
# So "rank-implied" approximates Polymarket's pricing better than e.g. an
# Elo-implied proxy would. The selector's task in EV mode becomes: find
# events where GBT's prediction will diverge from rank-implied → high EV.
#
# Realized return (the headline metric):
#   - For each pick, compute the dollar return if you'd bet $1 on GBT's
#     predicted side at the rank-implied market_p:
#         WON  → +payoff_ratio  =  (1 - market_p) / market_p
#         LOST → −1
#   - Aggregated across picks, this measures TRUE alpha: a random scorer
#     gets ~0, a perfect EV scorer earns mean(model_p − market_p) per pick.
# ---------------------------------------------------------------------------


def _market_proxy_p_anchor(
    a_rank_points: int | None, b_rank_points: int | None
) -> float | None:
    """Rank-points-implied probability that anchor wins, via Bradley-Terry.

    Returns None when either side has missing or non-positive points
    (mirrors the v1 selector's null-on-missing posture and `compute_ev_per
    _dollar`'s degenerate-edge guard). Polymarket pricing for tennis
    closely follows ranking, so this proxies retail-market consensus
    cheaply and deterministically.

    Bradley-Terry with k=1 (linear in points) is the simplest defensible
    form. Higher k sharpens the favorite; lower k flattens. k=1 matches
    empirical Polymarket midprices for top-50 matchups within ±5pp
    (informal check on settled tennis events; not load-bearing — this
    is the BACKTEST proxy, production reads live Polymarket mids).
    """
    if a_rank_points is None or b_rank_points is None:
        return None
    if a_rank_points <= 0 or b_rank_points <= 0:
        return None
    total = a_rank_points + b_rank_points
    return a_rank_points / total


def _synthesize_ev_labels(
    p_anchor_won: float | None,
    market_p_anchor: float | None,
    anchor_won: int,
) -> tuple[float | None, str | None, float | None, bool | None]:
    """Compute the EV-side of the synthetic labels for one match.

    Returns `(ev_per_dollar, ev_bucket, realized_return, gbt_pick_is_anchor)`
    where every field is None when EV cannot be computed (missing GBT
    prediction OR missing rank-points-implied market_p OR degenerate
    market_p at 0/1). `realized_return` is the dollar P&L per $1 staked
    on GBT's predicted side at the synthetic market_p (positive
    `payoff_ratio` if the GBT pick won, −1 if it lost). The EV bucket is
    the same `classify.classify_ev` bucketing the production pipeline
    persists on every PredictionRow.
    """
    if p_anchor_won is None or market_p_anchor is None:
        return None, None, None, None

    gbt_picks_anchor = p_anchor_won >= 0.5
    if gbt_picks_anchor:
        model_p_winner = p_anchor_won
        market_p_winner = market_p_anchor
    else:
        model_p_winner = 1.0 - p_anchor_won
        market_p_winner = 1.0 - market_p_anchor

    ev = compute_ev_per_dollar(model_p_winner, market_p_winner)
    if ev is None:
        return None, None, None, gbt_picks_anchor

    bucket, _ = classify_ev(model_p_winner, market_p_winner)

    # Realized return — what you'd actually earn betting $1 on GBT's
    # pick at the synthetic market_p. Win: payoff_ratio; loss: −1.
    actual_anchor_won = anchor_won == 1
    pick_was_right = gbt_picks_anchor == actual_anchor_won
    if pick_was_right:
        payoff_ratio = (1.0 - market_p_winner) / market_p_winner
        realized = payoff_ratio
    else:
        realized = -1.0
    return ev, bucket, realized, gbt_picks_anchor


# ---------------------------------------------------------------------------
# Baseline scorers — minimal set for phase 0; expand in phase 0.5 to
# port the full current 10-tier algorithm for a fair baseline.
# ---------------------------------------------------------------------------


def make_score_random(seed: int = 42) -> tuple[str, ScoreFn]:
    """Uniform-random scorer with a fixed seed. Pure noise floor —
    a meaningful selector should beat this comfortably.

    Returns (name, fn) so the harness can label the result.
    """
    rng = random.Random(seed)

    def score(**_: Any) -> float:
        return rng.random()

    return f"random_seed{seed}", score


def score_rank_points_ratio(
    *,
    a_stats: TennisPlayerStats,
    b_stats: TennisPlayerStats,
    **_: Any,
) -> float:
    """Tier-1 of the current production algorithm: log10 of the
    rank-points ratio, normalized to [0, 1] at a 10× ratio cap.

    Identical math to `selection._tennis_imbalance`'s base term. Serves
    as the minimal-faithful baseline before phase 0.5 ports the full
    10-tier composition.
    """
    a_points = a_stats.rank_points
    b_points = b_stats.rank_points
    if a_points is None or b_points is None or a_points <= 0 or b_points <= 0:
        return 0.0
    ratio = max(a_points, b_points) / min(a_points, b_points)
    return min(1.0, math.log10(ratio) / math.log10(10.0))


# ---------------------------------------------------------------------------
# Per-match processing — used by the slate walk.
# ---------------------------------------------------------------------------


@dataclass
class _MatchRecord:
    """Per-match record assembled during the slate snapshot pass."""

    tour: str
    match_date: date
    surface: str | None
    best_of: int
    # Selector's score under the candidate algorithm.
    selector_score: float
    # GBT probability that the anchor won. None when cold-start.
    p_anchor_won: float | None
    # Ground truth — did the anchor actually win this match?
    anchor_won: int
    # Synthetic Lock labels at the two thresholds. None when no GBT
    # prediction was possible.
    is_lock: bool | None
    is_lock_or_lean: bool | None
    # EV-side labels — populated when both GBT prediction AND the
    # rank-implied market_p are available; None otherwise. See
    # `_synthesize_ev_labels` for the bucket / realized-return contract.
    ev_per_dollar: float | None = None
    ev_bucket: str | None = None
    realized_return: float | None = None
    market_p_anchor: float | None = None


@dataclass
class _MatchContext:
    """All the data a scorer (or the label/GBT computation) needs to
    evaluate one match. Built ONCE per match in the multi-scorer
    path so the expensive operations (GBT inference, projection)
    don't get duplicated per scorer.
    """

    tour: str
    match_date: date
    surface: str | None
    best_of: int
    round_id: int | None
    rank_id: int | None
    a_stats: TennisPlayerStats
    b_stats: TennisPlayerStats
    a_history: PlayerHistory | None
    b_history: PlayerHistory | None
    p_anchor_won: float | None
    anchor_won: int
    is_lock: bool | None
    is_lock_or_lean: bool | None
    # EV-side labels — see `_MatchRecord` for the contract.
    ev_per_dollar: float | None = None
    ev_bucket: str | None = None
    realized_return: float | None = None
    market_p_anchor: float | None = None


def _build_match_context(
    row: pd.Series,
    *,
    store: HistoryStore,
    rank_lookup: dict,
    profile_lookup: dict,
    gbt_model: CatBoostClassifier,
    cat_idx: list[int],
    min_priors: int,
) -> _MatchContext | None:
    """All the expensive per-match work — projection, GBT prediction,
    synthetic Lock labeling — without the score_fn call. Used by both
    the single-scorer (`_process_match`) and multi-scorer paths.

    Returns None for unusable rows (missing ids, unresolvable winner).
    """
    p1 = _row_get_int(row, "p1_id")
    p2 = _row_get_int(row, "p2_id")
    winner_side = _row_get_int(row, "winner_side")
    if p1 is None or p2 is None or winner_side not in (1, 2):
        return None

    anchor = min(p1, p2)
    opp = max(p1, p2)
    anchor_won_int = 1 if (
        (p1 == anchor and winner_side == 1)
        or (p2 == anchor and winner_side == 2)
    ) else 0
    on_date = pd.Timestamp(row["match_date"]).date()
    tour = str(row["tour"])
    surface = surface_key(_row_get_int(row, "court_id"))
    bo_raw = _row_get_int(row, "best_of")
    best_of = 5 if bo_raw == 5 else 3

    ah = store.get(anchor)
    oh = store.get(opp)
    cold_start = (
        ah is None or oh is None
        or ah.matches < min_priors or oh.matches < min_priors
    )

    a_rank, a_pts = _lookup_rank_at(rank_lookup, tour, anchor, on_date)
    o_rank, o_pts = _lookup_rank_at(rank_lookup, tour, opp, on_date)
    a_prof = profile_lookup.get((tour, anchor))
    o_prof = profile_lookup.get((tour, opp))
    a_name = a_prof[0] if a_prof else f"player_{anchor}"
    o_name = o_prof[0] if o_prof else f"player_{opp}"
    a_plays = a_prof[1] if a_prof else None
    o_plays = o_prof[1] if o_prof else None
    a_bd = a_prof[2] if a_prof else None
    o_bd = o_prof[2] if o_prof else None

    ah_for_proj = ah if ah is not None else PlayerHistory(player_id=anchor)
    oh_for_proj = oh if oh is not None else PlayerHistory(player_id=opp)
    a_stats = _project_player(
        ah_for_proj, rank=a_rank, rank_points=a_pts,
        name=a_name, plays=a_plays, age_years=_age_at(a_bd, on_date),
    )
    b_stats = _project_player(
        oh_for_proj, rank=o_rank, rank_points=o_pts,
        name=o_name, plays=o_plays, age_years=_age_at(o_bd, on_date),
    )

    p_anchor_won: float | None = None
    is_lock: bool | None = None
    is_lock_or_lean: bool | None = None
    if not cold_start:
        try:
            p_anchor_won = _predict_p_anchor_won(
                gbt_model, cat_idx,
                anchor_history=ah, opp_history=oh,
                anchor_id=anchor, opp_id=opp,
                on_date=on_date, surface=surface, best_of=best_of,
                anchor_birthdate=a_bd, opp_birthdate=o_bd,
                anchor_rank=a_rank, opp_rank=o_rank,
                anchor_rank_points=a_pts, opp_rank_points=o_pts,
            )
        except Exception:  # noqa: BLE001
            log.exception("gbt predict crashed on match_id=%s", _row_get_int(row, "match_id"))
            p_anchor_won = None
        if p_anchor_won is not None:
            is_lock = _synthetic_lock_label(p_anchor_won, anchor_won_int, threshold=LOCK_THRESHOLD)
            is_lock_or_lean = _synthetic_lock_label(p_anchor_won, anchor_won_int, threshold=LEAN_THRESHOLD)

    market_p_anchor = _market_proxy_p_anchor(a_pts, o_pts)
    ev_per_dollar, ev_bucket, realized_return, _ = _synthesize_ev_labels(
        p_anchor_won, market_p_anchor, anchor_won_int,
    )

    return _MatchContext(
        tour=tour,
        match_date=on_date,
        surface=surface,
        best_of=best_of,
        round_id=_row_get_int(row, "round_id"),
        rank_id=_row_get_int(row, "rank_id"),
        a_stats=a_stats,
        b_stats=b_stats,
        a_history=ah,
        b_history=oh,
        p_anchor_won=p_anchor_won,
        anchor_won=anchor_won_int,
        is_lock=is_lock,
        is_lock_or_lean=is_lock_or_lean,
        ev_per_dollar=ev_per_dollar,
        ev_bucket=ev_bucket,
        realized_return=realized_return,
        market_p_anchor=market_p_anchor,
    )


def _score_context(ctx: _MatchContext, score_fn: ScoreFn) -> float:
    """Apply one score_fn to a pre-built context. Defensive: clamp to
    [0, 1] and swallow exceptions so a buggy scorer can't break the
    walk.

    Computes the primitive kwargs (h2h_total_meetings, match counts)
    from the histories so scorers don't have to know about
    PlayerHistory — the same primitives are computed in production
    from provider data, keeping the scorer signature portable.
    """
    h2h_total = 0
    a_total = 0
    b_total = 0
    a_surface = 0
    b_surface = 0
    if ctx.a_history is not None and ctx.b_history is not None:
        # H2H total — symmetric across sides; use one direction.
        _, n = ctx.a_history.h2h_against(ctx.b_history.player_id)
        h2h_total = int(n)
    if ctx.a_history is not None:
        a_total = int(ctx.a_history.matches)
        if ctx.surface is not None:
            bucket = ctx.a_history.by_surface.get(ctx.surface)
            a_surface = int(round(bucket.matches)) if bucket else 0
    if ctx.b_history is not None:
        b_total = int(ctx.b_history.matches)
        if ctx.surface is not None:
            bucket = ctx.b_history.by_surface.get(ctx.surface)
            b_surface = int(round(bucket.matches)) if bucket else 0

    try:
        s = float(
            score_fn(
                a_stats=ctx.a_stats, b_stats=ctx.b_stats,
                a_history=ctx.a_history, b_history=ctx.b_history,
                surface=ctx.surface, best_of=ctx.best_of,
                tour=ctx.tour, match_date=ctx.match_date,
                round_id=ctx.round_id, rank_id=ctx.rank_id,
                h2h_total_meetings=h2h_total,
                a_total_matches=a_total,
                b_total_matches=b_total,
                a_surface_matches=a_surface,
                b_surface_matches=b_surface,
            )
        )
    except Exception:  # noqa: BLE001
        log.exception("score_fn crashed")
        s = 0.0
    return max(0.0, min(1.0, s))


def _ctx_to_record(ctx: _MatchContext, selector_score: float) -> _MatchRecord:
    """Materialize a `_MatchRecord` from a context + score."""
    return _MatchRecord(
        tour=ctx.tour,
        match_date=ctx.match_date,
        surface=ctx.surface,
        best_of=ctx.best_of,
        selector_score=selector_score,
        p_anchor_won=ctx.p_anchor_won,
        anchor_won=ctx.anchor_won,
        is_lock=ctx.is_lock,
        is_lock_or_lean=ctx.is_lock_or_lean,
        ev_per_dollar=ctx.ev_per_dollar,
        ev_bucket=ctx.ev_bucket,
        realized_return=ctx.realized_return,
        market_p_anchor=ctx.market_p_anchor,
    )


def _process_match(
    row: pd.Series,
    *,
    store: HistoryStore,
    rank_lookup: dict,
    profile_lookup: dict,
    gbt_model: CatBoostClassifier,
    cat_idx: list[int],
    score_fn: ScoreFn,
    min_priors: int,
) -> _MatchRecord | None:
    """Snapshot stats for one match BEFORE folding it back into the
    store. Returns the record or None for unusable rows (missing ids,
    unresolvable winner side).

    Cold-start matches (history below `min_priors` on either side) get
    a record with `p_anchor_won=None` and `is_lock=None` — they can
    still be SCORED by the candidate algorithm (which gets whatever
    point-in-time projection is available), they just don't carry a
    synthetic Lock label. The aggregator treats them as not-Lock in
    precision-at-K.
    """
    ctx = _build_match_context(
        row,
        store=store,
        rank_lookup=rank_lookup,
        profile_lookup=profile_lookup,
        gbt_model=gbt_model,
        cat_idx=cat_idx,
        min_priors=min_priors,
    )
    if ctx is None:
        return None
    selector_score = _score_context(ctx, score_fn)
    return _ctx_to_record(ctx, selector_score)


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _aggregate_fold(
    picks: list[_MatchRecord],
    *,
    n_slates: int,
    full_pool: list[_MatchRecord],
) -> SlateMetrics:
    """Build the per-fold metrics block.

    `picks` = the union of all top-K selections across all evaluable
    slates in the fold.
    `full_pool` = every labelable match seen in the fold (regardless
    of whether it appeared in a slate big enough to be evaluable) —
    used for the base-rate denominator so the user can see whether
    the selector is meaningfully above noise.
    """
    n_picks = len(picks)
    if n_picks == 0:
        return SlateMetrics(
            n_slates=0, n_picks=0, n_labelable=0,
            precision_at_k_lock=float("nan"),
            precision_at_k_lock_or_lean=float("nan"),
            base_rate_lock=float("nan"),
            base_rate_lock_or_lean=float("nan"),
        )

    labelable_picks = [r for r in picks if r.is_lock is not None]
    n_labelable = len(labelable_picks)
    # Unlabelable picks count as NOT-Lock in precision — see module
    # docstring for the rationale.
    n_locks = sum(1 for r in picks if r.is_lock is True)
    n_lock_or_lean = sum(1 for r in picks if r.is_lock_or_lean is True)
    precision_lock = n_locks / n_picks
    precision_lol = n_lock_or_lean / n_picks

    # Base rate from the full labeled pool (NOT just picks) — that's
    # what the selector is competing against.
    labeled_pool = [r for r in full_pool if r.is_lock is not None]
    if labeled_pool:
        base_lock = sum(1 for r in labeled_pool if r.is_lock) / len(labeled_pool)
        base_lol = sum(1 for r in labeled_pool if r.is_lock_or_lean) / len(labeled_pool)
    else:
        base_lock = float("nan")
        base_lol = float("nan")

    # EV-side metrics. Unlike Lock, a pick with no EV label can't be
    # graded on EV — so EV-side denominators use n_ev_labelable rather
    # than n_picks (otherwise a scorer that picks lots of cold-start /
    # zero-points matchups would look artificially bad). Lock-side
    # keeps the "unlabelable = miss" convention because picking a cold-
    # start event genuinely wastes a slot in confidence mode.
    ev_picks = [r for r in picks if r.ev_per_dollar is not None]
    n_ev_labelable = len(ev_picks)
    if n_ev_labelable > 0:
        n_prime = sum(1 for r in ev_picks if r.ev_bucket == EV_BUCKET_PRIME)
        n_edge_plus = sum(
            1 for r in ev_picks
            if r.ev_bucket in (EV_BUCKET_PRIME, EV_BUCKET_EDGE)
        )
        prec_ev_prime = n_prime / n_ev_labelable
        prec_ev_edge = n_edge_plus / n_ev_labelable
        mean_ev = sum(r.ev_per_dollar for r in ev_picks) / n_ev_labelable
        mean_realized = sum(
            r.realized_return for r in ev_picks
            if r.realized_return is not None
        ) / n_ev_labelable
    else:
        prec_ev_prime = float("nan")
        prec_ev_edge = float("nan")
        mean_ev = float("nan")
        mean_realized = float("nan")

    ev_pool = [r for r in full_pool if r.ev_per_dollar is not None]
    if ev_pool:
        base_prime = sum(1 for r in ev_pool if r.ev_bucket == EV_BUCKET_PRIME) / len(ev_pool)
        base_edge_plus = sum(
            1 for r in ev_pool
            if r.ev_bucket in (EV_BUCKET_PRIME, EV_BUCKET_EDGE)
        ) / len(ev_pool)
        base_realized = sum(
            r.realized_return for r in ev_pool
            if r.realized_return is not None
        ) / len(ev_pool)
    else:
        base_prime = float("nan")
        base_edge_plus = float("nan")
        base_realized = float("nan")

    # Per-tour breakdown.
    by_tour: dict[str, list[_MatchRecord]] = defaultdict(list)
    for r in picks:
        by_tour[r.tour].append(r)
    per_tour: dict[str, dict[str, float | int]] = {}
    for t, group in by_tour.items():
        ev_group = [r for r in group if r.ev_per_dollar is not None]
        per_tour[t] = {
            "n_picks": len(group),
            "precision_lock": sum(1 for r in group if r.is_lock is True) / len(group),
            "precision_lock_or_lean": sum(1 for r in group if r.is_lock_or_lean is True) / len(group),
            "n_ev_labelable": len(ev_group),
            "mean_realized_return": (
                sum(r.realized_return for r in ev_group if r.realized_return is not None) / len(ev_group)
                if ev_group else float("nan")
            ),
            "mean_ev_per_dollar": (
                sum(r.ev_per_dollar for r in ev_group) / len(ev_group)
                if ev_group else float("nan")
            ),
            "precision_ev_prime": (
                sum(1 for r in ev_group if r.ev_bucket == EV_BUCKET_PRIME) / len(ev_group)
                if ev_group else float("nan")
            ),
        }

    return SlateMetrics(
        n_slates=n_slates,
        n_picks=n_picks,
        n_labelable=n_labelable,
        precision_at_k_lock=precision_lock,
        precision_at_k_lock_or_lean=precision_lol,
        base_rate_lock=base_lock,
        base_rate_lock_or_lean=base_lol,
        n_ev_labelable=n_ev_labelable,
        precision_at_k_ev_prime=prec_ev_prime,
        precision_at_k_ev_edge_or_better=prec_ev_edge,
        mean_ev_at_k=mean_ev,
        mean_realized_return_at_k=mean_realized,
        base_rate_ev_prime=base_prime,
        base_rate_ev_edge_or_better=base_edge_plus,
        base_mean_realized_return=base_realized,
        per_tour=per_tour,
    )


# ---------------------------------------------------------------------------
# Main backtest walk.
# ---------------------------------------------------------------------------


def run_selection_backtest_multi(
    score_fns: dict[str, ScoreFn],
    *,
    slate_cap: int = DEFAULT_SLATE_CAP,
    train_cutoff: date = TRAIN_CUTOFF,
    min_priors: int = MIN_PRIORS_PER_SIDE,
    start_date: date | None = None,
) -> dict[str, SelectionBacktestResult]:
    """Single-walk variant — score every match against EVERY scorer in
    `score_fns` in one pass. The expensive operations (GBT inference,
    HistoryStore walk) happen once, then each scorer just sorts +
    selects against the same labeled pool.

    Returns `{scorer_name: SelectionBacktestResult}`. Order in the
    returned dict matches `score_fns` insertion order.
    """
    log.info("loading parquets…")
    matches = pd.read_parquet(RAW_MATCHES_PATH).sort_values("match_date").reset_index(drop=True)
    rankings = pd.read_parquet(RANKINGS_HISTORY_PATH)
    profiles = pd.read_parquet(PLAYER_PROFILES_PATH)
    log.info(
        "loaded matches=%d, rankings=%d, profiles=%d",
        len(matches), len(rankings), len(profiles),
    )
    rank_lookup = _build_rank_lookup(rankings)
    profile_lookup = _build_profile_lookup(profiles)
    gbt_model, cat_idx = _load_gbt_model()
    log.info("loaded GBT model from %s", MODEL_PATH)

    store = HistoryStore()
    match_dates = pd.to_datetime(matches["match_date"]).dt.date

    # Per-scorer state.
    scorer_names = list(score_fns.keys())
    train_picks: dict[str, list[_MatchRecord]] = {n: [] for n in scorer_names}
    holdout_picks: dict[str, list[_MatchRecord]] = {n: [] for n in scorer_names}
    n_train_slates: dict[str, int] = {n: 0 for n in scorer_names}
    n_holdout_slates: dict[str, int] = {n: 0 for n in scorer_names}

    # Shared label pools (one per fold).
    train_pool: list[_MatchRecord] = []
    holdout_pool: list[_MatchRecord] = []
    n_slates_seen = 0
    n_dropped_other = 0
    n_dropped_cold = 0

    # Per-day buffer: maps scorer_name → list of records that scorer
    # scored. Each record has that scorer's score; the labels/metadata
    # are shared (set once when first recorded).
    current_day: date | None = None
    day_records_per_scorer: dict[str, list[_MatchRecord]] = {n: [] for n in scorer_names}
    day_rows: list[pd.Series] = []
    # Shared label pool for THIS day (one entry per match — used to
    # build the per-fold full_pool for base-rate computation).
    day_labels: list[_MatchRecord] = []

    def _flush_day(day: date) -> None:
        nonlocal n_slates_seen
        # Push labels into pool.
        pool = train_pool if day <= train_cutoff else holdout_pool
        pool.extend(r for r in day_labels if r.is_lock is not None)
        # Partition by tour for selection.
        by_tour_idx: dict[str, list[int]] = defaultdict(list)
        for i, lbl in enumerate(day_labels):
            by_tour_idx[lbl.tour].append(i)
        for _tour, idx_list in by_tour_idx.items():
            n_slates_seen += 1
            if len(idx_list) <= slate_cap:
                continue
            # For each scorer, sort indexes by THIS scorer's score and
            # take top-K.
            for sname in scorer_names:
                scored = [
                    (day_records_per_scorer[sname][i].selector_score, i)
                    for i in idx_list
                ]
                scored.sort(key=lambda t: -t[0])
                top_k_idx = [i for _, i in scored[:slate_cap]]
                top_k = [day_records_per_scorer[sname][i] for i in top_k_idx]
                if day <= train_cutoff:
                    train_picks[sname].extend(top_k)
                    n_train_slates[sname] += 1
                else:
                    holdout_picks[sname].extend(top_k)
                    n_holdout_slates[sname] += 1
        # Fold today's matches into history.
        for row in day_rows:
            _add_match_from_row(store, row)

    for i in range(len(matches)):
        row = matches.iloc[i]
        day = match_dates.iloc[i]
        if start_date is not None and day < start_date:
            if current_day is not None and day != current_day:
                _flush_day(current_day)
                day_records_per_scorer = {n: [] for n in scorer_names}
                day_labels = []
                day_rows = []
                current_day = day
            elif current_day is None:
                current_day = day
            day_rows.append(row)
            continue

        if current_day is not None and day != current_day:
            _flush_day(current_day)
            day_records_per_scorer = {n: [] for n in scorer_names}
            day_labels = []
            day_rows = []
        current_day = day

        # Build context ONCE (the expensive part — GBT inference +
        # projection). Then each scorer just gets a cheap call.
        ctx = _build_match_context(
            row,
            store=store,
            rank_lookup=rank_lookup,
            profile_lookup=profile_lookup,
            gbt_model=gbt_model,
            cat_idx=cat_idx,
            min_priors=min_priors,
        )
        if ctx is None:
            n_dropped_other += 1
            day_rows.append(row)
            continue
        if ctx.is_lock is None:
            n_dropped_cold += 1
        # Use the FIRST scorer's record to drive labels into day_labels
        # (labels are scorer-agnostic — just need one representative
        # per match). All scorers share the same context so the
        # per-record fields (is_lock, anchor_won, tour, etc.) are
        # identical across scorers.
        first_scorer = scorer_names[0]
        first_score = _score_context(ctx, score_fns[first_scorer])
        day_labels.append(_ctx_to_record(ctx, first_score))
        day_records_per_scorer[first_scorer].append(_ctx_to_record(ctx, first_score))
        for sname in scorer_names[1:]:
            s = _score_context(ctx, score_fns[sname])
            day_records_per_scorer[sname].append(_ctx_to_record(ctx, s))
        day_rows.append(row)

        if (i + 1) % 20000 == 0:
            log.info(
                "walked %d / %d  slates_seen=%d (multi: %d scorers)",
                i + 1, len(matches), n_slates_seen, len(scorer_names),
            )

    if current_day is not None:
        _flush_day(current_day)

    log.info(
        "done. slates_seen=%d cold=%d other=%d",
        n_slates_seen, n_dropped_cold, n_dropped_other,
    )

    results: dict[str, SelectionBacktestResult] = {}
    for sname in scorer_names:
        results[sname] = SelectionBacktestResult(
            train=_aggregate_fold(
                train_picks[sname], n_slates=n_train_slates[sname], full_pool=train_pool,
            ),
            holdout=_aggregate_fold(
                holdout_picks[sname], n_slates=n_holdout_slates[sname], full_pool=holdout_pool,
            ),
            n_dropped_cold_start=n_dropped_cold,
            n_dropped_other=n_dropped_other,
            n_total_matches=len(matches),
            n_slates_seen=n_slates_seen,
            n_slates_evaluable=n_train_slates[sname] + n_holdout_slates[sname],
            train_cutoff=train_cutoff.isoformat(),
            slate_cap=slate_cap,
            scorer_name=sname,
        )
    return results


def run_selection_backtest(
    score_fn: ScoreFn,
    *,
    scorer_name: str,
    slate_cap: int = DEFAULT_SLATE_CAP,
    train_cutoff: date = TRAIN_CUTOFF,
    min_priors: int = MIN_PRIORS_PER_SIDE,
    start_date: date | None = None,
) -> SelectionBacktestResult:
    """Walk the GBT parquet, group into per-day-per-tour slates, score
    every match with `score_fn`, take top-K per slate, measure
    synthetic-Lock precision-at-K.

    `start_date` lets you skip ahead to a recent year for fast
    iteration — full-corpus walks take a few minutes due to GBT
    inference, but a 2024+ walk runs in ~30s. The HISTORY still has
    to walk from the start of the corpus for point-in-time discipline;
    `start_date` only gates whether predictions are RECORDED.
    """
    log.info("loading parquets…")
    matches = pd.read_parquet(RAW_MATCHES_PATH).sort_values("match_date").reset_index(drop=True)
    rankings = pd.read_parquet(RANKINGS_HISTORY_PATH)
    profiles = pd.read_parquet(PLAYER_PROFILES_PATH)
    log.info(
        "loaded matches=%d, rankings=%d, profiles=%d",
        len(matches), len(rankings), len(profiles),
    )
    rank_lookup = _build_rank_lookup(rankings)
    profile_lookup = _build_profile_lookup(profiles)
    gbt_model, cat_idx = _load_gbt_model()
    log.info("loaded GBT model from %s", MODEL_PATH)

    store = HistoryStore()

    # Pre-extract date-only column so the groupby key is stable.
    match_dates = pd.to_datetime(matches["match_date"]).dt.date

    train_picks: list[_MatchRecord] = []
    holdout_picks: list[_MatchRecord] = []
    train_pool: list[_MatchRecord] = []
    holdout_pool: list[_MatchRecord] = []
    n_train_slates = 0
    n_holdout_slates = 0
    n_slates_seen = 0
    n_dropped_other = 0
    n_dropped_cold = 0

    # Iterate by (day, tour) preserving chronological order. groupby
    # with sort=False keeps the order matches appear in the sorted
    # frame, which is already chronological.
    current_day: date | None = None
    day_records: list[_MatchRecord] = []
    day_rows: list[pd.Series] = []

    def _flush_day(day: date) -> None:
        nonlocal n_slates_seen, n_train_slates, n_holdout_slates
        # Partition day's records by tour into per-tour slates.
        by_tour: dict[str, list[_MatchRecord]] = defaultdict(list)
        for rec in day_records:
            by_tour[rec.tour].append(rec)
        for _tour, slate_records in by_tour.items():
            n_slates_seen += 1
            pool = (
                train_pool if day <= train_cutoff else holdout_pool
            )
            pool.extend(r for r in slate_records if r.is_lock is not None)
            if len(slate_records) <= slate_cap:
                continue
            # Sort by selector score desc, take top K.
            slate_records.sort(key=lambda r: -r.selector_score)
            top_k = slate_records[:slate_cap]
            if day <= train_cutoff:
                train_picks.extend(top_k)
                n_train_slates += 1
            else:
                holdout_picks.extend(top_k)
                n_holdout_slates += 1
        # Fold all of today's matches into history AFTER scoring.
        for row in day_rows:
            _add_match_from_row(store, row)

    for i in range(len(matches)):
        row = matches.iloc[i]
        day = match_dates.iloc[i]
        if start_date is not None and day < start_date:
            # Still need to fold history; skip the scoring + label step
            # by NOT appending to day_records. We DO need to flush a
            # prior day if the day rolled over.
            if current_day is not None and day != current_day:
                _flush_day(current_day)
                day_records = []
                day_rows = []
                current_day = day
            elif current_day is None:
                current_day = day
            day_rows.append(row)
            continue

        if current_day is not None and day != current_day:
            _flush_day(current_day)
            day_records = []
            day_rows = []
        current_day = day

        record = _process_match(
            row,
            store=store,
            rank_lookup=rank_lookup,
            profile_lookup=profile_lookup,
            gbt_model=gbt_model,
            cat_idx=cat_idx,
            score_fn=score_fn,
            min_priors=min_priors,
        )
        if record is None:
            n_dropped_other += 1
        else:
            if record.is_lock is None:
                n_dropped_cold += 1
            day_records.append(record)
        day_rows.append(row)

        if (i + 1) % 20000 == 0:
            log.info(
                "walked %d / %d  slates_seen=%d train_picks=%d holdout_picks=%d",
                i + 1, len(matches),
                n_slates_seen, len(train_picks), len(holdout_picks),
            )

    # Flush the final day.
    if current_day is not None:
        _flush_day(current_day)

    log.info(
        "done. slates_seen=%d train_picks=%d holdout_picks=%d cold=%d other=%d",
        n_slates_seen, len(train_picks), len(holdout_picks), n_dropped_cold, n_dropped_other,
    )

    return SelectionBacktestResult(
        train=_aggregate_fold(train_picks, n_slates=n_train_slates, full_pool=train_pool),
        holdout=_aggregate_fold(holdout_picks, n_slates=n_holdout_slates, full_pool=holdout_pool),
        n_dropped_cold_start=n_dropped_cold,
        n_dropped_other=n_dropped_other,
        n_total_matches=len(matches),
        n_slates_seen=n_slates_seen,
        n_slates_evaluable=n_train_slates + n_holdout_slates,
        train_cutoff=train_cutoff.isoformat(),
        slate_cap=slate_cap,
        scorer_name=scorer_name,
    )


# ---------------------------------------------------------------------------
# Scorecard sidecar serialization.
# ---------------------------------------------------------------------------


SCORECARD_DIR = Path("models/selection_backtest")


def _serialize_metrics(m: SlateMetrics) -> dict[str, Any]:
    return {
        "n_slates": m.n_slates,
        "n_picks": m.n_picks,
        "n_labelable": m.n_labelable,
        "precision_at_k_lock": m.precision_at_k_lock,
        "precision_at_k_lock_or_lean": m.precision_at_k_lock_or_lean,
        "base_rate_lock": m.base_rate_lock,
        "base_rate_lock_or_lean": m.base_rate_lock_or_lean,
        "n_ev_labelable": m.n_ev_labelable,
        "precision_at_k_ev_prime": m.precision_at_k_ev_prime,
        "precision_at_k_ev_edge_or_better": m.precision_at_k_ev_edge_or_better,
        "mean_ev_at_k": m.mean_ev_at_k,
        "mean_realized_return_at_k": m.mean_realized_return_at_k,
        "base_rate_ev_prime": m.base_rate_ev_prime,
        "base_rate_ev_edge_or_better": m.base_rate_ev_edge_or_better,
        "base_mean_realized_return": m.base_mean_realized_return,
        "per_tour": m.per_tour,
    }


def write_scorecard(result: SelectionBacktestResult) -> Path:
    """Persist the result to a JSON scorecard so iterations are
    diffable and grep-able across sessions."""
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    path = SCORECARD_DIR / f"{result.scorer_name}.metrics.json"
    payload: dict[str, Any] = {
        "scorer_name": result.scorer_name,
        "slate_cap": result.slate_cap,
        "train_cutoff": result.train_cutoff,
        "n_total_matches": result.n_total_matches,
        "n_slates_seen": result.n_slates_seen,
        "n_slates_evaluable": result.n_slates_evaluable,
        "n_dropped_cold_start": result.n_dropped_cold_start,
        "n_dropped_other": result.n_dropped_other,
        "train": _serialize_metrics(result.train),
        "holdout": _serialize_metrics(result.holdout),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
