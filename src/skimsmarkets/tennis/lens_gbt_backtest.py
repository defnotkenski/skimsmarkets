"""Phase 0 backtest harness for the lens-GBT initiative.

Produces a single scorecard that answers the three-tier metric from
`memory/project_lens_gbt_initiative.md`:

  1. PRIMARY (ship/no-ship) — does combining the per-lens GBT with the
     full-feature GBT (`tennis_gbt_spike.cbm`) beat the full-feature
     GBT alone by ≥0.002 Brier? Tests 50/50 average + fitted weight.
  2. CO-PRIMARY — bucket-hit-rate at production thresholds
     (LOCK=0.75, LEAN=0.60), per-lens GBT vs algo lens baseline.
     Target: ≥0.02 absolute lift.
  3. SANITY — absolute Brier ≥0.002 better than algo lens
     (≤0.2231 for form_and_surface vs algo lens 0.2251).

Walks the parquet once via `build_training_table` (the same point-in-
time pipeline both the full-feature GBT and the algo lens use), then:

  - Trains the per-lens GBT on the augmented train fold (≤2024-12-31).
  - Predicts per-lens on the augmented holdout fold (>2024-12-31).
  - Loads `tennis_gbt_spike.cbm` and predicts full-feature on the
    same holdout rows.
  - Runs the algo lens (`_score_form_surface`) on the holdout matches
    via the existing `algo_backtest` walk; aligns its per-row
    predictions to the GBT holdout's single-direction subset.
  - Combines per-lens + full-feature via (a) 50/50 average and
    (b) fitted-weight search on the train fold.
  - Computes Brier, bucket-hit-rate at LOCK/LEAN, per-tour breakdown
    for all four prediction streams.
  - Persists the scorecard JSON.

Anchor alignment: the per-lens and full-feature GBTs predict on the
augmented table (both anchor orientations per match). The algo lens
runs on single-direction rows (anchor = lower MatchStat id). For
Brier-on-the-same-rows comparison, we evaluate everything on the
single-direction subset of the augmented holdout, matched by
`(match_id, anchor_id=min_id)`.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from skimsmarkets.tennis.algo_backtest import run_backtest as run_algo_backtest
from skimsmarkets.tennis.gbt_backfill import (
    PLAYER_PROFILES_PATH,
    RAW_MATCHES_PATH,
)
from skimsmarkets.tennis.gbt_features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)
from skimsmarkets.tennis.gbt_rankings_backfill import RANKINGS_HISTORY_PATH
from skimsmarkets.tennis.gbt_train import MODEL_PATH as FULL_GBT_MODEL_PATH
from skimsmarkets.tennis.gbt_train import TRAIN_CUTOFF
from skimsmarkets.tennis.lens_gbt_form_surface import (
    LENS_FORM_SURFACE_ALL_COLUMNS,
    LensGbtTrainOutput,
    train_lens_gbt,
)

log = logging.getLogger(__name__)

# Mirror `classify.THRESHOLD_LOCK` / `THRESHOLD_LEAN` so bucket-hit-rate
# in this harness maps 1:1 onto production's risk-classifier
# thresholds. Held locally so the harness stays standalone from the
# live config import chain (same pattern as `selection_backtest.py`).
LOCK_THRESHOLD = 0.75
LEAN_THRESHOLD = 0.60

# Where the Phase 0 scorecard lands. Subsequent ablation iterations
# can overwrite this with new variants; the JSON carries the variant
# label inside so diffs are visible.
SCORECARD_PATH = Path("models/tennis_lens_gbt_form_surface.phase0_scorecard.json")


# ---------------------------------------------------------------------------
# Metrics primitives — kept local so this module doesn't depend on
# gbt_train's private helpers.
# ---------------------------------------------------------------------------


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p_clip = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)))


def _bucket_hit_rate(
    y: np.ndarray, p: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Production-faithful bucket-hit-rate.

    Mirrors `selection_backtest._synthetic_lock_label`: a row is a
    bucket hit at threshold T iff `max(p, 1-p) ≥ T` AND the
    prediction was correct (winner-side prediction matched the
    actual winner).

    Returns the count of qualifying rows, the count of correct
    predictions among them, and the hit-rate (precision among
    high-confidence predictions). All three are useful — a model with
    LOTS of high-confidence picks at decent precision is preferable
    to one with very few perfect picks.
    """
    winner_side_p = np.maximum(p, 1.0 - p)
    high_conf_mask = winner_side_p >= threshold
    n_high_conf = int(high_conf_mask.sum())
    if n_high_conf == 0:
        return {
            "threshold": threshold,
            "n_high_confidence": 0,
            "n_correct": 0,
            "hit_rate": float("nan"),
            "frac_of_holdout": 0.0,
        }
    pred_anchor = (p >= 0.5).astype(int)
    correct = (pred_anchor == y).astype(int)
    n_correct = int(correct[high_conf_mask].sum())
    return {
        "threshold": threshold,
        "n_high_confidence": n_high_conf,
        "n_correct": n_correct,
        "hit_rate": n_correct / n_high_conf,
        "frac_of_holdout": n_high_conf / len(y),
    }


def _per_tour_brier(
    y: np.ndarray, p: np.ndarray, tour: np.ndarray
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for t in np.unique(tour):
        mask = tour == t
        if mask.sum() == 0:
            continue
        out[str(t)] = {
            "n": int(mask.sum()),
            "brier": _brier(y[mask], p[mask]),
        }
    return out


# ---------------------------------------------------------------------------
# Fitted-weight search for combining per-lens + full-feature.
# 1-D bracketed minimisation over w ∈ [0, 1] minimising train-fold NLL.
# Same golden-section pattern as `algo_backtest._fit_temperature`.
# ---------------------------------------------------------------------------


def _combined_nll(
    p_lens: np.ndarray, p_full: np.ndarray, y: np.ndarray, w: float
) -> float:
    """Mean NLL of `w * p_lens + (1-w) * p_full` against y."""
    p = w * p_lens + (1.0 - w) * p_full
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_combination_weight(
    p_lens: np.ndarray,
    p_full: np.ndarray,
    y: np.ndarray,
    *,
    tol: float = 1e-3,
) -> float:
    """Golden-section search for the weight w on the per-lens prediction
    minimising NLL of the combination on the train fold. Returns w in
    [0, 1]. w=1 means "use per-lens only"; w=0 means "use full only".

    NLL surface in w is convex on [0,1] when both predictors are
    proper probabilities, so golden-section converges quickly.
    """
    phi = (math.sqrt(5) - 1) / 2
    a, b = 0.0, 1.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _combined_nll(p_lens, p_full, y, c)
    fd = _combined_nll(p_lens, p_full, y, d)
    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _combined_nll(p_lens, p_full, y, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _combined_nll(p_lens, p_full, y, d)
    return (a + b) / 2


# ---------------------------------------------------------------------------
# Full-feature GBT inference — loads `tennis_gbt_spike.cbm` and
# predicts on a holdout DataFrame whose columns already match
# `ALL_FEATURE_COLUMNS`. Re-uses the same Pool construction the
# trainer uses so categoricals are handled identically.
# ---------------------------------------------------------------------------


def _full_gbt_predict(df: pd.DataFrame) -> np.ndarray:
    if not FULL_GBT_MODEL_PATH.exists():
        raise RuntimeError(
            f"{FULL_GBT_MODEL_PATH} not found — re-train tennis_gbt_spike "
            "before running the lens-GBT backtest"
        )
    model = CatBoostClassifier()
    model.load_model(str(FULL_GBT_MODEL_PATH))
    X = df[list(ALL_FEATURE_COLUMNS)].copy()
    for col in NUMERIC_FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURE_COLUMNS:
        X[col] = X[col].astype(str)
    from catboost import Pool
    cat_indices = [list(ALL_FEATURE_COLUMNS).index(c) for c in CATEGORICAL_FEATURE_COLUMNS]
    pool = Pool(data=X, cat_features=cat_indices)
    return model.predict_proba(pool)[:, 1]


# ---------------------------------------------------------------------------
# Phase 0 scorecard.
# ---------------------------------------------------------------------------


@dataclass
class Phase0Scorecard:
    """All numbers needed to evaluate PRIMARY + CO-PRIMARY + SANITY in
    one pass. Persisted to JSON; printable summary returned alongside.
    """

    variant_label: str
    train_cutoff: str
    train_n_augmented: int
    holdout_n_augmented: int
    holdout_n_single_direction: int
    n_paired_with_algo: int
    feature_columns: list[str]
    feature_importance: dict[str, float]

    # Brier on the single-direction holdout — all four streams paired
    # 1:1 with algo lens.
    algo_brier: float
    full_gbt_brier: float
    per_lens_brier: float
    combined_5050_brier: float
    combined_fitted_brier: float
    combined_fitted_weight: float

    algo_log_loss: float
    full_gbt_log_loss: float
    per_lens_log_loss: float

    # Bucket-hit-rate at LOCK + LEAN per stream.
    algo_lock_hit: dict[str, float | int]
    algo_lean_hit: dict[str, float | int]
    full_gbt_lock_hit: dict[str, float | int]
    full_gbt_lean_hit: dict[str, float | int]
    per_lens_lock_hit: dict[str, float | int]
    per_lens_lean_hit: dict[str, float | int]

    # Per-tour Brier — diagnostic, not gating.
    algo_per_tour: dict[str, dict[str, float | int]]
    full_gbt_per_tour: dict[str, dict[str, float | int]]
    per_lens_per_tour: dict[str, dict[str, float | int]]

    # Verdict booleans.
    sanity_passes: bool = field(init=False)
    co_primary_passes_lock: bool = field(init=False)
    co_primary_passes_lean: bool = field(init=False)
    primary_passes_5050: bool = field(init=False)
    primary_passes_fitted: bool = field(init=False)

    def __post_init__(self) -> None:
        self.sanity_passes = (self.algo_brier - self.per_lens_brier) >= 0.002
        algo_lock = self.algo_lock_hit.get("hit_rate", float("nan"))
        algo_lean = self.algo_lean_hit.get("hit_rate", float("nan"))
        pl_lock = self.per_lens_lock_hit.get("hit_rate", float("nan"))
        pl_lean = self.per_lens_lean_hit.get("hit_rate", float("nan"))
        self.co_primary_passes_lock = (
            not math.isnan(algo_lock)
            and not math.isnan(pl_lock)
            and (pl_lock - algo_lock) >= 0.02
        )
        self.co_primary_passes_lean = (
            not math.isnan(algo_lean)
            and not math.isnan(pl_lean)
            and (pl_lean - algo_lean) >= 0.02
        )
        self.primary_passes_5050 = (
            self.full_gbt_brier - self.combined_5050_brier
        ) >= 0.002
        self.primary_passes_fitted = (
            self.full_gbt_brier - self.combined_fitted_brier
        ) >= 0.002

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_label": self.variant_label,
            "train_cutoff": self.train_cutoff,
            "train_n_augmented": self.train_n_augmented,
            "holdout_n_augmented": self.holdout_n_augmented,
            "holdout_n_single_direction": self.holdout_n_single_direction,
            "n_paired_with_algo": self.n_paired_with_algo,
            "feature_columns": self.feature_columns,
            "feature_importance": self.feature_importance,
            "brier": {
                "algo_lens": self.algo_brier,
                "full_gbt": self.full_gbt_brier,
                "per_lens_gbt": self.per_lens_brier,
                "combined_5050": self.combined_5050_brier,
                "combined_fitted": self.combined_fitted_brier,
                "combined_fitted_weight_on_per_lens": self.combined_fitted_weight,
            },
            "log_loss": {
                "algo_lens": self.algo_log_loss,
                "full_gbt": self.full_gbt_log_loss,
                "per_lens_gbt": self.per_lens_log_loss,
            },
            "bucket_hit_rate": {
                "algo_lens": {"lock": self.algo_lock_hit, "lean": self.algo_lean_hit},
                "full_gbt": {"lock": self.full_gbt_lock_hit, "lean": self.full_gbt_lean_hit},
                "per_lens_gbt": {"lock": self.per_lens_lock_hit, "lean": self.per_lens_lean_hit},
            },
            "per_tour": {
                "algo_lens": self.algo_per_tour,
                "full_gbt": self.full_gbt_per_tour,
                "per_lens_gbt": self.per_lens_per_tour,
            },
            "verdict": {
                "primary_threshold_brier_lift": 0.002,
                "co_primary_threshold_hit_rate_lift": 0.02,
                "sanity_threshold_brier_lift_vs_algo": 0.002,
                "sanity_passes": self.sanity_passes,
                "co_primary_passes_lock": self.co_primary_passes_lock,
                "co_primary_passes_lean": self.co_primary_passes_lean,
                "primary_passes_5050": self.primary_passes_5050,
                "primary_passes_fitted": self.primary_passes_fitted,
            },
        }


@dataclass
class _AblationCache:
    """Caches the per-variant-invariant slow pieces — algo lens
    predictions and full-feature GBT predictions — across ablation
    runs. The PER-LENS GBT changes per variant; everything else
    doesn't, so caching shaves ~2.5 of the ~3 min per variant.
    """

    algo_holdout_rows: list[tuple[Any, ...]]
    full_train_p: np.ndarray
    full_holdout_p: np.ndarray
    train_df: pd.DataFrame
    holdout_df: pd.DataFrame
    matches_df: pd.DataFrame
    profiles_df: pd.DataFrame
    rankings_df: pd.DataFrame | None


def build_ablation_cache() -> _AblationCache:
    """One-shot precompute: load parquets, build training table (via
    a no-op feature subset), run algo backtest, predict full-feature
    GBT on both folds. Pass to `run_phase0_scorecard(cache=…)` to
    amortise across ablation variants.
    """
    log.info("ablation cache: loading parquets…")
    matches_df = pd.read_parquet(RAW_MATCHES_PATH)
    profiles_df = (
        pd.read_parquet(PLAYER_PROFILES_PATH) if PLAYER_PROFILES_PATH.exists()
        else pd.DataFrame()
    )
    rankings_df = (
        pd.read_parquet(RANKINGS_HISTORY_PATH) if RANKINGS_HISTORY_PATH.exists()
        else None
    )

    # Build the training table by piggybacking on a single per-lens
    # train (full feature set) — this gives us train_df + holdout_df
    # AND the full_train_p / full_holdout_p in one pass without
    # duplicating the build_training_table walk.
    log.info("ablation cache: building training table (one walk)…")
    bootstrap = train_lens_gbt(
        matches_df, profiles_df, rankings_df=rankings_df,
        # Use the full feature columns so the model itself is
        # discardable — we only want the train/holdout split DataFrames.
        feature_columns=ALL_FEATURE_COLUMNS,
    )

    log.info("ablation cache: full-feature GBT predict on holdout…")
    full_holdout_p = _full_gbt_predict(bootstrap.holdout_df)
    log.info("ablation cache: full-feature GBT predict on train…")
    full_train_p = _full_gbt_predict(bootstrap.train_df)

    log.info("ablation cache: running algo lens backtest…")
    algo_result = run_algo_backtest()
    algo_holdout_rows = algo_result.holdout_predictions  # type: ignore[attr-defined]

    return _AblationCache(
        algo_holdout_rows=algo_holdout_rows,
        full_train_p=full_train_p,
        full_holdout_p=full_holdout_p,
        train_df=bootstrap.train_df,
        holdout_df=bootstrap.holdout_df,
        matches_df=matches_df,
        profiles_df=profiles_df,
        rankings_df=rankings_df,
    )


def run_phase0_scorecard(
    *,
    variant_label: str = "v1_first_cut",
    feature_columns: tuple[str, ...] = LENS_FORM_SURFACE_ALL_COLUMNS,
    persist: bool = True,
    cache: _AblationCache | None = None,
) -> Phase0Scorecard:
    """End-to-end Phase 0 scorecard.

    Trains the per-lens GBT, runs the algo lens via the existing
    backtest harness, loads the full-feature GBT, aligns predictions
    on the single-direction holdout, computes all three tiers of the
    metric and writes the JSON scorecard.

    `feature_columns` lets ablation runs pass alternative subsets while
    keeping everything else identical. `cache` short-circuits the slow
    pieces (algo backtest, full-feature GBT inference, training-table
    walk) — pass one from `build_ablation_cache()` to iterate cheaply.
    """
    log.info("phase 0 scorecard: variant=%s, features=%d",
             variant_label, len(feature_columns))

    if cache is None:
        log.info("loading parquets…")
        matches_df = pd.read_parquet(RAW_MATCHES_PATH)
        profiles_df = (
            pd.read_parquet(PLAYER_PROFILES_PATH) if PLAYER_PROFILES_PATH.exists()
            else pd.DataFrame()
        )
        rankings_df = (
            pd.read_parquet(RANKINGS_HISTORY_PATH) if RANKINGS_HISTORY_PATH.exists()
            else None
        )
    else:
        matches_df = cache.matches_df
        profiles_df = cache.profiles_df
        rankings_df = cache.rankings_df
    log.info("matches=%d, profiles=%d, rankings=%s",
             len(matches_df), len(profiles_df),
             "absent" if rankings_df is None else len(rankings_df))

    log.info("training per-lens GBT…")
    lens_out: LensGbtTrainOutput = train_lens_gbt(
        matches_df, profiles_df,
        rankings_df=rankings_df,
        feature_columns=feature_columns,
    )

    # 2. Full-feature GBT predictions on the SAME holdout rows. Use
    # cache when available — the predictions come from the same
    # underlying parquet and same train_cutoff so they're directly
    # reusable across ablation variants.
    if cache is not None:
        log.info("using cached full-feature GBT predictions")
        full_holdout_p = cache.full_holdout_p
        full_train_p = cache.full_train_p
        # Sanity: cache predictions must align row-for-row with the
        # current lens_out tables (same training-table walk → identical
        # row ordering by construction, but verify shape).
        assert len(full_holdout_p) == len(lens_out.holdout_df), (
            f"cache mismatch: full_holdout_p={len(full_holdout_p)} "
            f"vs lens holdout={len(lens_out.holdout_df)}"
        )
        assert len(full_train_p) == len(lens_out.train_df), (
            f"cache mismatch: full_train_p={len(full_train_p)} "
            f"vs lens train={len(lens_out.train_df)}"
        )
    else:
        log.info("predicting full-feature GBT on holdout…")
        full_holdout_p = _full_gbt_predict(lens_out.holdout_df)
        log.info("predicting full-feature GBT on train (for weight fit)…")
        full_train_p = _full_gbt_predict(lens_out.train_df)

    # 3. Algo lens predictions — also cache-able.
    if cache is not None:
        log.info("using cached algo lens predictions")
        algo_holdout_rows = cache.algo_holdout_rows
    else:
        log.info("running algo lens backtest…")
        algo_result = run_algo_backtest()
        algo_holdout_rows = algo_result.holdout_predictions  # type: ignore[attr-defined]

    # 4. Align: algo lens emits one row per HOLDOUT MATCH (anchor =
    # lower MatchStat id). The augmented GBT holdout has 2 rows per
    # match — keep the (anchor_id < opponent_id) direction so the row
    # set matches the algo's single-direction shape. Join by
    # (match_id, anchor_id).
    holdout = lens_out.holdout_df.copy()
    holdout["per_lens_p"] = lens_out.holdout_p
    holdout["full_p"] = full_holdout_p
    single_dir_mask = holdout["anchor_id"] < holdout["opponent_id"]
    single_dir = holdout.loc[single_dir_mask].reset_index(drop=True)

    algo_df = pd.DataFrame(
        algo_holdout_rows,
        columns=["algo_p", "anchor_won", "surface", "tour", "match_id", "anchor_id"],
    )

    # Inner join on (match_id, anchor_id). Both sides anchor on the
    # lower-id player, so the join is 1:1 by construction.
    merged = single_dir.merge(
        algo_df[["match_id", "anchor_id", "algo_p"]],
        on=["match_id", "anchor_id"],
        how="inner",
    )
    log.info(
        "alignment: single_dir holdout=%d, algo holdout=%d, merged=%d",
        len(single_dir), len(algo_df), len(merged),
    )
    if merged.empty:
        raise RuntimeError("alignment produced 0 rows — anchor convention mismatch?")

    y = merged["target"].to_numpy().astype(int)
    p_lens = merged["per_lens_p"].to_numpy()
    p_full = merged["full_p"].to_numpy()
    p_algo = merged["algo_p"].to_numpy()
    tour = merged["tour"].to_numpy()

    # 5. Combinations.
    p_5050 = 0.5 * p_lens + 0.5 * p_full
    # Fit the combination weight on the TRAIN fold to keep it honest —
    # the holdout fold sees only the fitted weight, not the search.
    y_train = lens_out.train_df["target"].to_numpy().astype(int)
    fitted_w = _fit_combination_weight(
        lens_out.train_p, full_train_p, y_train,
    )
    p_fitted = fitted_w * p_lens + (1.0 - fitted_w) * p_full
    log.info("fitted combination weight w_per_lens=%.4f", fitted_w)

    # 6. Build the scorecard.
    scorecard = Phase0Scorecard(
        variant_label=variant_label,
        train_cutoff=TRAIN_CUTOFF.isoformat(),
        train_n_augmented=len(lens_out.train_df),
        holdout_n_augmented=len(lens_out.holdout_df),
        holdout_n_single_direction=len(single_dir),
        n_paired_with_algo=len(merged),
        feature_columns=list(feature_columns),
        feature_importance=lens_out.feature_importance,
        algo_brier=_brier(y, p_algo),
        full_gbt_brier=_brier(y, p_full),
        per_lens_brier=_brier(y, p_lens),
        combined_5050_brier=_brier(y, p_5050),
        combined_fitted_brier=_brier(y, p_fitted),
        combined_fitted_weight=fitted_w,
        algo_log_loss=_log_loss(y, p_algo),
        full_gbt_log_loss=_log_loss(y, p_full),
        per_lens_log_loss=_log_loss(y, p_lens),
        algo_lock_hit=_bucket_hit_rate(y, p_algo, LOCK_THRESHOLD),
        algo_lean_hit=_bucket_hit_rate(y, p_algo, LEAN_THRESHOLD),
        full_gbt_lock_hit=_bucket_hit_rate(y, p_full, LOCK_THRESHOLD),
        full_gbt_lean_hit=_bucket_hit_rate(y, p_full, LEAN_THRESHOLD),
        per_lens_lock_hit=_bucket_hit_rate(y, p_lens, LOCK_THRESHOLD),
        per_lens_lean_hit=_bucket_hit_rate(y, p_lens, LEAN_THRESHOLD),
        algo_per_tour=_per_tour_brier(y, p_algo, tour),
        full_gbt_per_tour=_per_tour_brier(y, p_full, tour),
        per_lens_per_tour=_per_tour_brier(y, p_lens, tour),
    )

    if persist:
        SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCORECARD_PATH.write_text(json.dumps(scorecard.to_dict(), indent=2, default=str))
        log.info("scorecard → %s", SCORECARD_PATH)

    return scorecard


def format_summary(sc: Phase0Scorecard) -> str:
    """One-screen scorecard summary for the CLI output."""
    lines: list[str] = []
    lines.append(f"=== Phase 0 lens-GBT scorecard — variant: {sc.variant_label} ===")
    lines.append(
        f"holdout_n: {sc.n_paired_with_algo} (paired w/ algo on single-direction subset)"
    )
    lines.append(f"feature_columns ({len(sc.feature_columns)}): "
                 f"{', '.join(sc.feature_columns)}")
    lines.append("")
    lines.append("BRIER (lower is better):")
    lines.append(f"  algo_lens                   : {sc.algo_brier:.5f}")
    lines.append(f"  full_gbt (tennis_gbt_spike) : {sc.full_gbt_brier:.5f}")
    lines.append(f"  per_lens_gbt                : {sc.per_lens_brier:.5f}")
    lines.append(f"  combined 50/50              : {sc.combined_5050_brier:.5f}")
    lines.append(
        f"  combined fitted (w={sc.combined_fitted_weight:.3f}) : "
        f"{sc.combined_fitted_brier:.5f}"
    )
    lines.append("")
    lines.append("BUCKET-HIT-RATE @ LOCK (≥0.75 winner-side prob):")
    for name, b in [
        ("algo_lens", sc.algo_lock_hit),
        ("full_gbt", sc.full_gbt_lock_hit),
        ("per_lens_gbt", sc.per_lens_lock_hit),
    ]:
        hr = b.get("hit_rate", float("nan"))
        n = b.get("n_high_confidence", 0)
        c = b.get("n_correct", 0)
        f = b.get("frac_of_holdout", 0.0)
        lines.append(
            f"  {name:25s}: {hr:.4f}  (n_high={n}, correct={c}, frac={f:.3f})"
        )
    lines.append("")
    lines.append("BUCKET-HIT-RATE @ LEAN (≥0.60 winner-side prob):")
    for name, b in [
        ("algo_lens", sc.algo_lean_hit),
        ("full_gbt", sc.full_gbt_lean_hit),
        ("per_lens_gbt", sc.per_lens_lean_hit),
    ]:
        hr = b.get("hit_rate", float("nan"))
        n = b.get("n_high_confidence", 0)
        c = b.get("n_correct", 0)
        f = b.get("frac_of_holdout", 0.0)
        lines.append(
            f"  {name:25s}: {hr:.4f}  (n_high={n}, correct={c}, frac={f:.3f})"
        )
    lines.append("")
    lines.append("VERDICT:")
    sanity = "PASS" if sc.sanity_passes else "FAIL"
    lines.append(
        f"  SANITY (Brier per-lens ≥0.002 better than algo): {sanity}  "
        f"(Δ = {sc.algo_brier - sc.per_lens_brier:+.5f})"
    )
    p5050 = "PASS" if sc.primary_passes_5050 else "FAIL"
    pfit = "PASS" if sc.primary_passes_fitted else "FAIL"
    lines.append(
        f"  PRIMARY 50/50 (Brier combined ≥0.002 better than full): {p5050}  "
        f"(Δ = {sc.full_gbt_brier - sc.combined_5050_brier:+.5f})"
    )
    lines.append(
        f"  PRIMARY fitted (Brier combined ≥0.002 better than full): {pfit}  "
        f"(Δ = {sc.full_gbt_brier - sc.combined_fitted_brier:+.5f})"
    )
    co_lock = "PASS" if sc.co_primary_passes_lock else "FAIL"
    co_lean = "PASS" if sc.co_primary_passes_lean else "FAIL"
    lines.append(
        f"  CO-PRIMARY LOCK (hit-rate per-lens ≥0.02 better than algo): {co_lock}  "
        f"(Δ = {sc.per_lens_lock_hit.get('hit_rate', 0.0) - sc.algo_lock_hit.get('hit_rate', 0.0):+.4f})"
    )
    lines.append(
        f"  CO-PRIMARY LEAN (hit-rate per-lens ≥0.02 better than algo): {co_lean}  "
        f"(Δ = {sc.per_lens_lean_hit.get('hit_rate', 0.0) - sc.algo_lean_hit.get('hit_rate', 0.0):+.4f})"
    )
    return "\n".join(lines)


__all__ = [
    "LEAN_THRESHOLD",
    "LOCK_THRESHOLD",
    "Phase0Scorecard",
    "build_ablation_cache",
    "format_summary",
    "run_phase0_scorecard",
]
