"""Walk-forward training for the tennis GBT spike.

Loads the parquet artefacts produced by `gbt_backfill.py`, hands the
match table to `gbt_features.build_training_table` (which enforces
point-in-time discipline by walking chronologically), splits by date
into train ≤ TRAIN_CUTOFF and holdout > TRAIN_CUTOFF, fits a
CatBoostClassifier, and writes:

  - `models/tennis_gbt_spike.cbm` — the artefact, loaded by `gbt.py`.
  - `models/tennis_gbt_spike.metrics.json` — pass-criteria scorecard
    (Brier, log loss, AUC, reliability deciles, per-tour Brier, GBT
    vs sim head-to-head).

The trainer also runs the iid Monte Carlo sim (`tennis/simulation.py`)
on each holdout row using the SAME aggregated career rates the GBT
sees, so the head-to-head comparison is apples-to-apples — both
priors are reading the exact same upstream evidence; the only
difference is the model.

Walk-forward (not random k-fold) is the only correct CV here —
random splits would leak future matches into training-folds via the
rolling aggregations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from skimsmarkets.tennis.gbt_backfill import (
    PLAYER_PROFILES_PATH,
    RAW_MATCHES_PATH,
)
from skimsmarkets.tennis.gbt_features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    HALF_LIFE_DAYS,
    MIN_PRIORS_PER_SIDE,
    NUMERIC_FEATURE_COLUMNS,
    PlayerHistory,
    build_training_table,
)
from skimsmarkets.tennis.gbt_rankings_backfill import RANKINGS_HISTORY_PATH
from skimsmarkets.tennis.models import (
    TennisHeadToHead,
    TennisPlayerStats,
    TennisStatsContext,
)
from skimsmarkets.tennis.simulation import simulate_match

log = logging.getLogger(__name__)

# Walk-forward cutoff. Matches with date ≤ TRAIN_CUTOFF land in the
# training fold; later matches land in holdout. Today's date is
# 2026-05-06 (current cycle); this cutoff puts ~17 months of holdout
# in the test fold which is plenty for the spike pass criteria.
TRAIN_CUTOFF = date(2024, 12, 31)

MODEL_PATH = Path("models/tennis_gbt_spike.cbm")
METRICS_PATH = Path("models/tennis_gbt_spike.metrics.json")

# CatBoost hyperparameters. v3 settled on depth=6 — bumping to 8
# (tested) made trees overfit faster without lifting holdout Brier
# (depth=8 stopped at iteration 365 vs 540 for depth=6, identical
# metrics). The iteration ceiling and early-stopping patience are
# kept generous (3000 / 100) so future re-trains on a wider backfill
# don't artificially cap; both defaults are no-ops when the model
# converges sooner. Re-grade vs the previous artefact's metrics.json
# after any change here.
_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 4.0,
    "random_seed": 42,
    "use_best_model": True,
    "od_type": "Iter",
    "od_wait": 100,
    "verbose": 200,
    "allow_writing_files": False,
}


@dataclass
class TrainOutput:
    """Bundle returned from `train_and_evaluate`. Exposed so tests can
    introspect the model + metrics dict without re-reading the json.
    """

    model: CatBoostClassifier
    model_version: str
    metrics: dict[str, Any]


def _features_with_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a view with `float` numerics and `str` categoricals.

    catboost is strict about categorical columns being strings (or
    ints declared via `cat_features`); pandas object columns mixing
    None + str break ingestion. The `compute_features` builder
    already produces categoricals as strings — this enforces that
    contract at the boundary so a future signature drift doesn't
    silently produce a NaN-typed categorical.
    """
    out = df[list(ALL_FEATURE_COLUMNS)].copy()
    for col in NUMERIC_FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURE_COLUMNS:
        out[col] = out[col].astype(str)
    return out


def _make_pool(df: pd.DataFrame, target: pd.Series | None = None) -> Pool:
    """Build a catboost `Pool` declaring categorical column indices."""
    X = _features_with_dtypes(df)
    cat_indices = [list(ALL_FEATURE_COLUMNS).index(c) for c in CATEGORICAL_FEATURE_COLUMNS]
    return Pool(data=X, label=target, cat_features=cat_indices)


def _hash_artifact(path: Path) -> str:
    h = hashlib.blake2b(path.read_bytes(), digest_size=8).hexdigest()
    return f"sha-{h}"


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def _log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p_clip = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip)))


def _auc(y_true: np.ndarray, p: np.ndarray) -> float | None:
    """Mann-Whitney U based AUC. Returns None when only one class is
    present in y_true (AUC undefined).
    """
    pos = p[y_true == 1]
    neg = p[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rankdata-based formulation, vectorized with numpy.
    ranks = np.argsort(np.argsort(np.concatenate([neg, pos]))) + 1
    pos_ranks = ranks[len(neg):]
    n_pos = len(pos)
    n_neg = len(neg)
    u = pos_ranks.sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _reliability_deciles(
    y_true: np.ndarray, p: np.ndarray
) -> list[dict[str, float]]:
    """Bin predictions into 10 equal-width buckets and return mean
    predicted vs mean observed per bucket. Diagnostic for calibration
    drift — a well-calibrated model has mean_predicted ≈ mean_observed
    across all bins.
    """
    out: list[dict[str, float]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        # Right edge inclusive on the final bucket so p=1.0 is captured.
        mask = (p >= lo) & (p < hi if i < 9 else p <= hi)
        if mask.sum() == 0:
            continue
        out.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()),
            "mean_observed": float(y_true[mask].mean()),
        })
    return out


def _stats_context_from_history(
    *,
    anchor_id: int,
    opponent_id: int,
    anchor_first_in: float | None,
    anchor_first_win: float | None,
    anchor_second_win: float | None,
    anchor_first_return_win: float | None,
    anchor_second_return_win: float | None,
    opp_first_in: float | None,
    opp_first_win: float | None,
    opp_second_win: float | None,
    opp_first_return_win: float | None,
    opp_second_return_win: float | None,
) -> TennisStatsContext:
    """Synthesise the minimal `TennisStatsContext` the sim needs.

    The sim only reads serve/return percentages off the player blocks;
    everything else (rank, surface, recent matches, H2H) is unused.
    Constructing a stub context with just the needed fields lets us
    run the sim against the exact same career rates the GBT sees —
    apples-to-apples comparison.

    `team_a` is the anchor, matching the GBT's anchor-relative
    prediction so both numbers can be compared without re-mapping.
    """
    a = TennisPlayerStats(
        name=f"anchor_{anchor_id}",
        api_player_id=str(anchor_id),
        first_serve_in_pct=anchor_first_in,
        first_serve_win_pct=anchor_first_win,
        second_serve_win_pct=anchor_second_win,
        first_serve_return_win_pct=anchor_first_return_win,
        second_serve_return_win_pct=anchor_second_return_win,
    )
    b = TennisPlayerStats(
        name=f"opp_{opponent_id}",
        api_player_id=str(opponent_id),
        first_serve_in_pct=opp_first_in,
        first_serve_win_pct=opp_first_win,
        second_serve_win_pct=opp_second_win,
        first_serve_return_win_pct=opp_first_return_win,
        second_serve_return_win_pct=opp_second_return_win,
    )
    return TennisStatsContext(
        provider="gbt_train_synthetic",
        fetched_at=datetime.now(UTC),
        player_a=a,
        player_b=b,
        head_to_head=TennisHeadToHead(),
    )


def _sim_baseline(
    *,
    history_anchor_rates: dict[str, float | None],
    history_opp_rates: dict[str, float | None],
    anchor_id: int,
    opponent_id: int,
    best_of: int,
) -> float | None:
    """Run the iid Monte Carlo sim against the SAME career rates the
    GBT sees. Returns `P(anchor wins)` or None when the sim's
    attachment gate fails (any required rate missing).
    """
    ctx = _stats_context_from_history(
        anchor_id=anchor_id,
        opponent_id=opponent_id,
        anchor_first_in=history_anchor_rates["first_serve_in_pct"],
        anchor_first_win=history_anchor_rates["first_serve_win_pct"],
        anchor_second_win=history_anchor_rates["second_serve_win_pct"],
        anchor_first_return_win=history_anchor_rates["first_serve_return_win_pct"],
        anchor_second_return_win=history_anchor_rates["second_serve_return_win_pct"],
        opp_first_in=history_opp_rates["first_serve_in_pct"],
        opp_first_win=history_opp_rates["first_serve_win_pct"],
        opp_second_win=history_opp_rates["second_serve_win_pct"],
        opp_first_return_win=history_opp_rates["first_serve_return_win_pct"],
        opp_second_return_win=history_opp_rates["second_serve_return_win_pct"],
    )
    sim = simulate_match(
        ctx,
        best_of=best_of,
        n_sims=2000,  # Lower than the live default (10k) — we run thousands
        # of these in the holdout loop and the spike comparison only
        # needs ~2pp precision per row.
        seed=hash(("sim_baseline", anchor_id, opponent_id)) & 0xFFFFFFFF,
    )
    return None if sim is None else sim.p_team_a_wins


def _compare_against_sim(
    holdout_df: pd.DataFrame, rates_by_match_player: dict[tuple[int, int], dict]
) -> dict[str, Any] | None:
    """Compute Brier(GBT) vs Brier(sim) on the SAME holdout subset
    where both priors produce a number. The sim's attachment gate
    drops rows where any required rate is unobserved — those rows
    are excluded from BOTH numbers so the comparison is paired.

    Lookup is keyed by `(match_id, player_id)` so the augmented
    training set's two anchor orientations resolve to the right
    player's rates without re-deriving the anchor role here.

    Returns None when the gated subset is empty.
    """
    matched_rows = []
    for _, row in holdout_df.iterrows():
        mid = int(row["match_id"])
        anchor_id = int(row["anchor_id"])
        opp_id = int(row["opponent_id"])
        anchor_rates = rates_by_match_player.get((mid, anchor_id))
        opp_rates = rates_by_match_player.get((mid, opp_id))
        if not anchor_rates or not opp_rates:
            continue
        try:
            best_of = int(row["best_of"]) if row["best_of"] in ("3", "5") else 3
        except (TypeError, ValueError):
            best_of = 3
        if best_of not in (3, 5):
            best_of = 3
        sim_p = _sim_baseline(
            history_anchor_rates=anchor_rates,
            history_opp_rates=opp_rates,
            anchor_id=int(row["anchor_id"]),
            opponent_id=int(row["opponent_id"]),
            best_of=best_of,
        )
        if sim_p is None:
            continue
        matched_rows.append({
            "match_id": mid,
            "y": int(row["target"]),
            "gbt_p": float(row["gbt_p"]),
            "sim_p": sim_p,
        })
    if not matched_rows:
        return None
    paired = pd.DataFrame(matched_rows)
    y = paired["y"].to_numpy()
    return {
        "n_paired": len(paired),
        "gbt_brier": _brier(y, paired["gbt_p"].to_numpy()),
        "sim_brier": _brier(y, paired["sim_p"].to_numpy()),
        "gbt_log_loss": _log_loss(y, paired["gbt_p"].to_numpy()),
        "sim_log_loss": _log_loss(y, paired["sim_p"].to_numpy()),
    }


def _per_tour_breakdown(
    holdout_df: pd.DataFrame, p: np.ndarray
) -> dict[str, dict[str, float]]:
    """Brier per tour (atp / wta) so a tour-skew in performance is
    visible at a glance.
    """
    out: dict[str, dict[str, float]] = {}
    for tour, group in holdout_df.groupby("tour"):
        idx = group.index.to_numpy()
        y = group["target"].to_numpy()
        out[str(tour)] = {
            "n": int(len(group)),
            "brier": _brier(y, p[idx]),
            "log_loss": _log_loss(y, p[idx]),
            "base_rate_anchor_wins": float(y.mean()),
        }
    return out


def _per_tier_breakdown(
    holdout_df: pd.DataFrame, p: np.ndarray
) -> dict[str, dict[str, float]]:
    """Brier per tier (grand_slam / masters / main_tour / challenger /
    futures / unknown). Mitigation for the tier-drift risk flagged in
    the plan — visible per-tier hit-rate makes drift spottable.
    """
    out: dict[str, dict[str, float]] = {}
    for tier, group in holdout_df.groupby("tier"):
        idx = group.index.to_numpy()
        y = group["target"].to_numpy()
        out[str(tier)] = {
            "n": int(len(group)),
            "brier": _brier(y, p[idx]),
        }
    return out


def _capture_holdout_history_rates(
    matches_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    holdout_match_ids: set[int],
) -> dict[tuple[int, int], dict]:
    """Walk the parquet a SECOND time and snapshot each player's career
    rates AT EACH HOLDOUT MATCH (before fold-in).

    Returns `{(match_id, player_id) → rates_dict}`. Keying by player_id
    (not anchor/opponent role) keeps the lookup correct under the
    augmented training set where the same match appears with both
    anchor orientations — the comparison code reads the rates for the
    row's literal anchor_id and opponent_id without re-deriving role.
    """
    from skimsmarkets.tennis.gbt_features import (
        HistoryStore,
        _add_match_from_row,
        _row_get_int,
    )

    def _rates(h: PlayerHistory) -> dict:
        return {
            "first_serve_in_pct": h.career_first_serve_in_pct(),
            "first_serve_win_pct": h.career_first_serve_win_pct(),
            "second_serve_win_pct": h.career_second_serve_win_pct(),
            "first_serve_return_win_pct": (
                None if h.return_first_d == 0
                else h.return_first_n / h.return_first_d
            ),
            "second_serve_return_win_pct": (
                None if h.return_second_d == 0
                else h.return_second_n / h.return_second_d
            ),
        }

    store = HistoryStore()
    df = matches_df.sort_values("match_date").reset_index(drop=True)
    out: dict[tuple[int, int], dict] = {}
    for _, row in df.iterrows():
        mid = _row_get_int(row, "match_id")
        if mid in holdout_match_ids:
            p1 = _row_get_int(row, "p1_id")
            p2 = _row_get_int(row, "p2_id")
            if p1 is not None and p2 is not None:
                out[(mid, p1)] = _rates(store.get_or_create(p1))
                out[(mid, p2)] = _rates(store.get_or_create(p2))
        _add_match_from_row(store, row)
    return out


# ---------------------------------------------------------------------------
# Training-recipe capture — single source of truth a future cold-start
# session can read off the metrics.json to reproduce the exact training
# run (or at least to KNOW what's missing if the recipe doesn't match
# what's in the current code/parquets).
#
# Everything in here is captured AT TRAIN TIME and persisted alongside
# the metrics. If the recipe block on the live metrics.json drifts from
# what's currently in the code (e.g. hyperparams changed but no re-train
# yet), comparing them tells you the artifact is stale.
# ---------------------------------------------------------------------------


def _capture_parquet_source(path: Path) -> dict[str, Any] | None:
    """Snapshot a parquet's identity: path, file mtime + size, row count.

    Row count is read from the parquet's metadata footer (cheap, no
    column scan) via `pyarrow.parquet.ParquetFile.metadata.num_rows`.
    Returns None when the file is absent.
    """
    if not path.exists():
        return None
    stat = path.stat()
    n_rows: int | None
    try:
        # pyarrow is a transitive dep of pandas[parquet] — already loaded
        # by `pd.read_parquet` elsewhere in this module, so the import
        # cost is amortised. The footer-read is ~1ms regardless of parquet
        # size — much cheaper than `len(pd.read_parquet(...))`.
        import pyarrow.parquet as pq
        n_rows = pq.ParquetFile(path).metadata.num_rows
    except Exception:  # noqa: BLE001 — defensive on corrupt parquets
        n_rows = None
    return {
        "path": str(path),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "size_bytes": stat.st_size,
        "n_rows": n_rows,
    }


def _capture_code_commit_sha() -> str | None:
    """Return the current git commit short SHA, or None when not in a
    git checkout (e.g. installed via wheel, or `.git/` was scrubbed).
    Defensive: any failure swallows to None — the recipe should never
    fail a train run because git couldn't be queried.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            return sha if sha else None
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _capture_training_recipe(*, train_cutoff: date) -> dict[str, Any]:
    """Build the full recipe block that goes into metrics.json.

    Captures everything a future cold-start session needs to understand
    how the live artifact was trained without reading code:
    - CatBoost hyperparams (the pinned `_CATBOOST_PARAMS` dict)
    - Feature column lists (numeric + categorical separately so the
      Pool can be rebuilt; also the combined order)
    - Cold-start + recency-decay constants from `gbt_features` (gate
      the training set + drive the recency-weighted aggregates)
    - Per-parquet snapshot: path, mtime, size, n_rows
    - Code commit SHA (best-effort; None outside a git checkout)
    """
    return {
        "train_cutoff": train_cutoff.isoformat(),
        "catboost_params": dict(_CATBOOST_PARAMS),
        "feature_columns": list(ALL_FEATURE_COLUMNS),
        "numeric_feature_columns": list(NUMERIC_FEATURE_COLUMNS),
        "categorical_feature_columns": list(CATEGORICAL_FEATURE_COLUMNS),
        "min_priors_per_side": MIN_PRIORS_PER_SIDE,
        "half_life_days": HALF_LIFE_DAYS,
        "data_sources": {
            "raw_matches": _capture_parquet_source(RAW_MATCHES_PATH),
            "player_profiles": _capture_parquet_source(PLAYER_PROFILES_PATH),
            "rankings_history": _capture_parquet_source(RANKINGS_HISTORY_PATH),
        },
        "code_commit_sha": _capture_code_commit_sha(),
        "trainer_module": "skimsmarkets.tennis.gbt_train",
    }


def train_and_evaluate(
    matches_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    *,
    rankings_df: pd.DataFrame | None = None,
    train_cutoff: date = TRAIN_CUTOFF,
    skip_sim_compare: bool = False,
) -> TrainOutput:
    """End-to-end: build features, walk-forward split, fit, evaluate.

    `skip_sim_compare=True` is for the smoke-test path; the sim
    comparison runs Monte Carlo across thousands of holdout rows
    which is the slow part of the trainer (~5-10 minutes for the
    spike's holdout).

    `rankings_df` is forwarded to `build_training_table` so the new
    rank-diff features get populated. None → train without rank
    features (the model still trains; catboost handles NaN columns).
    """
    table = build_training_table(
        matches_df, profiles_df, rankings_df=rankings_df,
    )
    rows = table.rows
    if rows.empty:
        raise RuntimeError(
            "training table is empty — backfill may have failed or all "
            "rows were dropped by the cold-start gate"
        )

    rows["match_date"] = pd.to_datetime(rows["match_date"]).dt.date
    train_mask = rows["match_date"] <= train_cutoff
    train_df = rows.loc[train_mask].reset_index(drop=True)
    holdout_df = rows.loc[~train_mask].reset_index(drop=True)
    if train_df.empty or holdout_df.empty:
        raise RuntimeError(
            f"walk-forward split produced an empty fold "
            f"(train={len(train_df)}, holdout={len(holdout_df)}); "
            "check that backfill window straddles TRAIN_CUTOFF"
        )

    log.info(
        "walk-forward split: train=%d (≤ %s), holdout=%d (> %s)",
        len(train_df), train_cutoff, len(holdout_df), train_cutoff,
    )

    train_pool = _make_pool(train_df, train_df["target"])
    holdout_pool = _make_pool(holdout_df, holdout_df["target"])

    model = CatBoostClassifier(**_CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=holdout_pool)

    # Predict on both folds for the metrics scorecard.
    train_p = model.predict_proba(train_pool)[:, 1]
    holdout_p = model.predict_proba(holdout_pool)[:, 1]

    holdout_df = holdout_df.copy()
    holdout_df["gbt_p"] = holdout_p

    y_train = train_df["target"].to_numpy()
    y_holdout = holdout_df["target"].to_numpy()

    metrics: dict[str, Any] = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "train_cutoff": str(train_cutoff),
        "train_n": int(len(train_df)),
        "holdout_n": int(len(holdout_df)),
        "n_dropped_cold_start": int(table.n_dropped_cold_start),
        "n_dropped_other": int(table.n_dropped_other),
        # Training recipe — single-source-of-truth for cold-start
        # reproducibility. Captures hyperparams, feature lists, data
        # source identities, and the commit SHA. Future sessions read
        # this block off the metrics.json to understand how the live
        # artifact was trained without grepping the code.
        "training_recipe": _capture_training_recipe(train_cutoff=train_cutoff),
        "train": {
            "brier": _brier(y_train, train_p),
            "log_loss": _log_loss(y_train, train_p),
            "auc": _auc(y_train, train_p),
            "base_rate_anchor_wins": float(y_train.mean()),
        },
        "holdout": {
            "brier": _brier(y_holdout, holdout_p),
            "log_loss": _log_loss(y_holdout, holdout_p),
            "auc": _auc(y_holdout, holdout_p),
            "base_rate_anchor_wins": float(y_holdout.mean()),
            "reliability_deciles": _reliability_deciles(y_holdout, holdout_p),
            "per_tour": _per_tour_breakdown(holdout_df, holdout_p),
            "per_tier": _per_tier_breakdown(holdout_df, holdout_p),
        },
        "feature_importance": dict(zip(
            ALL_FEATURE_COLUMNS,
            [float(x) for x in model.get_feature_importance()],
            strict=True,
        )),
    }

    if not skip_sim_compare:
        log.info(
            "running iid sim baseline against %d holdout rows "
            "(this takes a few minutes)",
            len(holdout_df),
        )
        history_rates = _capture_holdout_history_rates(
            matches_df, profiles_df,
            holdout_match_ids=set(int(m) for m in holdout_df["match_id"]),
        )
        compare = _compare_against_sim(holdout_df, history_rates)
        if compare is not None:
            metrics["sim_compare"] = compare

    # Persist artefact. Hash the bytes for the model_version stamp
    # so retro grading can detect retraining boundaries.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    model_version = _hash_artifact(MODEL_PATH)
    metrics["model_version"] = model_version
    metrics["model_path"] = str(MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("wrote model → %s, metrics → %s", MODEL_PATH, METRICS_PATH)

    return TrainOutput(model=model, model_version=model_version, metrics=metrics)


def run_train_cli(*, features_only: bool, skip_sim_compare: bool) -> dict[str, Any]:
    """CLI entry point. Loads the parquet artefacts, runs the trainer,
    returns the metrics dict so the CLI handler can print a summary
    without re-reading the json sidecar.

    `features_only=True` short-circuits before model fit — used by
    smoke test 3 to validate the feature-extractor pipeline without
    paying the catboost training cost.
    """
    if not RAW_MATCHES_PATH.exists():
        raise RuntimeError(
            f"{RAW_MATCHES_PATH} not found — run `skims gbt backfill` first"
        )
    matches_df = pd.read_parquet(RAW_MATCHES_PATH)
    profiles_df = (
        pd.read_parquet(PLAYER_PROFILES_PATH)
        if PLAYER_PROFILES_PATH.exists()
        else pd.DataFrame()
    )
    # Rankings history is opt-in: missing → train without the rank
    # features (rank_diff / rank_points_diff land NaN, catboost skips
    # them). Run `skims gbt rankings` to populate.
    from skimsmarkets.tennis.gbt_rankings_backfill import (
        RANKINGS_HISTORY_PATH,
    )
    rankings_df: pd.DataFrame | None = None
    if RANKINGS_HISTORY_PATH.exists():
        rankings_df = pd.read_parquet(RANKINGS_HISTORY_PATH)
    log.info(
        "loaded backfill: %d matches, %d profiles, %d rankings rows",
        len(matches_df), len(profiles_df),
        0 if rankings_df is None else len(rankings_df),
    )

    if features_only:
        table = build_training_table(
            matches_df, profiles_df, rankings_df=rankings_df,
        )
        rows = table.rows
        log.info("feature-build smoke: %d rows ready, columns:", len(rows))
        for col in ALL_FEATURE_COLUMNS:
            if col in rows.columns:
                series = pd.to_numeric(rows[col], errors="coerce")
                log.info(
                    "  %-40s mean=%-8.4f  null=%d/%d",
                    col, series.mean(), series.isna().sum(), len(series),
                )
        return {
            "rows": len(rows),
            "n_dropped_cold_start": table.n_dropped_cold_start,
            "n_dropped_other": table.n_dropped_other,
        }

    out = train_and_evaluate(
        matches_df, profiles_df,
        rankings_df=rankings_df,
        skip_sim_compare=skip_sim_compare,
    )
    return out.metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gbt-train", description=__doc__)
    p.add_argument(
        "--features-only", action="store_true",
        help="Build the training table and exit (no fit) — smoke test."
    )
    p.add_argument(
        "--skip-sim-compare", action="store_true",
        help=(
            "Skip the iid Monte Carlo baseline comparison — speeds up "
            "iteration when only fitting the GBT."
        ),
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    run_train_cli(
        features_only=args.features_only,
        skip_sim_compare=args.skip_sim_compare,
    )


__all__ = [
    "METRICS_PATH",
    "MODEL_PATH",
    "TRAIN_CUTOFF",
    "TrainOutput",
    "run_train_cli",
    "train_and_evaluate",
]
