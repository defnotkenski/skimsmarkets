"""Learning-curve diagnostic for the tennis GBT spike.

Question: is the v1 model data-bound (more rows would help) or
feature-bound (it's plateaued; better features would help)?

Method: subset the existing parquet to top_n players per tour for
n ∈ {10, 20, 35, 50}, retrain the same catboost configuration on
each subset, and compare holdout Brier vs train-set size.

Reading the result:
  - Holdout Brier still descending ≥ 0.5 pp per doubling of train_n
    → data-bound. Wider backfill (top 200 × deeper history) is the
    right v2 lever.
  - Holdout Brier flat within ±0.2 pp across consecutive doublings
    → feature-bound. Better features (recency weighting, split
    return%, age) will pay off more than more data.
  - Train Brier << holdout Brier (≥ 5 pp gap) at the largest n
    → overfit. Regularise before scaling either axis.

Cost: 2 cheap rankings API calls (one per tour) to get the top-N
player IDs in rank order; everything else reuses the existing
backfilled parquet.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pandas as pd
from catboost import CatBoostClassifier

from skimsmarkets.tennis.gbt_backfill import (
    PLAYER_PROFILES_PATH,
    RAW_MATCHES_PATH,
    _fetch_top_ids,
    _headers,
    _resolve_api_key,
)
from skimsmarkets.tennis.gbt_features import build_training_table
from skimsmarkets.tennis.gbt_train import (
    TRAIN_CUTOFF,
    _CATBOOST_PARAMS,
    _auc,
    _brier,
    _log_loss,
    _make_pool,
)
from skimsmarkets.tennis.matchstat import (
    _BURST_TOKENS,
    _REQUESTS_PER_SECOND,
    _TokenBucket,
)

log = logging.getLogger(__name__)

DEFAULT_TOP_N_VALUES: tuple[int, ...] = (10, 20, 35, 50)


async def _get_ranked_ids(top_n: int) -> dict[str, list[int]]:
    """Fetch top-N IDs per tour in rank order. Reuses the backfill's
    rate-limited HTTP path so we stay inside the 5 req/sec vendor
    budget even when the live pipeline is concurrently hitting
    MatchStat for a slate.
    """
    api_key = _resolve_api_key()
    bucket = _TokenBucket(_REQUESTS_PER_SECOND, _BURST_TOKENS)
    out: dict[str, list[int]] = {}
    async with httpx.AsyncClient(
        headers=_headers(api_key), timeout=30.0
    ) as client:
        for tour in ("atp", "wta"):
            pairs = await _fetch_top_ids(client, bucket, tour, top_n)
            out[tour] = [pid for pid, _ in pairs]
            log.info("learning curve: fetched top %d %s players", len(pairs), tour)
    return out


def _subset(
    matches_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    allowed_by_tour: dict[str, list[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only matches where BOTH players are in the allowed top-N
    set, AND the profiles for those players.

    Filtering on BOTH sides (rather than EITHER) is the right semantics
    here: the cold-start gate already drops rows where one side has <
    20 priors, but allowing one-side-in-allowed matches would let the
    aggregator see opponents we're saying we 'don't know about' for
    the experiment. Filtering both sides keeps the experiment honest:
    each top-N subset uses ONLY the players in the subset to build
    histories.
    """
    allowed_ids: set[int] = set()
    for ids in allowed_by_tour.values():
        allowed_ids.update(ids)
    mask = matches_df["p1_id"].isin(allowed_ids) & matches_df["p2_id"].isin(
        allowed_ids
    )
    filtered_matches = matches_df.loc[mask].reset_index(drop=True)
    pmask = profiles_df["player_id"].isin(allowed_ids)
    filtered_profiles = profiles_df.loc[pmask].reset_index(drop=True)
    return filtered_matches, filtered_profiles


def _train_one(
    matches_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Build features → walk-forward split → fit → score. Returns None
    when either fold is empty (subset too small to span the cutoff).
    """
    table = build_training_table(matches_df, profiles_df)
    if table.rows.empty:
        return None
    table.rows["match_date"] = pd.to_datetime(
        table.rows["match_date"]
    ).dt.date
    train_mask = table.rows["match_date"] <= TRAIN_CUTOFF
    train_df = table.rows.loc[train_mask].reset_index(drop=True)
    holdout_df = table.rows.loc[~train_mask].reset_index(drop=True)
    if train_df.empty or holdout_df.empty:
        return None

    train_pool = _make_pool(train_df, train_df["target"])
    holdout_pool = _make_pool(holdout_df, holdout_df["target"])
    # Quiet catboost for the curve sweep — we don't need per-iter logs
    # times 4 model fits.
    params = dict(_CATBOOST_PARAMS, verbose=False)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=holdout_pool)

    train_p = model.predict_proba(train_pool)[:, 1]
    holdout_p = model.predict_proba(holdout_pool)[:, 1]
    y_train = train_df["target"].to_numpy()
    y_holdout = holdout_df["target"].to_numpy()
    return {
        "n_unique_matches": int(len(matches_df)),
        "n_dropped_cold_start": int(table.n_dropped_cold_start),
        "n_train_rows": int(len(train_df)),
        "n_holdout_rows": int(len(holdout_df)),
        "train_brier": float(_brier(y_train, train_p)),
        "holdout_brier": float(_brier(y_holdout, holdout_p)),
        "holdout_log_loss": float(_log_loss(y_holdout, holdout_p)),
        "holdout_auc": float(_auc(y_holdout, holdout_p) or float("nan")),
    }


def _verdict(curve: pd.DataFrame) -> str:
    """One-line interpretation of the curve.

    Decisions:
      - Final-step Brier improvement: how much did the last doubling
        buy us? If ≥ 0.5 pp, data-bound. If ≤ 0.2 pp, feature-bound.
        In between, ambiguous — bias toward data-bound since features
        are a bigger engineering investment.
      - Overfit check: train_brier vs holdout_brier at the largest n.
        ≥ 5 pp gap means the model is memorising; regularise first.
    """
    if len(curve) < 2:
        return "VERDICT: insufficient data points to interpret"
    last = curve.iloc[-1]
    prev = curve.iloc[-2]
    delta_brier = prev["holdout_brier"] - last["holdout_brier"]
    overfit_gap = last["holdout_brier"] - last["train_brier"]
    bits: list[str] = []
    if overfit_gap >= 0.05:
        bits.append(
            f"OVERFIT (holdout − train = {overfit_gap:+.4f}); "
            "regularise before scaling either axis"
        )
    elif delta_brier >= 0.005:
        bits.append(
            f"DATA-BOUND (last doubling improved Brier by {delta_brier:+.4f}); "
            "wider backfill (top 200 × 5y) is the highest-leverage v2 move"
        )
    elif delta_brier <= 0.002:
        bits.append(
            f"FEATURE-BOUND (last doubling improved Brier by only {delta_brier:+.4f}); "
            "feature engineering (recency weighting, split return%, age) "
            "will pay off more than more data"
        )
    else:
        bits.append(
            f"AMBIGUOUS (last doubling improved Brier by {delta_brier:+.4f}); "
            "lean data-bound since wider backfill is cheaper engineering"
        )
    return "VERDICT: " + "  ".join(bits)


def run_row_subsample_curve(
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
    seed: int = 42,
) -> pd.DataFrame:
    """Cleaner data-boundness experiment: hold feature richness CONSTANT
    (every player has full as-of-today history from the entire parquet)
    and vary ONLY the training-row count by random subsampling.

    The top-N variant conflates two effects — fewer matches AND thinner
    per-player histories. This variant isolates the training-rows axis
    so the Brier-vs-rows curve directly answers "does more training
    data help, holding feature quality fixed?"
    """
    matches_df = pd.read_parquet(RAW_MATCHES_PATH)
    profiles_df = pd.read_parquet(PLAYER_PROFILES_PATH)
    log.info(
        "row-subsample: loaded backfill (%d matches, %d profiles) and "
        "building full training table once",
        len(matches_df), len(profiles_df),
    )
    table = build_training_table(matches_df, profiles_df)
    table.rows["match_date"] = pd.to_datetime(
        table.rows["match_date"]
    ).dt.date
    train_mask = table.rows["match_date"] <= TRAIN_CUTOFF
    full_train_df = table.rows.loc[train_mask].reset_index(drop=True)
    holdout_df = table.rows.loc[~train_mask].reset_index(drop=True)
    log.info(
        "row-subsample: full train=%d, fixed holdout=%d",
        len(full_train_df), len(holdout_df),
    )

    rng_master = pd.Series(range(len(full_train_df))).sample(
        frac=1.0, random_state=seed
    ).to_numpy()
    holdout_pool = _make_pool(holdout_df, holdout_df["target"])
    y_holdout = holdout_df["target"].to_numpy()

    rows: list[dict[str, Any]] = []
    for frac in fractions:
        n = max(50, int(round(frac * len(full_train_df))))
        idx = rng_master[:n]
        train_subset = full_train_df.iloc[idx].reset_index(drop=True)
        train_pool = _make_pool(train_subset, train_subset["target"])
        params = dict(_CATBOOST_PARAMS, verbose=False)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=holdout_pool)
        train_p = model.predict_proba(train_pool)[:, 1]
        holdout_p = model.predict_proba(holdout_pool)[:, 1]
        y_train = train_subset["target"].to_numpy()
        rows.append({
            "fraction": frac,
            "n_train_rows": int(len(train_subset)),
            "n_holdout_rows": int(len(holdout_df)),
            "train_brier": float(_brier(y_train, train_p)),
            "holdout_brier": float(_brier(y_holdout, holdout_p)),
            "holdout_log_loss": float(_log_loss(y_holdout, holdout_p)),
            "holdout_auc": float(_auc(y_holdout, holdout_p) or float("nan")),
        })
        log.info(
            "row-subsample frac=%.2f (n=%d): holdout brier=%.4f auc=%.4f",
            frac, n, rows[-1]["holdout_brier"], rows[-1]["holdout_auc"],
        )
    return pd.DataFrame(rows)


def run_learning_curve(
    top_n_values: tuple[int, ...] = DEFAULT_TOP_N_VALUES,
) -> pd.DataFrame:
    """End-to-end driver. Returns the curve DataFrame."""
    matches_df = pd.read_parquet(RAW_MATCHES_PATH)
    profiles_df = pd.read_parquet(PLAYER_PROFILES_PATH)
    log.info(
        "loaded backfill: %d matches, %d profiles",
        len(matches_df), len(profiles_df),
    )

    rankings = asyncio.run(_get_ranked_ids(max(top_n_values)))
    rows: list[dict[str, Any]] = []
    for n in top_n_values:
        allowed = {tour: ids[:n] for tour, ids in rankings.items()}
        m_sub, p_sub = _subset(matches_df, profiles_df, allowed)
        log.info(
            "top_n=%d: %d unique matches, %d profiles",
            n, len(m_sub), len(p_sub),
        )
        result = _train_one(m_sub, p_sub)
        if result is None:
            log.warning("top_n=%d: insufficient data after filtering, skipping", n)
            continue
        rows.append({"top_n": n, **result})

    curve = pd.DataFrame(rows)
    return curve


def _print_curve(curve: pd.DataFrame) -> None:
    """Pretty-print the curve as a table + verdict."""
    if curve.empty:
        print("learning curve produced no data points")
        return
    cols = [
        "top_n",
        "n_unique_matches",
        "n_train_rows",
        "n_holdout_rows",
        "n_dropped_cold_start",
        "train_brier",
        "holdout_brier",
        "holdout_log_loss",
        "holdout_auc",
    ]
    print(curve[cols].to_string(index=False))
    print()
    print(_verdict(curve))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print("=== Top-N curve (varies player pool AND history depth) ===")
    curve = run_learning_curve()
    _print_curve(curve)
    print()
    print("=== Row-subsample curve (varies train rows ONLY; histories full) ===")
    rs = run_row_subsample_curve()
    print(rs.to_string(index=False))
    if len(rs) >= 2:
        last, prev = rs.iloc[-1], rs.iloc[-2]
        delta = prev["holdout_brier"] - last["holdout_brier"]
        if delta >= 0.005:
            tag = "DATA-BOUND"
        elif delta <= 0.002:
            tag = "FEATURE-BOUND"
        else:
            tag = "AMBIGUOUS"
        print(f"\nROW-SUBSAMPLE VERDICT: {tag} (last step Δ Brier = {delta:+.4f})")


__all__ = [
    "DEFAULT_TOP_N_VALUES",
    "run_learning_curve",
    "run_row_subsample_curve",
]
