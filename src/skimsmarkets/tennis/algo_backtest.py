"""Walk-forward backtest for the algorithmic tennis form-and-surface lens.

Reuses the GBT spike's data layout — `raw_matches.parquet` for the
match corpus + ground truth, `player_profiles.parquet` for handedness
and birthdate, `rankings_history.parquet` for point-in-time rank.
Walks the parquet chronologically with `gbt_features.HistoryStore`'s
point-in-time discipline (snapshot BEFORE folding the current row),
projects each per-player snapshot into a synthetic
`TennisPlayerStats`, calls the algo's pure scoring function, and
scores predictions against `winner_side`.

Same fold cutoff as `gbt_train.py` (2024-12-31) so the algo's holdout
metrics line up directly against the GBT's scorecard.

Anchor convention mirrors the GBT exactly: anchor = player with the
LOWER MatchStat id, target = 1 if anchor won. Anchor-swapping flips
every feature's sign and the predicted probability — symmetric by
construction.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from skimsmarkets.agents.sports.tennis.schemas import TennisFormSurfaceReport
from skimsmarkets.tennis.algo_lens import _score_form_surface
from skimsmarkets.tennis.gbt_backfill import (
    PLAYER_PROFILES_PATH,
    RAW_MATCHES_PATH,
)
from skimsmarkets.tennis.gbt_features import (
    MIN_PRIORS_PER_SIDE,
    HistoryStore,
    PlayerHistory,
    _add_match_from_row,
    _row_get_int,
    surface_key,
)
from skimsmarkets.tennis.gbt_rankings_backfill import RANKINGS_HISTORY_PATH
from skimsmarkets.tennis.models import (
    TennisPlayerStats,
    TennisStatsContext,
)

log = logging.getLogger(__name__)

# Same cutoff as gbt_train.py:TRAIN_CUTOFF — holdout starts 2025-01-01.
TRAIN_CUTOFF = date(2024, 12, 31)


@dataclass(frozen=True)
class BacktestMetrics:
    n: int
    brier: float
    log_loss: float
    accuracy: float
    base_rate: float
    per_surface: dict[str, dict[str, float | int]] = field(default_factory=dict)
    per_tour: dict[str, dict[str, float | int]] = field(default_factory=dict)
    reliability: list[dict[str, float | int]] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestResult:
    train: BacktestMetrics
    holdout: BacktestMetrics
    holdout_calibrated: BacktestMetrics | None
    fitted_temperature: float | None
    n_dropped_cold_start: int
    n_dropped_other: int
    train_cutoff: str
    algo_version: str


# ---------------------------------------------------------------------------
# Rank lookup — pre-build a (tour, player_id) → ranking trajectory dict so
# each per-match lookup is O(log n) via bisect.
# ---------------------------------------------------------------------------


def _build_rank_lookup(
    rankings_df: pd.DataFrame,
) -> dict[tuple[str, int], list[tuple[int, int, int | None]]]:
    """Return `{(tour, player_id): [(epoch_seconds, rank, points), ...]}`
    sorted ascending by date. Caller uses `bisect_right` for the
    "most recent rank on or before match_date" lookup.
    """
    out: dict[tuple[str, int], list[tuple[int, int, int | None]]] = defaultdict(list)
    # Convert ranking_date to int epoch seconds once.
    ts = pd.to_datetime(rankings_df["ranking_date"]).astype("int64") // 10**9
    tours = rankings_df["tour"].to_numpy()
    pids = rankings_df["player_id"].to_numpy()
    ranks = rankings_df["rank"].to_numpy()
    points = rankings_df["rank_points"].to_numpy()
    for i in range(len(rankings_df)):
        pt = int(points[i]) if pd.notna(points[i]) else None
        out[(str(tours[i]), int(pids[i]))].append((int(ts.iloc[i]), int(ranks[i]), pt))
    for k in out:
        out[k].sort()
    return out


def _lookup_rank_at(
    lookup: dict[tuple[str, int], list[tuple[int, int, int | None]]],
    tour: str,
    pid: int,
    on_date: date,
) -> tuple[int | None, int | None]:
    """Return (rank, points) at or before `on_date`, or (None, None)."""
    entries = lookup.get((tour, pid))
    if not entries:
        return None, None
    target_ts = int(pd.Timestamp(on_date).timestamp())
    # Bisect on the timestamps. Pythonic + fast enough at 768k entries.
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


# ---------------------------------------------------------------------------
# PlayerHistory + profile + rank → synthetic TennisPlayerStats projection.
# ---------------------------------------------------------------------------


def _form_string_from_recent(recent_deque: Any) -> str | None:
    """Encode the recent ring buffer as a "WLWWL..." string. Newest at
    the right matches the renderer convention in `tennis/rendering.py`.
    """
    if not recent_deque:
        return None
    return "".join("W" if w else "L" for w in recent_deque)


def _project_player(
    history: PlayerHistory,
    *,
    rank: int | None,
    rank_points: int | None,
    name: str,
    plays: str | None,
    age_years: int | None,
) -> TennisPlayerStats:
    """Project a `PlayerHistory` snapshot + profile + rank into the
    `TennisPlayerStats` shape the algo consumes in production.

    Career win/loss totals go into `ytd_win_loss` — the field is
    documented as a vendor-defined window, but the algo treats it as
    a general career proxy; using career counters here keeps the
    backtest internally consistent (the algo reads the same field in
    production from the vendor's actual YTD).
    """
    # Surface buckets. PlayerHistory stores floats (recency-decayed);
    # we round to int (wins, losses) for the schema, preserving
    # ordering at the cost of fractional precision the algo doesn't
    # actually use.
    surface_win_loss: dict[str, tuple[int, int]] = {}
    for surf, bucket in history.by_surface.items():
        if surf is None:
            continue
        w = int(round(bucket.wins))
        total = int(round(bucket.matches))
        if total == 0:
            continue
        surface_win_loss[surf] = (w, max(0, total - w))

    return TennisPlayerStats(
        name=name,
        api_player_id=str(history.player_id),
        rank_singles=rank,
        rank_points=rank_points,
        age_years=age_years,
        plays=plays,
        # Career counters as the ytd proxy. Algo reads this via
        # _career_winrate when serve/return rates are missing.
        ytd_win_loss=(history.wins, max(0, history.matches - history.wins)),
        surface_win_loss=surface_win_loss or None,
        last_10_form=_form_string_from_recent(history.recent),
        # Career rates — recency-weighted via the 365-day half-life
        # decay in PlayerHistory._decay_to.
        first_serve_in_pct=history.career_first_serve_in_pct(),
        first_serve_win_pct=history.career_first_serve_win_pct(),
        second_serve_win_pct=history.career_second_serve_win_pct(),
        first_serve_return_win_pct=history.career_first_serve_return_win_pct(),
        second_serve_return_win_pct=history.career_second_serve_return_win_pct(),
        break_point_save_pct=history.career_bp_save_pct(),
        break_point_convert_pct=history.career_bp_convert_pct(),
        last_match_date=history.last_match_date,
    )


# ---------------------------------------------------------------------------
# Backtest walk.
# ---------------------------------------------------------------------------


def _build_profile_lookup(
    profiles_df: pd.DataFrame,
) -> dict[tuple[str, int], tuple[str, str | None, date | None]]:
    """Return `{(tour, player_id): (name, plays, birthdate)}`."""
    out: dict[tuple[str, int], tuple[str, str | None, date | None]] = {}
    for _, row in profiles_df.iterrows():
        tour = str(row["tour"])
        pid = int(row["player_id"])
        name = str(row["name"])
        plays = row["plays"] if pd.notna(row["plays"]) else None
        bd = row["birthdate"]
        bdate: date | None = None
        if pd.notna(bd):
            bdate = pd.Timestamp(bd).date()
        out[(tour, pid)] = (name, plays, bdate)
    return out


def _age_at(birthdate: date | None, on_date: date) -> int | None:
    if birthdate is None:
        return None
    yrs = on_date.year - birthdate.year
    if (on_date.month, on_date.day) < (birthdate.month, birthdate.day):
        yrs -= 1
    return max(0, yrs)


def run_backtest(
    *,
    compute_fn: Callable[
        ..., TennisFormSurfaceReport | None
    ] = _score_form_surface,
    min_priors: int = MIN_PRIORS_PER_SIDE,
    train_cutoff: date = TRAIN_CUTOFF,
    algo_version: str = "v1",
) -> BacktestResult:
    """Walk the GBT parquet, predict via `compute_fn` for each row that
    clears the cold-start gate, and return train + holdout metrics.

    `compute_fn` defaults to the algo lens's `_score_form_surface` but
    can be swapped to test alternate scoring rules. Signature:
    `(team_a_name, team_b_name, ts) -> TennisFormSurfaceReport | None`.
    The harness applies `baseline + form_shift + surface_shift` and
    clips to `[0, 1]` to get the final probability scored against
    `winner_side`.
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

    store = HistoryStore()
    train: list[tuple[float, int, str | None, str]] = []  # (pred, label, surface, tour)
    holdout: list[tuple[float, int, str | None, str]] = []
    n_cold = 0
    n_other = 0
    n_processed = 0

    for _, row in matches.iterrows():
        n_processed += 1
        if n_processed % 20000 == 0:
            log.info(
                "walked %d / %d  train=%d holdout=%d cold=%d other=%d",
                n_processed, len(matches),
                len(train), len(holdout), n_cold, n_other,
            )

        p1 = _row_get_int(row, "p1_id")
        p2 = _row_get_int(row, "p2_id")
        winner_side = _row_get_int(row, "winner_side")
        if p1 is None or p2 is None or winner_side not in (1, 2):
            n_other += 1
            continue

        anchor = min(p1, p2)
        opp = max(p1, p2)
        anchor_won = 1 if ((p1 == anchor and winner_side == 1) or (p2 == anchor and winner_side == 2)) else 0
        on_date = pd.Timestamp(row["match_date"]).date()
        tour = str(row["tour"])
        surface = surface_key(_row_get_int(row, "court_id"))
        # Slams ship best_of=5; everything else bo3. Vendor field
        # occasionally arrives as None or as a string — coerce
        # defensively so the algo always sees a clean int.
        bo_raw = _row_get_int(row, "best_of")
        best_of = 5 if bo_raw == 5 else 3

        ah = store.get(anchor)
        oh = store.get(opp)
        if ah is None or oh is None or ah.matches < min_priors or oh.matches < min_priors:
            n_cold += 1
            _add_match_from_row(store, row)
            continue

        # Rank + profile lookups.
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

        ts = TennisStatsContext(
            provider="backtest",
            fetched_at=pd.Timestamp(on_date).to_pydatetime(),
            surface=surface,
            tournament=None,
            player_a=_project_player(
                ah, rank=a_rank, rank_points=a_pts,
                name=a_name, plays=a_plays, age_years=_age_at(a_bd, on_date),
            ),
            player_b=_project_player(
                oh, rank=o_rank, rank_points=o_pts,
                name=o_name, plays=o_plays, age_years=_age_at(o_bd, on_date),
            ),
            head_to_head=None,
        )

        try:
            report = compute_fn(a_name, o_name, ts, best_of=best_of)
        except Exception:  # noqa: BLE001
            log.exception("compute_fn crashed on match_id=%s", _row_get_int(row, "match_id"))
            n_other += 1
            _add_match_from_row(store, row)
            continue

        if report is None:
            n_other += 1
            _add_match_from_row(store, row)
            continue

        pred = report.team_a_win_probability + report.form_signed_shift + report.surface_signed_shift
        pred = max(0.001, min(0.999, pred))

        bucket = train if on_date <= train_cutoff else holdout
        bucket.append((pred, anchor_won, surface, tour))

        # Fold AFTER snapshot.
        _add_match_from_row(store, row)

    log.info(
        "done. train=%d holdout=%d cold=%d other=%d",
        len(train), len(holdout), n_cold, n_other,
    )

    # Fit a single temperature on the train fold and apply to the
    # holdout fold. Cheap (1-D bracketed minimisation over [0.5, 3.0])
    # and tightens extreme-bin reliability without touching the algo.
    fitted_t: float | None = None
    holdout_cal: BacktestMetrics | None = None
    if train and holdout:
        fitted_t = _fit_temperature([(p, y) for p, y, _, _ in train])
        if fitted_t is not None and abs(fitted_t - 1.0) > 1e-3:
            calibrated = [
                (_apply_temperature(p, fitted_t), y, s, t)
                for p, y, s, t in holdout
            ]
            holdout_cal = _compute_metrics(calibrated)

    return BacktestResult(
        train=_compute_metrics(train),
        holdout=_compute_metrics(holdout),
        holdout_calibrated=holdout_cal,
        fitted_temperature=fitted_t,
        n_dropped_cold_start=n_cold,
        n_dropped_other=n_other,
        train_cutoff=train_cutoff.isoformat(),
        algo_version=algo_version,
    )


# ---------------------------------------------------------------------------
# Temperature scaling — fit a single scalar T on the train fold by
# minimising NLL, apply to the holdout. Hand-rolled to keep
# algo_backtest standalone from retro/.
# ---------------------------------------------------------------------------


def _apply_temperature(p: float, t: float) -> float:
    """Apply temperature `t` to a probability via logit space.
    T > 1 softens toward 0.5; T < 1 sharpens away from 0.5.
    """
    p = max(1e-6, min(1 - 1e-6, p))
    logit = math.log(p / (1 - p))
    scaled = logit / t
    return 1.0 / (1.0 + math.exp(-scaled))


def _nll(pairs: list[tuple[float, int]], t: float) -> float:
    """Mean negative log-likelihood at temperature `t`."""
    total = 0.0
    for p, y in pairs:
        q = _apply_temperature(p, t)
        q = max(1e-6, min(1 - 1e-6, q))
        total += -(y * math.log(q) + (1 - y) * math.log(1 - q))
    return total / len(pairs)


def _fit_temperature(
    pairs: list[tuple[float, int]],
    *,
    t_min: float = 0.5,
    t_max: float = 3.0,
    tol: float = 1e-3,
) -> float:
    """Golden-section search for the temperature minimising NLL.

    1-D bracketed minimisation; the NLL surface is convex in T over
    [t_min, t_max] so golden-section converges quickly without needing
    derivatives. Bounds chosen wide enough that the optimum sits
    interior except in pathological cases (a fitted T at the boundary
    indicates the underlying predictions are degenerate).
    """
    if not pairs:
        return 1.0
    phi = (math.sqrt(5) - 1) / 2  # 0.618...
    a, b = t_min, t_max
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _nll(pairs, c)
    fd = _nll(pairs, d)
    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _nll(pairs, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _nll(pairs, d)
    return (a + b) / 2


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _compute_metrics(
    rows: list[tuple[float, int, str | None, str]],
) -> BacktestMetrics:
    if not rows:
        return BacktestMetrics(n=0, brier=float("nan"), log_loss=float("nan"),
                               accuracy=float("nan"), base_rate=float("nan"))
    n = len(rows)
    brier = sum((p - y) ** 2 for p, y, _, _ in rows) / n
    log_loss = sum(
        -(y * math.log(p) + (1 - y) * math.log(1 - p))
        for p, y, _, _ in rows
    ) / n
    accuracy = sum(1 for p, y, _, _ in rows if (p >= 0.5) == bool(y)) / n
    base_rate = sum(y for _, y, _, _ in rows) / n

    # Per-surface Brier.
    by_surf: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for p, y, s, _ in rows:
        by_surf[s or "unknown"].append((p, y))
    per_surface = {
        s: {
            "n": len(group),
            "brier": sum((p - y) ** 2 for p, y in group) / len(group),
        }
        for s, group in by_surf.items()
    }
    # Per-tour Brier.
    by_tour: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for p, y, _, t in rows:
        by_tour[t].append((p, y))
    per_tour = {
        t: {
            "n": len(group),
            "brier": sum((p - y) ** 2 for p, y in group) / len(group),
        }
        for t, group in by_tour.items()
    }
    # Reliability deciles.
    deciles: list[dict[str, float | int]] = []
    for i in range(10):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        bin_rows = [(p, y) for p, y, _, _ in rows if (p >= lo and (p < hi or (i == 9 and p <= hi)))]
        if not bin_rows:
            continue
        deciles.append({
            "bin_low": lo,
            "bin_high": hi,
            "n": len(bin_rows),
            "mean_predicted": sum(p for p, _ in bin_rows) / len(bin_rows),
            "mean_observed": sum(y for _, y in bin_rows) / len(bin_rows),
        })

    return BacktestMetrics(
        n=n,
        brier=brier,
        log_loss=log_loss,
        accuracy=accuracy,
        base_rate=base_rate,
        per_surface=per_surface,
        per_tour=per_tour,
        reliability=deciles,
    )


# ---------------------------------------------------------------------------
# CLI entry — write a metrics scorecard sidecar similar to gbt_train.
# ---------------------------------------------------------------------------


METRICS_PATH = Path("models/tennis_algo_form_surface.metrics.json")


def _serialize_metrics(m: BacktestMetrics) -> dict[str, Any]:
    return {
        "n": m.n,
        "brier": m.brier,
        "log_loss": m.log_loss,
        "accuracy": m.accuracy,
        "base_rate_anchor_wins": m.base_rate,
        "per_surface": m.per_surface,
        "per_tour": m.per_tour,
        "reliability_deciles": m.reliability,
    }


def run_backtest_cli(*, algo_version: str = "v1") -> BacktestResult:
    result = run_backtest(algo_version=algo_version)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "algo_version": result.algo_version,
        "train_cutoff": result.train_cutoff,
        "train_n": result.train.n,
        "holdout_n": result.holdout.n,
        "n_dropped_cold_start": result.n_dropped_cold_start,
        "n_dropped_other": result.n_dropped_other,
        "fitted_temperature": result.fitted_temperature,
        "train": _serialize_metrics(result.train),
        "holdout": _serialize_metrics(result.holdout),
    }
    if result.holdout_calibrated is not None:
        payload["holdout_calibrated"] = _serialize_metrics(result.holdout_calibrated)
    METRICS_PATH.write_text(json.dumps(payload, indent=2, default=str))
    return result
