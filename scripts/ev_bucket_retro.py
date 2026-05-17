"""Hit rate + realized PnL by expected-value bucket.

Tests whether the asymmetric-payoff principle (high EV → high realized
return) actually holds in this codebase's settled tennis predictions.
If yes: green light to rewrite `trader.py` sizing to scale by EV magnitude
rather than uniform `--bet-size`. If no (or if high-EV bucket realizes
LOWER returns than mid-EV): the model probabilities are likely too
optimistic at the underdog tail, and the asymmetric-sizing strategy
would amplify miscalibration losses.

Reads `logs/runs/*.jsonl` (predictions) + `*.resolutions.jsonl`
(settlement). For rows where `ev_per_dollar` was persisted at decision
time, uses the snapshot value. For older rows that predate the field,
recomputes from `predicted_yes_probability` and
`polymarket_implied_probability` via `compute_ev_per_dollar` so the
analysis works on all historical data.

Realized PnL calculation: for each settled bet of $1 staked,
  win → returns $(1 - market_p) / market_p in profit
  loss → returns -$1
Both legs are deterministic given (predicted_correct, market_p) so we
can compute realized $ return per $1 staked across buckets without
needing actual fill data.

Usage:
    uv run python scripts/ev_bucket_retro.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from skimsmarkets.pipeline import compute_ev_per_dollar
from skimsmarkets.retro.jsonl import (
    iter_predictions,
    list_run_files,
    resolutions_sidecar_path,
)


def _load_resolutions(path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("settled"):
                continue
            pc = row.get("predicted_correct")
            if pc is None:
                continue
            out[row["event_id"]] = bool(pc)
    return out


def _realized_pnl_per_dollar(market_p: float | None, won: bool) -> float | None:
    """Realized $ return per $1 staked on the predicted side.

    Win: $1 stake returns (1 - market_p) / market_p in profit
    Loss: $1 stake loses the full $1
    """
    if market_p is None or not (0.0 < market_p < 1.0):
        return None
    if won:
        return (1.0 - market_p) / market_p
    return -1.0


def _bucket(ev: float | None) -> str:
    """EV bucket label. Boundaries chosen to separate fair-market noise
    from each major edge tier."""
    if ev is None:
        return "no_ev"
    if ev < 0:
        return "negative"
    if ev < 0.15:
        return "0-15%"
    if ev < 0.30:
        return "15-30%"
    if ev < 0.50:
        return "30-50%"
    return "50%+"


BUCKET_ORDER = ["negative", "0-15%", "15-30%", "30-50%", "50%+", "no_ev"]


def main() -> None:
    rows: list[dict] = []  # joined predictions with their outcome + EV
    runs_scanned = 0
    runs_with_resolutions = 0

    for run_path in list_run_files():
        runs_scanned += 1
        resolutions = _load_resolutions(resolutions_sidecar_path(run_path))
        if not resolutions:
            continue
        runs_with_resolutions += 1
        for pred in iter_predictions(run_path):
            if pred.event_id not in resolutions:
                continue
            # Prefer the snapshot, fall back to on-the-fly compute for
            # rows that predate the field (handles historical data).
            ev = pred.ev_per_dollar
            if ev is None:
                ev = compute_ev_per_dollar(
                    pred.predicted_yes_probability,
                    pred.polymarket_implied_probability,
                )
            won = resolutions[pred.event_id]
            realized = _realized_pnl_per_dollar(
                pred.polymarket_implied_probability, won
            )
            rows.append({
                "event_id": pred.event_id,
                "model_p": pred.predicted_yes_probability,
                "market_p": pred.polymarket_implied_probability,
                "ev": ev,
                "won": won,
                "realized": realized,
                "bucket": _bucket(ev),
            })

    print(f"Runs scanned:              {runs_scanned}")
    print(f"Runs with resolutions:     {runs_with_resolutions}")
    print(f"Settled predictions:       {len(rows)}")
    if not rows:
        print("\nNothing to analyze yet.")
        return

    # Baseline overall stats
    all_won = [r["won"] for r in rows]
    overall_hr = sum(all_won) / len(all_won)
    rows_with_realized = [r for r in rows if r["realized"] is not None]
    if rows_with_realized:
        overall_realized = mean(r["realized"] for r in rows_with_realized)
    else:
        overall_realized = None
    print(f"Baseline hit rate:         {overall_hr:.3f}")
    if overall_realized is not None:
        print(f"Baseline realized PnL/$:   {overall_realized:+.4f}  "
              f"(across {len(rows_with_realized)} rows with market_p)")

    # Per-bucket breakdown
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    print()
    print("=" * 92)
    print(f"{'EV bucket':<12}  {'n':>5}  {'hit_rate':>10}  "
          f"{'mean_ev':>10}  {'mean_realized':>14}  {'edge_vs_ev':>10}")
    print("=" * 92)
    for bucket in BUCKET_ORDER:
        items = by_bucket.get(bucket, [])
        if not items:
            continue
        won = [r["won"] for r in items]
        hr = sum(won) / len(won)
        evs = [r["ev"] for r in items if r["ev"] is not None]
        mean_ev = mean(evs) if evs else None
        reals = [r["realized"] for r in items if r["realized"] is not None]
        mean_real = mean(reals) if reals else None
        # "edge_vs_ev" = realized minus expected. Positive = model is well-
        # calibrated or pessimistic; negative = model overestimates edge.
        edge_vs_ev = (mean_real - mean_ev) if (mean_ev is not None and mean_real is not None) else None
        ev_s = f"{mean_ev:+.3f}" if mean_ev is not None else "—"
        real_s = f"{mean_real:+.3f}" if mean_real is not None else "—"
        edge_s = f"{edge_vs_ev:+.3f}" if edge_vs_ev is not None else "—"
        print(f"{bucket:<12}  {len(items):>5}  {hr:>10.3f}  "
              f"{ev_s:>10}  {real_s:>14}  {edge_s:>10}")

    # Asymmetric-strategy assessment — does high-EV outperform low-EV in realized terms?
    print()
    high_ev = sum((by_bucket.get(b, []) for b in ("15-30%", "30-50%", "50%+")), [])
    low_ev = by_bucket.get("0-15%", [])
    if high_ev and low_ev:
        h_real = [r["realized"] for r in high_ev if r["realized"] is not None]
        l_real = [r["realized"] for r in low_ev if r["realized"] is not None]
        if h_real and l_real:
            h_mean = mean(h_real)
            l_mean = mean(l_real)
            print(f"ASYMMETRIC-STRATEGY SIGNAL:")
            print(f"  High-EV buckets (15%+):  n={len(h_real)}  mean realized/$ = {h_mean:+.3f}")
            print(f"  Low-EV bucket (0-15%):   n={len(l_real)}  mean realized/$ = {l_mean:+.3f}")
            print(f"  Delta (high - low):      {h_mean - l_mean:+.3f}")
            if h_mean > l_mean and h_mean > 0:
                print(f"  → High-EV bets are outperforming. EV-scaled sizing in trader.py is worth shipping.")
            elif h_mean > 0 and l_mean > 0:
                print(f"  → Both buckets profitable; EV-scaling would amplify edge but not flip sign.")
            else:
                print(f"  → High-EV bucket not outperforming. Likely model miscalibration at "
                      f"underdog prices; do NOT ship EV-scaled sizing without recalibration.")
            if len(h_real) < 20 or len(l_real) < 20:
                print(f"  WARNING: one or both bins have n<20 — directional read only, not statistical proof.")
    else:
        missing = []
        if not high_ev:
            missing.append("high-EV")
        if not low_ev:
            missing.append("low-EV")
        print(f"Cannot assess asymmetric strategy — no settled events in bucket(s): {', '.join(missing)}")

    # Coverage caveat
    no_ev_n = len(by_bucket.get("no_ev", []))
    if no_ev_n > 0:
        print(f"\n{no_ev_n}/{len(rows)} predictions have no EV (missing market_p OR degenerate "
              f"price). Excluded from bucket analysis.")


if __name__ == "__main__":
    main()
