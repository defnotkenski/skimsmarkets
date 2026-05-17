"""Validate the UW trader-profile enrichment by hit-rate-by-tier.

Hypothesis being tested: events where the director saw at least one ★ SMART
insider (UW-classified informed-flow wallet sized at z≥2 on YES) win more
often than events with only ⚑ NOTABLE-but-not-SMART insiders or no insiders
at all. If true, the enrichment is load-bearing and the prompt-level
weighting guidance is justified. If false, the enrichment is harmless
overhead and can be reverted by deleting the profile-fetch step in
`pipeline._one`.

Data shape required: each prediction row in `logs/runs/*.jsonl` must carry
`uw_insiders_total` / `uw_insiders_notable` / `uw_insiders_smart` (added
2026-05-17). Rows persisted BEFORE that change have None values and are
classified as "no snapshot" — excluded from the tier bins but counted in
the baseline.

Usage:
    uv run python scripts/uw_enrichment_retro.py

Honest expectations on sample size: at the current hourly-cron / 5-event
cadence, ~100 settled tennis events with the new snapshot field accumulate
in ~3-5 days. With smaller N, this script will still run — it just won't
have statistical weight. Read the per-tier `n` columns; tiers with n<20
are suggestive at best.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from skimsmarkets.retro.jsonl import (
    iter_predictions,
    list_run_files,
    resolutions_sidecar_path,
)


def _load_resolutions(path: Path) -> dict[str, bool]:
    """{event_id: predicted_correct} from a resolutions sidecar."""
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


def _tier(notable: int | None, smart: int | None) -> str:
    """Classify an event's UW enrichment tier from its persisted counts."""
    if notable is None and smart is None:
        return "no_snapshot"
    notable = notable or 0
    smart = smart or 0
    if smart >= 1:
        return "smart_present"
    if notable >= 1:
        return "notable_only"
    return "no_insiders"


def main() -> None:
    # Walk all runs, build (tier, predicted_correct, predicted_yes_prob, event_id)
    rows: list[tuple[str, bool, float | None, str, str]] = []
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
            tier = _tier(pred.uw_insiders_notable, pred.uw_insiders_smart)
            rows.append((
                tier,
                resolutions[pred.event_id],
                pred.predicted_yes_probability,
                pred.event_id,
                run_path.stem,
            ))

    print(f"Runs scanned:              {runs_scanned}")
    print(f"Runs with resolutions:     {runs_with_resolutions}")
    print(f"Settled predictions:       {len(rows)}")
    if not rows:
        print("\nNothing to analyze.")
        return

    # Overall baseline
    all_correct = [r[1] for r in rows]
    baseline_hr = sum(all_correct) / len(all_correct)
    print(f"Baseline hit rate:         {baseline_hr:.3f}")

    # Per-tier breakdown
    by_tier: dict[str, list[bool]] = defaultdict(list)
    by_tier_brier: dict[str, list[float]] = defaultdict(list)
    for tier, correct, yes_p, _, _ in rows:
        by_tier[tier].append(correct)
        if yes_p is not None:
            # Brier on the YES probability vs the outcome's YES-side truth.
            # Without the directional join we can't compute the director's
            # actual Brier (we don't know if `predicted_correct=True` means
            # the YES side won or the predicted side won). Heuristic: use
            # `predicted_correct` as the binary outcome relative to the
            # director's call — Brier here is "how close was the director's
            # confidence to being right". For a unidirectional comparison
            # across tiers this is consistent.
            confidence = yes_p if correct else 1 - yes_p
            by_tier_brier[tier].append((1 - confidence) ** 2)

    print()
    print("=" * 75)
    print(f"{'tier':<20}  {'n':>5}  {'hit_rate':>10}  {'vs_base':>10}  {'mean_brier':>11}")
    print("=" * 75)

    tier_order = ["smart_present", "notable_only", "no_insiders", "no_snapshot"]
    tier_label = {
        "smart_present": "★ SMART present",
        "notable_only":  "⚑ NOTABLE only",
        "no_insiders":   "UW (no insiders)",
        "no_snapshot":   "no UW snapshot",
    }
    for tier in tier_order:
        items = by_tier.get(tier, [])
        if not items:
            print(f"{tier_label[tier]:<20}  {'—':>5}  {'—':>10}  {'—':>10}  {'—':>11}")
            continue
        hr = sum(items) / len(items)
        delta = hr - baseline_hr
        briers = by_tier_brier.get(tier, [])
        mb = sum(briers) / len(briers) if briers else None
        mb_s = f"{mb:.4f}" if mb is not None else "—"
        print(f"{tier_label[tier]:<20}  {len(items):>5}  {hr:>10.3f}  {delta:>+10.3f}  {mb_s:>11}")

    print()
    smart_n = len(by_tier.get("smart_present", []))
    notable_n = len(by_tier.get("notable_only", []))
    if smart_n > 0 and notable_n > 0:
        smart_hr = sum(by_tier["smart_present"]) / smart_n
        notable_hr = sum(by_tier["notable_only"]) / notable_n
        print(f"SMART-vs-NOTABLE-only hit-rate delta: {(smart_hr - notable_hr):+.3f}")
        print(f"  (SMART n={smart_n}, NOTABLE-only n={notable_n})")
        if smart_n < 20 or notable_n < 20:
            print("  WARNING: one or both bins have n<20 — directional read only.")
    else:
        missing = []
        if smart_n == 0:
            missing.append("SMART")
        if notable_n == 0:
            missing.append("NOTABLE-only")
        print(f"Cannot compute delta — no settled events in tier(s): {', '.join(missing)}")

    snap_n = sum(1 for r in rows if r[0] != "no_snapshot")
    if snap_n < len(rows):
        old_n = len(rows) - snap_n
        print(f"\n{old_n}/{len(rows)} predictions predate the UW snapshot field "
              f"(no `uw_insiders_*` populated). Re-run after {old_n} more "
              f"settled events to retire this caveat.")


if __name__ == "__main__":
    main()
