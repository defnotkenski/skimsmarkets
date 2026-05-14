"""End-to-end retro orchestration — wires Steps 1, 2, and 3 together.

Single entry point per CLI invocation. Reads the same logs/run sidecars
the lower-level modules expose individually, but centralises the
"join features with post-match" step and the output-file naming.

Output-file naming uses a timestamp so repeat runs don't clobber:
`logs/retro/YYYYMMDD-HHMMSS.{calibrate.json,findings.json,report.md}`.
The latest of each can always be found with a sort-newest glob.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from anthropic import AsyncAnthropic

from skimsmarkets import config as cfg
from skimsmarkets.calibration import (
    CALIBRATION_PATH,
    apply_temperature,
    load_temperature,
    write_calibration,
)
from skimsmarkets.retro.analyze import analyze_all_sports
from skimsmarkets.retro.calibrate import (
    CalibrateReport,
    aggregate,
    collect_features,
    render_report,
)
from skimsmarkets.retro.features import extract_features
from skimsmarkets.retro.jsonl import (
    iter_predictions,
    list_run_files,
    log_root,
    resolutions_sidecar_path,
    run_path_for_id,
)
from skimsmarkets.retro.metrics import (
    MIN_CLASS_N,
    MIN_FIT_N,
    compute_metrics,
    fit_temperature,
)
from skimsmarkets.retro.models import (
    EventFeatures,
    ResolvedOutcome,
)
from skimsmarkets.retro.post_match import fetch_post_match_for_settled
from skimsmarkets.retro.report import write_report as write_md_report
from skimsmarkets.retro.resolve import resolve_all_runs, resolve_run
from skimsmarkets.tennis.provider import build_tennis_provider

log = logging.getLogger(__name__)


def _retro_root() -> Path:
    """Resolved at call-time (not import) so test setups can swap the
    working dir without losing track of the output root.
    """
    return log_root().parent / "retro"


def _retro_id() -> str:
    """Timestamp-based id so back-to-back retros don't overwrite each
    other. Operator can always find the latest with `ls -t logs/retro/`.
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


async def run_step_resolve(run_id: str | None = None) -> list[Path]:
    """Step 1 — write resolution sidecars.

    `run_id=None` resolves every run log under `logs/runs/`. A specific
    `run_id` resolves only that run.
    """
    if run_id is not None:
        return [await resolve_run(run_id)]
    return await resolve_all_runs()


def _features_with_post_match(
    post_match_by_event: dict | None = None,
    run_id: str | None = None,
) -> tuple[list[EventFeatures], int]:
    """Build the feature list, optionally joining post-match data.
    Returns `(deduped_features, n_dropped)`.

    `post_match_by_event` is keyed by `event_id`; when None, features
    have no post-match divergence populated (Step 2 doesn't need it).
    `run_id` filters to a single run (used by `--run-id` invocations).

    **Dedup**: same `market_slug` can appear across multiple runs.
    Counting each instance double-counts settled outcomes in hit-rate
    cuts AND feeds the same divergence to the Step 3 LLM analyzer
    multiple times. Dedup by `market_slug`, keeping the
    chronologically EARLIEST prediction (oldest-run wins). `run_id`
    filtered invocations dedup within one run (no-op since a single
    run shouldn't have duplicate slugs); the cross-run case (no
    `run_id`) is where dedup actually fires. Mirrors
    `calibrate.collect_features` so both paths report consistent
    market counts.
    """
    feats: list[EventFeatures] = []
    seen_slugs: set[str] = set()
    duplicates_dropped = 0
    paths = (
        [run_path_for_id(run_id)]
        if run_id is not None
        # Oldest-first so the earliest prediction wins dedup.
        # `list_run_files` returns mtime-descending; reverse here.
        else list(reversed(list_run_files()))
    )
    for run_path in paths:
        if not run_path.exists():
            log.warning("retro: run log not found: %s", run_path)
            continue
        sidecar = resolutions_sidecar_path(run_path)
        outcomes_by_slug: dict[str, ResolvedOutcome] = {}
        if sidecar.exists():
            with sidecar.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        outcome = ResolvedOutcome.model_validate_json(line)
                    except Exception:  # noqa: BLE001
                        continue
                    outcomes_by_slug[outcome.slug] = outcome
        for row in iter_predictions(run_path):
            if row.market_slug in seen_slugs:
                duplicates_dropped += 1
                continue
            seen_slugs.add(row.market_slug)
            outcome = outcomes_by_slug.get(row.market_slug)
            pm = (
                post_match_by_event.get(row.event_id)
                if post_match_by_event is not None
                else None
            )
            feats.append(extract_features(row, outcome, pm))
    if duplicates_dropped > 0:
        log.info(
            "retro: dedup'd %d duplicate prediction rows by market_slug "
            "(kept earliest)",
            duplicates_dropped,
        )
    return feats, duplicates_dropped


async def run_step_calibrate(
    run_id: str | None = None,
) -> CalibrateReport:
    """Step 2 — print hit-rate cuts + proper scoring metrics to stdout.
    Returns the report.

    Renders Brier / log-loss / ECE before and after the live committed
    calibration temperature (`models/tennis_calibration.json`); with no
    artefact committed the two are identical.

    Auto-runs Step 1 (`run_step_resolve`) first so the calibrate cuts
    always reflect the freshest gamma settlements without the operator
    needing a separate `--step resolve` invocation. Resolve is
    idempotent (skips slugs already settled in the sidecar), so this
    only fetches gamma for predictions that don't yet have a settled
    outcome cached. The standalone `--step resolve` command is still
    available for batch refresh without the calibrate render.

    Intentionally does NOT persist anything to disk: the terminal
    output is the artefact, and `--step all` folds the tables into
    the combined `report.md`. A previous version wrote a per-call
    `<retro_id>.calibrate.json` sidecar; the operator asked for it
    to be dropped because it cluttered `logs/retro/` without adding
    information not already in the markdown digest. If a
    machine-readable form is needed later, re-add `write_json=True`
    here and a `--write-json` flag on the CLI.

    Uses the same `collect_features` path Step 3 shares — keeps the
    feature-extraction definition single-source-of-truth. Both paths
    now dedup by `market_slug` (keeping the earliest prediction per
    market) so hit-rate cuts don't double-count rows that appear in
    multiple JSONL run files.
    """
    await run_step_resolve(run_id)
    feats, duplicates_dropped = (
        _features_with_post_match(run_id=run_id)
        if run_id is not None
        else collect_features()
    )
    # The live committed temperature, so the rendered metrics show
    # before/after what the current artefact would produce. Absent
    # artefact → 1.0 → "after" equals "before".
    temperature = load_temperature("tennis")
    report = aggregate(
        feats,
        n_duplicates_dropped=duplicates_dropped,
        temperature=temperature,
    )
    render_report(report)
    return report


async def run_step_fit_calibration(run_id: str | None = None) -> None:
    """Fit + commit the calibration temperature — the operator-gated
    artefact-writing step behind `skims retro --step fit-calibration`.

    Auto-resolves first (mirrors `run_step_calibrate`), collects every
    settled tennis `(predicted_prob, won)` pair, fits a single
    temperature by minimising NLL, and — only on a real fit — writes
    `models/tennis_calibration.json` with the before/after scorecard.

    Refuses (writes nothing, leaves any existing artefact intact) when
    there is too little data to trust a one-parameter fit; the operator
    sees a "refused — <reason>" line. Mirrors `skims gbt train`:
    operator-run, writes a committed artefact, never auto-triggered.

    The fit only ever sees `(predicted_prob, won)` — win/loss outcomes,
    never a market price — so calibration stays market-blind.
    """
    await run_step_resolve(run_id)
    feats, _ = (
        _features_with_post_match(run_id=run_id)
        if run_id is not None
        else collect_features()
    )
    pairs = [
        (f.predicted_prob, f.won)
        for f in feats
        if f.settled and f.won is not None and f.sport_type == "tennis"
    ]
    t = fit_temperature(pairs)
    if t is None:
        # Re-derive which guard tripped for an operator-facing message.
        n_pos = sum(1 for _, y in pairs if y)
        n_neg = len(pairs) - n_pos
        if len(pairs) < MIN_FIT_N:
            reason = (
                f"only {len(pairs)} settled tennis events, "
                f"need >= {MIN_FIT_N}"
            )
        else:
            reason = (
                f"need >= {MIN_CLASS_N} of each outcome class "
                f"(have {n_pos} wins, {n_neg} losses)"
            )
        log.warning("retro fit-calibration: refused — %s", reason)
        print(f"retro fit-calibration: refused — {reason}")
        return
    before = compute_metrics(pairs)
    after = compute_metrics([(apply_temperature(p, t), y) for p, y in pairs])
    write_calibration(
        {
            "tennis": {
                "temperature": t,
                "n": len(pairs),
                "fit_at_utc": datetime.now(UTC),
                "brier_before": before.brier,
                "brier_after": after.brier,
                "log_loss_before": before.log_loss,
                "log_loss_after": after.log_loss,
                "ece_before": before.ece,
                "ece_after": after.ece,
            }
        }
    )
    print(
        f"retro fit-calibration: tennis T={t:.4f} (n={len(pairs)})  "
        f"Brier {before.brier:.4f} → {after.brier:.4f}  "
        f"log-loss {before.log_loss:.4f} → {after.log_loss:.4f}  "
        f"ECE {before.ece:.4f} → {after.ece:.4f}  → {CALIBRATION_PATH}"
    )
    log.info(
        "retro fit-calibration: wrote tennis T=%.4f (n=%d) to %s",
        t, len(pairs), CALIBRATION_PATH,
    )


async def run_step_analyze(
    *,
    sports_filter: set[str] | None = None,
    run_id: str | None = None,
) -> Path:
    """Full retro pass — calibrate cuts + post-match fetch + LLM
    pattern call per sport, joined into one `report.md`.

    Inlines `run_step_calibrate` so the operator sees the hit-rate
    tables alongside the LLM findings without needing a second
    invocation. Calibrate auto-resolves at its start; this function
    re-runs resolve before the post-match fetch (idempotent — second
    pass is a no-op on already-settled markets) so a long-running
    analyze won't act on stale resolutions if the calibrate render
    completed minutes earlier.

    Returns the path to the written `report.md`. The `findings.json`
    sidecar is gone — the markdown digest is the canonical artefact;
    callers that want machine-readable findings can re-instantiate
    them via `analyze_all_sports` directly.
    """
    rid = _retro_id()
    calibrate_report = await run_step_calibrate(run_id=run_id)

    # Re-resolve before the post-match path — calibrate's resolve may
    # have run minutes ago and another match could have settled since.
    # Idempotent on already-settled rows, so the cost is one "all
    # already resolved" sweep when no new settlements landed.
    await run_step_resolve(run_id)
    config = cfg.Config.from_env()
    async with build_tennis_provider(config) as provider:
        post_match = await fetch_post_match_for_settled(provider)

    feats, _duplicates_dropped = _features_with_post_match(
        post_match_by_event=post_match, run_id=run_id
    )
    anthropic = AsyncAnthropic(api_key=config.anthropic_api_key)
    findings = await analyze_all_sports(anthropic, feats, sports_filter)

    md_path = _retro_root() / f"{rid}.report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    write_md_report(md_path, calibrate_report, findings)
    log.info("retro report written to %s", md_path)
    return md_path
