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
from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic

from skimsmarkets import config as cfg
from skimsmarkets.retro.analyze import analyze_all_sports
from skimsmarkets.retro.calibrate import (
    CalibrateReport,
    aggregate,
    collect_features,
    render_report,
    write_report,
)
from skimsmarkets.retro.features import extract_features
from skimsmarkets.retro.jsonl import (
    iter_predictions,
    list_run_files,
    log_root,
    resolutions_sidecar_path,
    run_path_for_id,
)
from skimsmarkets.retro.models import (
    EventFeatures,
    ResolvedOutcome,
    RetroFindings,
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
) -> list[EventFeatures]:
    """Build the feature list, optionally joining post-match data.

    `post_match_by_event` is keyed by `event_id`; when None, features
    have no post-match divergence populated (Step 2 doesn't need it).
    `run_id` filters to a single run (used by `--run-id` invocations).
    """
    feats: list[EventFeatures] = []
    paths = (
        [run_path_for_id(run_id)]
        if run_id is not None
        else list_run_files()
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
            outcome = outcomes_by_slug.get(row.market_slug)
            pm = (
                post_match_by_event.get(row.event_id)
                if post_match_by_event is not None
                else None
            )
            feats.append(extract_features(row, outcome, pm))
    return feats


def run_step_calibrate(
    run_id: str | None = None, retro_id: str | None = None
) -> tuple[CalibrateReport, Path]:
    """Step 2 — print + write hit-rate tables. Returns (report, json_path).

    Uses the same `collect_features` path Step 3 shares — keeps the
    feature-extraction definition single-source-of-truth.
    """
    feats = (
        _features_with_post_match(run_id=run_id)
        if run_id is not None
        else collect_features()
    )
    report = aggregate(feats)
    render_report(report)
    rid = retro_id or _retro_id()
    out_path = _retro_root() / f"{rid}.calibrate.json"
    write_report(report, out_path)
    log.info("retro calibrate written to %s", out_path)
    return report, out_path


async def run_step_analyze(
    *,
    sports_filter: set[str] | None = None,
    run_id: str | None = None,
    retro_id: str | None = None,
) -> tuple[dict[str, RetroFindings], Path]:
    """Step 3 — fetch post-match stats + run LLM pattern call per sport.

    Returns `(findings_by_sport, json_path)`. Calibration is also
    re-aggregated internally with post-match-enriched features so the
    Step 3 LLM call sees divergence columns; the JSON output for
    Step 2 still uses non-post-match features (they're the same
    numbers — divergence doesn't change hit-rate cuts).
    """
    config = cfg.Config.from_env()
    async with build_tennis_provider(config) as provider:
        post_match = await fetch_post_match_for_settled(provider)

    feats = _features_with_post_match(
        post_match_by_event=post_match, run_id=run_id
    )
    anthropic = AsyncAnthropic(api_key=config.anthropic_api_key)
    findings = await analyze_all_sports(anthropic, feats, sports_filter)

    rid = retro_id or _retro_id()
    out_path = _retro_root() / f"{rid}.findings.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "[\n"
        + ",\n".join(f.model_dump_json(indent=2) for f in findings.values())
        + "\n]"
    )
    log.info("retro findings written to %s", out_path)
    return findings, out_path


async def run_step_all(
    *,
    sports_filter: set[str] | None = None,
    run_id: str | None = None,
) -> Path:
    """Steps 1 → 2 → 3, then write the combined `report.md`.

    Returns the report.md path so the CLI can print "open this file".
    """
    rid = _retro_id()
    await run_step_resolve(run_id)
    calibrate_report, _ = run_step_calibrate(run_id=run_id, retro_id=rid)
    findings, _ = await run_step_analyze(
        sports_filter=sports_filter, run_id=run_id, retro_id=rid,
    )
    md_path = _retro_root() / f"{rid}.report.md"
    write_md_report(md_path, calibrate_report, findings)
    return md_path
