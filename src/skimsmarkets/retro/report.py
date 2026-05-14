"""Markdown digest combining Step 2 calibration tables and Step 3 LLM
findings into a single human-readable file. The CLI writes this when
`--step all` runs end-to-end so the operator has one place to read.

JSON outputs (`<retro_id>.calibrate.json`, `<retro_id>.findings.json`)
remain as the machine-readable substrate; this is the human view.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from skimsmarkets.retro.calibrate import CalibrateReport, CutTable
from skimsmarkets.retro.metrics import ScoringMetrics
from skimsmarkets.retro.models import RetroFindings

log = logging.getLogger(__name__)


def _md_table(cut: CutTable) -> str:
    """Render one cut as a Markdown pipe-table. Suppresses empty buckets
    to match the terminal renderer.
    """
    rows = ["| Bucket | Hits | Total | Hit rate |", "| --- | ---: | ---: | ---: |"]
    for b in cut.buckets:
        if b.total == 0:
            continue
        rate = f"{b.hit_rate:.1%}" if b.hit_rate is not None else "—"
        rows.append(f"| {b.label} | {b.hits} | {b.total} | {rate} |")
    return "\n".join(rows)


def _md_metric(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "—"


def _md_metric_cell(
    before: float | None, after: float | None, *, has_after: bool
) -> str:
    """A metric cell: just the value, or `before → after` when a
    calibration temperature was applied.
    """
    if not has_after:
        return _md_metric(before)
    return f"{_md_metric(before)} → {_md_metric(after)}"


def _md_metrics_table(
    scopes: list[tuple[str, ScoringMetrics, ScoringMetrics | None]],
    *,
    has_after: bool,
) -> str:
    """Brier / log-loss / ECE summary as a Markdown pipe-table, one row
    per scope (Overall + each sport). Cells show `before → after` when a
    calibration temperature was applied.
    """
    rows = [
        "| Scope | n | Brier | Log-loss | ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scope, before, after in scopes:
        brier = _md_metric_cell(
            before.brier, after.brier if after else None, has_after=has_after
        )
        ll = _md_metric_cell(
            before.log_loss,
            after.log_loss if after else None,
            has_after=has_after,
        )
        ece = _md_metric_cell(
            before.ece, after.ece if after else None, has_after=has_after
        )
        rows.append(f"| {scope} | {before.n} | {brier} | {ll} | {ece} |")
    return "\n".join(rows)


def _md_curve_table(metrics: ScoringMetrics) -> str:
    """One scope's calibration curve as a Markdown pipe-table. Trims the
    always-empty leading/trailing bins, keeps interior coverage gaps
    visible — mirrors the terminal `_render_curve_table`.
    """
    nonempty = [i for i, b in enumerate(metrics.curve) if b.n > 0]
    if not nonempty:
        return "_No settled events._"
    rows = [
        "| Bin | n | Mean pred | Observed | Gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for b in metrics.curve[nonempty[0] : nonempty[-1] + 1]:
        if b.n == 0:
            rows.append(f"| {b.lo:.2f}-{b.hi:.2f} | 0 | — | — | — |")
            continue
        gap = b.mean_predicted - b.observed_freq
        rows.append(
            f"| {b.lo:.2f}-{b.hi:.2f} | {b.n} | {b.mean_predicted:.1%} | "
            f"{b.observed_freq:.1%} | {gap:+.1%} |"
        )
    return "\n".join(rows)


def _metrics_section(calibrate: CalibrateReport) -> list[str]:
    """The "## Proper scoring metrics" section: a summary table plus a
    per-scope calibration curve. When the report carries an
    after-calibration pass the summary shows `before → after` and the
    section notes the committed temperature. Empty list when no metrics
    exist (a report built before `aggregate` ran).
    """
    has_after = calibrate.metrics_after_calibration is not None
    scopes: list[tuple[str, ScoringMetrics, ScoringMetrics | None]] = []
    if calibrate.overall_metrics is not None:
        scopes.append((
            "Overall",
            calibrate.overall_metrics,
            calibrate.metrics_after_calibration,
        ))
    for sport, m in calibrate.per_sport_metrics.items():
        scopes.append(
            (sport, m, calibrate.per_sport_metrics_after.get(sport))
        )
    if not scopes:
        return []
    lines = ["## Proper scoring metrics", ""]
    if has_after:
        lines.append(
            f"_Before → after the committed calibration temperature "
            f"T={calibrate.calibration_temperature:.4f}._"
        )
        lines.append("")
    lines.append(_md_metrics_table(scopes, has_after=has_after))
    lines.append("")
    # The calibration curve is the raw reliability diagram — always the
    # "before" metrics, mirroring the terminal renderer.
    for scope, before, _after in scopes:
        lines.append(f"### {scope} — calibration curve")
        lines.append("")
        lines.append(_md_curve_table(before))
        lines.append("")
    return lines


def _findings_md(findings: RetroFindings) -> str:
    lines = [
        f"### {findings.sport} — {findings.n_events} settled events "
        f"({findings.n_wins} wins, {findings.n_losses} losses)",
        "",
    ]
    if findings.n_wins == 0 or findings.n_losses == 0:
        lines.append(
            "_Skipped: differential pattern-finding needs both wins and "
            "losses._"
        )
        return "\n".join(lines)
    if findings.recurring_patterns:
        lines.append("**Recurring patterns overrepresented in losses:**")
        lines.append("")
        for p in findings.recurring_patterns:
            lines.append(f"- {p}")
        lines.append("")
    if findings.lens_underperformance:
        lines.append("**Lens attribution:**")
        lines.append("")
        for lu in findings.lens_underperformance:
            lines.append(f"- `{lu.lens_name}` — {lu.failure_mode}")
        lines.append("")
    if findings.prompt_recommendations:
        lines.append("**Prompt-edit recommendations (operator review):**")
        lines.append("")
        for r in findings.prompt_recommendations:
            lines.append(f"- {r}")
        lines.append("")
    return "\n".join(lines)


def write_report(
    out_path: Path,
    calibrate: CalibrateReport,
    findings_by_sport: dict[str, RetroFindings],
) -> None:
    """Write the combined report.md to `out_path`."""
    overall_rate = (
        f"{calibrate.n_correct / calibrate.n_settled:.1%}"
        if calibrate.n_settled > 0
        else "n/a"
    )
    dupe_note = (
        f" (dedup'd {calibrate.n_duplicates_dropped} duplicate "
        f"prediction{'s' if calibrate.n_duplicates_dropped != 1 else ''} "
        f"by market_slug — kept earliest per market)"
        if calibrate.n_duplicates_dropped > 0
        else ""
    )
    sections = [
        f"# Retro report — {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"**Overall:** {calibrate.n_settled} settled / "
        f"{calibrate.n_predictions_total} unique markets{dupe_note}, "
        f"{calibrate.n_correct} correct ({overall_rate})",
        "",
        *_metrics_section(calibrate),
        "## Step 2 — Hit-rate cuts",
        "",
        "### Overall",
        "",
    ]
    for cut in calibrate.overall_cuts:
        sections.append(f"#### {cut.name}")
        sections.append("")
        sections.append(_md_table(cut))
        sections.append("")
    for sport, cuts in calibrate.per_sport_cuts.items():
        sections.append(f"### {sport}")
        sections.append("")
        for cut in cuts:
            sections.append(f"#### {cut.name}")
            sections.append("")
            sections.append(_md_table(cut))
            sections.append("")
    if findings_by_sport:
        sections.append("## Step 3 — LLM pattern findings")
        sections.append("")
        for sport in sorted(findings_by_sport):
            sections.append(_findings_md(findings_by_sport[sport]))
            sections.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections))
    log.info("retro report written to %s", out_path)
