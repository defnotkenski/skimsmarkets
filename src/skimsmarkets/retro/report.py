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
