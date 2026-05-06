"""Step 2 — deterministic hit-rate aggregation.

Walks every prediction row + its resolution sidecar, joins them via
(run_id, slug), runs `extract_features`, then bins by three cuts:
Case bucket, Confidence tier, and Market-favorite-vs-underdog. No LLM.

Renders rich.Table to stdout and writes the same numbers to JSON.
Drops unsettled events from the denominator — they're not informative
for hit-rate questions. Per-sport breakdown plus an overall aggregate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from skimsmarkets.retro.features import extract_features
from skimsmarkets.retro.jsonl import (
    iter_predictions,
    list_run_files,
    resolutions_sidecar_path,
)
from skimsmarkets.retro.models import EventFeatures, ResolvedOutcome

log = logging.getLogger(__name__)


@dataclass
class _Bucket:
    """Hit-count + total for one bucket cell. Hit rate computed lazily."""

    label: str
    hits: int = 0
    total: int = 0

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.total if self.total > 0 else None


@dataclass
class CutTable:
    """One cut's worth of buckets, in the order they should render."""

    name: str
    buckets: list[_Bucket]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "buckets": [
                {
                    "label": b.label,
                    "hits": b.hits,
                    "total": b.total,
                    "hit_rate": b.hit_rate,
                }
                for b in self.buckets
            ],
        }


@dataclass
class CalibrateReport:
    """Top-level shape: an Overall section + per-sport sections, each
    carrying the three cut tables. Sports without any settled events
    are omitted from the per-sport block (no point rendering an empty
    table)."""

    n_predictions_total: int = 0
    n_settled: int = 0
    n_correct: int = 0
    overall_cuts: list[CutTable] = field(default_factory=list)
    per_sport_cuts: dict[str, list[CutTable]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_predictions_total": self.n_predictions_total,
            "n_settled": self.n_settled,
            "n_correct": self.n_correct,
            "overall_cuts": [c.to_dict() for c in self.overall_cuts],
            "per_sport_cuts": {
                sport: [c.to_dict() for c in cuts]
                for sport, cuts in self.per_sport_cuts.items()
            },
        }


# Cut definitions live as ordered (label, predicate) pairs so the
# render order is stable across reports — alphabetical sort would
# scramble Case 1🔥 below Case 5🔥 etc.

# Labels deliberately ASCII-only. The leaderboard renders 🔥 emojis in
# a dedicated column with fixed-width padding to compensate for font
# fallback drift; embedding the emoji inline in a label string would
# break that trick (rich computes wcwidth=2 for 🔥 but most terminal
# fonts render it at 1 or 1.5 cells, shifting the right border of any
# cell that contains it). ASCII labels render at predictable widths
# everywhere.
_CASE_ORDER: list[str] = [
    "5 (>=0.85)",
    "4 (>=0.65)",
    "3 (>=0.45)",
    "2 (>=0.25)",
    "1 (<0.25)",
    "no judge",
]


def _case_label(bucket: int | None) -> str:
    return {
        5: "5 (>=0.85)",
        4: "4 (>=0.65)",
        3: "3 (>=0.45)",
        2: "2 (>=0.25)",
        1: "1 (<0.25)",
        None: "no judge",
    }[bucket]


_CONFIDENCE_ORDER: list[str] = ["high", "medium", "low"]


_FAVORITE_ORDER: list[str] = ["favorite", "underdog", "no market price"]


def _favorite_label(flag: bool | None) -> str:
    if flag is None:
        return "no market price"
    return "favorite" if flag else "underdog"


def _aggregate_one_cut(
    feats: Iterable[EventFeatures],
    name: str,
    order: list[str],
    label_for: callable,  # noqa: ANN001 — callable arity varies by cut
) -> CutTable:
    buckets = {label: _Bucket(label=label) for label in order}
    for f in feats:
        if not f.settled or f.won is None:
            continue
        label = label_for(f)
        if label not in buckets:
            # Defensive: if extract_features ever produces a label
            # outside the static order, include it in the table without
            # crashing. Keeps the renderer honest about real data.
            buckets[label] = _Bucket(label=label)
            order.append(label)
        b = buckets[label]
        b.total += 1
        if f.won:
            b.hits += 1
    return CutTable(name=name, buckets=[buckets[label] for label in order])


def aggregate(feats: list[EventFeatures]) -> CalibrateReport:
    """Build the full report from a flat feature list."""
    report = CalibrateReport(n_predictions_total=len(feats))
    settled = [f for f in feats if f.settled and f.won is not None]
    report.n_settled = len(settled)
    report.n_correct = sum(1 for f in settled if f.won is True)

    def _build_cuts(scope: list[EventFeatures]) -> list[CutTable]:
        return [
            _aggregate_one_cut(
                scope,
                "Case bucket vs hit rate",
                list(_CASE_ORDER),
                lambda f: _case_label(f.case_bucket),
            ),
            _aggregate_one_cut(
                scope,
                "Confidence tier vs hit rate",
                list(_CONFIDENCE_ORDER),
                lambda f: f.confidence,
            ),
            _aggregate_one_cut(
                scope,
                "Market-favorite vs underdog pick",
                list(_FAVORITE_ORDER),
                lambda f: _favorite_label(f.market_favorite_pick),
            ),
        ]

    report.overall_cuts = _build_cuts(feats)

    # Per-sport. Sport=None rolls into "unknown" rather than getting
    # silently dropped — surfaces any pre-sport-tagging rows in the
    # historical log.
    by_sport: dict[str, list[EventFeatures]] = {}
    for f in feats:
        sport = f.sport_type or "unknown"
        by_sport.setdefault(sport, []).append(f)
    for sport in sorted(by_sport):
        scope = by_sport[sport]
        # Only render per-sport cuts for sports that have at least one
        # settled event — empty tables clutter the output.
        if not any(f.settled for f in scope):
            continue
        report.per_sport_cuts[sport] = _build_cuts(scope)
    return report


def collect_features() -> list[EventFeatures]:
    """Walk every run log + sidecar and produce the flat feature list.

    Joins prediction rows to their resolution by (sidecar slug). When
    a sidecar is missing for a run, predictions from that run come
    through with `settled=False` (and are dropped from hit-rate cuts
    anyway). Step 3 fetches its own post-match data, so this function
    deliberately returns features WITHOUT post-match divergence
    populated — Step 2 doesn't use it.
    """
    feats: list[EventFeatures] = []
    for run_path in list_run_files():
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
                    except Exception as e:  # noqa: BLE001
                        log.warning(
                            "calibrate: skipping malformed sidecar line in %s: %s",
                            sidecar.name, e,
                        )
                        continue
                    outcomes_by_slug[outcome.slug] = outcome
        for row in iter_predictions(run_path):
            outcome = outcomes_by_slug.get(row.market_slug)
            feats.append(extract_features(row, outcome, post_match=None))
    return feats


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_cut_table(console: Console, scope: str, cut: CutTable) -> None:
    table = Table(
        title=f"{scope} — {cut.name}",
        title_justify="left",
        show_lines=False,
    )
    table.add_column("Bucket")
    table.add_column("Hits", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Hit rate", justify="right")
    for b in cut.buckets:
        rate = (
            f"{b.hit_rate:.1%}"
            if b.hit_rate is not None
            else "—"
        )
        # Suppress empty-bucket rows — they're noise in the report
        # (e.g. "5 🔥: 0/0" carries no signal and squeezes the cells
        # that DO have data).
        if b.total == 0:
            continue
        table.add_row(b.label, str(b.hits), str(b.total), rate)
    console.print(table)


def render_report(report: CalibrateReport) -> None:
    console = Console()
    overall_rate = (
        f"{report.n_correct / report.n_settled:.1%}"
        if report.n_settled > 0
        else "n/a"
    )
    console.print(
        f"[bold]Retro calibrate[/bold] — "
        f"{report.n_settled} settled / {report.n_predictions_total} total, "
        f"{report.n_correct} correct ({overall_rate})"
    )
    console.print()
    for cut in report.overall_cuts:
        _render_cut_table(console, "Overall", cut)
        console.print()
    for sport, cuts in report.per_sport_cuts.items():
        for cut in cuts:
            _render_cut_table(console, sport, cut)
            console.print()


def write_report(report: CalibrateReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2))
