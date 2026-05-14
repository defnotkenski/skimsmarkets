"""Step 2 — deterministic hit-rate aggregation.

Walks every prediction row + its resolution sidecar, joins them via
(run_id, slug), runs `extract_features`, then bins by three cuts:
Case bucket, Confidence tier, and Market-favorite-vs-underdog. No LLM.

Renders rich.Table to stdout and writes the same numbers to JSON.
Drops unsettled events from the denominator — they're not informative
for hit-rate questions. Per-sport breakdown plus an overall aggregate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from skimsmarkets.calibration import apply_temperature
from skimsmarkets.retro.features import extract_features
from skimsmarkets.retro.jsonl import (
    iter_predictions,
    list_run_files,
    resolutions_sidecar_path,
)
from skimsmarkets.retro.metrics import ScoringMetrics, compute_metrics
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
    table).

    `n_predictions_total` counts UNIQUE markets after the dedup pass
    in `collect_features` — same `market_slug` predicted across
    multiple runs collapses to one row (the chronologically earliest
    instance wins). `n_duplicates_dropped` carries the count of
    same-slug rows the dedup discarded so the operator can see how
    load-bearing dedup was on this particular run.
    """

    n_predictions_total: int = 0
    n_settled: int = 0
    n_correct: int = 0
    n_duplicates_dropped: int = 0
    overall_cuts: list[CutTable] = field(default_factory=list)
    per_sport_cuts: dict[str, list[CutTable]] = field(default_factory=dict)
    # Proper-scoring metrics (Brier / log-loss / ECE / calibration curve)
    # over the same settled rows the hit-rate cuts use. `overall_metrics`
    # is None only on a report built before `aggregate` ran (it always
    # populates it, even to an n=0 scorecard).
    overall_metrics: ScoringMetrics | None = None
    per_sport_metrics: dict[str, ScoringMetrics] = field(default_factory=dict)
    # Before/after-calibration comparison. Populated only when `aggregate`
    # is called with a `temperature` (the live committed scalar) — `--step
    # calibrate` passes it; Phase-1-only callers leave these at defaults.
    # The `*_after` fields mirror the `*_metrics` shape, recomputed on the
    # raw probabilities pushed through `apply_temperature`.
    calibration_temperature: float | None = None
    metrics_after_calibration: ScoringMetrics | None = None
    per_sport_metrics_after: dict[str, ScoringMetrics] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "n_predictions_total": self.n_predictions_total,
            "n_settled": self.n_settled,
            "n_correct": self.n_correct,
            "n_duplicates_dropped": self.n_duplicates_dropped,
            "overall_cuts": [c.to_dict() for c in self.overall_cuts],
            "per_sport_cuts": {
                sport: [c.to_dict() for c in cuts]
                for sport, cuts in self.per_sport_cuts.items()
            },
            "overall_metrics": (
                self.overall_metrics.to_dict()
                if self.overall_metrics is not None
                else None
            ),
            "per_sport_metrics": {
                sport: m.to_dict()
                for sport, m in self.per_sport_metrics.items()
            },
            "calibration_temperature": self.calibration_temperature,
            "metrics_after_calibration": (
                self.metrics_after_calibration.to_dict()
                if self.metrics_after_calibration is not None
                else None
            ),
            "per_sport_metrics_after": {
                sport: m.to_dict()
                for sport, m in self.per_sport_metrics_after.items()
            },
        }


# Cut definitions live as ordered (label, predicate) pairs so the
# render order is stable across reports — alphabetical sort would
# scramble Case 1 below Case 5 etc.

# Labels deliberately ASCII-only — keeps cell widths predictable
# regardless of terminal font.
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


_NEGATIVE_EDGE_ORDER: list[str] = [
    "negative edge",
    "non-negative edge",
    "no market price",
]


def _negative_edge_label(flag: bool | None) -> str:
    if flag is None:
        return "no market price"
    return "negative edge" if flag else "non-negative edge"


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


def aggregate(
    feats: list[EventFeatures],
    *,
    n_duplicates_dropped: int = 0,
    temperature: float | None = None,
) -> CalibrateReport:
    """Build the full report from a flat feature list.

    `n_duplicates_dropped` is plumbed through from `collect_features`
    (the dedup happens there, not here) so the report can surface
    "we dropped N duplicate predictions" alongside the deduped totals.

    `temperature`, when passed, is the live committed calibration scalar:
    the report then also carries the "after-calibration" scoring metrics,
    recomputed by pushing every raw probability through
    `apply_temperature`. None (the default) skips that pass — Phase-1
    callers that only want the raw metrics pass nothing.
    """
    report = CalibrateReport(
        n_predictions_total=len(feats),
        n_duplicates_dropped=n_duplicates_dropped,
    )
    settled = [f for f in feats if f.settled and f.won is not None]
    report.n_settled = len(settled)
    report.n_correct = sum(1 for f in settled if f.won is True)
    # Proper-scoring metrics over the same settled rows the cuts use, so
    # the headline Brier / ECE and the hit-rate tables can't disagree
    # about which events they're describing.
    report.overall_metrics = compute_metrics(
        [(f.predicted_prob, f.won) for f in settled]
    )
    # Before/after the live committed temperature, when one was passed.
    # "After" is recomputed on the fly — every raw probability pushed
    # through `apply_temperature` — so no persisted feature is needed.
    if temperature is not None:
        report.calibration_temperature = temperature
        report.metrics_after_calibration = compute_metrics(
            [
                (apply_temperature(f.predicted_prob, temperature), f.won)
                for f in settled
            ]
        )

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
            _aggregate_one_cut(
                scope,
                "Negative-edge vs non-negative-edge pick",
                list(_NEGATIVE_EDGE_ORDER),
                lambda f: _negative_edge_label(f.negative_edge),
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
        sport_settled = [
            f for f in scope if f.settled and f.won is not None
        ]
        report.per_sport_metrics[sport] = compute_metrics(
            [(f.predicted_prob, f.won) for f in sport_settled]
        )
        if temperature is not None:
            report.per_sport_metrics_after[sport] = compute_metrics(
                [
                    (apply_temperature(f.predicted_prob, temperature), f.won)
                    for f in sport_settled
                ]
            )
    return report


def collect_features() -> tuple[list[EventFeatures], int]:
    """Walk every run log + sidecar; return `(deduped_features, n_dropped)`.

    Joins prediction rows to their resolution by (sidecar slug). When
    a sidecar is missing for a run, predictions from that run come
    through with `settled=False` (and are dropped from hit-rate cuts
    anyway). Step 3 fetches its own post-match data, so this function
    deliberately returns features WITHOUT post-match divergence
    populated — Step 2 doesn't use it.

    **Dedup**: same `market_slug` can appear across multiple runs (the
    operator runs `skims rank` repeatedly before a match settles, all
    snapshots land in `logs/runs/`). Counting each instance in the
    hit-rate cuts double-counts settled outcomes. We dedup by
    `market_slug` keeping the chronologically EARLIEST prediction —
    the most-defensible anchor for hit-rate calibration because it
    was made with the least market-price drift since prediction time.
    `list_run_files` sorts mtime-DESCENDING (newest first), so we
    iterate `reversed(...)` to walk oldest → newest, and the first
    occurrence of each slug wins the dedup race.

    Returned tuple's second element is the count of rows dropped by
    dedup, plumbed through to `aggregate(...)` so the report can
    surface it alongside the deduped totals.
    """
    feats: list[EventFeatures] = []
    seen_slugs: set[str] = set()
    duplicates_dropped = 0
    # Oldest-first so the earliest prediction for each market wins
    # dedup. `list_run_files` returns mtime-descending; reverse for
    # ascending traversal without re-sorting.
    for run_path in reversed(list_run_files()):
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
            if row.market_slug in seen_slugs:
                duplicates_dropped += 1
                continue
            seen_slugs.add(row.market_slug)
            outcome = outcomes_by_slug.get(row.market_slug)
            feats.append(extract_features(row, outcome, post_match=None))
    if duplicates_dropped > 0:
        log.info(
            "calibrate: dedup'd %d duplicate prediction rows by market_slug "
            "(kept earliest)",
            duplicates_dropped,
        )
    return feats, duplicates_dropped


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
        # (e.g. "5: 0/0" carries no signal and squeezes the cells
        # that DO have data).
        if b.total == 0:
            continue
        table.add_row(b.label, str(b.hits), str(b.total), rate)
    console.print(table)


def _fmt_metric(x: float | None) -> str:
    """4-decimal metric cell, or the same em-dash the cut tables use for
    a missing value (no settled events in scope).
    """
    return f"{x:.4f}" if x is not None else "—"


def _metric_cell(
    before: float | None, after: float | None, *, has_after: bool
) -> str:
    """A metric cell: just the value, or `before → after` when a
    calibration temperature was applied.
    """
    if not has_after:
        return _fmt_metric(before)
    return f"{_fmt_metric(before)} → {_fmt_metric(after)}"


def _render_metrics_summary(console: Console, report: CalibrateReport) -> None:
    """One table: Brier / log-loss / ECE for Overall + each sport.

    These are the headline numbers — they say *how miscalibrated* the
    probabilities are, which hit-rate alone can't. Rendered before the
    cut tables for that reason. When the report carries an
    after-calibration pass, each cell shows `before → after`.
    """
    has_after = report.metrics_after_calibration is not None
    title = "Proper scoring metrics"
    if has_after:
        title += (
            f"  (before → after, T={report.calibration_temperature:.4f})"
        )
    table = Table(title=title, title_justify="left", show_lines=False)
    table.add_column("Scope")
    table.add_column("n", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("Log-loss", justify="right")
    table.add_column("ECE", justify="right")
    scopes: list[
        tuple[str, ScoringMetrics | None, ScoringMetrics | None]
    ] = [("Overall", report.overall_metrics, report.metrics_after_calibration)]
    for sport, m in report.per_sport_metrics.items():
        scopes.append((sport, m, report.per_sport_metrics_after.get(sport)))
    for scope, before, after in scopes:
        if before is None:
            continue
        table.add_row(
            scope,
            str(before.n),
            _metric_cell(
                before.brier, after.brier if after else None,
                has_after=has_after,
            ),
            _metric_cell(
                before.log_loss, after.log_loss if after else None,
                has_after=has_after,
            ),
            _metric_cell(
                before.ece, after.ece if after else None,
                has_after=has_after,
            ),
        )
    console.print(table)


def _render_curve_table(
    console: Console, scope: str, metrics: ScoringMetrics
) -> None:
    """Reliability diagram for one scope — predicted vs observed per bin.

    Trims the always-empty leading/trailing bins (predictions cluster in
    [0.5, 1.0] under the contrarian-call discipline, so the low bins
    never fill) but keeps *interior* empty bins visible — those are real
    coverage gaps, not noise.
    """
    nonempty = [i for i, b in enumerate(metrics.curve) if b.n > 0]
    if not nonempty:
        return
    table = Table(
        title=f"{scope} — calibration curve",
        title_justify="left",
        show_lines=False,
    )
    table.add_column("Bin")
    table.add_column("n", justify="right")
    table.add_column("Mean pred", justify="right")
    table.add_column("Observed", justify="right")
    table.add_column("Gap", justify="right")
    for b in metrics.curve[nonempty[0] : nonempty[-1] + 1]:
        if b.n == 0:
            table.add_row(f"{b.lo:.2f}-{b.hi:.2f}", "0", "—", "—", "—")
            continue
        gap = b.mean_predicted - b.observed_freq
        table.add_row(
            f"{b.lo:.2f}-{b.hi:.2f}",
            str(b.n),
            f"{b.mean_predicted:.1%}",
            f"{b.observed_freq:.1%}",
            f"{gap:+.1%}",
        )
    console.print(table)


def render_report(report: CalibrateReport) -> None:
    console = Console()
    overall_rate = (
        f"{report.n_correct / report.n_settled:.1%}"
        if report.n_settled > 0
        else "n/a"
    )
    dupe_note = (
        f" (dedup'd {report.n_duplicates_dropped} duplicate "
        f"prediction{'s' if report.n_duplicates_dropped != 1 else ''})"
        if report.n_duplicates_dropped > 0
        else ""
    )
    console.print(
        f"[bold]Retro calibrate[/bold] — "
        f"{report.n_settled} settled / {report.n_predictions_total} unique markets"
        f"{dupe_note}, {report.n_correct} correct ({overall_rate})"
    )
    console.print()
    _render_metrics_summary(console, report)
    console.print()
    if report.overall_metrics is not None:
        _render_curve_table(console, "Overall", report.overall_metrics)
        console.print()
    for sport, m in report.per_sport_metrics.items():
        _render_curve_table(console, sport, m)
        console.print()
    for cut in report.overall_cuts:
        _render_cut_table(console, "Overall", cut)
        console.print()
    for sport, cuts in report.per_sport_cuts.items():
        for cut in cuts:
            _render_cut_table(console, sport, cut)
            console.print()


