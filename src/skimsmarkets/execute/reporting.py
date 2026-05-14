"""Rich-backed live display + final summary for `skims execute`.

Two pieces:

  - `ExecuteDisplay` — async context manager wrapping a single
    `rich.live.Live` whose renderable is a `Group(Progress, Table)`.
    The Progress shows the four pre-flight phases (load / filter /
    exposure / events); the Table starts empty and grows one row per
    filtered prediction as each trade resolves. Single Live keeps the
    two regions cooperating cleanly — no context-manager juggling.

  - `print_execute_summary` — final Rich panel that replaces the bare
    `print(f"...")` line previously at the end of `_cmd_execute`.

Both render to stdout via the same `rich.console.Console`. Logging
goes to stderr (see `cli._setup_logging`), so the display and the
existing `log.info` / `log.warning` breadcrumbs don't fight.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Self

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from skimsmarkets.retro.models import PredictionRow, TradeRow

if TYPE_CHECKING:
    # `ExecuteSummary` lives in `execute.trader`, which imports this
    # module — TYPE_CHECKING avoids the circular import while keeping
    # the annotation precise for IDE/static analysis.
    from skimsmarkets.execute.trader import ExecuteSummary

# Pastel palette — matches reporting.py for visual consistency across
# `skims rank` and `skims execute`.
_MINT = "#a8e6cf"
_ROSE = "#ffaaa5"
_PEACH = "#ffd3b6"
_SKY = "#a8dadc"
_LAVENDER = "#d4a5e8"
_DIM = "#b0b0b0"
_CREAM = "#fff3b0"

_PHASE_LABELS: dict[str, str] = {
    "load": "Loading predictions",
    "filter": "Applying filters",
    "exposure": "Fetching Kalshi exposure",
    "events": "Fetching Kalshi events",
}

# Maps the wire `fill_status` (Literal on TradeRow) to (rich-style,
# display-label) tuples. `skipped_dry_run` → "DRY-RUN" because the audit
# value has an underscore that reads badly in a 10-char column.
_STATUS_DISPLAY: dict[str, tuple[str, str]] = {
    "filled": (f"bold {_MINT}", "FILLED"),
    "partial": (f"bold {_PEACH}", "PARTIAL"),
    "submitted": (f"bold {_LAVENDER}", "SUBMITTED"),
    "skipped_dry_run": (f"bold {_DIM}", "DRY-RUN"),
    "skipped": (f"bold {_ROSE}", "SKIPPED"),
}

_CONF_STYLE: dict[str, str] = {
    "high": _MINT,
    "medium": _PEACH,
    "low": _ROSE,
}


def _format_result(audit: TradeRow) -> str:
    """Render the `Result` cell text for one resolved trade row.

    Filled / partial: `$X.XX @ Y.YYY` (cost + average fill price in
    dollars). Submitted: `pending fill`. Dry-run: `would buy @ ask`
    when ask is known, else `dry-run`. Skipped: the audit's
    `skip_reason` (e.g. `no_kalshi_match`, `exposure_cap_exceeded`).
    """
    if audit.fill_status in ("filled", "partial"):
        cost = audit.fill_total_cost_cents / 100
        if audit.fill_avg_price_cents:
            avg = audit.fill_avg_price_cents / 100
            return f"${cost:.2f} @ {avg:.3f}"
        return f"${cost:.2f}"
    if audit.fill_status == "submitted":
        return "pending fill"
    if audit.fill_status == "skipped_dry_run":
        ask = audit.kalshi_yes_ask_dollars_at_decision
        if ask is not None:
            return f"would buy @ {ask:.3f}"
        return "dry-run"
    return audit.skip_reason or "skipped"


class ExecuteDisplay(AbstractAsyncContextManager["ExecuteDisplay"]):
    """Unified live display for `skims execute`.

    Two regions in one `rich.live.Live`, stacked top-to-bottom:

      1. `Progress` with four pre-flight phases (load / filter /
         exposure / events). Bars fill as `start_phase` / `complete_phase`
         fire during `run_execute`.
      2. `Table` of trades. Seeded with one row per filtered prediction
         via `add_pending`; each row updates in place via `update_trade`
         when the trade resolves.

    Use as an async context manager:

        async with ExecuteDisplay() as display:
            summary, exposure = await run_execute(
                opts, config=config, display=display,
            )

    `None` is a valid `display` argument anywhere in `trader.py` — the
    pipeline guards each call with `if display is not None`, so non-CLI
    callers (tests, future automation) work without instantiating one.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        # `transient=False` on the outer `Live` keeps the rendered state
        # visible after the context exits — the static table + finished
        # bars stay on screen while `print_execute_summary` renders the
        # panel below them.
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        self._phase_tasks: dict[str, int] = {}
        # Each entry: {event_id, event_title, pick, conf, status (None
        # until resolved), result}. The status `None` sentinel renders
        # as a dim `...` so the user can see queued rows distinctly
        # from resolved-but-edge-case rows.
        self._rows: list[dict[str, str | None]] = []
        self._live: Live | None = None

    async def __aenter__(self) -> Self:
        # `refresh_per_second=2` mirrors `ProgressReporter` (rank). The
        # default 10 Hz causes every spinner-tick redraw to interleave
        # with stderr log writes (logging's StreamHandler bypasses
        # Rich's Live redirect because it caches sys.stderr at config
        # time). 2 Hz keeps the bars + table responsive without
        # flooding scrollback between log lines.
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=2,
            transient=False,
        )
        self._live.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.stop()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Group:
        """Build the current renderable: progress bars on top, trades
        table below (when any rows have been added)."""
        if self._rows:
            return Group(self._progress, self._build_table())
        return Group(self._progress)

    def _build_table(self) -> Table:
        table = Table(
            title=f"[{_CREAM}]Trades[/]",
            title_justify="left",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            header_style=_LAVENDER,
        )
        table.add_column(
            "Event", style=_SKY, min_width=32, overflow="fold"
        )
        table.add_column(
            "Pick", style=f"bold {_LAVENDER}", min_width=10, overflow="fold"
        )
        table.add_column("Conf", justify="center", min_width=6)
        table.add_column("Status", justify="center", min_width=10)
        table.add_column(
            "Result", style=_DIM, overflow="fold", min_width=24
        )
        for r in self._rows:
            conf = r["conf"] or ""
            conf_style = _CONF_STYLE.get(conf, _DIM)
            conf_cell = f"[{conf_style}]{conf}[/]" if conf else "—"
            status = r["status"]
            if status is None:
                status_cell = f"[{_DIM}]...[/]"
            else:
                style, label = _STATUS_DISPLAY.get(
                    status, (_DIM, status.upper())
                )
                status_cell = f"[{style}]{label}[/]"
            table.add_row(
                r["event_title"] or "",
                r["pick"] or "",
                conf_cell,
                status_cell,
                r["result"] or "",
            )
        return table

    # ── Progress API ───────────────────────────────────────────────
    def start_phase(self, phase: str, total: int = 1) -> None:
        """Begin a pre-flight phase. Idempotent — second call is a
        no-op, so the pipeline can call it inside any of several entry
        points without branching."""
        if phase in self._phase_tasks:
            return
        label = _PHASE_LABELS.get(phase, phase)
        self._phase_tasks[phase] = self._progress.add_task(label, total=total)
        self._refresh()

    def complete_phase(self, phase: str) -> None:
        """Snap a phase to its total so it renders as 100%."""
        if phase not in self._phase_tasks:
            return
        task_id = self._phase_tasks[phase]
        total = self._progress.tasks[task_id].total or 1
        self._progress.update(task_id, completed=total)
        self._refresh()

    # ── Trades API ─────────────────────────────────────────────────
    def add_pending(self, row: PredictionRow) -> None:
        """Append a row showing this prediction as queued. Called
        before the trade loop so the user sees the full pipeline of
        trades that are coming."""
        self._rows.append(
            {
                "event_id": row.event_id,
                "event_title": row.event_title or row.event_id,
                "pick": row.predicted_winner,
                "conf": row.confidence,
                "status": None,
                "result": "queued",
            }
        )
        self._refresh()

    def update_trade(self, row: PredictionRow, audit: TradeRow) -> None:
        """Update the pending row for `row.event_id` to reflect the
        trade outcome. If no pending row exists (shouldn't happen in
        practice — `add_pending` is called for every filtered row),
        appends a fresh row so nothing is lost."""
        for r in self._rows:
            if r["event_id"] == row.event_id:
                r["status"] = audit.fill_status
                r["result"] = _format_result(audit)
                self._refresh()
                return
        # Defensive fallback — should not fire.
        self._rows.append(
            {
                "event_id": row.event_id,
                "event_title": row.event_title or row.event_id,
                "pick": row.predicted_winner,
                "conf": row.confidence,
                "status": audit.fill_status,
                "result": _format_result(audit),
            }
        )
        self._refresh()


def print_execute_summary(
    summary: ExecuteSummary,
    *,
    open_exposure_cents: int | None = None,
    max_open_exposure_cents: int | None = None,
    console: Console | None = None,
) -> None:
    """Final summary panel printed after the live display exits.

    Replaces the previous bare `print(f"...")` line in `_cmd_execute`.
    Highlights total $ committed and breaks down skip reasons.
    `open_exposure_cents` + `max_open_exposure_cents` are optional —
    when provided, the panel surfaces the post-run exposure utilisation
    next to the dollar total. Both default to None so the function
    stays usable in test contexts that synthesize an `ExecuteSummary`
    without Kalshi-side state.
    """
    console = console or Console()

    committed_dollars = summary.total_filled_cost_cents / 100
    exposure_note = ""
    if (
        open_exposure_cents is not None
        and max_open_exposure_cents is not None
    ):
        exposure_dollars = open_exposure_cents / 100
        cap_dollars = max_open_exposure_cents / 100
        exposure_note = (
            f"  [{_DIM}](exposure: ${exposure_dollars:,.0f} / "
            f"${cap_dollars:,.0f} cap)[/]"
        )

    lines = [
        f"  [{_SKY}]predictions:[/] {summary.total_predictions}    "
        f"[{_SKY}]passed filters:[/] {summary.passed_filters}",
        "",
        f"  [{_MINT}]filled:[/] {summary.filled}     "
        f"[{_PEACH}]partial:[/] {summary.partial}     "
        f"[{_LAVENDER}]submitted:[/] {summary.submitted}     "
        f"[{_DIM}]dry-run:[/] {summary.skipped_dry_run}     "
        f"[{_ROSE}]skipped:[/] {summary.skipped}",
        "",
        f"  [{_CREAM}]total committed:[/] "
        f"[bold {_MINT}]${committed_dollars:,.2f}[/]"
        f"{exposure_note}",
    ]
    if summary.skip_reasons:
        lines.append("")
        lines.append(f"  [{_SKY}]skip reasons:[/]")
        # Sort by descending count, then alpha, so the dominant skip
        # reason leads.
        sorted_reasons = sorted(
            summary.skip_reasons.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        # Pad after the colon so the counts line up. (Padding inside
        # the name pushed the colon away from short names and looked
        # ragged.)
        max_name_with_colon = max(len(r) for r, _ in sorted_reasons) + 1
        for reason, count in sorted_reasons:
            label = f"{reason}:"
            lines.append(
                f"    [{_DIM}]{label}[/]"
                f"{' ' * (max_name_with_colon - len(label) + 2)}{count}"
            )

    panel = Panel(
        "\n".join(lines),
        title=f"[{_CREAM}]Execute summary[/]",
        title_align="left",
        border_style=_LAVENDER,
        padding=(0, 0),
    )
    console.print(panel)
