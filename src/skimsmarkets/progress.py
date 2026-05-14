"""Rich-backed progress reporter for `skims rank`.

Wraps a `rich.progress.Progress` with named pipeline phases so the
pipeline can call `start` / `set_total` / `advance` / `complete` at
phase boundaries without importing Rich types itself. The CLI is the
only caller that instantiates one; non-CLI entry points (tests,
backtest, retro) pass `None` and the pipeline's `_advance_if` helpers
no-op cleanly.

Phases (declared up front so the order in the rendered display is
stable run-to-run):
  - "slate"   : gamma /events listing + horizon + selection. Single
                step.
  - "enrich"  : UW context + CLOB book + CLOB price history + tennis
                stats + sim + GBT prior. Single step (the per-stage
                breakdown stays in the post-run `lens-stage totals`
                log line).
  - "predict" : per-event lens chain + director. Determinate bar —
                total = number of events surviving lens dispatch;
                advances as each event's `process_event` task
                resolves.
  - "judge"   : slate-wide defensibility judgement. Single step.

Logging goes to stderr (configured in `cli._setup_logging`), Rich
output to stdout, so the progress display and the existing `log.info`
breadcrumbs don't fight each other.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

_PHASE_LABELS: dict[str, str] = {
    "slate": "Fetching slate",
    "enrich": "Enriching events",
    "predict": "Predicting events",
    "judge": "Judging slate",
}


class ProgressReporter(AbstractAsyncContextManager["ProgressReporter"]):
    """Rich-backed progress display for the rank pipeline.

    Use as an async context manager:

        async with ProgressReporter() as p:
            result = await run_pipeline(..., progress=p)

    `start(phase)` is idempotent — calling it twice is a no-op so the
    pipeline can call it inside any of several entry points (e.g. each
    enrichment stage starts/completes the same "enrich" task) without
    branching. `complete(phase)` snaps the bar to 100% so the user
    sees the full phase sequence persist after the pipeline returns.
    """

    def __init__(self, console: Console | None = None) -> None:
        # `refresh_per_second=2` keeps spam down. Rich's default is 10
        # which causes the bars to redraw on every spinner-tick cycle;
        # because stdlib logging writes directly to stderr (bypassing
        # Rich's Live redirect — the StreamHandler caches sys.stderr
        # at config time, before Live.start replaces it), the bars and
        # log lines interleave and every redraw shows up in scrollback.
        # 2 Hz still gives a visibly-spinning spinner during long
        # phases (predict can run for minutes) without flooding the
        # terminal between log lines.
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console or Console(),
            transient=False,
            refresh_per_second=2,
        )
        self._tasks: dict[str, int] = {}

    async def __aenter__(self) -> Self:
        self._progress.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._progress.stop()

    def start(self, phase: str, total: int = 1) -> None:
        """Begin a phase. Idempotent — second call is a no-op."""
        if phase in self._tasks:
            return
        label = _PHASE_LABELS.get(phase, phase)
        self._tasks[phase] = self._progress.add_task(label, total=total)

    def set_total(self, phase: str, total: int) -> None:
        """Adjust a phase's total after start — used by `predict` when
        the post-dispatch event count is known."""
        if phase in self._tasks:
            self._progress.update(self._tasks[phase], total=total)

    def advance(self, phase: str, n: int = 1) -> None:
        """Bump a phase's `completed` by `n`. No-op if the phase
        hasn't been started (defensive — keeps a stray callback from
        a cancelled task from raising)."""
        if phase in self._tasks:
            self._progress.advance(self._tasks[phase], n)

    def complete(self, phase: str) -> None:
        """Mark a phase done. Snaps the bar to its `total` so a
        single-step phase renders as 100% rather than the implicit
        50% it'd otherwise show after one advance."""
        if phase not in self._tasks:
            return
        task_id = self._tasks[phase]
        total = self._progress.tasks[task_id].total or 1
        self._progress.update(task_id, completed=total)
