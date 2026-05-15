"""SKIMS CLI banner — the ANSI Shadow wordmark shown atop an interactive run.

TTY-gated (see `print_banner`): never prints when stdout/stderr is piped or
captured, so it stays out of `skims positions`' parseable output and the
captured logs of scheduled cloud routines. Purely cosmetic — no command
behaviour depends on it.
"""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from rich.console import Console

# "SKIMS" in the figlet "ANSI Shadow" font.
_WORDMARK: tuple[str, ...] = (
    "███████╗██╗  ██╗██╗ ███╗   ███╗███████╗",
    "██╔════╝██║ ██╔╝██║ ████╗ ████║██╔════╝",
    "███████╗█████╔╝ ██║ ██╔████╔██║███████╗",
    "╚════██║██╔═██╗ ██║ ██║╚██╔╝██║╚════██║",
    "███████║██║  ██╗██║ ██║ ╚═╝ ██║███████║",
    "╚══════╝╚═╝  ╚═╝╚═╝ ╚═╝     ╚═╝╚══════╝",
)

# Per-row "ember" gradient — warm gold → clay → deep rust. Truecolor hex;
# Rich downgrades gracefully on 256/16-colour terminals.
_EMBER: tuple[str, ...] = (
    "#e8b563",
    "#e29c5e",
    "#dc8359",
    "#cf6e50",
    "#ba5c41",
    "#a54a32",
)

# Dim slate palette for the cosmetic stars bracketing the wordmark. `·` is
# the faint dust glyph; `✦` is the brighter focal star. Two tones so the
# constellation reads as foreground/background rather than a uniform field.
_STAR_DIM = "#5a6b7e"
_STAR_BRIGHT = "#7a8a9c"

# Top + bottom decorative lines outside the wordmark. Hand-placed so they
# look scattered rather than uniform.
_STAR_LINE_TOP = (
    f"  [{_STAR_DIM}]·[/]   [{_STAR_BRIGHT}]✦[/]            "
    f"[{_STAR_DIM}]·[/]           [{_STAR_BRIGHT}]✦[/]     [{_STAR_DIM}]·[/]"
)
_STAR_LINE_BOTTOM = (
    f"       [{_STAR_DIM}]·[/]          [{_STAR_BRIGHT}]✦[/]              "
    f"[{_STAR_DIM}]·[/]         [{_STAR_DIM}]·[/]"
)

# Per-wordmark-row side stars. Each entry renders to exactly 5 visible cols
# (1 glyph + 4 spaces, or 5 spaces when blank) so every wordmark row stays
# horizontally aligned with the subtitle / status / star-line indent. Sparse
# pattern — only 3 of 6 rows carry a glyph per side, alternating so the
# constellation doesn't form a vertical column.
_LEFT_STARS: tuple[str, ...] = (
    f"[{_STAR_DIM}]·[/]    ",     # row 0
    "     ",                       # row 1
    f"[{_STAR_BRIGHT}]✦[/]    ",  # row 2
    "     ",                       # row 3
    "     ",                       # row 4
    f"[{_STAR_DIM}]·[/]    ",     # row 5
)
_RIGHT_STARS: tuple[str, ...] = (
    f"    [{_STAR_BRIGHT}]✦[/]",  # row 0
    "     ",                       # row 1
    "     ",                       # row 2
    f"    [{_STAR_DIM}]·[/]",     # row 3
    f"    [{_STAR_BRIGHT}]✦[/]",  # row 4
    "     ",                       # row 5
)

# Ember gold for the live status values; matches the top wordmark row.
_STATUS_ACCENT = "#e8b563"


def print_banner(console: Console, command: str) -> None:
    """Print the SKIMS banner through `console`, padded with a blank line
    above and below so it stands clear of the surrounding output.

    No-op unless `console` is an interactive terminal — that keeps the art
    out of piped stdout and captured scheduled-routine logs. Callers still
    decide *which* commands get a banner (the parseable `positions` command
    opts out at the call site).
    """
    if not console.is_terminal:
        return
    console.print()
    console.print(_STAR_LINE_TOP, highlight=False)
    for i, (line, color) in enumerate(zip(_WORDMARK, _EMBER)):
        # Wordmark contains only box-drawing glyphs — no `[`/`]` — so we can
        # safely interpolate it into a markup row alongside the side stars.
        row = f"  {_LEFT_STARS[i]}[{color}]{line}[/]{_RIGHT_STARS[i]}"
        console.print(row, highlight=False)
    console.print(
        f"  risk-graded sports markets · {command} · v{version('skimsmarkets')}",
        style="dim italic",
        markup=False,
        highlight=False,
    )
    status = _status_line()
    if status is not None:
        # Hanging branch: a `│` riser drops from the subtitle, then `╰─`
        # elbows out to the status. Skipped entirely when no runs exist on
        # disk so a fresh install doesn't show a dangling branch.
        console.print(f"  [{_STAR_DIM}]│[/]", highlight=False)
        console.print(f"  [{_STAR_DIM}]╰─[/] {status}", highlight=False)
    console.print(_STAR_LINE_BOTTOM, highlight=False)
    console.print()


def _status_line() -> str | None:
    """Build the status content shown under the hanging branch.

    Reads `logs/runs/` only — no network. Returns None when there are no
    runs on disk yet (fresh install). Any unexpected error degrades to
    None so the banner can never crash a command.
    """
    try:
        # Lazy import: keeps the banner-module import path light and avoids
        # dragging Pydantic / pipeline imports into `--help` flows.
        from skimsmarkets.retro.jsonl import iter_predictions, list_run_files

        runs = list_run_files()
        if not runs:
            return None
        latest = runs[0]
        age = _format_age(latest)

        # Count + sport from the latest run. Small JSONL parse — microseconds
        # for a typical 10-20 event slate.
        count = 0
        sport: str | None = None
        for pred in iter_predictions(latest):
            count += 1
            if sport is None and pred.sport_type:
                sport = pred.sport_type
        sport_pill = f"{count} {sport}" if sport else f"{count} predictions"

        return (
            f"[dim]latest run[/] [{_STATUS_ACCENT}]{latest.stem}[/] "
            f"[dim]({age})[/]   [dim]·[/]   "
            f"[dim]slate[/] [{_STATUS_ACCENT}]{sport_pill}[/]"
        )
    except Exception:  # noqa: BLE001 — banner must never raise
        return None


def _format_age(path: Path) -> str:
    """Human-friendly mtime age (e.g. `2h ago`)."""
    secs = int(
        (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime))
        .total_seconds()
    )
    if secs < 3600:
        return f"{max(1, secs // 60)}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"
