"""NOCTA CLI banner — the ANSI Shadow wordmark shown atop an interactive run.

NOCTA is the system's user-facing wordmark (post the 2026-05-17 dual-mode
shift — the brand reflects the new direction; the underlying CLI command
and package name stay `skims` / `skimsmarkets` to avoid breaking cron
invocations and import paths).

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

# "NOCTA" in the figlet "ANSI Shadow" font (regenerate via
# `uvx pyfiglet -f ansi_shadow NOCTA` if the wordmark needs updating).
_WORDMARK: tuple[str, ...] = (
    "███╗   ██╗ ██████╗  ██████╗████████╗ █████╗ ",
    "████╗  ██║██╔═══██╗██╔════╝╚══██╔══╝██╔══██╗",
    "██╔██╗ ██║██║   ██║██║        ██║   ███████║",
    "██║╚██╗██║██║   ██║██║        ██║   ██╔══██║",
    "██║ ╚████║╚██████╔╝╚██████╗   ██║   ██║  ██║",
    "╚═╝  ╚═══╝ ╚═════╝  ╚═════╝   ╚═╝   ╚═╝  ╚═╝",
)

# Per-row "DAWN" gradient — peach top fading through clay → slate at the
# baseline. Replaced the original ember sweep 2026-05-17 alongside the
# rebrand to Nocta; DAWN reads as sunrise to a system that's about to
# start a trading window, matching the brand mood better than the
# strictly-sunset ember palette. Truecolor hex; Rich downgrades
# gracefully on 256/16-colour terminals.
_DAWN: tuple[str, ...] = (
    "#f5d4b8",
    "#eebfa4",
    "#e0a890",
    "#c89580",
    "#a88378",
    "#807070",
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

# Mode-themed status accent for live status values (run id, mode tag).
# Confidence mode = frost blue; EV mode = ember gold; tail mode = vermillion
# (warmer than gold, signals "higher-variance strategy"). Picked at render
# time via `_status_accent_for_mode` so the same banner code serves all
# three. The wordmark stays on the neutral DAWN gradient regardless — only
# the status pill carries the mode color, keeping mode visualization where
# it matters (current-state at-a-glance) without flipping the brand identity.
_STATUS_ACCENT_CONFIDENCE = "#7fa1ce"   # frost blue
_STATUS_ACCENT_EV = "#e8b563"           # ember gold
_STATUS_ACCENT_TAIL = "#d96a4f"         # vermillion


def _status_accent_for_mode(mode: str) -> str:
    """Resolve the status-pill accent color from the configured trade mode."""
    if mode == "tail":
        return _STATUS_ACCENT_TAIL
    if mode == "ev":
        return _STATUS_ACCENT_EV
    return _STATUS_ACCENT_CONFIDENCE

# Indent for body text (subtitle + status branch) so it left-aligns with the
# first glyph of the wordmark rather than with the outer star constellation.
# Derived as: 2 (the shared `  ` base indent applied to every line) + 5 (the
# fixed visible-column width of `_LEFT_STARS[i]`, which sits between the
# base indent and the wordmark glyph on each wordmark row). Hard-coded as a
# literal rather than computed at runtime because the star-column width is
# itself hard-coded into `_LEFT_STARS`; if you change the star column, bump
# this too.
_WORDMARK_INDENT = "       "  # 7 spaces


def print_banner(
    console: Console, command: str, mode: str | None = None,
) -> None:
    """Print the NOCTA banner through `console`, padded with a blank line
    above and below so it stands clear of the surrounding output.

    No-op unless `console` is an interactive terminal — that keeps the art
    out of piped stdout and captured scheduled-routine logs. Callers still
    decide *which* commands get a banner (the parseable `positions` command
    opts out at the call site).

    `mode` is the per-call CLI override (from `args.mode` on rank / fetch /
    execute). When None, the status pill falls through to
    `cfg.KALSHI_DEFAULT_TRADE_MODE` so commands without a `--mode` flag
    (e.g. `retro`) still get a sensible mode display.
    """
    if not console.is_terminal:
        return
    console.print()
    console.print(_STAR_LINE_TOP, highlight=False)
    for i, (line, color) in enumerate(zip(_WORDMARK, _DAWN)):
        # Wordmark contains only box-drawing glyphs — no `[`/`]` — so we can
        # safely interpolate it into a markup row alongside the side stars.
        row = f"  {_LEFT_STARS[i]}[{color}]{line}[/]{_RIGHT_STARS[i]}"
        console.print(row, highlight=False)
    console.print(
        f"{_WORDMARK_INDENT}dual-mode sports markets · {command} · "
        f"v{version('skimsmarkets')}",
        style="dim italic",
        markup=False,
        highlight=False,
    )
    status = _status_line(mode_override=mode)
    if status is not None:
        # Hanging branch: a `│` riser drops from the subtitle, then `╰─`
        # elbows out to the status. Skipped entirely when no runs exist on
        # disk so a fresh install doesn't show a dangling branch. Both lines
        # share the wordmark indent so the branch hangs from the subtitle
        # column rather than from the outer star line.
        console.print(
            f"{_WORDMARK_INDENT}[{_STAR_DIM}]│[/]", highlight=False,
        )
        console.print(
            f"{_WORDMARK_INDENT}[{_STAR_DIM}]╰─[/] {status}", highlight=False,
        )
    console.print(_STAR_LINE_BOTTOM, highlight=False)
    console.print()


def _status_line(mode_override: str | None = None) -> str | None:
    """Build the status content shown under the hanging branch.

    Reads `logs/runs/` + `config.KALSHI_DEFAULT_TRADE_MODE` only — no
    network, no JSONL parsing. Returns None when there are no runs on
    disk yet (fresh install). Any unexpected error degrades to None so
    the banner can never crash a command.

    Pill shape: `latest run <id> (<age>)   ·   mode <X>`. Accent color
    is picked from the mode — frost blue for confidence, ember gold for
    ev, vermillion for tail — so the operator's first glance at the
    banner answers "which mode is this run using?" through color alone.

    `mode_override` is the per-call `--mode` arg from the active command
    (rank / fetch / execute). When provided it wins; when None the
    status falls through to `cfg.KALSHI_DEFAULT_TRADE_MODE`. This keeps
    the banner accurate for a one-off `skims rank --mode tail` invocation
    even though the cron default stays "confidence".
    """
    try:
        # Lazy imports: keep the banner-module import path light and
        # avoid dragging config / Pydantic into `--help` flows.
        from skimsmarkets import config as cfg
        from skimsmarkets.retro.jsonl import list_run_files

        runs = list_run_files()
        if not runs:
            return None
        latest = runs[0]
        age = _format_age(latest)
        mode = mode_override or cfg.KALSHI_DEFAULT_TRADE_MODE
        accent = _status_accent_for_mode(mode)

        return (
            f"[dim]latest run[/] [{accent}]{latest.stem}[/] "
            f"[dim]({age})[/]   [dim]·[/]   "
            f"[dim]mode[/] [bold {accent}]{mode}[/]"
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
