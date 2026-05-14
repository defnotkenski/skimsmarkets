"""SKIMS CLI banner — the ANSI Shadow wordmark shown atop an interactive run.

TTY-gated (see `print_banner`): never prints when stdout/stderr is piped or
captured, so it stays out of `skims positions`' parseable output and the
captured logs of scheduled cloud routines. Purely cosmetic — no command
behaviour depends on it.
"""

from __future__ import annotations

from importlib.metadata import version

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
    for line, color in zip(_WORDMARK, _EMBER):
        console.print(f"  {line}", style=color, markup=False, highlight=False)
    console.print(
        f"  risk-graded sports markets · {command} · v{version('skimsmarkets')}",
        style="dim italic",
        markup=False,
        highlight=False,
    )
    console.print()
