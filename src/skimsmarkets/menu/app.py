"""Splash-style interactive launcher behind `skims menu`.

`run_menu` is the loop: print the banner, show the arrow-key command
picker, walk the chosen command's common-flag flow, then hand the
assembled argv to the *existing* argparse parser + dispatch table. It
reimplements no command logic — see the package docstring for the
argv-not-Namespace invariant.
"""

from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable

import readchar
from rich.console import Console

from skimsmarkets.banner import print_banner
from skimsmarkets.menu import flows
from skimsmarkets.menu.widgets import select_option

# Subcommand → one-line description, shown in the picker. Order is the
# order they appear in the menu — roughly daily-use first.
_COMMANDS: tuple[tuple[str, str], ...] = (
    ("rank", "Build the slate + run the full ranking pipeline"),
    ("fetch", "Print the slate as a table (zero LLM cost)"),
    ("execute", "Place Kalshi orders against a ranked run"),
    ("positions", "Live Kalshi open-exposure summary"),
    ("retro", "Resolve past runs + calibration / pattern pass"),
    ("backtest", "Build / refresh the backtest dataset cache"),
    ("gbt", "Tennis GBT spike — backfill / train the prior"),
)

DispatchFn = Callable[[argparse.Namespace], Awaitable[int]]


async def run_menu(
    *,
    console: Console,
    parser: argparse.ArgumentParser,
    dispatch: dict[str, DispatchFn],
) -> int:
    """Interactive launcher loop. Returns a process exit code.

    TTY-only: bails with a hint when `console` isn't a terminal, so
    scripted and scheduled callers that reach `skims menu` by mistake
    get a clear message instead of a hang.
    """
    if not console.is_terminal:
        console.print(
            "skims menu is interactive and needs a terminal. Use the "
            "subcommands directly when scripting — e.g. "
            "`skims rank --sport tennis`.",
            style="yellow",
        )
        return 1

    try:
        while True:
            print_banner(console, "menu")
            command = select_option(
                console,
                title="main menu",
                options=[(name, name, desc) for name, desc in _COMMANDS],
                footer="↑↓ move    ⏎ select    q quit",
            )
            if command is None:
                console.print("  [dim]bye.[/]", highlight=False)
                return 0

            argv = flows.collect(console, command)
            if argv is None:
                continue  # backed out of the flow — redraw the main menu

            console.print(
                f"  [dim]→[/] [bold]skims[/] {' '.join(argv)}",
                highlight=False,
            )
            try:
                args = parser.parse_args(argv)
            except SystemExit:
                # argparse already printed the error to stderr; don't let
                # it kill the menu — fall back to the picker.
                console.print(
                    "  [yellow]that didn't parse — back to the menu[/]",
                    highlight=False,
                )
                if not _wait_for_key(console):
                    return 0
                continue

            await dispatch[args.command](args)
            if not _wait_for_key(console):
                return 0
    except KeyboardInterrupt:
        console.print()
        console.print("  [dim]bye.[/]", highlight=False)
        return 0


def _wait_for_key(console: Console) -> bool:
    """Block for a keypress between commands. Returns False when the user
    asked to quit (q / Esc / Ctrl-C), True to redraw the menu.
    """
    console.print()
    console.print(
        "  [dim]any key → menu    ·    q → quit[/]", highlight=False,
    )
    try:
        key = readchar.readkey()
    except KeyboardInterrupt:
        return False
    return key not in ("q", "Q", readchar.key.ESC, readchar.key.CTRL_C)
