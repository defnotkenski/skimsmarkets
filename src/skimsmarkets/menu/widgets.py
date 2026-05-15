"""Arrow-key input primitive for `skims menu`.

`select_option` drives a Rich `Live` region (manual refresh — no
background thread to race the blocking key read) and reads keys through
`readchar`, which normalises arrow escape sequences across platforms.
Returns the user's pick, or `None` when the level is dismissed so the
caller can fall back; Ctrl-C raises `KeyboardInterrupt` so the menu
loop can quit cleanly.

This is the only sub-panel surface in the menu: the main command
picker uses it directly and the form uses it for the LIVE-armed
danger confirm. Field editing in the form is inline (cycle keys,
typed digits) — no sub-panel for that path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import readchar
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from skimsmarkets.menu._palette import CLAY, GOLD

V = TypeVar("V")

_DEFAULT_FOOTER = "↑↓ move    ⏎ select    esc back"


def _option_panel(
    *,
    title: str,
    options: Sequence[tuple[V, str, str]],
    selected_index: int,
    footer: str,
) -> Panel:
    """Render the option list as a rounded clay panel — the Splash look.

    `options` rows are `(value, label, description)`; `value` is opaque
    here, only `label` / `description` are drawn.
    """
    label_width = max(len(label) for _, label, _ in options) + 2
    rows: list[Text] = []
    for i, (_value, label, desc) in enumerate(options):
        if i == selected_index:
            rows.append(Text.assemble(
                ("❯ ", f"bold {GOLD}"),
                (f"{label:<{label_width}}", f"bold {GOLD}"),
                (desc, CLAY),
            ))
        else:
            rows.append(Text.assemble(
                ("  ", ""),
                (f"{label:<{label_width}}", "grey70"),
                (desc, "grey42"),
            ))
    body = Group(*rows, Text(""), Text(footer, style="dim"))
    return Panel(
        body,
        title=f"[bold]{title}[/]",
        border_style=CLAY,
        box=ROUNDED,
        padding=(1, 2),
    )


def select_option(
    console: Console,
    *,
    title: str,
    options: Sequence[tuple[V, str, str]],
    footer: str = _DEFAULT_FOOTER,
) -> V | None:
    """Arrow-key picker over `options` (value, label, description).

    Returns the chosen `value`, or `None` if the user dismissed this
    level (q / Esc). Raises `KeyboardInterrupt` on Ctrl-C so the caller
    can quit the whole menu cleanly. The panel is transient — it clears
    on exit, leaving the surrounding output (banner, command echo)
    uncluttered.
    """
    if not options:
        return None
    idx = 0
    with Live(
        _option_panel(
            title=title, options=options, selected_index=idx, footer=footer,
        ),
        console=console,
        auto_refresh=False,
        transient=True,
    ) as live:
        live.refresh()
        while True:
            try:
                key = readchar.readkey()
            except KeyboardInterrupt:
                raise
            if key == readchar.key.CTRL_C:
                raise KeyboardInterrupt
            if key in (readchar.key.UP, "k"):
                idx = (idx - 1) % len(options)
            elif key in (readchar.key.DOWN, "j"):
                idx = (idx + 1) % len(options)
            elif key in (readchar.key.ENTER, "\r", "\n"):
                return options[idx][0]
            elif key in (readchar.key.ESC, "q", "Q"):
                return None
            else:
                continue
            live.update(
                _option_panel(
                    title=title, options=options,
                    selected_index=idx, footer=footer,
                ),
                refresh=True,
            )
