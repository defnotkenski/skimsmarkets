"""Declarative form panel for `skims menu` — every flag for a command
on one screen, plus a `▶ run` row at the bottom.

A command is described by a list of `Field` specs (label + how the
value is picked + what argv tokens it contributes). One shared engine
(`run_form`) renders the panel, drives editing, and assembles the
argv. Adding a flag is a single spec entry; no new interactive code.

The engine handles the LIVE-style safety gate generically: any choice
flagged `danger=True` tints the value + run row red and pops a confirm
panel before `▶ run` returns its argv. That's what the
`execute · mode = LIVE` choice uses today; future destructive choices
get the same treatment for free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as dc_field
from typing import Literal

import readchar
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from skimsmarkets.menu._palette import CLAY, GOLD, GREEN, RED
from skimsmarkets.menu.widgets import select_option

_FOOTER = "↑↓ rows    ←→ cycle    type number    ⌫ delete    ⏎ run    esc cancel"


# --- spec types ------------------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """One option for a `kind="choice"` field.

    `argv` is the tokens this choice contributes when assembled — empty
    for the implicit-default option (so the flag is omitted), `("--live",)`
    for a boolean flag, `("--horizon", "12")` for a value flag,
    `("backfill",)` for a positional. `danger=True` arms the safety gate.
    """

    label: str
    argv: tuple[str, ...] = ()
    danger: bool = False


ChoicesProvider = list[Choice] | Callable[[], list[Choice]]


@dataclass(frozen=True)
class Field:
    """One row in the form. Choice fields cycle with `←→`; number fields
    open a type-prompt on `⏎`. `choices` may be a list or a zero-arg
    callable (used for fields whose options depend on filesystem state,
    e.g. `execute · run-id`).
    """

    label: str
    kind: Literal["choice", "number"]
    choices: ChoicesProvider | None = None
    flag: str | None = None         # required for kind="number"
    default_label: str = "default"  # number fields: shown when value is None


# --- form session state ----------------------------------------------------


@dataclass
class _State:
    # Per-field, parallel arrays:
    choice_resolved: list[list[Choice]]  # resolved at form open (snapshots
                                          # callable choices so the form
                                          # doesn't re-fetch on every keypress)
    choice_index: list[int]              # current selection per choice field
    number_value: list[str | None] = dc_field(default_factory=list)
                                          # typed value per number field;
                                          # None = use the command's default


def _init_state(fields: list[Field]) -> _State:
    resolved: list[list[Choice]] = []
    for f in fields:
        if f.kind == "choice":
            cs = f.choices() if callable(f.choices) else (f.choices or [])
            resolved.append(list(cs))
        else:
            resolved.append([])
    return _State(
        choice_resolved=resolved,
        choice_index=[0] * len(fields),
        number_value=[None] * len(fields),
    )


# --- pure helpers (unit-tested) -------------------------------------------


def _assemble_argv(
    command: str, fields: list[Field], state: _State,
) -> list[str]:
    argv = [command]
    for i, f in enumerate(fields):
        if f.kind == "choice":
            choices = state.choice_resolved[i]
            if choices and 0 <= state.choice_index[i] < len(choices):
                argv.extend(choices[state.choice_index[i]].argv)
        else:  # number
            v = state.number_value[i]
            if v is not None and f.flag is not None:
                argv.extend([f.flag, v])
    return argv


def _any_danger(fields: list[Field], state: _State) -> bool:
    for i, f in enumerate(fields):
        if f.kind != "choice":
            continue
        choices = state.choice_resolved[i]
        if choices and choices[state.choice_index[i]].danger:
            return True
    return False


def _danger_summary(fields: list[Field], state: _State) -> str:
    parts = []
    for i, f in enumerate(fields):
        if f.kind != "choice":
            continue
        choices = state.choice_resolved[i]
        if not choices:
            continue
        c = choices[state.choice_index[i]]
        if c.danger:
            parts.append(f"{f.label}={c.label}")
    return ", ".join(parts)


# --- rendering -------------------------------------------------------------


def _render_form_panel(
    command: str, fields: list[Field], state: _State, cursor: int,
) -> Panel:
    label_width = max((len(f.label) for f in fields), default=0) + 2
    rows: list[Text] = []
    for i, f in enumerate(fields):
        rows.append(_render_field_row(
            f, state, i, label_width, focused=(i == cursor),
        ))
    rows.append(Text(""))
    rows.append(_render_run_row(
        focused=(cursor == len(fields)),
        danger=_any_danger(fields, state),
    ))
    rows.append(Text(""))
    rows.append(Text(_FOOTER, style="dim"))
    return Panel(
        Group(*rows),
        title=f"[bold]{command}[/]",
        border_style=CLAY,
        box=ROUNDED,
        padding=(1, 2),
    )


def _render_field_row(
    f: Field, state: _State, i: int, label_width: int, focused: bool,
) -> Text:
    if f.kind == "choice":
        choices = state.choice_resolved[i]
        if choices and 0 <= state.choice_index[i] < len(choices):
            choice = choices[state.choice_index[i]]
            value = choice.label
            value_style = RED if choice.danger else "white"
        else:
            value = "—"
            value_style = "grey42"
    else:  # number
        v = state.number_value[i]
        value = v if v else f.default_label
        value_style = "white"

    if focused:
        parts: list[tuple[str, str]] = [
            ("❯ ", f"bold {GOLD}"),
            (f"{f.label:<{label_width}}", f"bold {GOLD}"),
        ]
        if f.kind == "choice":
            parts += [
                ("‹ ", f"bold {GOLD}"),
                (value, f"bold {value_style}"),
                (" ›", f"bold {GOLD}"),
            ]
        else:
            parts.append((value, f"bold {value_style}"))
    else:
        parts = [
            ("  ", ""),
            (f"{f.label:<{label_width}}", "grey70"),
            (value, value_style),
        ]

    if f.kind == "number" and state.number_value[i] is None:
        parts.append(("  (default)", CLAY if focused else "grey42"))
    return Text.assemble(*parts)


def _render_run_row(*, focused: bool, danger: bool) -> Text:
    """`▶ run` always reads bold green (or red when armed) so it stands
    out from grey field labels even when the cursor is elsewhere. The
    `❯` pointer is the focus signal; the warning text only appears
    once you're actually about to run a danger choice.
    """
    color = RED if danger else GREEN
    if focused:
        parts: list[tuple[str, str]] = [
            ("❯ ", f"bold {color}"),
            ("▶ run", f"bold {color}"),
        ]
        if danger:
            parts.append(
                ("    LIVE — ⏎ asks you to confirm", f"bold {color}"),
            )
    else:
        parts = [("  ", ""), ("▶ run", f"bold {color}")]
    return Text.assemble(*parts)


# --- interactive driver ----------------------------------------------------


def run_form(
    console: Console,
    *,
    command: str,
    fields: list[Field],
) -> list[str] | None:
    """Render the form, drive editing, return the assembled argv (or
    `None` on cancel). Empty `fields` short-circuits to `[command]` so
    a command with no flags worth prompting (`positions`) just runs.

    Field editing happens **inline** inside the form panel: `←→` cycles
    a choice field's value; digit keys append to a number field, `⌫`
    deletes the last char. `⏎` on a field advances the cursor; `⏎` on
    `▶ run` submits (with a danger-confirm sub-panel when armed).

    Each render session runs in its own `with Live(...)` block; the
    danger confirm and the main menu picker are the only sub-panels
    the form will surface, and they happen *between* Live sessions.
    Reusing a single Live across teardowns via `stop()` / `start()`
    leaves Rich's `LiveRender` shape state stale and walks the next
    paint up into the banner above — fresh Live per session avoids it.
    """
    if not fields:
        return [command]
    state = _init_state(fields)
    cursor = 0
    while True:
        action, value, cursor = _drive_session(
            console, command, fields, state, cursor,
        )
        if action == "cancel":
            return None
        if action == "run":
            return value
        if action == "confirm_run":
            if _confirm_danger(console, command, fields, state):
                return _assemble_argv(command, fields, state)
            continue  # declined — re-open form, cursor still on run row
        raise AssertionError(f"unreachable form action: {action}")


def _drive_session(
    console: Console,
    command: str,
    fields: list[Field],
    state: _State,
    start_cursor: int,
) -> tuple[str, list[str] | None, int]:
    """Run one form Live session. Field edits are inline — the session
    only exits when something genuinely needs to happen outside Live:
    cancel, run, or confirm_run (danger choice armed). Tuple shape:
    `(action, value, final_cursor)`.
    """
    cursor = start_cursor
    n_rows = len(fields) + 1  # field rows + ▶ run

    def _render() -> Panel:
        return _render_form_panel(command, fields, state, cursor)

    with Live(
        _render(), console=console, auto_refresh=False, transient=True,
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
                cursor = (cursor - 1) % n_rows
            elif key in (readchar.key.DOWN, "j"):
                cursor = (cursor + 1) % n_rows
            elif key in (readchar.key.LEFT, "h"):
                if cursor < len(fields):
                    _cycle(fields, state, cursor, -1)
            elif key in (readchar.key.RIGHT, "l"):
                if cursor < len(fields):
                    _cycle(fields, state, cursor, +1)
            elif key in (readchar.key.BACKSPACE, "\x7f", "\x08"):
                _backspace_number(fields, state, cursor)
            elif len(key) == 1 and key.isdigit():
                _append_number(fields, state, cursor, key)
            elif key in (readchar.key.ENTER, "\r", "\n"):
                if cursor == len(fields):
                    # ▶ run row
                    if _any_danger(fields, state):
                        return ("confirm_run", None, cursor)
                    return (
                        "run",
                        _assemble_argv(command, fields, state),
                        cursor,
                    )
                # On a field — advance to the next row (Tab-like). Inline
                # editing already happened via cycle / typing keys.
                cursor = (cursor + 1) % n_rows
            elif key in (readchar.key.ESC, "q", "Q"):
                return ("cancel", None, cursor)
            else:
                continue  # unknown key — don't redraw
            live.update(_render(), refresh=True)


def _cycle(fields: list[Field], state: _State, i: int, delta: int) -> None:
    if fields[i].kind != "choice":
        return  # ←→ is a no-op on number fields; digit keys edit those
    n = len(state.choice_resolved[i])
    if n:
        state.choice_index[i] = (state.choice_index[i] + delta) % n


def _append_number(
    fields: list[Field], state: _State, i: int, ch: str,
) -> None:
    if i >= len(fields) or fields[i].kind != "number":
        return
    state.number_value[i] = (state.number_value[i] or "") + ch


def _backspace_number(fields: list[Field], state: _State, i: int) -> None:
    if i >= len(fields) or fields[i].kind != "number":
        return
    v = state.number_value[i] or ""
    v = v[:-1]
    # Empty string → None so the row reverts to the "(default)" display
    # and the flag is omitted from argv.
    state.number_value[i] = v if v else None


def _confirm_danger(
    console: Console, command: str, fields: list[Field], state: _State,
) -> bool:
    summary = _danger_summary(fields, state) or "the danger choice"
    confirmed = select_option(
        console,
        title=f"{command} · confirm",
        options=[
            ("no", "back out", "return to the form, nothing run"),
            ("yes", f"proceed with {summary}",
             "this can spend real money on Kalshi"),
        ],
    )
    return confirmed == "yes"
