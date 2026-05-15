"""`skims menu` — interactive arrow-key launcher for the CLI subcommands.

Splash look: the existing CLI banner printed verbatim, then a Rich
`Live` menu panel rendered below it. The menu is a **front-end only** —
it collects choices and assembles the argv string the user would
otherwise have typed, then hands that to the same argparse parser and
dispatch table the manual CLI uses.

Load-bearing invariant: the menu never hand-builds an
`argparse.Namespace` or calls a `_cmd_*` handler directly. It always
goes argv → parser → dispatch, so every default and validation rule
stays single-sourced in `cli.py` and the manual / scheduled paths can't
drift from the interactive one.

Interactive / TTY-only: `run_menu` bails with a hint when stdout isn't a
terminal, so scripted and scheduled callers can't trip into it.

Nothing is re-exported here — the only caller (`cli._cmd_menu`) imports
`run_menu` from `skimsmarkets.menu.app` directly.
"""

from __future__ import annotations
