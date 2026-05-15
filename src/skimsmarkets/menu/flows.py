"""Per-command field specs for `skims menu`.

Each command is a list of `Field` definitions that the form engine
(`form.run_form`) renders on one screen and assembles into argv.
Adding a new flag is one entry in the list — no new interactive code,
no new builder. Rare flags intentionally stay CLI-only.

The pure `_assemble_argv` (and its helpers) live in `form.py`; tests
exercise them with representative spec / state pairs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.console import Console

from skimsmarkets import config as cfg
from skimsmarkets.menu.form import Choice, Field, run_form
from skimsmarkets.retro.jsonl import list_run_files


# --- shared choice providers ----------------------------------------------


def _registered_sports() -> list[str]:
    """Sports the ranker has lens sets for. Lazy import keeps the menu's
    import path light; the v1 set is the fallback if the registry can't
    be loaded for any reason.
    """
    try:
        from skimsmarkets.agents.sports import SPORT_LENS_SETS
    except ImportError:
        return ["tennis"]
    return sorted(SPORT_LENS_SETS) or ["tennis"]


def _age(path: Path) -> str:
    """Human-friendly mtime age for a run-log path (e.g. `2h ago`)."""
    secs = int(
        (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime))
        .total_seconds()
    )
    if secs < 3600:
        return f"{max(1, secs // 60)}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _sport_choices() -> list[Choice]:
    return [
        Choice(label=s, argv=("--sport", s))
        for s in _registered_sports()
    ]


def _run_id_choices() -> list[Choice]:
    """For `execute · run-id`. The first option omits `--run-id` (execute
    uses its own latest-run default); subsequent options pin a specific
    run from the most recent ones on disk.
    """
    runs = list_run_files()
    if not runs:
        return [Choice(label="latest (no runs yet)")]
    out = [Choice(label=f"latest   {runs[0].stem} · {_age(runs[0])}")]
    for p in runs[1:8]:
        out.append(Choice(
            label=f"{p.stem} · {_age(p)}",
            argv=("--run-id", p.stem),
        ))
    return out


def _retro_scope_choices() -> list[Choice]:
    """For `retro · scope`. `all runs` omits `--run-id` (retro processes
    every log). `latest only` pins to the newest run. With no logs on
    disk we still offer `all runs` so retro's own empty-input message
    speaks.
    """
    runs = list_run_files()
    out = [Choice(label="all runs")]
    if runs:
        out.append(Choice(
            label=f"latest only   {runs[0].stem} · {_age(runs[0])}",
            argv=("--run-id", runs[0].stem),
        ))
    return out


# --- per-command field specs ----------------------------------------------


# Each command's flag form. Adding a flag = append one Field. The first
# choice in a list is the implicit-default (selected when the form
# opens); for fields where the invariant requires a value (`rank --sport`)
# every choice carries a non-empty argv.
_COMMAND_FORMS: dict[str, list[Field]] = {
    "rank": [
        Field(label="sport", kind="choice", choices=_sport_choices),
        Field(
            label="horizon", kind="number", flag="--horizon",
            default_label=f"{cfg.DEFAULT_HORIZON_HOURS}h",
        ),
    ],
    "fetch": [
        Field(label="sport", kind="choice", choices=_sport_choices),
        Field(
            label="horizon", kind="number", flag="--horizon",
            default_label=f"{cfg.DEFAULT_HORIZON_HOURS}h",
        ),
    ],
    "execute": [
        Field(label="run-id", kind="choice", choices=_run_id_choices),
        Field(label="mode", kind="choice", choices=[
            Choice(label="dry-run"),
            Choice(label="LIVE", argv=("--live",), danger=True),
        ]),
    ],
    "retro": [
        Field(label="step", kind="choice", choices=[
            Choice(label="analyze",         argv=("--step", "analyze")),
            Choice(label="calibrate",       argv=("--step", "calibrate")),
            Choice(label="fit-calibration", argv=("--step", "fit-calibration")),
        ]),
        Field(label="scope", kind="choice", choices=_retro_scope_choices),
    ],
    "backtest": [
        Field(
            label="max events", kind="number", flag="--max-events",
            default_label="800",
        ),
    ],
    "gbt": [
        Field(label="cmd", kind="choice", choices=[
            Choice(label="backfill", argv=("backfill",)),
            Choice(label="train",    argv=("train",)),
        ]),
    ],
    # `positions` has no flow — `collect` short-circuits to ["positions"].
}


# --- dispatch -------------------------------------------------------------


def collect(console: Console, command: str) -> list[str] | None:
    """Open the form for `command` and return the assembled argv, or
    `None` if the user cancelled. Commands with no field spec just run
    with defaults — the form never opens.
    """
    fields = _COMMAND_FORMS.get(command)
    if not fields:
        return [command]
    return run_form(console, command=command, fields=fields)
