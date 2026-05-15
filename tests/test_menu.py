"""Plain-assert tests for the `skims menu` form engine.

Usage:

    uv run python tests/test_menu.py

The interactive layer (Rich `Live` + `readchar` key handling) isn't
unit-tested — like the execute display, it's the thin I/O shell. What
*is* tested is the load-bearing part: the spec dataclasses, the
`_assemble_argv` builder, the `_any_danger` detector, and the
contract that every argv assembled from the registered specs parses
through the real `skims` parser.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skimsmarkets.cli import _build_parser  # noqa: E402
from skimsmarkets.menu.flows import _COMMAND_FORMS, collect  # noqa: E402
from skimsmarkets.menu.form import (  # noqa: E402
    Choice,
    Field,
    _State,
    _any_danger,
    _assemble_argv,
    _init_state,
)


def _state_for(fields: list[Field], **selections: object) -> _State:
    """Build a _State with specific picks per field label.

    `selections` values: int (choice index) or str (typed number value).
    """
    state = _init_state(fields)
    for i, f in enumerate(fields):
        if f.label not in selections:
            continue
        sel = selections[f.label]
        if f.kind == "choice":
            assert isinstance(sel, int)
            state.choice_index[i] = sel
        else:
            assert isinstance(sel, str)
            state.number_value[i] = sel
    return state


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def _t_assemble_simple_choice() -> int:
    fields = [
        Field(label="mode", kind="choice", choices=[
            Choice(label="off"),
            Choice(label="on", argv=("--mode", "on")),
        ]),
    ]
    assert _assemble_argv("foo", fields, _state_for(fields, mode=0)) == ["foo"]
    assert _assemble_argv("foo", fields, _state_for(fields, mode=1)) == [
        "foo", "--mode", "on",
    ]
    return 2


def _t_assemble_number_default_vs_typed() -> int:
    fields = [
        Field(label="n", kind="number", flag="--n", default_label="100"),
    ]
    # default → flag omitted
    assert _assemble_argv("foo", fields, _state_for(fields)) == ["foo"]
    # typed → flag emitted
    assert _assemble_argv("foo", fields, _state_for(fields, n="42")) == [
        "foo", "--n", "42",
    ]
    return 2


def _t_assemble_mixed() -> int:
    fields = [
        Field(label="sport", kind="choice", choices=[
            Choice(label="tennis", argv=("--sport", "tennis")),
        ]),
        Field(
            label="horizon", kind="number", flag="--horizon",
            default_label="6h",
        ),
    ]
    # sport=tennis (only choice), horizon default → omit horizon
    assert _assemble_argv("rank", fields, _state_for(fields)) == [
        "rank", "--sport", "tennis",
    ]
    # sport=tennis, horizon=12 typed
    assert _assemble_argv(
        "rank", fields, _state_for(fields, horizon="12"),
    ) == ["rank", "--sport", "tennis", "--horizon", "12"]
    return 2


def _t_any_danger() -> int:
    fields = [
        Field(label="mode", kind="choice", choices=[
            Choice(label="safe"),
            Choice(label="DANGER", danger=True),
        ]),
    ]
    assert _any_danger(fields, _state_for(fields, mode=0)) is False
    assert _any_danger(fields, _state_for(fields, mode=1)) is True
    # Number-only fields can never be "danger".
    f_n = [Field(label="x", kind="number", flag="--x", default_label="0")]
    assert _any_danger(f_n, _state_for(f_n)) is False
    return 3


# ---------------------------------------------------------------------------
# registered command specs
# ---------------------------------------------------------------------------


def _t_specs_well_formed() -> int:
    """Every Field in every registered command spec is well-formed —
    choice fields carry a (callable or list) `choices` entry, number
    fields carry a `flag`. `_init_state` resolves callable choices, so
    it must not crash on any registered spec.
    """
    n = 0
    for command, fields in _COMMAND_FORMS.items():
        for f in fields:
            if f.kind == "choice":
                assert f.choices is not None, (command, f.label)
            elif f.kind == "number":
                assert f.flag is not None, (command, f.label)
            else:
                raise AssertionError(f"unknown kind: {f.kind} in {command}")
        state = _init_state(fields)
        assert len(state.choice_index) == len(fields)
        n += 1
    return n


def _t_collect_short_circuit() -> int:
    """Commands with no field spec short-circuit to `[command]` and
    never touch the console. `positions` is the in-tree case; an unknown
    name falls into the same branch (the parser upstream rejects bad
    names — the menu is defensive).
    """
    assert collect(None, "positions") == ["positions"]
    assert collect(None, "no-such-command") == ["no-such-command"]
    return 2


def _t_execute_live_is_danger() -> int:
    """The execute form's `mode=LIVE` must arm `_any_danger` so the
    confirm gate fires. Regression check: this is the load-bearing
    safety affordance the menu inherited from the standalone confirm.
    """
    fields = _COMMAND_FORMS["execute"]
    state = _init_state(fields)
    mode_i = next(i for i, f in enumerate(fields) if f.label == "mode")
    live_idx = next(
        j for j, c in enumerate(state.choice_resolved[mode_i])
        if c.label == "LIVE"
    )
    assert state.choice_resolved[mode_i][live_idx].danger is True
    # Default state (mode=dry-run): no danger.
    assert _any_danger(fields, state) is False
    # Cycle to LIVE: danger.
    state.choice_index[mode_i] = live_idx
    assert _any_danger(fields, state) is True
    return 3


# ---------------------------------------------------------------------------
# the contract: every assembled argv parses
# ---------------------------------------------------------------------------


def _t_argv_round_trip() -> int:
    """For each registered command, the argv assembled from the default
    state — plus a danger-armed execute and a typed-horizon rank — must
    parse through the real `skims` parser. This is the contract that
    lets the menu skip its own validation.
    """
    parser = _build_parser()
    n = 0
    for command, fields in _COMMAND_FORMS.items():
        state = _init_state(fields)
        argv = _assemble_argv(command, fields, state)
        args = parser.parse_args(argv)
        assert args.command == command, (command, argv, args.command)
        n += 1
    # execute with mode=LIVE armed must still parse (and land live=True).
    fields = _COMMAND_FORMS["execute"]
    state = _init_state(fields)
    mode_i = next(i for i, f in enumerate(fields) if f.label == "mode")
    live_idx = next(
        j for j, c in enumerate(state.choice_resolved[mode_i])
        if c.label == "LIVE"
    )
    state.choice_index[mode_i] = live_idx
    argv = _assemble_argv("execute", fields, state)
    args = parser.parse_args(argv)
    assert args.command == "execute" and args.live is True, (argv, args)
    n += 1
    # rank with a typed horizon must parse and land as int.
    fields = _COMMAND_FORMS["rank"]
    state = _init_state(fields)
    h_i = next(i for i, f in enumerate(fields) if f.label == "horizon")
    state.number_value[h_i] = "12"
    argv = _assemble_argv("rank", fields, state)
    args = parser.parse_args(argv)
    assert args.horizon == 12, (argv, args.horizon)
    n += 1
    # Bare `positions` and `menu` — taken straight from cli, sanity check.
    assert parser.parse_args(["positions"]).command == "positions"
    assert parser.parse_args(["menu"]).command == "menu"
    n += 2
    return n


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    groups = [
        ("_assemble_argv simple choice", _t_assemble_simple_choice),
        ("_assemble_argv number default/typed", _t_assemble_number_default_vs_typed),
        ("_assemble_argv mixed", _t_assemble_mixed),
        ("_any_danger", _t_any_danger),
        ("registered specs well-formed", _t_specs_well_formed),
        ("collect short-circuit", _t_collect_short_circuit),
        ("execute LIVE is danger", _t_execute_live_is_danger),
        ("argv round-trips through parser", _t_argv_round_trip),
    ]
    failures = 0
    for name, fn in groups:
        try:
            n = fn()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok    {name} ({n} asserts)")
    if failures:
        print(f"\n{failures} group(s) failed")
        return 1
    print(f"\nall {len(groups)} groups passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
