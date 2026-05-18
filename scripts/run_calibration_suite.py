"""Run the full read-only calibration suite end-to-end.

One command runs three checks in sequence:

  1. `skims retro --step calibrate` — Brier / log-loss / ECE + hit-rate
     cuts BEFORE and AFTER the live committed temperature. Tells you
     whether the existing temperature is still appropriate.
  2. `validate_ev_calibration.py --mode ev` — per-bin predicted vs
     realized win rate in the moderate-EV market_p range.
  3. `validate_ev_calibration.py --mode tail` — same in the deep-
     underdog range (the harder calibration problem — temperature
     scaling can't fix selective tail miscalibration).

This script does NOT fit a new temperature. That step (`skims retro
--step fit-calibration`) writes `models/tennis_calibration.json` and
is operator-gated by design. The suite's job is to tell you whether
fitting is needed; the fit itself stays your call.

Exit code: 0 = OK / INCONCLUSIVE everywhere, 1 = at least one
miscalibration warning fired. CI-able.

Run from project root:

    uv run python scripts/run_calibration_suite.py
    uv run python scripts/run_calibration_suite.py --skip-retro   # validation only
    uv run python scripts/run_calibration_suite.py --modes ev tail
"""

from __future__ import annotations

import argparse
import subprocess
import sys

ALL_MODES = ("ev", "tail", "confidence")


def _section(title: str) -> None:
    print()
    print("#" * 78)
    print(f"#  {title}")
    print("#" * 78)


def _run(cmd: list[str]) -> int:
    """Run a subprocess, stream output, return exit code."""
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-retro", action="store_true",
        help="Skip `skims retro --step calibrate` (run validation checks only).",
    )
    ap.add_argument(
        "--modes", nargs="+", choices=ALL_MODES, default=list(ALL_MODES),
        help=f"Modes to validate via the per-mode script (default: {' '.join(ALL_MODES)}).",
    )
    ap.add_argument(
        "--min-samples", type=int, default=50,
        help="Pass through to validate_ev_calibration.py.",
    )
    args = ap.parse_args()

    exit_codes: list[int] = []

    if not args.skip_retro:
        _section("STEP 1 — Retro calibrate (Brier / log-loss / ECE)")
        rc = _run(["uv", "run", "skims", "retro", "--step", "calibrate"])
        exit_codes.append(rc)

    for mode in args.modes:
        _section(f"STEP 2.{mode} — Validate calibration + realized EV ({mode} mode)")
        rc = _run([
            "uv", "run", "python", "scripts/validate_ev_calibration.py",
            "--mode", mode,
            "--min-samples", str(args.min_samples),
        ])
        exit_codes.append(rc)

    _section("SUMMARY + RECOMMENDED NEXT STEP")
    worst = max(exit_codes) if exit_codes else 0
    if worst == 0:
        print("\nAll checks returned OK or INCONCLUSIVE.")
        print()
        print("If INCONCLUSIVE: corpus is too small for a verdict. Re-run after")
        print("more settled predictions accumulate (`logs/runs/`). At 24 rows")
        print("today, the corpus needs to grow to ~50+ in each mode's market_p")
        print("range before calibration becomes assessable.")
        print()
        print("If OK on the mode you want to trade: safe to consider --live (after")
        print("the usual safety review). Pair with the right CLI flags per mode:")
        print("  - confidence: `skims execute --mode confidence` (defaults are fine)")
        print("  - ev:         `skims execute --mode ev`")
        print("  - tail:       `skims execute --mode tail --min-market-implied 0.0 "
              "--min-ev 0.30 --bet-size-cents 100`")
    else:
        print("\nAt least one check fired a miscalibration WARNING.")
        print()
        print("Recommended action:")
        print("  1. If the calibration check fired (Brier or per-bin predicted vs")
        print("     realized diverged by > 5pp): fit a new temperature with")
        print("     `uv run skims retro --step fit-calibration`. Writes")
        print("     `models/tennis_calibration.json`. Re-run this suite afterward.")
        print()
        print("  2. If the realized-EV check fired (predicted ev_per_dollar exceeds")
        print("     realized by > $0.05/$1): the math is over-optimistic. Possible")
        print("     causes: (a) calibration drift (fix per #1), (b) market_p drift")
        print("     between rank-time and Kalshi trade-time (look at trader logs),")
        print("     (c) tail-selective miscalibration that temperature can't fix")
        print("     (isotonic on 6+ month window would help but isn't built).")
        print()
        print("  3. DO NOT flip the affected mode to --live until the warning")
        print("     clears or you've consciously accepted the bleed risk.")
        print()
        print("See `playbooks/calibration.md` for the full decision tree.")
    return worst


if __name__ == "__main__":
    sys.exit(main())
