"""Probability calibration — temperature scaling for the risk classifier.

Lives on the deterministic-ranker side next to `classify.py`. The
director emits raw stacking-math probabilities with zero calibration;
this module applies a single scalar temperature T (fit offline by the
retro layer on resolved win/loss outcomes — NEVER on price, so
calibration stays market-blind) to the magnitude term of
`classify_risk`.

Temperature scaling has 0.5 as its fixed point and is monotone, so it
structurally cannot move a probability across 0.5 — it rescales
confidence without ever flipping the director's pick. That property is
why temperature scaling was chosen over Platt / isotonic, which can.

This module holds only the live-path primitives the ranker needs. The
scoring rules and the offline fitter (`fit_temperature`) live in
`retro/metrics.py` — nothing on the `rank` path fits a T.

Artefact: `models/tennis_calibration.json` (committed to the repo, same
posture as the GBT artefacts). Absent artefact → T=1.0 everywhere →
exact pre-calibration behaviour (no-op cold start).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)

# Relative path — same cwd-dependent posture as `gbt_train.MODEL_PATH`;
# the CLI is always run from the repo root.
CALIBRATION_PATH = Path("models/tennis_calibration.json")

# Clamp keeping logit() finite when a probability lands exactly on 0/1.
_P_CLAMP = 1e-6

# Fit search bounds, and the sanity range `load_temperature` clamps to —
# a stored value outside this band is treated as corrupt and ignored.
T_MIN, T_MAX = 0.25, 5.0


def _logit(p: float) -> float:
    pc = min(max(p, _P_CLAMP), 1.0 - _P_CLAMP)
    return math.log(pc / (1.0 - pc))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def apply_temperature(p: float, t: float) -> float:
    """`sigmoid(logit(clamp(p)) / t)`.

    Identity at `t == 1.0`. Fixed point 0.5 — `apply_temperature(0.5, t)`
    is 0.5 for every t, and the map is monotone, so a probability stays
    on whichever side of 0.5 it started: temperature scaling can never
    flip the director's pick. `t > 1` pulls toward 0.5 (de-confidence),
    `t < 1` pushes away.
    """
    return _sigmoid(_logit(p) / t)


def load_temperature(sport: str) -> float:
    """Per-sport temperature from `models/tennis_calibration.json`.

    Returns 1.0 (the identity / no-op transform) on every failure mode:
    file absent, unreadable, not a JSON object, sport key missing, value
    non-numeric, non-finite, or outside [T_MIN, T_MAX]. Cold start with
    no committed artefact is therefore exactly the pre-calibration path.

    `sport` is keyed now for forward-compatibility — v1 fits tennis only;
    other sports miss the key and correctly get 1.0.
    """
    if not CALIBRATION_PATH.exists():
        return 1.0
    try:
        data = json.loads(CALIBRATION_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning(
            "calibration: could not read %s (%s) — using T=1.0",
            CALIBRATION_PATH, e,
        )
        return 1.0
    if not isinstance(data, dict):
        return 1.0
    entry = data.get(sport)
    if not isinstance(entry, dict):
        return 1.0
    raw = entry.get("temperature")
    try:
        t = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t) or not (T_MIN <= t <= T_MAX):
        log.warning(
            "calibration: %s temperature %r outside [%.2f, %.2f] — using T=1.0",
            sport, raw, T_MIN, T_MAX,
        )
        return 1.0
    return t


def write_calibration(
    entries: dict[str, dict], *, path: Path = CALIBRATION_PATH
) -> None:
    """Persist the fitted artefact. `entries` is keyed by sport; each
    value carries at least `temperature`, plus the fit metadata and the
    before/after scorecard the operator reviews. Single `write_text`
    call (mirrors `gbt_train`), so a crash mid-fit leaves any existing
    artefact intact rather than half-written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, default=str) + "\n")
