"""Proper scoring metrics for the retro calibrate step.

Retro-only: the live `rank` path never computes a Brier score. The
scoring rules here operate on `(probability, binary-outcome)` pairs in
the predicted-winner frame — for each settled event,
`(EventFeatures.predicted_prob, EventFeatures.won)` is exactly that pair
(`predicted_prob` is the director's probability for whoever it picked;
`won` is whether that pick won).

`skims retro --step calibrate` only ever measured hit-rate, which says
nothing about *how miscalibrated* the probabilities are — a model that
calls every match 0.99 and goes 80/20 has a fine hit rate and a terrible
Brier. These rules surface that gap, overall and per-sport, and are the
measurement layer Phase 2's temperature fit corrects against.

`fit_temperature` (the offline fitter behind `--step fit-calibration`)
lives here too — fitting code belongs next to the offline retro step,
not on the hot path, mirroring how `tennis/gbt_train.py` carries its
own `_brier` / `_log_loss`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from skimsmarkets.calibration import T_MAX, T_MIN, apply_temperature

# Matches the clamp `tennis/gbt_train.py:_log_loss` uses — keeps log()
# finite when a probability lands exactly on 0.0 or 1.0.
_LOG_LOSS_EPS = 1e-15


@dataclass
class CalibrationCurveBin:
    """One reliability-diagram bin: mean predicted probability vs the
    observed win frequency, with the event count behind it. The two
    rate fields are None for an empty bin so the renderer can show the
    coverage gap honestly rather than dropping the row.
    """

    lo: float
    hi: float
    n: int
    mean_predicted: float | None
    observed_freq: float | None

    def to_dict(self) -> dict:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "mean_predicted": self.mean_predicted,
            "observed_freq": self.observed_freq,
        }


@dataclass
class ScoringMetrics:
    """Proper-scoring scorecard for one scope (overall, or one sport).

    `brier` / `log_loss` / `ece` are None when there are no settled
    events in scope; `n` is then 0 and `curve` is all-empty bins.
    """

    n: int
    brier: float | None
    log_loss: float | None
    ece: float | None
    curve: list[CalibrationCurveBin]

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "curve": [b.to_dict() for b in self.curve],
        }


def brier_score(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """Mean squared error between predicted probability and outcome.
    None on empty input.
    """
    if not pairs:
        return None
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / len(pairs)


def log_loss(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """Mean negative log-likelihood. p is clamped to [eps, 1-eps] so a
    probability of exactly 0.0 or 1.0 never produces -inf. None on empty
    input.
    """
    if not pairs:
        return None
    total = 0.0
    for p, y in pairs:
        pc = min(max(p, _LOG_LOSS_EPS), 1.0 - _LOG_LOSS_EPS)
        yf = 1.0 if y else 0.0
        total += -(yf * math.log(pc) + (1.0 - yf) * math.log(1.0 - pc))
    return total / len(pairs)


def calibration_curve(
    pairs: Sequence[tuple[float, bool]], *, n_bins: int = 10
) -> list[CalibrationCurveBin]:
    """Reliability-diagram bins over [0, 1]. Always returns exactly
    `n_bins` bins in order; bins with no events carry n=0 and None
    rates. Bin index is `min(int(p * n_bins), n_bins - 1)` so p=1.0
    lands in the last bin rather than overflowing.
    """
    sums_p = [0.0] * n_bins
    sums_y = [0.0] * n_bins
    counts = [0] * n_bins
    for p, y in pairs:
        idx = min(max(int(p * n_bins), 0), n_bins - 1)
        sums_p[idx] += p
        sums_y[idx] += 1.0 if y else 0.0
        counts[idx] += 1
    bins: list[CalibrationCurveBin] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if counts[i] == 0:
            bins.append(
                CalibrationCurveBin(
                    lo=lo, hi=hi, n=0, mean_predicted=None, observed_freq=None
                )
            )
        else:
            bins.append(
                CalibrationCurveBin(
                    lo=lo,
                    hi=hi,
                    n=counts[i],
                    mean_predicted=sums_p[i] / counts[i],
                    observed_freq=sums_y[i] / counts[i],
                )
            )
    return bins


def expected_calibration_error(
    pairs: Sequence[tuple[float, bool]], *, n_bins: int = 10
) -> float | None:
    """Count-weighted average gap between predicted probability and
    observed frequency across bins. Empty bins contribute 0 (no events,
    no defined mean to take a distance from). None on empty input.
    """
    if not pairs:
        return None
    n = len(pairs)
    ece = 0.0
    for b in calibration_curve(pairs, n_bins=n_bins):
        if b.n == 0:
            continue
        ece += (b.n / n) * abs(b.mean_predicted - b.observed_freq)
    return ece


def compute_metrics(
    pairs: Sequence[tuple[float, bool]], *, n_bins: int = 10
) -> ScoringMetrics:
    """Bundle brier + log-loss + ECE + calibration curve into one
    `ScoringMetrics`. The single entry the retro aggregate step calls
    per scope — threads one `n_bins` through ECE and the curve so the
    two can't drift.
    """
    return ScoringMetrics(
        n=len(pairs),
        brier=brier_score(pairs),
        log_loss=log_loss(pairs),
        ece=expected_calibration_error(pairs, n_bins=n_bins),
        curve=calibration_curve(pairs, n_bins=n_bins),
    )


# A 1-parameter fit on a binary target needs enough resolved events that
# the NLL minimum isn't just sampling noise. 75 is a few weeks of
# accumulated blind-mode runs — same cold-start posture as the GBT prior
# (degrade to the no-op until there's real data). Tunable.
MIN_FIT_N = 75

# Second guardrail: a 90/3 win/loss split clears MIN_FIT_N but the fit is
# still junk — require at least this many of *each* class. Also subsumes
# the all-one-class case (NLL is then monotone, no interior minimum).
MIN_CLASS_N = 10

_GOLDEN_RATIO = (5.0 ** 0.5 - 1.0) / 2.0  # ~0.618


def _nll(pairs: Sequence[tuple[float, bool]], t: float) -> float:
    """Mean negative log-likelihood of the data under temperature `t` —
    `log_loss` of the temperature-scaled probabilities. The objective
    `fit_temperature` minimises.
    """
    scaled = [(apply_temperature(p, t), y) for p, y in pairs]
    return log_loss(scaled)  # pairs non-empty by the time this is called


def fit_temperature(
    pairs: Sequence[tuple[float, bool]], *, min_n: int = MIN_FIT_N
) -> float | None:
    """Fit a single temperature T by minimising NLL over [T_MIN, T_MAX].

    Golden-section search — scipy isn't a dependency, and NLL(T) is
    unimodal in T for a logistic model (one interior minimum), so a
    derivative-free bracket search converges reliably.

    Returns None — the caller keeps T=1.0 and writes no artefact — when:
      - `len(pairs) < min_n`: too little data to trust a 1-param fit.
      - fewer than `MIN_CLASS_N` of either outcome class: the fit is
        junk (and the all-one-class case has no interior NLL minimum at
        all — the optimum runs to a bound).
    Otherwise returns the fitted T, clamped to [T_MIN, T_MAX].

    Market-blind: `pairs` are `(predicted_prob, won)` — the fit only ever
    sees win/loss outcomes, never a market price.
    """
    if len(pairs) < min_n:
        return None
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if n_pos < MIN_CLASS_N or n_neg < MIN_CLASS_N:
        return None

    lo, hi = T_MIN, T_MAX
    c = hi - _GOLDEN_RATIO * (hi - lo)
    d = lo + _GOLDEN_RATIO * (hi - lo)
    fc = _nll(pairs, c)
    fd = _nll(pairs, d)
    while hi - lo > 1e-4:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - _GOLDEN_RATIO * (hi - lo)
            fc = _nll(pairs, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _GOLDEN_RATIO * (hi - lo)
            fd = _nll(pairs, d)
    return min(max((lo + hi) / 2.0, T_MIN), T_MAX)
