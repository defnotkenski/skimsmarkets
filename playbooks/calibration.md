# Calibration playbook

Pre-flight checklist before flipping any mode (`confidence`, `ev`, `tail`) to `--live`. Confirms the director's calibrated probabilities are accurate in the mode's market_p range AND that realized EV matches the math.

## TL;DR — one command

```bash
uv run python scripts/run_calibration_suite.py
```

Runs all three read-only checks in sequence (retro calibrate + validate_ev_calibration for ev/tail/confidence) and prints a verdict + recommended next step. Does NOT fit a new temperature — that step is operator-gated. Exit code 0 = OK or INCONCLUSIVE; 1 = at least one miscalibration warning fired.

## When to run

- **Always** before flipping a mode to `--live` for the first time
- **Periodically** as `logs/runs/` accumulates (the corpus is too small today for stable verdicts; expect INCONCLUSIVE until ~50+ rows land in each mode's market_p range)
- **After** any significant change to the GBT model, director prompt, or temperature artifact
- **When** trading PnL diverges materially from backtest expectations

## The full pipeline

### Step 1 — Read-only Brier / log-loss / ECE report

```bash
uv run skims retro --step calibrate
```

What it does: auto-refreshes gamma resolutions, computes Brier / log-loss / Expected Calibration Error BEFORE and AFTER the live committed temperature in `models/tennis_calibration.json`. With no artifact committed (current state), the two are identical.

Reads: every settled run in `logs/runs/*.jsonl` joined with `*.resolutions.jsonl`.

Output: terminal tables only (no file artifact).

### Step 2 — Per-mode calibration + realized-EV validation

```bash
uv run python scripts/validate_ev_calibration.py --mode ev          # default
uv run python scripts/validate_ev_calibration.py --mode tail
uv run python scripts/validate_ev_calibration.py --mode confidence
```

Two checks per mode invocation:

- **Calibration check**: filters predictions to the mode's market_p range, bins by model confidence, compares predicted probability vs realized win rate.
- **Realized-EV check** (mode-agnostic, always runs): per ev_bucket, compares mean predicted `ev_per_dollar` vs mean realized $-return per $1 staked at the persisted market_p. Answers "did the EV math hold?"

Reads: `logs/runs/*.jsonl` + `*.resolutions.jsonl`.

Output: per-mode terminal report + exit code (0 = OK / INCONCLUSIVE, 1 = miscalibration warning).

### Step 3 (CONDITIONAL) — Fit a new temperature

```bash
uv run skims retro --step fit-calibration
```

Only run this if Step 1 or Step 2 fired a miscalibration warning AND the cause is uniform miscalibration (model is over/under-confident across the board, not just at the tails).

What it does: fits a single scalar T by minimizing NLL on resolved win/loss outcomes (golden-section search, range [0.25, 5.0]). Writes `models/tennis_calibration.json`. Operator-gated by design — the ONLY step that writes a model artifact.

Prerequisites:
- ≥ 75 settled predictions in the corpus (the `MIN_FIT_N` floor in `retro/metrics.py`)
- ≥ 10 of each class (wins + losses) in the corpus

After fitting, **re-run Step 1 + Step 2** to confirm the new temperature actually improves the metrics.

## Decision matrix

| Suite output | Diagnosis | Action |
|---|---|---|
| All checks INCONCLUSIVE | Corpus too small | Re-run after more settled predictions accumulate. At ~30-60 predictions/day cadence, ev/confidence ranges fill in days; tail range fills in weeks. |
| All checks OK | Mode is calibrated; EV math holds | Safe to consider `--live` for that mode (after the usual safety review). Pair with the right per-mode CLI flags. |
| Step 1 reports Brier degraded vs old temp | Calibration has drifted | Fit a new temperature (Step 3). Re-run suite. |
| Step 2 calibration check fires WARNING for ev/confidence | Uniform miscalibration in the mode's range | Fit a new temperature (Step 3). Re-run suite. |
| Step 2 calibration check fires WARNING for tail | **Selective tail miscalibration** | Temperature scaling cannot fix this. Either: (a) accept tail bet bleed risk, (b) build isotonic recalibration on a 6+ month window (NOT shipped — would need new code), (c) defer tail mode until corpus supports the isotonic fit. |
| Step 2 realized-EV check fires WARNING | EV math is over-optimistic vs real outcomes | Possible causes: (1) calibration drift → Step 3; (2) market_p drift between rank-time and trade-time (look at trader logs); (3) tail-selective miscalibration (see row above). |

## Per-mode trade-time flags (after suite clears)

The suite tells you whether the model is calibrated. It does NOT configure the trader. Recommended flag combinations:

```bash
# Confidence mode (default, lowest risk)
skims execute --mode confidence

# Moderate-EV mode
skims execute --mode ev

# Tail / asymmetric-payoff mode
skims execute --mode tail \
              --min-market-implied 0.0 \
              --min-ev 0.30 \
              --bet-size-cents 100
```

The tail flags are load-bearing: `--min-market-implied 0.0` allows directional disagreement (the strategy's whole point), `--min-ev 0.30` requires Prime-bucket EV to compensate for variance, `--bet-size-cents 100` ($1 bets) caps per-bet downside while you accumulate convergence data.

## What this playbook does NOT cover

- **Isotonic recalibration on a 6+ month window**: the right tool for selective tail miscalibration but not built. See `project_tennis_iteration_archive.md` for the rationale (90-day variant was tested and got worse; 6+ month window needs data we don't have yet).
- **Per-tour calibration**: ATP and WTA may calibrate differently. The current temperature is single-tour. Splitting would require ~150+ settled predictions per tour.
- **Per-surface calibration**: same issue, even thinner data.
- **Per-sport calibration**: only tennis is wired up today. Other sports would need their own temperature artifact + validation pass.

## Maintenance

If you tweak the calibration analysis logic, do it in:
- `scripts/validate_ev_calibration.py` — per-mode validation
- `src/skimsmarkets/retro/metrics.py` — `fit_temperature` + `_nll` (the offline fitter)
- `src/skimsmarkets/calibration.py` — `apply_temperature` (the live-path primitive)

The suite wrapper (`scripts/run_calibration_suite.py`) is intentionally thin — it just calls the underlying tools in sequence. New analyses should land in the per-mode script, not in the wrapper.
