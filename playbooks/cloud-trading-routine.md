# Cloud trading routine playbook

Use when triggered by a scheduled run (cron / cloud scheduler) to walk the full end-to-end trading flow: exposure pre-flight → slate probe → rank → execute live. This playbook IS the trigger prompt — the scheduler invokes Claude with this file as the instruction set.

**Routine scope: RUN + READ + REPORT only.** Run the commands below verbatim, read their stdout/stderr, report the result. **DO NOT** edit any file in the repo, install or update packages, perform git write operations (`add` / `commit` / `push` / `branch` / `checkout -b` / `reset` / `tag`), or "fix" anything that looks broken. If a step fails, abort and report — the human handles any fixes outside the routine. The trading commands write their own audit logs internally (`logs/runs/*.jsonl`, `logs/trades/*.jsonl`); the routine never writes files directly.

## Flow

```
Step 1: skims positions      → headroom?     → no  → abort clean
Step 2: skims fetch          → events?       → no  → abort clean
Step 3: skims rank           → predictions?  → no  → abort clean
Step 4: skims execute --live → fills logged
Step 5: report
```

Each step gates the next. Failing the gate at any step aborts the routine and reports — that's the normal happy path for "nothing to do today."

## Step 1 — Exposure pre-flight

```
uv run skims positions
```

Output is key=value lines. Parse:

- `can_place_bet=true|false` — primary gate
- `open_exposure_dollars` — current exposure
- `headroom_dollars` — `cap - exposure`

**Decision:**
- `can_place_bet=false` → **abort clean.** Report: `"open exposure $X / cap $Y, headroom $Z < bet size $B. No trade today."` Exit.
- Subcommand fails (network, auth, etc.) → **abort error.** Report the stderr. Don't proceed without an exposure read — that's the load-bearing safety check.
- `can_place_bet=true` → continue.

## Step 2 — Slate probe

```
uv run skims fetch --sport tennis
```

Zero LLM cost — just gamma `/events` + horizon filter. Prints a Rich-formatted table.

**Decision:**
- "No live markets found" in stdout → **abort clean.** Report: `"no in-window tennis events. Skipping rank."` Exit.
- Table has ≥ 1 row → continue.

Don't try to count rows precisely from the Rich-styled output. The presence vs absence of the "No live markets found" string is the reliable signal.

## Step 3 — Rank

```
uv run skims rank --sport tennis
```

Runs the full LLM pipeline. Produces `logs/runs/<run_id>.jsonl`. The run_id is embedded in Rich-styled stdout, which isn't reliable to parse — instead, after the command completes, grab the most recently written file:

```
RUN_ID=$(ls -t logs/runs/*.jsonl | head -1 | xargs basename | sed 's/\.jsonl$//')
```

**Decision:**
- Command exits non-zero → **abort error.** Report stderr. Don't retry — partial LLM state may have leaked partial rows.
- Command exits zero, file has zero `record_type="prediction"` rows → **abort clean.** Report: `"rank produced 0 predictions (all events errored or dropped). No trade today."` Exit. (Inspect with `grep -c '"record_type":"prediction"' logs/runs/$RUN_ID.jsonl`.)
- ≥ 1 prediction row → continue.

## Step 4 — Execute live

```
uv run skims execute --run-id $RUN_ID --live
```

The trader will:
- Re-read open exposure (defense in depth — your step 1 might be stale by minutes).
- Apply the deterministic filter set.
- Match each prediction to a Kalshi market by surname pair.
- Place one IOC limit buy per match, sized by `bet_size_cents`.
- Write one audit row per prediction to `logs/trades/$RUN_ID.jsonl`.

The final stdout line:

```
execute: predictions=N passed=K filled=A partial=B submitted=C dry_run=D skipped=E total_cost_cents=F
```

**Decision:**
- Exits non-zero → **abort error.** Report stderr + the audit log at `logs/trades/$RUN_ID.jsonl` for forensics. Don't retry — Kalshi orders are not safe to blind-retry (the `client_order_id` dedupe protects against duplicates within ONE call, but an outer retry uses a fresh UUID per row).
- Exits zero with `filled=0 partial=0 submitted=0` → no orders placed (everything filtered or unmatched). Not an error — report and exit.
- Exits zero with at least one fill → continue to report.

## Step 5 — Report

Compose a short summary for the scheduler log. Pull from the per-step outputs:

```
Trading routine $RUN_ID
- Exposure before: $X.XX / $Y.YY cap (headroom $Z.ZZ)
- Slate: <N events found>
- Rank: <P predictions ranked>
- Execute: <K passed filters, A filled / B partial / C submitted / E skipped>
  - filled cost: $F.FF
  - skip reasons: <reasons map from stdout if present>
- Exposure after: <re-run `skims positions` and quote new exposure, or
  estimate as exposure_before + total_cost_cents>
```

Keep it tight — the scheduler will surface this in dashboards and digests.

## Abort taxonomy

| Where | Reason | Severity | Operator action |
|---|---|---|---|
| Step 1 | `can_place_bet=false` | clean | none — wait for positions to settle / close |
| Step 1 | network / auth failure | error | check Kalshi credentials, API status |
| Step 2 | no in-window events | clean | none — wait for slate |
| Step 3 | rank crash | error | check LLM provider, Polymarket reachability |
| Step 3 | 0 predictions | clean | event-level errors logged in JSONL; inspect if recurring |
| Step 4 | execute crash | error | inspect `logs/trades/$RUN_ID.jsonl` for partial state |
| Step 4 | 0 trades placed | clean | filter set may be too strict; inspect skip reasons |

Only "error" rows warrant alerting the operator. "Clean" aborts are the expected behavior for off-hours / empty-slate days.

## What NOT to do

- **NEVER edit code in this repo.** The routine is an orchestrator — the trade layer was authored deliberately by a human and is checked into git. Even if you see what looks like a bug, a missing import, a TODO, or a "small improvement", do NOT use Edit / Write / NotebookEdit on any file. Abort the run and report instead; the human will fix it outside the routine.
- **NEVER perform git write operations.** No `git add`, `git commit`, `git push`, `git branch`, `git checkout -b`, `git reset`, `git tag`, `git stash`, etc. Read-only git (`git status`, `git log`, `git diff`) is fine for diagnostics if you need it. Code in this repo is published to GitHub on a human's schedule, not the routine's.
- **NEVER install or update packages.** No `uv add`, `uv lock`, `pip install`, `pip uninstall`, modifications to `pyproject.toml` or `uv.lock`. If `uv run skims ...` fails because dependencies aren't available, that's a sandbox-provisioning issue — abort and report; don't try to fix it from inside the routine. (Running `uv sync` once to populate the venv against the existing `uv.lock` is OK if needed; that's read-only against the lockfile.)
- **NEVER "fix" a failing step.** If a command fails, abort with the stderr + relevant log paths and exit. Trying to patch state mid-routine risks corrupting the audit log or leaving Kalshi in an inconsistent state.
- **Don't retry on Kalshi-side failure.** A request that times out mid-POST leaves order state unknown. The trader's `client_order_id` dedupe handles single-call retries; a blind outer retry uses fresh UUIDs and would risk duplicates.
- **Don't override `--max-open-exposure-cents` upward to fit a trade.** The cap is the portfolio safety limit; if there's no headroom, the answer is "trade fewer / smaller" or "wait", not "widen the limit".
- **Don't skip step 1** even though `skims execute --live` re-reads exposure internally. The pre-flight saves the LLM cost of `skims rank` on days when there's no headroom anyway.
- **Don't add edge gates, EV thresholds, or sizing logic.** Per CLAUDE.md the trade layer is deterministic by design. All knobs are CLI flags: `--bet-size-cents`, `--max-open-exposure-cents`, `--confidence`, `--min-defensibility`, `--no-negative-edge`, `--sport`.
