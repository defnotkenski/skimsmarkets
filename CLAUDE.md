# CLAUDE.md

Orientation for future sessions. Read the code for specifics.

## What this is

A confidence-ranked list of today's Kalshi tennis matches, **plus
an opt-in Kalshi trader (`skims execute`) that consumes the ranker's
JSONL**. The two halves are deliberately separated:

- **Ranker** (`skims rank` and everything under `pipeline.py`,
  `agents/`, lens providers) is **not** an edge finder. No buy/pass
  gates, no edge thresholds, no position sizes. The LLM ranks by
  predicted probability; trade decisions don't live here.
- **Executor** (`skims execute` under `src/skimsmarkets/execute/`,
  Kalshi adapter under `src/skimsmarkets/kalshi/`) is the opt-in,
  deterministic trade layer. Reads `logs/runs/<run_id>.jsonl`,
  applies a flag-based filter (no LLM), places Kalshi orders.

Kalshi = data + execution (single venue, tighter loop). Polymarket
survives only as a UW bridge: gamma `/events?tag_slug=tennis`
provides the slug Kalshi events resolve to so Unusual Whales wallet
flow can attach to `event.uw_context`. If you're adding trade logic
to the ranker, you're in the wrong package — push it into `execute/`.

`PolymarketEvent` is the venue-neutral pipeline event type despite
the misleading name — Kalshi-sourced events are also `PolymarketEvent`
instances, built by `kalshi/slate.py`. Rename to `MarketEvent` is
deferred to avoid touching ~84 call sites + the JSONL schema.

## Toolchain

- `uv` for everything (`uv sync`, `uv run …`). Never `pip` / `python`.
- `ruff` for linting. No formatter, no type-checker.
- Python 3.13+. Secrets in `.env` (see `.env.example`).

## Pipeline shape

```
kalshi /series + /events  →  pre-LLM selection (fundamental imbalance)
                          →  per-sport lens chains (provider fetcher → Claude reasoner, parallel)
                          →  Claude director per event (synthesises lens reports)
                          →  Claude judge over the slate (defensibility score)
                          →  deterministic post-processing (rendering, ranking, JSONL persistence)
```

LLMs only in the agent layer; everything else is deterministic.
Data + execution venue: `api.elections.kalshi.com` (public reads:
`/series`, `/events`, `/markets/{ticker}/orderbook`,
`/series/{s}/markets/{t}/candlesticks`. RSA-signed POST `/portfolio/
orders` — opt-in via `skims execute`). UW bridge: `gamma-api.
polymarket.com /events?tag_slug=tennis` (one HTTP per pipeline run).

## Where things live

| Path | What |
| --- | --- |
| `src/skimsmarkets/pipeline.py` | Orchestrator, JSONL persistence |
| `src/skimsmarkets/cli.py` | `skims` entry point (rank / fetch / backtest / retro / gbt / execute) |
| `src/skimsmarkets/agents/` | LLM layer (director, reasoners, judge, fetcher providers) |
| `src/skimsmarkets/agents/sports/<sport>/` | Per-sport lens registration |
| `src/skimsmarkets/polymarket/` | UW bridge: `PolymarketEvent` model (venue-neutral type), `find_polymarket_slug` reverse matcher |
| `src/skimsmarkets/tennis/` | Tennis stats vendor (MatchStat) + sim |
| `src/skimsmarkets/unusual_whales/` | UW flow context (gamma slug → asset_id → wallet flow) |
| `src/skimsmarkets/retro/` | Self-improvement layer (`skims retro`) |
| `src/skimsmarkets/kalshi/` | Kalshi venue adapter — `slate.py` (Polymarket → Kalshi data swap), `enrichment.py` (book + history), `client.py` (read + RSA-signed orders), `matcher.py` |
| `src/skimsmarkets/execute/` | Deterministic trader: ranked JSONL → Kalshi orders |
| `logs/runs/<run_id>.jsonl` | One file per pipeline run |
| `logs/trades/<run_id>.jsonl` | Audit log for one `skims execute` invocation |

## Load-bearing invariants

- **Sport-keyed lens registry is strict.** Events with no registered
  sport drop at `lens_dispatch` BEFORE enrichment fan-out. Adding a
  sport: build `agents/sports/<sport>/`, register in `SPORT_LENS_SETS`.
- **Kalshi bid/ask is a PRIOR, not a ceiling.** The director can
  name the market underdog as winner when synthesis genuinely supports
  it. Don't soften "pull back toward the market" language back into
  the prompt.
- **Both Kalshi YES sides are independent books.** A tennis match
  event holds two `KalshiMarket` records (one per player), each with
  its own native YES book. The slate adapter constructs one
  `PolymarketMarket` per side reading prices/depth directly from that
  side — never inverts the favorite's book. Don't reintroduce
  `inverted_no_side` semantics on the Kalshi data path.
- **`confidence` measures real-world contingency robustness**, not
  matchup lopsidedness. Lens-internal agreement is what
  `defensibility_score` measures separately.
- **`team_a_name` is canonical.** Flows verbatim from market label
  through prompts → typed reports → `predicted_winner` → market lookup.
  Never reformat / capitalise / abbreviate.
- **Lenses are siloed.** Each lens runs its own fetcher → reasoner
  chain. Director sees all reports. UW flow + tennis sim are
  director-only — don't pipe them into lens prompts.
- **Per-event drops, not per-run aborts.** All failures degrade
  silently into `ErrorRecord` rows in the JSONL.
- **Prices are dollars `[0.0, 1.0]`** — implied probabilities, not
  cents.

## Don't

- Flip the LLM's pick in deterministic code (raise on mismatch).
- Conflate game-start time (slate filter) with settlement time.
- Pipe one lens's output into another lens's context.
- Add `if enabled` branches for optional vendors — use stub providers.
- Import `execute/` or `kalshi/` from anywhere in `pipeline.py` /
  `agents/` / lens code. The ranker doesn't know trades exist.
- Put any LLM call in `execute/` or `kalshi/`. The trade path is
  deterministic on purpose — that's what makes scheduled cloud
  routines safe to run without human review.

## Code style

Terse. `from __future__ import annotations`. Pydantic
`BaseModel(ConfigDict(extra="ignore"))` for external payloads,
`dataclass` for internal. Comment the *why* of non-obvious choices,
not the *what*. Flat package layout — sub-packages own one vendor or
capability.

## Note to future Claude sessions

This file is an **architectural scaffold** — orientation, not
documentation. When editing it, keep these in mind:

- The codebase is the source of truth. Anything findable by reading
  the relevant module or its docstring belongs there, not here.
- Keep load-bearing principles (rules a future session could violate
  without realising). Drop implementation specifics (field lists,
  endpoints, bucket boundaries, exact thresholds).
- Favour the *why* over the *how* — mechanics live in the code.
- Prefer deleting stale lines over padding new ones.
