# CLAUDE.md

Orientation for future sessions. Read the code for specifics.

## What this is

A confidence-ranked list of today's Polymarket sports markets (tennis
in v1), **plus an opt-in Kalshi trader (`skims execute`) that consumes
the ranker's JSONL**. The two halves are deliberately separated:

- **Ranker** (`skims rank` and everything under `pipeline.py`,
  `agents/`, lens providers) is **not** an edge finder. No buy/pass
  gates, no edge thresholds, no position sizes. The LLM ranks by
  predicted probability; trade decisions don't live here.
- **Executor** (`skims execute` under `src/skimsmarkets/execute/`,
  Kalshi adapter under `src/skimsmarkets/kalshi/`) is the opt-in,
  deterministic trade layer. Reads `logs/runs/<run_id>.jsonl`,
  applies a flag-based filter (no LLM), places Kalshi orders.

**Dual-venue.** Polymarket = data (gamma `/events` for slate, gamma
`/markets?slug=` for token-id resolution, CLOB `/book` + `/prices-
history` for per-market enrichment). Kalshi = execution (RSA-signed
`POST /portfolio/orders`). Cross-venue bridge: surname matching at
trade time in `kalshi/matcher.py` — Polymarket-sourced predictions
that have no Kalshi counterpart drop with `MatchOutcome.kind=
"no_kalshi_match"`. The single-venue collapse was tried briefly
(2026-05-11 → 2026-05-12) and reverted on data-quality grounds; the
split is intentional. If you're adding trade logic to the ranker,
you're in the wrong package — push it into `execute/`.

`PolymarketEvent` is the pipeline event type (now correctly named
since the slate is Polymarket-sourced again). Built by
`polymarket/slate.py:fetch_gamma_slate` → `PolymarketEvent.from_gamma`.

## Toolchain

- `uv` for everything (`uv sync`, `uv run …`). Never `pip` / `python`.
- `ruff` for linting. No formatter, no type-checker.
- Python 3.13+. Secrets in `.env` (see `.env.example`).

## Pipeline shape

```
gamma /events  →  CLOB book + price-history enrichment
               →  pre-LLM selection (fundamental imbalance)
               →  per-sport lens chains (provider fetcher → Claude reasoner, parallel)
               →  Claude director per event (synthesises lens reports)
               →  Claude judge over the slate (defensibility score)
               →  deterministic post-processing (rendering, ranking, JSONL persistence)
```

LLMs only in the agent layer; everything else is deterministic.
Data hosts: `gamma-api.polymarket.com` (slate listing, slug→token-id),
`clob.polymarket.com` (order book + price history). Execution host:
`api.elections.kalshi.com` (RSA-signed POST `/portfolio/orders`,
opt-in via `skims execute`).

## Where things live

| Path | What |
| --- | --- |
| `src/skimsmarkets/pipeline.py` | Orchestrator, JSONL persistence |
| `src/skimsmarkets/cli.py` | `skims` entry point (rank / fetch / backtest / retro / gbt / execute) |
| `src/skimsmarkets/agents/` | LLM layer (director, reasoners, judge, fetcher providers) |
| `src/skimsmarkets/agents/sports/<sport>/` | Per-sport lens registration |
| `src/skimsmarkets/polymarket/` | Slate + CLOB enrichment: `slate.py` (gamma `/events` listing), `enrichment.py` (CLOB book + history), `models.py` (`PolymarketEvent`, `PolymarketMarket`, `from_gamma`) |
| `src/skimsmarkets/clob/` | Bare CLOB HTTP fetchers (book + price history) + summarizers — shared by live + backtest |
| `src/skimsmarkets/tennis/` | Tennis stats vendor (MatchStat) + sim |
| `src/skimsmarkets/unusual_whales/` | UW flow context + gamma slug → asset_id resolver (shared cache with CLOB enrichment) |
| `src/skimsmarkets/retro/` | Self-improvement layer (`skims retro`) |
| `src/skimsmarkets/kalshi/` | Kalshi execution venue — `client.py` (events read + RSA-signed orders), `matcher.py` (surname-pair matching, used at trade time) |
| `src/skimsmarkets/execute/` | Deterministic trader: ranked JSONL → Kalshi orders |
| `logs/runs/<run_id>.jsonl` | One file per pipeline run |
| `logs/trades/<run_id>.jsonl` | Audit log for one `skims execute` invocation |

## Load-bearing invariants

- **Sport-keyed lens registry is strict.** Events with no registered
  sport drop at `lens_dispatch` BEFORE enrichment fan-out. Adding a
  sport: build `agents/sports/<sport>/`, register in `SPORT_LENS_SETS`.
- **Polymarket bid/ask is a PRIOR, not a ceiling.** The director can
  name the market underdog as winner when synthesis genuinely supports
  it. Don't soften "pull back toward the market" language back into
  the prompt.
- **Tennis NO-clone inversion is one-sided in CLOB enrichment.**
  `PolymarketEvent.from_gamma` synthesises one inverted NO clone per
  binary head-to-head via `PolymarketMarket.inverted_no_side`. The
  CLOB book + history endpoints expose data per-token, so the YES
  token's data is fetched once and applied to both clones — bid/ask
  sides swap on the NO clone for the book, scalars sign-flip for
  history. Don't fetch the NO token separately; it's the same book
  inverted.
- **Cross-venue surname matching is the trade-time bridge.** The
  Polymarket-sourced JSONL row carries no Kalshi ticker. `execute/`
  pre-fetches Kalshi events at trade time and routes each prediction
  to a `KalshiMarket` via player-surname pair (in `kalshi/matcher.py`).
  Some Polymarket events have no Kalshi counterpart — that's an
  expected `MatchOutcome.kind="no_kalshi_match"` skip, not an error.
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
