# CLAUDE.md

Orientation for future sessions. Read the code for specifics.

## What this is

A risk-graded list of today's Polymarket sports markets (tennis
in v1), **plus an opt-in Kalshi trader (`skims execute`) that consumes
the ranker's JSONL**. The two halves are deliberately separated:

- **Ranker** (`skims rank` and everything under `pipeline.py`,
  `agents/`, lens providers) is **not** an edge finder. No buy/pass
  gates, no edge thresholds, no position sizes. The LLMs produce a
  market-blind probability estimate; deterministic post-processing then
  grades the slate into risk buckets. Trade decisions don't live here.
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
               →  deterministic post-processing (risk classification, ranking, JSONL persistence)
```

LLMs only in the agent layer; everything else is deterministic.
Data hosts: `gamma-api.polymarket.com` (slate listing, slug→token-id),
`clob.polymarket.com` (order book + price history). Execution host:
`api.elections.kalshi.com` (RSA-signed POST `/portfolio/orders`,
opt-in via `skims execute`).

## Important notes

- **LLM stages are blind to market price.** Fetchers, reasoners,
  director, and judge never see Polymarket bid/ask, implied probability,
  or price history — the agreement between their independent estimate
  and the market is itself a deterministic signal. Never pipe price
  data into any agent prompt.
- **Ranker and trader are one-way separated.** `pipeline.py` / `agents/`
  / lens code never imports `execute/` or `kalshi/`. The trade path is
  deterministic — no LLM calls in `execute/` or `kalshi/` — which is
  what makes scheduled routines safe to run unattended.
