"""`skims execute` — opt-in deterministic trader.

Consumes the ranker's JSONL (`logs/runs/<run_id>.jsonl`) and places
Kalshi market-buy orders against the predicted winner of each
filtered row. Never imported by `pipeline.py` — the ranker stays pure.

The `__init__` keeps re-exports minimal — `ExecuteOptions` / `run_execute`
live in `execute.trader` and import `KalshiClient`, which transitively
pulls `cryptography`. Importing the trader eagerly here would force
that into every code path touching filter logic. Import directly:
`from skimsmarkets.execute.trader import run_execute`.
"""

from skimsmarkets.execute.filters import filter_rows

__all__ = ["filter_rows"]
