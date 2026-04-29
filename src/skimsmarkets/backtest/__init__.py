"""Backtest scaffolding — pull closed-event history from Polymarket gamma + CLOB.

This sub-package is *separate* from the live pipeline: it talks to the same
gamma-api and the public CLOB price-history endpoint, but caches everything to
disk under `backtest_cache/` so analyses are reproducible without re-hitting
the API. Live code path does not import from here.
"""
