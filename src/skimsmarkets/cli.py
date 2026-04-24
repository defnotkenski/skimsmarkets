from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from skimsmarkets import config as cfg
from skimsmarkets.pipeline import (
    fetch_polymarket_slate,
    resolve_market_prices,
    run_pipeline,
)
from skimsmarkets.polymarket import PolymarketClient
from skimsmarkets.reporting import print_events_table, print_run_summary


class _HttpxMinLevelFilter(logging.Filter):
    """Hide sub-threshold httpx / httpcore records from the terminal handler.

    Attached to the stream handler (not the logger itself) so the records
    keep their original INFO severity and any additional handler — e.g. a
    file log, or pytest's capture — still sees them.
    """

    def __init__(self, min_level: int) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith(("httpx", "httpcore")):
            return record.levelno >= self.min_level
        return True


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # httpx / httpcore emit one INFO line per request, which drowns out the
    # pipeline's own INFO logs during a normal run. In normal mode, hide
    # anything below WARNING from the terminal; in verbose mode, show everything.
    if not verbose:
        handler_filter = _HttpxMinLevelFilter(logging.WARNING)
        for handler in logging.getLogger().handlers:
            handler.addFilter(handler_filter)


async def _fetch_only(league: str | None) -> int:
    poly_sem = asyncio.Semaphore(cfg.POLYMARKET_FETCH_SEM)
    async with PolymarketClient() as pm:
        events = await fetch_polymarket_slate(pm, league, cfg.DEFAULT_HORIZON_HOURS)
        await resolve_market_prices(pm, events, poly_sem)
    print_events_table(events, league, horizon_hours=cfg.DEFAULT_HORIZON_HOURS)
    return 0


async def _full_run(league: str | None, dry_run: bool) -> int:
    result = await run_pipeline(league=league, dry_run=dry_run)
    print_run_summary(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="skims",
        description=(
            "Fetch live Polymarket sports markets and run the multi-agent "
            f"confidence-ranker pipeline. Horizon fixed at "
            f"{cfg.DEFAULT_HORIZON_HOURS}h (set DEFAULT_HORIZON_HOURS in config.py to change)."
        ),
    )
    parser.add_argument(
        "--league",
        default=None,
        help=(
            "Restrict to a single Polymarket league by series-slug prefix "
            "(e.g. 'nba' matches 'nba-2025'). Default: all live sports."
        ),
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Print live events + markets as a table without invoking LLMs (zero cost).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline against a single event only (~$0.30 of LLM spend).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.fetch_only:
        return asyncio.run(_fetch_only(args.league))
    return asyncio.run(_full_run(args.league, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
