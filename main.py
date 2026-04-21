from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from skimsmarkets.kalshi import KalshiClient
from skimsmarkets.pipeline import fetch_live_sports, run_pipeline
from skimsmarkets.reporting import print_events_table, print_run_summary


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


async def _fetch_only(series_ticker: str | None) -> int:
    async with KalshiClient() as c:
        events = await fetch_live_sports(c, series_ticker)
    print_events_table(events, series_ticker)
    return 0


async def _full_run(series_ticker: str | None, dry_run: bool) -> int:
    result = await run_pipeline(series_filter=series_ticker, dry_run=dry_run)
    print_run_summary(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="skimsmarkets",
        description="Fetch live Kalshi sports markets and run the multi-agent prediction pipeline.",
    )
    parser.add_argument(
        "--series-ticker",
        default=None,
        help="Restrict to a single Kalshi series (e.g. KXNBAGAME). Default: all live sports.",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Print live events + markets as a table without invoking LLMs (zero cost).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline against a single market only (~$0.30 of LLM spend).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.fetch_only:
        return asyncio.run(_fetch_only(args.series_ticker))
    return asyncio.run(_full_run(args.series_ticker, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
