from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from skimsmarkets import config as cfg
from skimsmarkets.kalshi import KalshiClient
from skimsmarkets.pipeline import fetch_live_sports, run_pipeline
from skimsmarkets.reporting import print_events_table, print_run_summary


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


async def _fetch_only(series_ticker: str | None, horizon_hours: int) -> int:
    async with KalshiClient() as c:
        events = await fetch_live_sports(c, series_ticker)
    print_events_table(events, series_ticker, horizon_hours=horizon_hours)
    return 0


async def _full_run(series_ticker: str | None, dry_run: bool, horizon_hours: int) -> int:
    result = await run_pipeline(
        series_filter=series_ticker, dry_run=dry_run, horizon_hours=horizon_hours,
    )
    print_run_summary(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="skims",
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
        "--horizon-hours",
        type=int,
        default=cfg.MAX_HOURS_UNTIL_EXPIRATION,
        help=(
            "Only include markets whose expected settlement is within this many hours. "
            f"Default: {cfg.MAX_HOURS_UNTIL_EXPIRATION} (today's slate). Use 48-72 to include tomorrow."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.fetch_only:
        return asyncio.run(_fetch_only(args.series_ticker, args.horizon_hours))
    return asyncio.run(_full_run(args.series_ticker, args.dry_run, args.horizon_hours))


if __name__ == "__main__":
    raise SystemExit(main())