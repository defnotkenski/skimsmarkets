"""Run the backtest dataset builder. Output: backtest_cache/dataset.parquet."""

from __future__ import annotations

import argparse
import asyncio
import logging

from skimsmarkets.backtest.dataset import build_dataset


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=800)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    df = await build_dataset(max_events=args.max_events)
    print(f"rows={len(df)} unique_events={df['event_slug'].nunique() if not df.empty else 0}")
    if not df.empty:
        print(df.head())
        print()
        print("by league:")
        print(df.groupby("league").size().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    asyncio.run(main())
