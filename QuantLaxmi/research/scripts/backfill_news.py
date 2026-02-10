#!/usr/bin/env python3
"""Backfill historical news headlines from GDELT + score with FinBERT.

Usage:
    # Full backfill (India + crypto, last 12 months)
    python research/scripts/backfill_news.py

    # Custom date range
    python research/scripts/backfill_news.py --start 2025-01-01 --end 2026-02-10

    # Only crypto
    python research/scripts/backfill_news.py --categories crypto

    # Only India
    python research/scripts/backfill_news.py --categories india_market india_stocks

    # Skip FinBERT scoring (just fetch headlines)
    python research/scripts/backfill_news.py --no-score

    # Verify features build correctly
    python research/scripts/backfill_news.py --verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill_news")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill news from GDELT + score with FinBERT")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: 12 months ago)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--categories", nargs="+",
        default=["india_market", "india_stocks", "crypto", "us_market", "us_stocks", "europe", "intl"],
        choices=["india_market", "india_stocks", "crypto", "us_market", "us_stocks", "europe", "intl"],
        help="Which categories to fetch (default: all)",
    )
    parser.add_argument("--chunk-days", type=int, default=7, help="Days per GDELT request chunk")
    parser.add_argument("--no-score", action="store_true", help="Skip FinBERT scoring")
    parser.add_argument("--verify", action="store_true", help="Build features after backfill to verify")
    args = parser.parse_args()

    # Date range defaults
    end_date = args.end or date.today().isoformat()
    start_date = args.start or (date.today() - timedelta(days=365)).isoformat()

    # --- Step 1: GDELT backfill ---
    logger.info("=" * 60)
    logger.info("Step 1: GDELT historical backfill")
    logger.info("  Date range: %s to %s", start_date, end_date)
    logger.info("  Categories: %s", args.categories)
    logger.info("=" * 60)

    from quantlaxmi.data.collectors.news.gdelt import backfill_headlines

    n_new = backfill_headlines(
        start_date=start_date,
        end_date=end_date,
        categories=args.categories,
        chunk_days=args.chunk_days,
    )
    logger.info("GDELT: %d new headlines archived", n_new)

    # --- Step 2: FinBERT scoring ---
    if not args.no_score:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 2: FinBERT sentiment scoring")
        logger.info("=" * 60)

        from quantlaxmi.data.collectors.news.headline_archive import read_archive
        from quantlaxmi.features.news_sentiment import score_headlines_bulk, _ScoreCache
        from datetime import datetime, timezone

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc,
        )

        headlines = read_archive(start=start_dt, end=end_dt)
        logger.info("Loaded %d headlines from archive", len(headlines))

        if headlines:
            cache = _ScoreCache()
            uncached = sum(1 for h in headlines if not cache.has(h.get("title", "")))
            logger.info("  %d already cached, %d to score", len(headlines) - uncached, uncached)

            if uncached > 0:
                scored = score_headlines_bulk(headlines, cache=cache)
                logger.info("  Scored %d headlines, cache now has %d entries", len(scored), len(cache))
            else:
                logger.info("  All headlines already scored — nothing to do")
    else:
        logger.info("Skipping FinBERT scoring (--no-score)")

    # --- Step 3: Verify features ---
    if args.verify:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 3: Verify feature builder")
        logger.info("=" * 60)

        from quantlaxmi.features.news_sentiment import NewsSentimentBuilder

        builder = NewsSentimentBuilder(use_finbert=not args.no_score)
        df = builder.build(start_date, end_date)

        if df.empty:
            logger.warning("Feature builder returned empty DataFrame")
        else:
            logger.info("Features built: %d days x %d columns", len(df), len(df.columns))
            logger.info("Columns: %s", list(df.columns))
            logger.info("\nSample (last 5 days):")
            print(df.tail().to_string())
            print()

            # Sanity checks
            assert "ns_sent_mean" in df.columns, "Missing ns_sent_mean"
            assert "ns_news_count" in df.columns, "Missing ns_news_count"
            assert df["ns_pos_ratio"].between(0, 1).all(), "ns_pos_ratio out of [0,1]"
            assert df["ns_neg_ratio"].between(0, 1).all(), "ns_neg_ratio out of [0,1]"
            assert df["ns_news_count"].ge(0).all(), "Negative news count"
            logger.info("All sanity checks PASSED")

    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
