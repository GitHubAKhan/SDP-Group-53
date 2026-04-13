#!/usr/bin/env python3
"""
Gold Price Data Fetcher

Pulls GLD (SPDR Gold Shares ETF) prices from Yahoo Finance as a free fallback
when Bloomberg is unavailable. Also supports IAU as an alternative.

Usage:
    python gold_fetcher.py --start 2010-01-01 --end 2025-12-31
    python gold_fetcher.py --ticker IAU --start 2010-01-01
"""

import os
import argparse
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Install yfinance: pip install yfinance")


def fetch_gold_prices(ticker="GLD", start_date="2010-01-01", end_date=None,
                      output_dir=None):
    """
    Fetch gold ETF daily prices from Yahoo Finance.

    Args:
        ticker: Gold ETF ticker (GLD or IAU)
        start_date: Start date string
        end_date: End date string (defaults to today)
        output_dir: If set, save CSV to this directory

    Returns:
        DataFrame with columns: date, ticker, px_last, volume
    """
    print(f"Fetching {ticker} prices from Yahoo Finance...")

    gold = yf.Ticker(ticker)
    hist = gold.history(start=start_date, end=end_date, auto_adjust=True)

    if hist.empty:
        print(f"  No data returned for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame({
        "date": hist.index,
        "ticker": ticker,
        "px_last": hist["Close"].values,
        "volume": hist["Volume"].values,
    })

    # Remove timezone info
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    print(f"  Fetched {len(df)} daily observations")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{ticker.lower()}_prices.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")

        # If running in Lambda, also upload to S3
        s3_bucket = os.getenv("S3_BUCKET")
        if s3_bucket:
            import boto3, io
            s3 = boto3.client("s3")
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            s3_key = f"cache/{ticker.lower()}_prices.csv"
            s3.put_object(Bucket=s3_bucket, Key=s3_key,
                          Body=buf.getvalue(), ContentType="text/csv")
            print(f"  Uploaded to s3://{s3_bucket}/{s3_key}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch gold ETF prices")
    parser.add_argument("--ticker", default="GLD", help="Gold ETF ticker (GLD or IAU)")
    parser.add_argument("--start", default="2010-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")
    parser.add_argument("--output-dir", default="data/cache", help="Output directory")
    args = parser.parse_args()

    fetch_gold_prices(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
