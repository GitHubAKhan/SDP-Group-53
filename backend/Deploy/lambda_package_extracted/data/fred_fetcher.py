#!/usr/bin/env python3
"""
FRED API Data Fetcher

Pulls macro data from the Federal Reserve Economic Data (FRED) API.
Used for macro regime detection: VIX, yield curve, credit spreads, etc.

Usage:
    python fred_fetcher.py --api-key YOUR_KEY --start 2010-01-01 --end 2025-12-31
    python fred_fetcher.py --start 2010-01-01 --end 2025-12-31  # uses FRED_API_KEY env var
"""

import os
import argparse
import pandas as pd
import yaml

try:
    from fredapi import Fred
except ImportError:
    raise ImportError("Install fredapi: pip install fredapi")


# Default FRED series for macro regime detection
DEFAULT_SERIES = {
    "VIXCLS": "VIX daily close",
    "T10Y2Y": "10Y-2Y Treasury spread (yield curve)",
    "BAA10Y": "Baa corporate - 10Y Treasury spread (credit spread)",
    "FEDFUNDS": "Federal funds effective rate",
    "ICSA": "Initial jobless claims (weekly)",
}


def load_settings():
    settings_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def get_fred_client(api_key=None):
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED API key not found. Either:\n"
            "  1. Pass --api-key YOUR_KEY\n"
            "  2. Set FRED_API_KEY environment variable\n"
            "  3. Add to config/credentials.env\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=key)


def fetch_fred_series(fred, series_id, start_date=None, end_date=None, use_first_release=False):
    """
    Fetch a single FRED series.

    Args:
        fred: Fred client instance
        series_id: FRED series ID (e.g., 'VIXCLS')
        start_date: Start date string
        end_date: End date string
        use_first_release: If True, use first-release data (avoids look-ahead bias in backtests)
    """
    try:
        if use_first_release:
            # Use first-release data for backtesting to avoid look-ahead from revisions
            data = fred.get_series_first_release(series_id)
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
        else:
            data = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
        return data
    except Exception as e:
        print(f"  Warning: Could not fetch {series_id}: {e}")
        return pd.Series(dtype=float)


def fetch_all_macro_data(api_key=None, start_date="2010-01-01", end_date=None,
                         series_dict=None, use_first_release=False, output_dir=None):
    """
    Fetch all macro series from FRED and return as a DataFrame.
    """
    fred = get_fred_client(api_key)

    if series_dict is None:
        series_dict = DEFAULT_SERIES

    print(f"Fetching {len(series_dict)} FRED series...")
    all_series = {}

    for series_id, description in series_dict.items():
        print(f"  {series_id}: {description}...")
        data = fetch_fred_series(fred, series_id, start_date, end_date, use_first_release)
        if not data.empty:
            all_series[series_id] = data
            print(f"    -> {len(data)} observations ({data.index.min().date()} to {data.index.max().date()})")
        else:
            print(f"    -> No data returned")

    if not all_series:
        print("No macro data fetched.")
        return pd.DataFrame()

    # Combine into a single DataFrame, forward-fill missing dates
    df = pd.DataFrame(all_series)
    df.index.name = "date"
    df = df.sort_index()

    # Forward-fill to align daily/weekly/monthly series
    df = df.ffill()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "macro_data.csv")
        df.to_csv(out_path)
        print(f"\nSaved macro data to {out_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

        # If running in Lambda, also upload to S3
        s3_bucket = os.getenv("S3_BUCKET")
        if s3_bucket:
            import boto3, io
            s3 = boto3.client("s3")
            buf = io.StringIO()
            df.to_csv(buf)
            s3.put_object(Bucket=s3_bucket, Key="cache/macro_data.csv",
                          Body=buf.getvalue(), ContentType="text/csv")
            print(f"  Uploaded to s3://{s3_bucket}/cache/macro_data.csv")

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch macro data from FRED API")
    parser.add_argument("--api-key", default=None, help="FRED API key")
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="data/cache", help="Output directory")
    parser.add_argument("--first-release", action="store_true",
                        help="Use first-release data (for backtesting)")
    args = parser.parse_args()

    fetch_all_macro_data(
        api_key=args.api_key,
        start_date=args.start,
        end_date=args.end,
        use_first_release=args.first_release,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
