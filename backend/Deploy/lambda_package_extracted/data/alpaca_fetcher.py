#!/usr/bin/env python3
"""
Alpaca Market Data Fetcher

Backup data source when Bloomberg is unavailable.
Uses Alpaca's free market data API for stock bars and crypto.

Usage:
    python alpaca_fetcher.py --tickers AAPL MSFT NVDA --start 2024-01-01
"""

import os
import argparse
import pandas as pd
from datetime import datetime, timedelta

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    raise ImportError("Install alpaca-py: pip install alpaca-py")


def get_alpaca_keys():
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        env_path = os.path.join(os.path.dirname(__file__), "..", "config", "credentials.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip()
                    if key == "ALPACA_API_KEY" and not api_key:
                        api_key = val
                    elif key == "ALPACA_API_SECRET" and not api_secret:
                        api_secret = val

    return api_key, api_secret


def fetch_stock_bars(tickers, start_date, end_date=None, api_key=None, api_secret=None):
    """
    Fetch daily stock bars from Alpaca.

    Args:
        tickers: List of stock tickers (e.g., ['AAPL', 'MSFT'])
        start_date: Start date string
        end_date: End date string
        api_key: Alpaca API key
        api_secret: Alpaca API secret

    Returns:
        DataFrame with columns: date, ticker, px_last, volume
    """
    if api_key is None or api_secret is None:
        key, secret = get_alpaca_keys()
        api_key = api_key or key
        api_secret = api_secret or secret

    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_API_SECRET.")

    client = StockHistoricalDataClient(api_key, api_secret)

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(end_date) if end_date else None,
    )

    print(f"Fetching {len(tickers)} stock tickers from Alpaca...")
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

    # Normalize to match our data format
    result = pd.DataFrame({
        "date": pd.to_datetime(df["timestamp"]).dt.tz_localize(None),
        "ticker": df["symbol"],
        "px_last": df["close"],
        "volume": df["volume"],
    })

    print(f"  Fetched {len(result)} bars for {result['ticker'].nunique()} tickers")
    return result


def fetch_crypto_bars(symbols=None, start_date="2020-01-01", end_date=None):
    """
    Fetch daily crypto bars from Alpaca (no API key needed for crypto data).

    Args:
        symbols: List of crypto symbols (e.g., ['BTC/USD', 'ETH/USD'])
        start_date: Start date string
        end_date: End date string
    """
    if symbols is None:
        symbols = ["BTC/USD"]

    # Crypto data is free — no API key needed
    client = CryptoHistoricalDataClient()

    request = CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(end_date) if end_date else None,
    )

    print(f"Fetching {len(symbols)} crypto symbols from Alpaca...")
    bars = client.get_crypto_bars(request)
    df = bars.df.reset_index()

    result = pd.DataFrame({
        "date": pd.to_datetime(df["timestamp"]).dt.tz_localize(None),
        "ticker": df["symbol"],
        "px_last": df["close"],
        "volume": df["volume"],
    })

    print(f"  Fetched {len(result)} bars for {result['ticker'].nunique()} symbols")
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch market data from Alpaca")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA"],
                        help="Stock tickers to fetch")
    parser.add_argument("--crypto", nargs="*", default=None,
                        help="Crypto symbols (e.g., BTC/USD ETH/USD)")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")
    parser.add_argument("--output-dir", default="data/cache", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Fetch stocks
    if args.tickers:
        stock_df = fetch_stock_bars(args.tickers, args.start, args.end)
        if not stock_df.empty:
            out_path = os.path.join(args.output_dir, "alpaca_stock_bars.csv")
            stock_df.to_csv(out_path, index=False)
            print(f"  Saved to {out_path}")

    # Fetch crypto
    if args.crypto is not None:
        symbols = args.crypto if args.crypto else ["BTC/USD"]
        crypto_df = fetch_crypto_bars(symbols, args.start, args.end)
        if not crypto_df.empty:
            out_path = os.path.join(args.output_dir, "alpaca_crypto_bars.csv")
            crypto_df.to_csv(out_path, index=False)
            print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
