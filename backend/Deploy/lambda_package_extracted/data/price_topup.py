#!/usr/bin/env python3
"""
Price Top-Up

Fetches recent daily price data from Yahoo Finance for all tickers
in the existing Bloomberg parquet universe, starting from the last
available Bloomberg date through today.

Saves to data/cache/price_topup.parquet. The DataPipeline.load_prices()
method is patched via load_prices_with_topup() to merge both sources.

Usage:
    python data/price_topup.py
"""

import os
import pandas as pd
import yfinance as yf
from datetime import date, timedelta


PARQUET_DIR = "data/data/prices_parquet"
TOPUP_PATH  = "data/cache/price_topup.parquet"


def bbg_to_yf(ticker: str) -> str:
    """Convert 'AAPL US Equity' -> 'AAPL'."""
    return ticker.replace(" US Equity", "").replace(" Equity", "").strip()


def run_topup():
    print("Loading existing Bloomberg parquet to find last date and tickers...")
    df_bbg = pd.read_parquet(PARQUET_DIR)
    df_bbg["date"] = pd.to_datetime(df_bbg["date"]).dt.tz_localize(None)

    last_date = df_bbg["date"].max().date()
    start_date = last_date + timedelta(days=1)
    end_date = date.today()

    if start_date > end_date:
        print(f"Data already current through {last_date}. No top-up needed.")
        return

    print(f"Top-up window: {start_date} -> {end_date}")

    # Get all unique tickers from Bloomberg universe
    bbg_tickers = df_bbg["ticker"].unique().tolist()
    yf_tickers = [bbg_to_yf(t) for t in bbg_tickers]

    # Map yfinance ticker -> bloomberg ticker
    yf_to_bbg = {bbg_to_yf(t): t for t in bbg_tickers}

    print(f"Fetching {len(yf_tickers)} tickers from Yahoo Finance...")

    # Download unadjusted prices (px_last = raw close, volume)
    raw_unadj = yf.download(
        yf_tickers,
        start=str(start_date),
        end=str(end_date + timedelta(days=1)),
        auto_adjust=False,
        progress=True,
    )

    # Download adjusted prices (tri_gross = dividend-adjusted close)
    raw_adj = yf.download(
        yf_tickers,
        start=str(start_date),
        end=str(end_date + timedelta(days=1)),
        auto_adjust=True,
        progress=False,
    )

    if raw_unadj.empty:
        print("No data returned from yfinance.")
        return

    # Reshape: yfinance returns MultiIndex columns (field, ticker)
    frames = []
    for yf_ticker in yf_tickers:
        try:
            px_last  = raw_unadj["Close"][yf_ticker].squeeze()
            volume   = raw_unadj["Volume"][yf_ticker].squeeze()
            tri_gross = raw_adj["Close"][yf_ticker].squeeze()
        except (KeyError, TypeError):
            continue

        df_t = pd.DataFrame({
            "date":      px_last.index,
            "ticker":    yf_to_bbg.get(yf_ticker, yf_ticker),
            "px_last":   px_last.values,
            "tri_gross": tri_gross.reindex(px_last.index).values,
            "volume":    volume.values,
        })
        df_t = df_t.dropna(subset=["px_last"])
        frames.append(df_t)

    if not frames:
        print("Could not parse any ticker data.")
        return

    df_topup = pd.concat(frames, ignore_index=True)
    df_topup["date"] = pd.to_datetime(df_topup["date"]).dt.tz_localize(None)
    df_topup = df_topup.sort_values(["ticker", "date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(TOPUP_PATH), exist_ok=True)
    df_topup.to_parquet(TOPUP_PATH, index=False)

    print(f"\nTop-up saved to {TOPUP_PATH}")
    print(f"  Rows:    {len(df_topup):,}")
    print(f"  Tickers: {df_topup['ticker'].nunique()}")
    print(f"  Dates:   {df_topup['date'].min().date()} -> {df_topup['date'].max().date()}")


if __name__ == "__main__":
    run_topup()
