#!/usr/bin/env python3
"""
Momentum Signal Calculation

Computes the 12-1 cross-sectional momentum signal:
    Signal = cumulative return from t-252 to t-21

The skip period (most recent month) avoids short-term reversal effects.
This is the standard Jegadeesh & Titman (1993) formation with the widely
adopted skip-month convention.
"""

import numpy as np
import pandas as pd


def compute_returns(prices_df, price_col="tri_gross", fallback_col="px_last"):
    """
    Compute daily returns from price data.

    Args:
        prices_df: DataFrame with columns: date, ticker, [price_col]
        price_col: Primary price column (tri_gross preferred for total return)
        fallback_col: Fallback if primary is missing

    Returns:
        prices_df with 'ret' column added
    """
    df = prices_df.sort_values(["ticker", "date"]).copy()

    # Use total return index if available, otherwise price
    if price_col in df.columns and not df[price_col].isna().all():
        col = price_col
    elif fallback_col in df.columns:
        col = fallback_col
    else:
        raise ValueError(f"Neither {price_col} nor {fallback_col} found in DataFrame")

    df["ret"] = df.groupby("ticker")[col].pct_change()
    df["_price_col_used"] = col
    return df


def compute_momentum_signal(prices_df, formation_period=252, skip_period=21,
                             price_col="tri_gross", fallback_col="px_last"):
    """
    Compute 12-1 momentum signal for each stock on each date.

    Signal = (Price_{t-skip} / Price_{t-formation}) - 1

    Args:
        prices_df: DataFrame with columns: date, ticker, [price_col]
        formation_period: Lookback period in trading days (default 252 = 12 months)
        skip_period: Skip period in trading days (default 21 = 1 month)
        price_col: Price column to use
        fallback_col: Fallback price column

    Returns:
        prices_df with 'mom_12_1' and 'vol20' columns added
    """
    df = prices_df.sort_values(["ticker", "date"]).copy()

    # Determine which price column to use
    if price_col in df.columns and not df[price_col].isna().all():
        col = price_col
    elif fallback_col in df.columns:
        col = fallback_col
    else:
        raise ValueError(f"Neither {price_col} nor {fallback_col} found")

    # Compute returns if not already present
    if "ret" not in df.columns:
        df["ret"] = df.groupby("ticker")[col].pct_change(fill_method=None)

    g = df.groupby("ticker")

    # Momentum = price_{t-skip} / price_{t-formation} - 1
    val_t_skip = g[col].shift(skip_period)
    val_t_formation = g[col].shift(formation_period)
    df["mom_12_1"] = (val_t_skip / val_t_formation) - 1.0

    # 20-day trailing volatility (used for inverse-vol weighting)
    df["vol20"] = g["ret"].transform(lambda s: s.rolling(20).std())

    return df


def compute_momentum_table(prices_df, formation_period=252, skip_period=21):
    """
    Compute momentum signal and return as a pivot table.

    Returns:
        mom_tbl: DataFrame pivoted (index=date, columns=ticker, values=mom_12_1)
        vol_tbl: DataFrame pivoted (index=date, columns=ticker, values=vol20)
        ret_tbl: DataFrame pivoted (index=date, columns=ticker, values=ret)
    """
    df = compute_momentum_signal(prices_df, formation_period, skip_period)

    mom_tbl = df.pivot(index="date", columns="ticker", values="mom_12_1")
    vol_tbl = df.pivot(index="date", columns="ticker", values="vol20")
    ret_tbl = df.pivot(index="date", columns="ticker", values="ret")

    return mom_tbl, vol_tbl, ret_tbl
