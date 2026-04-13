#!/usr/bin/env python3
"""
Portfolio Constructor

Builds long/short portfolios from ranked signals.
Handles inverse-vol weighting, position caps, and sector caps.
"""

import numpy as np
import pandas as pd


def inverse_vol_weights(vol_series, tickers, max_weight=0.05):
    """
    Compute inverse-volatility weights.
    Lower-vol stocks get higher weights, capped per name.

    Args:
        vol_series: Series indexed by ticker with 20-day trailing vol values
        tickers: List of tickers to weight
        max_weight: Maximum weight per stock

    Returns:
        Series of weights summing to 1.0
    """
    v = vol_series.reindex(tickers).replace([np.inf, -np.inf], np.nan)
    med = np.nanmedian(v.values) if np.isfinite(v.values).any() else 0.02
    v = v.fillna(med).clip(lower=1e-6)

    w = 1.0 / v
    w = w / w.sum()
    w = w.clip(upper=max_weight)
    w = w / w.sum()
    return w


def equal_weights(tickers, max_weight=0.05):
    """Simple equal-weight portfolio."""
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    w = pd.Series(1.0 / n, index=tickers)
    w = w.clip(upper=max_weight)
    w = w / w.sum()
    return w


def apply_sector_cap(weights, sectors_map, sector_cap=0.30):
    """
    Apply sector concentration cap.
    If any sector exceeds sector_cap, redistribute excess proportionally.

    Args:
        weights: Series indexed by ticker with portfolio weights
        sectors_map: Series mapping ticker -> sector
        sector_cap: Maximum fraction in any single sector

    Returns:
        Adjusted weights Series
    """
    if weights.empty:
        return weights

    w = weights.copy()
    sec = sectors_map.reindex(w.index).fillna("Unknown")

    for _ in range(10):  # Iterate to convergence
        sector_weights = w.groupby(sec).sum()
        over_limit = sector_weights[sector_weights > sector_cap]

        if over_limit.empty:
            break

        for sector, sw in over_limit.items():
            sector_tickers = sec[sec == sector].index
            scale = sector_cap / sw
            w.loc[sector_tickers] *= scale

        # Renormalize
        w = w / w.sum()

    return w


def build_long_short_portfolio(long_tickers, short_tickers, vol_series,
                                sectors_map=None, max_weight=0.05, sector_cap=0.30):
    """
    Build a dollar-neutral long/short portfolio.

    Each leg (long and short) sums to 0.5 (so gross exposure = 1.0, net = 0.0).

    Args:
        long_tickers: List of tickers for the long leg
        short_tickers: List of tickers for the short leg
        vol_series: Series of 20-day trailing volatilities
        sectors_map: Series mapping ticker -> sector (optional)
        max_weight: Max weight per name
        sector_cap: Max sector weight per leg

    Returns:
        dict with 'long_weights' and 'short_weights' (each summing to 0.5)
    """
    # Compute weights for each leg
    long_w = inverse_vol_weights(vol_series, long_tickers, max_weight)
    short_w = inverse_vol_weights(vol_series, short_tickers, max_weight)

    # Apply sector caps if provided
    if sectors_map is not None:
        long_w = apply_sector_cap(long_w, sectors_map, sector_cap)
        short_w = apply_sector_cap(short_w, sectors_map, sector_cap)

    # Scale each leg to 0.5 (total gross = 1.0)
    long_w = long_w * 0.5
    short_w = short_w * 0.5

    return {
        "long_weights": long_w,
        "short_weights": short_w,
        "n_long": len(long_tickers),
        "n_short": len(short_tickers),
        "gross_exposure": float(long_w.sum() + short_w.sum()),
        "net_exposure": float(long_w.sum() - short_w.sum()),
    }


def build_long_only_portfolio(tickers, vol_series, sectors_map=None,
                               max_weight=0.05, sector_cap=0.30):
    """
    Build a long-only portfolio with inverse-vol weights.
    """
    w = inverse_vol_weights(vol_series, tickers, max_weight)

    if sectors_map is not None:
        w = apply_sector_cap(w, sectors_map, sector_cap)

    return {
        "weights": w,
        "n_positions": len(tickers),
        "gross_exposure": float(w.sum()),
    }


def compute_turnover(prev_weights, new_weights):
    """
    Compute portfolio turnover.

    Args:
        prev_weights: Previous period weights (Series)
        new_weights: New period weights (Series)

    Returns:
        Float: total turnover (sum of absolute weight changes)
    """
    if prev_weights is None or prev_weights.empty:
        return float(new_weights.abs().sum()) if not new_weights.empty else 0.0

    all_tickers = prev_weights.index.union(new_weights.index)
    prev = prev_weights.reindex(all_tickers).fillna(0.0)
    new = new_weights.reindex(all_tickers).fillna(0.0)
    return float((new - prev).abs().sum())
