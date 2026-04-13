#!/usr/bin/env python3
"""
Cross-Sectional Ranking Module

Ranks stocks by momentum signal within the universe.
Supports both cross-sectional and industry-neutral ranking.
"""

import numpy as np
import pandas as pd


def rank_cross_sectional(momentum_series, universe, long_pct=0.20, short_pct=0.20):
    """
    Rank stocks by momentum and select top/bottom quintiles.

    Args:
        momentum_series: Series indexed by ticker, values = momentum signal
        universe: Set of tickers in the investable universe
        long_pct: Fraction of universe to go long (default top 20%)
        short_pct: Fraction of universe to go short (default bottom 20%)

    Returns:
        dict with keys: 'long', 'short' (lists of tickers)
    """
    # Filter to universe and drop NaN/inf
    mom = momentum_series.copy()
    mom = mom[mom.index.isin(universe)]
    mom = mom.replace([np.inf, -np.inf], np.nan).dropna()

    if len(mom) == 0:
        return {"long": [], "short": []}

    n = len(mom)
    n_long = max(1, int(np.ceil(n * long_pct)))
    n_short = max(1, int(np.ceil(n * short_pct)))

    sorted_mom = mom.sort_values(ascending=False)
    longs = sorted_mom.head(n_long).index.tolist()
    shorts = sorted_mom.tail(n_short).index.tolist()

    return {"long": longs, "short": shorts}


def rank_by_sector(momentum_series, sectors_map, universe, top_pct=0.10):
    """
    Rank stocks within each GICS sector and pick top performers per sector.
    This avoids sector concentration and is academically defensible
    (Baltussen et al. 2026).

    Args:
        momentum_series: Series indexed by ticker, values = momentum signal
        sectors_map: Series mapping ticker -> sector
        universe: Set of tickers in the investable universe
        top_pct: Fraction of stocks to pick per sector

    Returns:
        List of selected tickers (long only)
    """
    mom = momentum_series.copy()
    mom = mom[mom.index.isin(universe)]
    mom = mom.replace([np.inf, -np.inf], np.nan).dropna()

    if len(mom) == 0:
        return []

    sec = sectors_map.reindex(mom.index).fillna("Unknown")
    picks = []

    for sector, idx in sec.groupby(sec).groups.items():
        sub = mom.loc[idx].dropna()
        if len(sub) == 0:
            continue
        k = max(1, int(np.ceil(len(sub) * top_pct)))
        top_names = sub.sort_values(ascending=False).head(k).index.tolist()
        picks.extend(top_names)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(picks))


def rank_industry_neutral(momentum_series, sectors_map, universe, top_pct=0.20, short_pct=0.20):
    """
    Industry-neutral momentum ranking: rank stocks within each industry
    and pick top/bottom from each. Forces diversification across industries.

    This is Option 1 from the buildspec addendum.

    Args:
        momentum_series: Series indexed by ticker, values = momentum signal
        sectors_map: Series mapping ticker -> sector
        universe: Set of tickers in the investable universe
        top_pct: Fraction of stocks to go long per industry
        short_pct: Fraction of stocks to go short per industry

    Returns:
        dict with keys: 'long', 'short' (lists of tickers)
    """
    mom = momentum_series.copy()
    mom = mom[mom.index.isin(universe)]
    mom = mom.replace([np.inf, -np.inf], np.nan).dropna()

    if len(mom) == 0:
        return {"long": [], "short": []}

    sec = sectors_map.reindex(mom.index).fillna("Unknown")
    longs = []
    shorts = []

    for sector, idx in sec.groupby(sec).groups.items():
        sub = mom.loc[idx].dropna()
        if len(sub) < 2:
            continue
        n_long = max(1, int(np.ceil(len(sub) * top_pct)))
        n_short = max(1, int(np.ceil(len(sub) * short_pct)))

        sorted_sub = sub.sort_values(ascending=False)
        longs.extend(sorted_sub.head(n_long).index.tolist())
        shorts.extend(sorted_sub.tail(n_short).index.tolist())

    return {
        "long": list(dict.fromkeys(longs)),
        "short": list(dict.fromkeys(shorts)),
    }


def compute_percentile_ranks(momentum_series, universe):
    """
    Compute percentile rank of each stock within the universe.

    Returns:
        Series indexed by ticker, values = percentile rank [0, 1]
    """
    mom = momentum_series.copy()
    mom = mom[mom.index.isin(universe)]
    mom = mom.replace([np.inf, -np.inf], np.nan).dropna()

    if len(mom) == 0:
        return pd.Series(dtype=float)

    return mom.rank(pct=True)
