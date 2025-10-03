#!/usr/bin/env python3
"""
Strategy Core Module

Shared logic for both backtesting and live trading.
All strategy calculations happen here to ensure consistency.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Strategy Configuration
TOP_PCT_PER_SECTOR = 0.10
MAX_NAME_WEIGHT = 0.05
VOL_WINDOW = 20
TREND_MA_WINDOW = 200
TREND_TICKER = "SPY US Equity"
TARGET_VOL = 0.10
USE_TREND_FILTER = True
USE_VOL_TARGETING = True
COST_PER_TURNOVER = 0.0010

US_EXCHANGE_CODES = {"UN", "UW", "UQ", "UA", "UR", "UT", "UV"}


def to_parse_keyable(member_code: str) -> str:
    """Convert raw ticker to parse-keyable format"""
    parts = str(member_code).strip().split()
    if len(parts) == 1:
        return f"{parts[0]} US Equity"
    ticker, code = parts[0], parts[1]
    if code in US_EXCHANGE_CODES:
        return f"{ticker} US Equity"
    if len(code) == 2:
        return f"{ticker} {code} Equity"
    return f"{ticker} US Equity"


def load_prices(data_dir: Path):
    """Load price data from parquet"""
    parquet_path = data_dir / "prices_parquet"
    if parquet_path.is_dir():
        df = pd.read_parquet(parquet_path)
    else:
        files = list(data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet data in {data_dir}")
        df = pd.read_parquet(files[0])
    
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    return df


def load_constituents(data_dir: Path):
    """Load SPX constituents"""
    path = data_dir / "constituents_long.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def load_sectors(data_dir: Path):
    """Load sector mappings"""
    path = data_dir / "sectors.csv"
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def compute_daily_returns(prices_df):
    """Calculate daily returns"""
    prices_df = prices_df.sort_values(["ticker", "date"]).copy()
    base_col = "tri_gross" if "tri_gross" in prices_df.columns else "px_last"
    prices_df["ret"] = prices_df.groupby("ticker")[base_col].pct_change()
    return prices_df, base_col


def compute_momentum(prices_df, base_col):
    """Calculate 12-1 momentum and 20-day volatility"""
    prices_df = prices_df.sort_values(["ticker", "date"]).copy()
    g = prices_df.groupby("ticker")
    
    # 12-1 momentum
    val_t_21 = g[base_col].shift(21)
    val_t_252 = g[base_col].shift(252)
    prices_df["mom_12_1"] = (val_t_21 / val_t_252) - 1.0
    
    # 20-day volatility
    vol20_values = g["ret"].apply(lambda x: x.rolling(VOL_WINDOW).std())
    prices_df["vol20"] = vol20_values.values
    
    return prices_df


def check_market_regime(prices_df, asof_date, ticker=TREND_TICKER, window=TREND_MA_WINDOW):
    """Check if market is in uptrend"""
    if not USE_TREND_FILTER:
        return True
    
    benchmark = prices_df[prices_df["ticker"] == ticker].copy()
    benchmark = benchmark[benchmark["date"] <= asof_date].sort_values("date")
    
    if len(benchmark) < window:
        return True
    
    price_col = "px_last" if "px_last" in benchmark.columns else "tri_gross"
    current_price = benchmark[price_col].iloc[-1]
    ma = benchmark[price_col].rolling(window).mean().iloc[-1]
    
    return current_price > ma


def build_universe(const_df, asof_date):
    """Get SPX constituents as of date"""
    df = const_df[const_df["date"] <= asof_date].sort_values(["ticker_raw", "date"])
    last = df.groupby("ticker_raw").tail(1)
    U_raw = last.loc[last["in_spx"] == 1, "ticker_raw"]
    U = set(to_parse_keyable(x) for x in U_raw)
    return U


def pick_by_sector(momentum_series, sectors_map, universe, top_pct=TOP_PCT_PER_SECTOR):
    """Select top % momentum stocks per sector"""
    mom = momentum_series.copy().replace([np.inf, -np.inf], np.nan).dropna()
    mom = mom[mom.index.isin(universe)]
    
    if len(mom) == 0:
        return []
    
    sec = sectors_map.loc[mom.index].fillna("Unknown")
    picks = []
    
    for sector, idx in sec.groupby(sec).groups.items():
        sub = mom.loc[idx].dropna()
        if len(sub) == 0:
            continue
        k = max(1, int(np.ceil(len(sub) * top_pct)))
        top_names = sub.sort_values(ascending=False).head(k).index.tolist()
        picks.extend(top_names)
    
    return list(dict.fromkeys(picks))


def inverse_vol_weights(vol_series, names, cap=MAX_NAME_WEIGHT):
    """Calculate inverse volatility weights"""
    if len(names) == 0:
        return pd.Series(dtype=float)
    
    v = vol_series.loc[names].replace([np.inf, -np.inf], np.nan)
    med = np.nanmedian(v.values) if np.isfinite(v.values).any() else 0.02
    v = v.fillna(med).clip(lower=1e-6)
    
    w = 1.0 / v
    w = w / w.sum()
    w = w.clip(upper=cap)
    w = w / w.sum()
    
    return w


def apply_vol_targeting(weights, nav_history, target_vol=TARGET_VOL):
    """Scale weights based on realized portfolio volatility"""
    if not USE_VOL_TARGETING or len(nav_history) < 60:
        return weights
    
    recent_returns = [h.get('return', 0) for h in nav_history[-60:]]
    if len(recent_returns) < 20:
        return weights
    
    realized_vol = np.std(recent_returns) * np.sqrt(12)  # Annualized
    
    if realized_vol < 0.01:
        return weights
    
    scalar = target_vol / realized_vol
    scalar = np.clip(scalar, 0.25, 1.0)
    
    return weights * scalar


def compute_turnover(prev_weights, new_weights):
    """Calculate portfolio turnover"""
    names = prev_weights.index.union(new_weights.index)
    prev = prev_weights.reindex(names).fillna(0.0)
    new = new_weights.reindex(names).fillna(0.0)
    return float((new - prev).abs().sum())


def generate_signals(data_dir, asof_date, nav_history=None):
    """
    Main signal generation function.
    Returns dict of target weights or empty dict if going to cash.
    
    Args:
        data_dir: Path to data directory
        asof_date: Date to generate signals as of
        nav_history: List of NAV history dicts for vol targeting
        
    Returns:
        dict: {ticker: weight} or {} if cash
    """
    data_dir = Path(data_dir)
    asof_date = pd.to_datetime(asof_date)
    
    # Load data
    prices_df = load_prices(data_dir)
    const_df = load_constituents(data_dir)
    sectors_df = load_sectors(data_dir)
    
    # Filter to historical data only
    prices_df = prices_df[prices_df["date"] <= asof_date]
    
    # Check market regime
    is_uptrend = check_market_regime(prices_df, asof_date)
    if not is_uptrend:
        return {}  # Go to cash
    
    # Compute returns and momentum
    prices_df, base_col = compute_daily_returns(prices_df)
    prices_df = compute_momentum(prices_df, base_col)
    
    # Get universe
    universe = build_universe(const_df, asof_date)
    
    # Get latest signals per ticker
    signals = prices_df[prices_df["date"] <= asof_date].groupby("ticker").tail(1).set_index("ticker")
    
    # Build sector map
    sectors_map = sectors_df.drop_duplicates(subset=["ticker"]).set_index("ticker")["sector"]
    
    # Select stocks
    picks = pick_by_sector(signals["mom_12_1"], sectors_map, universe)
    
    if len(picks) == 0:
        return {}
    
    # Calculate weights
    weights = inverse_vol_weights(signals["vol20"], picks)
    
    # Apply vol targeting if we have history
    if nav_history:
        weights = apply_vol_targeting(weights, nav_history)
    
    return weights.to_dict()


def calculate_trades(target_weights, current_positions, current_prices, nav):
    """
    Calculate trades needed to reach target portfolio.
    
    Args:
        target_weights: dict of {ticker: weight}
        current_positions: dict of {ticker: shares}
        current_prices: dict of {ticker: price}
        nav: current portfolio NAV
        
    Returns:
        list of trade dicts
    """
    trades = []
    
    # Current weights
    current_weights = {}
    for ticker, shares in current_positions.items():
        if ticker in current_prices:
            value = shares * current_prices[ticker]
            current_weights[ticker] = value / nav
    
    # Calculate required trades
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    
    for ticker in all_tickers:
        current_w = current_weights.get(ticker, 0)
        target_w = target_weights.get(ticker, 0)
        
        if abs(target_w - current_w) < 0.001:
            continue
        
        if ticker not in current_prices:
            continue
        
        target_dollars = target_w * nav
        current_dollars = current_w * nav
        trade_dollars = target_dollars - current_dollars
        
        price = current_prices[ticker]
        shares = int(trade_dollars / price)
        
        if shares != 0:
            trades.append({
                'ticker': ticker,
                'shares': shares,
                'side': 'BUY' if shares > 0 else 'SELL',
                'price': price,
                'value': abs(shares * price),
                'current_weight': current_w,
                'target_weight': target_w
            })
    
    return trades