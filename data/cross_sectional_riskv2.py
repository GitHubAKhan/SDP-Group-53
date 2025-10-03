#!/usr/bin/env python3
"""
Cross-Sectional Momentum Backtest (Refactored)

Now imports all strategy logic from strategy_core.py to ensure
consistency with live trading.

Usage:
    python cross_sectional_risk.py --data-dir data --start 1994-01-01 --end 2025-09-30 --out-dir results
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Import from strategy core
from strategy_core import (
    load_prices,
    load_constituents,
    load_sectors,
    compute_daily_returns,
    compute_momentum,
    check_market_regime,
    build_universe,
    pick_by_sector,
    inverse_vol_weights,
    apply_vol_targeting,
    compute_turnover,
    COST_PER_TURNOVER,
    TOP_PCT_PER_SECTOR,
    MAX_NAME_WEIGHT,
    VOL_WINDOW,
    TREND_MA_WINDOW
)


def backtest(prices, const_df, sectors_df, start: str, end: str, out_dir: str):
    """Run backtest using strategy_core functions"""
    os.makedirs(out_dir, exist_ok=True)

    # Prep data
    prices, base_col = compute_daily_returns(prices)
    prices = compute_momentum(prices, base_col)
    prices = prices.sort_values(["date", "ticker"])
    
    # Build sector map
    sectors_map = sectors_df.drop_duplicates(subset=["ticker"]).set_index("ticker")["sector"]

    # Filter date range
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    prices = prices[(prices["date"] >= start_dt) & (prices["date"] <= end_dt)]

    # Pivot tables for fast lookup
    daily_ret = prices.pivot(index="date", columns="ticker", values="ret")
    vol20_tbl = prices.pivot(index="date", columns="ticker", values="vol20")
    mom_tbl = prices.pivot(index="date", columns="ticker", values="mom_12_1")

    # Month-ends (rebalance dates)
    mes = pd.date_range(start=daily_ret.index.min(), end=daily_ret.index.max(), freq="BM")
    mes = mes[(mes >= start_dt) & (mes <= end_dt)]
    
    if len(mes) == 0:
        raise RuntimeError("No month-ends in selected range.")

    # Performance trackers
    daily_index = daily_ret.index
    port_ret = pd.Series(index=daily_index, dtype=float)
    positions_by_month = {}
    nav_history = []

    prev_w = pd.Series(dtype=float)

    for i, M in enumerate(mes):
        # Check market regime using strategy_core
        is_uptrend = check_market_regime(prices, M)
        
        if not is_uptrend:
            print(f"\n[{M.date()}] MARKET DOWNTREND - Moving to cash (SPY < {TREND_MA_WINDOW}MA)")
            
            # Hold cash until next rebalance
            if M != mes[-1]:
                next_M = mes[mes.get_loc(M)+1]
            else:
                next_M = daily_index.max()
            
            window = daily_ret.loc[(daily_ret.index > M) & (daily_ret.index <= next_M)]
            port_ret.loc[window.index] = 0.0
            
            # Track turnover cost if liquidating positions
            if not prev_w.empty:
                turnover = prev_w.sum()
                cost = turnover * COST_PER_TURNOVER
                if len(window) > 0:
                    port_ret.loc[window.index[0]] = -cost
                print(f"  Liquidated all positions. Turnover: {turnover:.2%}, Cost: {cost:.3%}")
            
            prev_w = pd.Series(dtype=float)
            positions_by_month[pd.to_datetime(M).date()] = {}
            continue
        
        # Build universe using strategy_core
        U = build_universe(const_df, M)

        # Get signals as of M
        mom_M = mom_tbl.loc[:M].tail(1).T.squeeze()
        vol_M = vol20_tbl.loc[:M].tail(1).T.squeeze()

        # Select stocks using strategy_core
        picks = pick_by_sector(momentum_series=mom_M, sectors_map=sectors_map, universe=U)

        if len(picks) == 0:
            next_window = daily_ret.loc[(daily_ret.index > M) & (daily_ret.index <= (mes[mes.get_loc(M)+1] if M != mes[-1] else daily_index.max()))]
            port_ret.loc[next_window.index] = 0.0
            continue

        # Calculate weights using strategy_core
        w = inverse_vol_weights(vol_M, picks, cap=MAX_NAME_WEIGHT)
        w.name = "weight"
        
        # Apply vol targeting using strategy_core
        w = apply_vol_targeting(w, nav_history)
        
        # Compute turnover
        turnover = compute_turnover(prev_w, w)
        cost = turnover * COST_PER_TURNOVER
        
        # Logging
        if i % 12 == 0:  # Log annually
            print(f"\n[{M.date()}] Rebalance:")
            print(f"  Positions: {len(picks)}")
            print(f"  Exposure: {w.sum():.1%}")
            print(f"  Turnover: {turnover:.2%}")
            print(f"  Top holding: {w.idxmax()} @ {w.max():.2%}")

        prev_w = w

        # Trading window
        if M != mes[-1]:
            next_M = mes[mes.get_loc(M)+1]
        else:
            next_M = daily_index.max()

        window = daily_ret.loc[(daily_ret.index > M) & (daily_ret.index <= next_M)]
        
        # Calculate portfolio returns
        sub = window[picks].copy()
        sub = sub.fillna(0.0)
        r_series = (sub @ w.reindex(sub.columns).fillna(0.0)).astype(float)
        
        # Apply transaction cost on first day
        if len(r_series) > 0 and cost != 0.0:
            first_day = r_series.index[0]
            r_series.loc[first_day] = r_series.loc[first_day] - cost

        port_ret.loc[r_series.index] = r_series.values

        # Store positions
        positions_by_month[pd.to_datetime(M).date()] = w.sort_values(ascending=False).to_dict()
        
        # Track NAV history for vol targeting
        if len(r_series) > 0:
            nav_history.append({
                'return': r_series.mean() * 100  # For vol targeting
            })

    # Build equity curve
    port_ret = port_ret.fillna(0.0)
    equity = (1.0 + port_ret).cumprod()
    equity.name = "NAV"

    # Performance metrics
    def ann_return(series):
        if len(series) == 0: return np.nan
        yrs = (series.index[-1] - series.index[0]).days / 365.25
        return series.iloc[-1]**(1/yrs) - 1 if yrs > 0 else np.nan

    def ann_vol(ret_daily):
        return ret_daily.std(ddof=0) * np.sqrt(252)

    cagr = ann_return(equity)
    vol = ann_vol(port_ret)
    sharpe = (port_ret.mean() * 252) / vol if vol and vol > 0 else np.nan
    mdd = (equity / equity.cummax() - 1.0).min()

    # Save outputs
    equity.to_csv(os.path.join(out_dir, "backtest_equity_curve.csv"))
    pd.DataFrame.from_dict(positions_by_month, orient="index").to_csv(
        os.path.join(out_dir, "backtest_positions.csv")
    )
    
    with open(os.path.join(out_dir, "backtest_summary.txt"), "w") as f:
        f.write("=== Cross-Sectional Momentum Backtest (Strategy Core) ===\n\n")
        f.write(f"Period: {start} to {end}\n")
        f.write(f"CAGR:   {cagr:.4%}\n")
        f.write(f"Vol:    {vol:.4%}\n")
        f.write(f"Sharpe: {sharpe:.2f}\n")
        f.write(f"MaxDD:  {mdd:.2%}\n\n")
        f.write("Strategy Parameters:\n")
        f.write(f"  Transaction costs: {COST_PER_TURNOVER*10000:.1f} bps\n")
        f.write(f"  Top % per sector: {TOP_PCT_PER_SECTOR:.0%}\n")
        f.write(f"  Vol window: {VOL_WINDOW} days\n")
        f.write(f"  Position cap: {MAX_NAME_WEIGHT:.2%}\n")
        f.write(f"  Trend MA: {TREND_MA_WINDOW} days\n")

    print("\n" + "="*70)
    print("=== Backtest Complete ===")
    print("="*70)
    print(f"Period: {start} to {end}")
    print(f"CAGR:   {cagr:.2%}")
    print(f"Vol:    {vol:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"MaxDD:  {mdd:.2%}")
    print(f"\nOutputs written to: {out_dir}/")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Cross-Sectional Momentum Backtest")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    print("Loading data...")
    data_dir = Path(args.data_dir)
    
    prices = load_prices(data_dir)
    const = load_constituents(data_dir)
    sectors = load_sectors(data_dir)
    
    # Keep only needed columns
    keep = [c for c in ["date", "ticker", "px_last", "tri_gross", "volume"] 
            if c in prices.columns]
    prices = prices[keep].dropna(subset=["date", "ticker"])
    
    print(f"Loaded {len(prices)} price rows")
    print(f"Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")
    print(f"Unique tickers: {prices['ticker'].nunique()}")
    
    # Run backtest
    backtest(prices, const, sectors, args.start, args.end, args.out_dir)


if __name__ == "__main__":
    main()