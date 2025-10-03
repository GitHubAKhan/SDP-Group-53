#!/usr/bin/env python3
"""
Multi-Month Paper Trading Simulator

Runs your momentum strategy through multiple months of historical data
to simulate what paper trading would have looked like.

Usage:
    python paper_trading_simulator.py --start 2024-01-01 --end 2024-12-31 --capital 100000
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Import the paper trading system
from paper_trader import PaperTradingSystem

def get_month_ends(start_date, end_date):
    """Get all business month ends in date range"""
    dates = pd.date_range(start=start_date, end=end_date, freq='BM')
    return [d.date() for d in dates]

def get_next_trading_day(date, prices_df):
    """Get next available trading day after given date"""
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    future_dates = prices_df[prices_df['date'] > pd.to_datetime(date)]['date'].unique()
    if len(future_dates) == 0:
        return None
    return pd.to_datetime(future_dates[0]).date()

def get_prices_on_date(prices_df, date):
    """Get closing prices for all stocks on a given date"""
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    date_data = prices_df[prices_df['date'] == pd.to_datetime(date)]
    
    if len(date_data) == 0:
        # Try to get most recent prices before this date
        date_data = prices_df[prices_df['date'] <= pd.to_datetime(date)].groupby('ticker').tail(1)
    
    price_col = "px_last" if "px_last" in date_data.columns else "tri_gross"
    return date_data.set_index('ticker')[price_col].to_dict()

def run_simulation(start_date, end_date, initial_capital, data_dir, output_dir):
    """Run multi-month paper trading simulation"""
    
    print("="*70)
    print("PAPER TRADING SIMULATION")
    print("="*70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Data Directory: {data_dir}")
    print("="*70)
    
    # Create fresh system
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Clear any existing state
    state_file = output_path / "paper_trading_state.json"
    if state_file.exists():
        state_file.unlink()
    
    system = PaperTradingSystem(
        initial_capital=initial_capital,
        data_dir=data_dir,
        output_dir=output_dir
    )
    
    # Load price data once
    print("\nLoading price data...")
    prices_df = system.load_prices()
    
    # Get rebalancing dates
    rebal_dates = get_month_ends(start_date, end_date)
    print(f"Found {len(rebal_dates)} rebalancing dates\n")
    
    if len(rebal_dates) == 0:
        print("No month-ends found in date range")
        return
    
    # Track monthly performance
    monthly_returns = []
    
    for i, signal_date in enumerate(rebal_dates, 1):
        print(f"\n{'='*70}")
        print(f"MONTH {i}/{len(rebal_dates)}: {signal_date}")
        print('='*70)
        
        try:
            # Generate signals
            print(f"\n1. Generating signals for {signal_date}...")
            target_weights = system.generate_target_portfolio(signal_date.strftime('%Y-%m-%d'))
            
            if not target_weights:
                print("   → No positions (market downtrend or no signals)")
                # Still record NAV
                prices_signal = get_prices_on_date(prices_df, signal_date)
                system.record_nav(signal_date.strftime('%Y-%m-%d'), prices_signal)
                system.save_state()
                continue
            
            # Get execution date (next trading day)
            exec_date = get_next_trading_day(signal_date, prices_df)
            if exec_date is None:
                print(f"   → No trading data after {signal_date}, skipping")
                continue
            
            print(f"\n2. Execution date: {exec_date}")
            
            # Get prices for trade calculation (signal date)
            prices_signal = get_prices_on_date(prices_df, signal_date)
            
            # Calculate trades
            print(f"\n3. Calculating trades...")
            trades = system.calculate_trades(target_weights, prices_signal)
            
            if len(trades) == 0:
                print("   → No rebalancing needed")
                system.record_nav(exec_date.strftime('%Y-%m-%d'), prices_signal)
                system.save_state()
                continue
            
            print(f"   → {len(trades)} trades generated")
            
            # Get execution prices (next day)
            prices_exec = get_prices_on_date(prices_df, exec_date)
            
            # Execute trades
            print(f"\n4. Executing trades...")
            system.execute_trades(trades, exec_date.strftime('%Y-%m-%d'), prices_exec)
            
            # Record NAV
            system.record_nav(exec_date.strftime('%Y-%m-%d'), prices_exec)
            system.save_state()
            
            # Calculate monthly return
            if len(system.nav_history) >= 2:
                prev_nav = system.nav_history[-2]['nav']
                curr_nav = system.nav_history[-1]['nav']
                monthly_ret = (curr_nav / prev_nav - 1) * 100
                monthly_returns.append({
                    'date': exec_date,
                    'nav': curr_nav,
                    'return': monthly_ret
                })
                print(f"\n5. Monthly Performance:")
                print(f"   NAV: ${curr_nav:,.2f}")
                print(f"   Monthly Return: {monthly_ret:+.2f}%")
            
        except Exception as e:
            print(f"\nError processing {signal_date}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final report
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    
    system.print_report()
    
    # Monthly returns summary
    if monthly_returns:
        print("\n" + "="*70)
        print("MONTHLY RETURNS")
        print("="*70)
        returns_df = pd.DataFrame(monthly_returns)
        print(returns_df.to_string(index=False))
        
        print(f"\nReturn Statistics:")
        print(f"  Average Monthly: {returns_df['return'].mean():.2f}%")
        print(f"  Std Dev:         {returns_df['return'].std():.2f}%")
        print(f"  Best Month:      {returns_df['return'].max():.2f}%")
        print(f"  Worst Month:     {returns_df['return'].min():.2f}%")
        print(f"  Positive Months: {(returns_df['return'] > 0).sum()}/{len(returns_df)}")
        
        # Save to CSV
        returns_df.to_csv(output_path / "monthly_returns.csv", index=False)
        print(f"\nMonthly returns saved to: {output_path / 'monthly_returns.csv'}")
    
    # Export equity curve
    if system.nav_history:
        nav_df = pd.DataFrame(system.nav_history)
        nav_df.to_csv(output_path / "equity_curve.csv", index=False)
        print(f"Equity curve saved to: {output_path / 'equity_curve.csv'}")
    
    print("\n" + "="*70)
    
    return system

def main():
    parser = argparse.ArgumentParser(description="Multi-Month Paper Trading Simulator")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="paper_trading_sim", help="Output directory")
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        sys.exit(1)
    
    if start_date >= end_date:
        print("Error: Start date must be before end date")
        sys.exit(1)
    
    # Run simulation
    run_simulation(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()