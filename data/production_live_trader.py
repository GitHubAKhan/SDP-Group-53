#!/usr/bin/env python3
"""
Production Live Trading System

Uses strategy_core.py for all signal generation to ensure
consistency with backtests.

Usage:
    python production_live_trader.py --init --capital 100000
    python production_live_trader.py --update-data
    python production_live_trader.py --check
    python production_live_trader.py --status
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Import strategy core
try:
    from strategy_core import generate_signals, calculate_trades
except ImportError:
    print("ERROR: strategy_core.py not found in current directory")
    sys.exit(1)

# Check for yfinance
try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionLiveTrader:
    def __init__(self, data_dir="data", state_file="production_trading_state.json"):
        self.data_dir = Path(data_dir)
        self.state_file = Path(state_file)
        self.state = self.load_state()
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_state(self):
        """Load trading state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                logger.info(f"Loaded state: NAV=${state.get('nav', 0):,.2f}, "
                           f"{len(state.get('positions', {}))} positions")
                return state
        return {
            'capital': 100000,
            'cash': 100000,
            'positions': {},
            'nav_history': [],
            'trade_history': [],
            'last_rebalance': None,
            'initialized': False
        }
    
    def save_state(self):
        """Save trading state"""
        self.state['last_updated'] = datetime.now().isoformat()
        self.state['nav'] = self.get_nav()
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        logger.info(f"State saved: NAV=${self.state['nav']:,.2f}")
    
    def initialize(self, capital):
        """Initialize paper trading"""
        self.state = {
            'capital': capital,
            'cash': capital,
            'positions': {},
            'nav_history': [],
            'trade_history': [],
            'last_rebalance': None,
            'initialized': True,
            'start_date': datetime.now().isoformat()
        }
        self.save_state()
        
        logger.info("="*60)
        logger.info("PRODUCTION LIVE TRADING INITIALIZED")
        logger.info("="*60)
        logger.info(f"Starting Capital: ${capital:,.2f}")
        logger.info(f"Start Date: {datetime.now().strftime('%Y-%m-%d')}")
        logger.info(f"Data Directory: {self.data_dir}")
        logger.info("="*60)
    
    def is_month_end(self):
        """Check if close to month end"""
        today = datetime.now()
        next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        days_until_next = (next_month - today).days
        return days_until_next <= 3
    
    def update_price_data(self):
        """
        Update parquet files with recent data from Yahoo Finance.
        Appends new data to existing files.
        """
        logger.info("Updating price data...")
        
        try:
            # Load existing prices
            prices_df = pd.read_parquet(self.data_dir / "prices_parquet")
            prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None)
            
            # Get last date in data
            last_date = prices_df["date"].max()
            logger.info(f"Last date in data: {last_date.date()}")
            
            # Get unique tickers
            tickers = prices_df["ticker"].unique().tolist()
            logger.info(f"Updating {len(tickers)} tickers...")
            
            # Download recent data
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            if start_date >= end_date:
                logger.info("Data is already up to date")
                return
            
            logger.info(f"Downloading {start_date} to {end_date}")
            
            # Download in batches to avoid errors
            batch_size = 50
            new_data = []
            
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i+batch_size]
                logger.info(f"Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
                
                try:
                    data = yf.download(
                        batch,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=False
                    )
                    
                    if len(batch) == 1:
                        # Single ticker returns DataFrame
                        if not data.empty:
                            ticker = batch[0]
                            df = data.reset_index()
                            df['ticker'] = ticker
                            new_data.append(df)
                    else:
                        # Multiple tickers returns MultiIndex
                        for ticker in batch:
                            try:
                                ticker_data = data[ticker].dropna(how='all')
                                if not ticker_data.empty:
                                    df = ticker_data.reset_index()
                                    df['ticker'] = ticker
                                    new_data.append(df)
                            except:
                                continue
                                
                except Exception as e:
                    logger.warning(f"Error downloading batch: {e}")
                    continue
            
            if new_data:
                # Combine new data
                new_df = pd.concat(new_data, ignore_index=True)
                new_df = new_df.rename(columns={
                    'Date': 'date',
                    'Open': 'px_open',
                    'High': 'px_high',
                    'Low': 'px_low',
                    'Close': 'px_last',
                    'Volume': 'volume',
                    'Adj Close': 'tri_gross'
                })
                
                # Append to existing data
                updated_df = pd.concat([prices_df, new_df], ignore_index=True)
                updated_df = updated_df.drop_duplicates(subset=['date', 'ticker'])
                updated_df = updated_df.sort_values(['ticker', 'date'])
                
                # Save back to parquet
                updated_df.to_parquet(self.data_dir / "prices_parquet", index=False)
                
                logger.info(f"Added {len(new_df)} new rows")
                logger.info(f"New last date: {updated_df['date'].max().date()}")
            else:
                logger.warning("No new data downloaded")
                
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            raise
    
    def get_current_prices(self):
        """Get current prices for all positions"""
        if not self.state['positions']:
            return {}
        
        tickers = list(self.state['positions'].keys())
        
        # Convert to Yahoo Finance format
        yf_tickers = [t.split()[0] for t in tickers]
        
        try:
            data = yf.download(yf_tickers, period='1d', progress=False, auto_adjust=True)
            
            prices = {}
            if len(yf_tickers) == 1:
                if 'Close' in data.columns:
                    prices[tickers[0]] = float(data['Close'].iloc[-1])
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    for i, ticker in enumerate(tickers):
                        yf_ticker = yf_tickers[i]
                        try:
                            prices[ticker] = float(data[yf_ticker]['Close'].iloc[-1])
                        except:
                            try:
                                prices[ticker] = float(data['Close'][yf_ticker].iloc[-1])
                            except:
                                logger.warning(f"Could not get price for {ticker}")
                else:
                    prices[tickers[0]] = float(data['Close'].iloc[-1])
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return {}
    
    def get_nav(self):
        """Calculate current NAV"""
        current_prices = self.get_current_prices()
        
        position_value = 0
        for ticker, shares in self.state['positions'].items():
            if ticker in current_prices:
                position_value += shares * current_prices[ticker]
        
        return self.state['cash'] + position_value
    
    def run_rebalance(self):
        """Execute monthly rebalance"""
        logger.info("="*70)
        logger.info(f"MONTHLY REBALANCE: {datetime.now().strftime('%Y-%m-%d')}")
        logger.info("="*70)
        
        # Get today's date for signal generation
        signal_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Generate signals using strategy core
            logger.info("Generating signals from strategy_core...")
            target_weights = generate_signals(
                data_dir=self.data_dir,
                asof_date=signal_date,
                nav_history=self.state['nav_history']
            )
            
            if not target_weights:
                logger.info("⚠️  Going to cash (downtrend or no signals)")
                
                # Liquidate all positions
                if self.state['positions']:
                    current_prices = self.get_current_prices()
                    for ticker, shares in list(self.state['positions'].items()):
                        price = current_prices.get(ticker, 0)
                        self.state['cash'] += shares * price
                        del self.state['positions'][ticker]
                        
                        self.state['trade_history'].append({
                            'date': datetime.now().isoformat(),
                            'ticker': ticker,
                            'shares': -shares,
                            'price': price,
                            'side': 'SELL'
                        })
                    
                    logger.info(f"Liquidated all positions. Cash: ${self.state['cash']:,.2f}")
                
                self.record_nav()
                self.state['last_rebalance'] = datetime.now().isoformat()
                self.save_state()
                return
            
            # Get current prices
            logger.info("Fetching current prices...")
            current_prices = {}
            
            # Need prices for all tickers (current and target)
            all_tickers = set(self.state['positions'].keys()) | set(target_weights.keys())
            yf_tickers = [t.split()[0] for t in all_tickers]
            
            data = yf.download(list(set(yf_tickers)), period='1d', progress=False, auto_adjust=True)
            
            for ticker in all_tickers:
                yf_ticker = ticker.split()[0]
                try:
                    if len(all_tickers) == 1:
                        current_prices[ticker] = float(data['Close'].iloc[-1])
                    else:
                        try:
                            current_prices[ticker] = float(data[yf_ticker]['Close'].iloc[-1])
                        except:
                            current_prices[ticker] = float(data['Close'][yf_ticker].iloc[-1])
                except:
                    logger.warning(f"Could not get price for {ticker}")
            
            # Calculate NAV
            current_nav = self.get_nav()
            logger.info(f"Current NAV: ${current_nav:,.2f}")
            
            # Display target portfolio
            logger.info(f"\nTarget Portfolio:")
            logger.info(f"  Total exposure: {sum(target_weights.values()):.1%}")
            logger.info(f"  Number of positions: {len(target_weights)}")
            logger.info(f"  Top 5 holdings:")
            sorted_weights = sorted(target_weights.items(), key=lambda x: x[1], reverse=True)
            for ticker, weight in sorted_weights[:5]:
                logger.info(f"    {ticker}: {weight:.2%}")
            
            # Calculate trades
            trades = calculate_trades(
                target_weights=target_weights,
                current_positions=self.state['positions'],
                current_prices=current_prices,
                nav=current_nav
            )
            
            if not trades:
                logger.info("\nNo rebalancing needed")
                self.record_nav()
                self.state['last_rebalance'] = datetime.now().isoformat()
                self.save_state()
                return
            
            # Display trades
            buys = [t for t in trades if t['shares'] > 0]
            sells = [t for t in trades if t['shares'] < 0]
            
            logger.info(f"\nTrade Summary:")
            logger.info(f"  Buys: {len(buys)}")
            logger.info(f"  Sells: {len(sells)}")
            
            # Execute trades
            self.execute_trades(trades)
            
            # Record NAV
            self.record_nav()
            self.state['last_rebalance'] = datetime.now().isoformat()
            self.save_state()
            
            logger.info("="*70)
            logger.info(f"Rebalance complete. NAV: ${self.get_nav():,.2f}")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Error during rebalance: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def execute_trades(self, trades):
        """Execute trades in paper account"""
        logger.info(f"\nExecuting {len(trades)} trades...")
        
        total_bought = 0
        total_sold = 0
        
        for trade in trades:
            ticker = trade['ticker']
            shares = trade['shares']
            price = trade['price']
            
            if shares > 0:  # Buy
                cost = shares * price
                self.state['cash'] -= cost
                total_bought += cost
                
                self.state['positions'][ticker] = self.state['positions'].get(ticker, 0) + shares
                
            else:  # Sell
                proceeds = abs(shares) * price
                self.state['cash'] += proceeds
                total_sold += proceeds
                
                self.state['positions'][ticker] = self.state['positions'].get(ticker, 0) + shares
                if self.state['positions'][ticker] == 0:
                    del self.state['positions'][ticker]
            
            # Log trade
            self.state['trade_history'].append({
                'date': datetime.now().isoformat(),
                'ticker': ticker,
                'shares': shares,
                'price': price,
                'side': trade['side']
            })
        
        logger.info(f"  Bought: ${total_bought:,.2f}")
        logger.info(f"  Sold: ${total_sold:,.2f}")
        logger.info(f"  Cash: ${self.state['cash']:,.2f}")
        logger.info(f"  Positions: {len(self.state['positions'])}")
    
    def record_nav(self):
        """Record NAV snapshot"""
        nav = self.get_nav()
        
        entry = {
            'date': datetime.now().isoformat(),
            'nav': nav,
            'cash': self.state['cash'],
            'num_positions': len(self.state['positions'])
        }
        
        if len(self.state['nav_history']) > 0:
            prev_nav = self.state['nav_history'][-1]['nav']
            entry['return'] = ((nav / prev_nav) - 1) * 100
        
        self.state['nav_history'].append(entry)
    
    def print_status(self):
        """Print portfolio status"""
        print("\n" + "="*70)
        print("PORTFOLIO STATUS")
        print("="*70)
        
        if not self.state['initialized']:
            print("Not initialized. Run --init first.")
            return
        
        nav = self.get_nav()
        total_return = ((nav / self.state['capital']) - 1) * 100
        
        print(f"Starting Capital: ${self.state['capital']:,.2f}")
        print(f"Current NAV: ${nav:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Cash: ${self.state['cash']:,.2f}")
        print(f"Positions: {len(self.state['positions'])}")
        
        if self.state['last_rebalance']:
            print(f"Last Rebalance: {self.state['last_rebalance'][:10]}")
        
        if self.state['positions']:
            print(f"\nCurrent Holdings:")
            current_prices = self.get_current_prices()
            
            holdings = []
            for ticker, shares in self.state['positions'].items():
                price = current_prices.get(ticker, 0)
                value = shares * price
                pct = (value / nav) * 100 if nav > 0 else 0
                holdings.append((ticker, shares, price, value, pct))
            
            # Sort by value descending
            holdings.sort(key=lambda x: x[3], reverse=True)
            
            for ticker, shares, price, value, pct in holdings:
                print(f"  {ticker:20s} {shares:6.0f} @ ${price:8.2f} = ${value:12,.2f} ({pct:5.2f}%)")
        
        print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Live Trading System")
    parser.add_argument('--init', action='store_true', help='Initialize trading')
    parser.add_argument('--capital', type=float, default=100000, help='Starting capital')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--update-data', action='store_true', help='Update price data from Yahoo Finance')
    parser.add_argument('--check', action='store_true', help='Check if month-end and rebalance')
    parser.add_argument('--run', action='store_true', help='Force run rebalance now')
    parser.add_argument('--status', action='store_true', help='Show portfolio status')
    
    args = parser.parse_args()
    
    try:
        trader = ProductionLiveTrader(data_dir=args.data_dir)
        
        if args.init:
            trader.initialize(args.capital)
        
        elif args.update_data:
            if not trader.state['initialized']:
                logger.error("Not initialized. Run --init first.")
                sys.exit(1)
            trader.update_price_data()
        
        elif args.check:
            if not trader.state['initialized']:
                logger.error("Not initialized. Run --init first.")
                sys.exit(1)
            
            if trader.is_month_end():
                logger.info("Month-end detected. Running rebalance...")
                trader.run_rebalance()
            else:
                logger.info("Not month-end yet. Check back in a few days.")
                trader.print_status()
        
        elif args.run:
            if not trader.state['initialized']:
                logger.error("Not initialized. Run --init first.")
                sys.exit(1)
            trader.run_rebalance()
        
        elif args.status:
            trader.print_status()
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()