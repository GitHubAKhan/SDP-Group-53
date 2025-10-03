#!/usr/bin/env python3
"""
Live Forward Paper Trading System

Runs your momentum strategy with real market data going forward.
Tracks a simulated $100k portfolio starting today.

Usage:
    # Initialize
    python live_paper_trader.py --init --capital 100000
    
    # Check if it's month-end and generate signals
    python live_paper_trader.py --check
    
    # Force run (for testing)
    python live_paper_trader.py --run
    
    # View current portfolio
    python live_paper_trader.py --status
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Check for yfinance
try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed.")
    print("Install it with: pip install yfinance")
    sys.exit(1)

# Strategy Configuration
TOP_PCT_PER_SECTOR = 0.10
MAX_NAME_WEIGHT = 0.05
TREND_MA_WINDOW = 200
TARGET_VOL = 0.10
USE_TREND_FILTER = True
USE_VOL_TARGETING = True

# S&P 500 ticker
SPY_TICKER = "SPY"

class LivePaperTrader:
    def __init__(self, state_file="live_paper_trading.json"):
        self.state_file = Path(state_file)
        self.state = self.load_state()
        
    def load_state(self):
        """Load trading state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
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
        """Save trading state to file"""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"State saved to {self.state_file}")
    
    def initialize(self, capital):
        """Initialize paper trading account"""
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
        print(f"\n{'='*60}")
        print(f"LIVE PAPER TRADING INITIALIZED")
        print(f"{'='*60}")
        print(f"Starting Capital: ${capital:,.2f}")
        print(f"Start Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"\nRun '--check' at month-end to generate signals")
        print(f"{'='*60}")
    
    def is_month_end(self):
        """Check if today is close to month end"""
        today = datetime.now()
        # Check if we're within last 3 days of month
        next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        days_until_next = (next_month - today).days
        return days_until_next <= 3
    
    def get_sp500_tickers(self):
        """Get current S&P 500 constituents"""
        # This is a simplified version - you'd ideally use your constituents file
        # For now, we'll get the top holdings
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = sp500['Symbol'].str.replace('.', '-').tolist()
            return tickers[:100]  # Use top 100 for speed in testing
        except:
            print("Warning: Could not fetch S&P 500 list. Using fallback list.")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'UNH']
    
    def download_prices(self, tickers, period='2y'):
        """Download historical prices from Yahoo Finance"""
        print(f"Downloading prices for {len(tickers)} stocks...")
        data = yf.download(tickers, period=period, progress=False, group_by='ticker', auto_adjust=True)
        return data
    
    def check_market_regime(self):
        """Check if SPY is above 200-day MA"""
        if not USE_TREND_FILTER:
            return True
        
        print("Checking market regime (SPY vs 200-day MA)...")
        spy = yf.download(SPY_TICKER, period='1y', progress=False, auto_adjust=True)
        
        if len(spy) < TREND_MA_WINDOW:
            print("  → Not enough data, assuming uptrend")
            return True
        
        # Handle both Series and DataFrame returns
        if isinstance(spy, pd.DataFrame):
            close = spy['Close'] if 'Close' in spy.columns else spy['Adj Close']
        else:
            close = spy
        
        current_price = float(close.iloc[-1])
        ma_200 = float(close.rolling(TREND_MA_WINDOW).mean().iloc[-1])
        
        is_uptrend = current_price > ma_200
        print(f"  → SPY: ${current_price:.2f}")
        print(f"  → 200MA: ${ma_200:.2f}")
        print(f"  → Trend: {'UPTREND ✓' if is_uptrend else 'DOWNTREND ✗'}")
        
        return is_uptrend
    
    def calculate_momentum(self, prices):
        """Calculate 12-1 momentum for each stock"""
        print("Calculating momentum signals...")
        momentum = {}
        volatility = {}
        
        for ticker in prices.columns.levels[0] if isinstance(prices.columns, pd.MultiIndex) else [prices.name]:
            try:
                if isinstance(prices.columns, pd.MultiIndex):
                    close = prices[ticker]['Close'].dropna()
                else:
                    close = prices['Close'].dropna()
                
                if len(close) < 252:
                    continue
                
                # 12-1 momentum
                mom = (close.iloc[-21] / close.iloc[-252]) - 1
                
                # 20-day volatility
                returns = close.pct_change()
                vol = returns.tail(20).std()
                
                if np.isfinite(mom) and np.isfinite(vol) and vol > 0:
                    momentum[ticker] = mom
                    volatility[ticker] = vol
            except:
                continue
        
        print(f"  → Calculated momentum for {len(momentum)} stocks")
        return momentum, volatility
    
    def select_stocks(self, momentum, top_pct=TOP_PCT_PER_SECTOR):
        """Select top momentum stocks"""
        # Simplified: just take top N overall since we don't have sector data
        mom_series = pd.Series(momentum).sort_values(ascending=False)
        n_select = max(10, int(len(mom_series) * top_pct))
        selected = mom_series.head(n_select).index.tolist()
        
        print(f"  → Selected {len(selected)} stocks (top {top_pct:.0%})")
        return selected
    
    def calculate_weights(self, selected, volatility):
        """Calculate inverse-vol weights"""
        vol_series = pd.Series({t: volatility[t] for t in selected})
        
        # Inverse vol weighting
        weights = 1.0 / vol_series
        weights = weights / weights.sum()
        
        # Cap at max weight
        weights = weights.clip(upper=MAX_NAME_WEIGHT)
        weights = weights / weights.sum()
        
        return weights
    
    def apply_vol_targeting(self, weights):
        """Scale portfolio by volatility targeting"""
        if not USE_VOL_TARGETING or len(self.state['nav_history']) < 60:
            return weights
        
        recent_returns = [h['return'] for h in self.state['nav_history'][-60:] if 'return' in h]
        if len(recent_returns) < 20:
            return weights
        
        realized_vol = np.std(recent_returns) * np.sqrt(12)  # Annualized
        
        if realized_vol < 0.01:
            return weights
        
        scalar = TARGET_VOL / realized_vol
        scalar = np.clip(scalar, 0.25, 1.0)
        
        print(f"  → Vol targeting: {scalar:.2f}x (target {TARGET_VOL:.0%}, realized {realized_vol:.1%})")
        
        return weights * scalar
    
    def generate_trades(self, target_weights, current_prices):
        """Generate trade list to reach target portfolio"""
        trades = []
        current_nav = self.get_nav(current_prices)
        
        # Current weights
        current_weights = {}
        for ticker, pos in self.state['positions'].items():
            if ticker in current_prices:
                value = pos['shares'] * current_prices[ticker]
                current_weights[ticker] = value / current_nav
        
        # Calculate trades
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        
        for ticker in all_tickers:
            current_w = current_weights.get(ticker, 0)
            target_w = target_weights.get(ticker, 0)
            
            if abs(target_w - current_w) < 0.001:  # Skip tiny changes
                continue
            
            if ticker not in current_prices:
                continue
            
            target_dollars = target_w * current_nav
            current_dollars = current_w * current_nav
            trade_dollars = target_dollars - current_dollars
            
            price = current_prices[ticker]
            shares = int(trade_dollars / price)
            
            if shares != 0:
                trades.append({
                    'ticker': ticker,
                    'shares': shares,
                    'side': 'BUY' if shares > 0 else 'SELL',
                    'price': price,
                    'value': abs(shares * price)
                })
        
        return trades
    
    def execute_trades(self, trades):
        """Execute trades in paper account"""
        print(f"\nExecuting {len(trades)} trades...")
        
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
                
                if ticker in self.state['positions']:
                    self.state['positions'][ticker]['shares'] += shares
                else:
                    self.state['positions'][ticker] = {'shares': shares, 'entry_price': price}
            else:  # Sell
                proceeds = abs(shares) * price
                self.state['cash'] += proceeds
                total_sold += proceeds
                
                if ticker in self.state['positions']:
                    self.state['positions'][ticker]['shares'] += shares
                    if self.state['positions'][ticker]['shares'] == 0:
                        del self.state['positions'][ticker]
            
            self.state['trade_history'].append({
                'date': datetime.now().isoformat(),
                'ticker': ticker,
                'shares': shares,
                'price': price,
                'side': trade['side']
            })
        
        print(f"  Bought: ${total_bought:,.2f}")
        print(f"  Sold: ${total_sold:,.2f}")
        print(f"  Cash: ${self.state['cash']:,.2f}")
        print(f"  Positions: {len(self.state['positions'])}")
    
    def get_nav(self, current_prices=None):
        """Calculate current NAV"""
        if current_prices is None:
            current_prices = self.get_current_prices()
        
        position_value = 0
        for ticker, pos in self.state['positions'].items():
            if ticker in current_prices:
                position_value += pos['shares'] * current_prices[ticker]
        
        return self.state['cash'] + position_value
    
    def get_current_prices(self):
        """Get current prices for all positions"""
        if not self.state['positions']:
            return {}
        
        tickers = list(self.state['positions'].keys())
        data = yf.download(tickers, period='1d', progress=False, auto_adjust=True)
        
        prices = {}
        
        # Handle single ticker (returns DataFrame) vs multiple tickers (returns MultiIndex)
        if len(tickers) == 1:
            if 'Close' in data.columns:
                prices[tickers[0]] = float(data['Close'].iloc[-1])
        else:
            # Multiple tickers - data has MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in tickers:
                    try:
                        prices[ticker] = float(data[ticker]['Close'].iloc[-1])
                    except:
                        # Try alternate format
                        try:
                            prices[ticker] = float(data['Close'][ticker].iloc[-1])
                        except:
                            pass
            else:
                # Single level columns (shouldn't happen but handle it)
                if 'Close' in data.columns:
                    prices[tickers[0]] = float(data['Close'].iloc[-1])
        
        return prices
    
    def record_nav(self):
        """Record current NAV"""
        current_prices = self.get_current_prices()
        nav = self.get_nav(current_prices)
        
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
    
    def run_rebalance(self):
        """Main rebalancing logic"""
        print(f"\n{'='*60}")
        print(f"MONTHLY REBALANCE: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Check market regime
        is_uptrend = self.check_market_regime()
        
        if not is_uptrend:
            print("\n⚠️  Market in downtrend - moving to cash")
            # Liquidate all positions
            if self.state['positions']:
                current_prices = self.get_current_prices()
                for ticker in list(self.state['positions'].keys()):
                    shares = self.state['positions'][ticker]['shares']
                    price = current_prices.get(ticker, 0)
                    self.state['cash'] += shares * price
                    del self.state['positions'][ticker]
                print(f"  Liquidated all positions. Cash: ${self.state['cash']:,.2f}")
            
            self.record_nav()
            self.state['last_rebalance'] = datetime.now().isoformat()
            self.save_state()
            return
        
        # Get S&P 500 tickers
        tickers = self.get_sp500_tickers()
        
        # Download prices
        prices = self.download_prices(tickers)
        
        # Calculate signals
        momentum, volatility = self.calculate_momentum(prices)
        
        if len(momentum) == 0:
            print("No valid momentum signals - staying in cash")
            self.record_nav()
            self.state['last_rebalance'] = datetime.now().isoformat()
            self.save_state()
            return
        
        # Select stocks
        selected = self.select_stocks(momentum)
        
        # Calculate weights
        weights = self.calculate_weights(selected, volatility)
        
        # Apply vol targeting
        weights = self.apply_vol_targeting(weights)
        
        # Get current prices
        current_prices = {}
        for ticker in selected:
            try:
                if isinstance(prices.columns, pd.MultiIndex):
                    current_prices[ticker] = prices[ticker]['Close'].iloc[-1]
                else:
                    current_prices[ticker] = prices['Close'].iloc[-1]
            except:
                pass
        
        print(f"\nTarget Portfolio:")
        print(f"  Total exposure: {weights.sum():.1%}")
        print(f"  Top 5 positions:")
        for ticker, weight in weights.sort_values(ascending=False).head(5).items():
            print(f"    {ticker}: {weight:.2%}")
        
        # Generate trades
        trades = self.generate_trades(weights, current_prices)
        
        if trades:
            print(f"\nTrade Summary:")
            buys = [t for t in trades if t['shares'] > 0]
            sells = [t for t in trades if t['shares'] < 0]
            print(f"  Buys: {len(buys)}")
            print(f"  Sells: {len(sells)}")
            
            # Execute
            self.execute_trades(trades)
        else:
            print("\nNo rebalancing needed")
        
        # Record NAV
        self.record_nav()
        self.state['last_rebalance'] = datetime.now().isoformat()
        self.save_state()
        
        print(f"\n{'='*60}")
        print(f"Rebalance complete. NAV: ${self.get_nav():,.2f}")
        print(f"{'='*60}")
    
    def print_status(self):
        """Print current portfolio status"""
        print(f"\n{'='*60}")
        print(f"PORTFOLIO STATUS")
        print(f"{'='*60}")
        
        if not self.state['initialized']:
            print("Not initialized. Run --init first.")
            return
        
        current_prices = self.get_current_prices()
        nav = self.get_nav(current_prices)
        
        print(f"Starting Capital: ${self.state['capital']:,.2f}")
        print(f"Current NAV: ${nav:,.2f}")
        print(f"Total Return: {((nav/self.state['capital'])-1)*100:+.2f}%")
        print(f"Cash: ${self.state['cash']:,.2f}")
        print(f"Positions: {len(self.state['positions'])}")
        
        if self.state['last_rebalance']:
            print(f"Last Rebalance: {self.state['last_rebalance'][:10]}")
        
        if self.state['positions']:
            print(f"\nCurrent Holdings:")
            total_value = 0
            for ticker, pos in sorted(self.state['positions'].items()):
                price = current_prices.get(ticker, 0)
                value = pos['shares'] * price
                total_value += value
                pct = (value / nav) * 100 if nav > 0 else 0
                print(f"  {ticker}: {pos['shares']} shares @ ${price:.2f} = ${value:,.2f} ({pct:.1f}%)")
        
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Live Forward Paper Trading")
    parser.add_argument('--init', action='store_true', help='Initialize paper trading')
    parser.add_argument('--capital', type=float, default=100000, help='Starting capital')
    parser.add_argument('--check', action='store_true', help='Check if month-end and run if so')
    parser.add_argument('--run', action='store_true', help='Force run rebalance now')
    parser.add_argument('--status', action='store_true', help='Show portfolio status')
    
    args = parser.parse_args()
    
    trader = LivePaperTrader()
    
    if args.init:
        trader.initialize(args.capital)
    
    elif args.check:
        if not trader.state['initialized']:
            print("ERROR: Not initialized. Run --init first.")
            sys.exit(1)
        
        if trader.is_month_end():
            print("Month-end detected. Running rebalance...")
            trader.run_rebalance()
        else:
            print("Not month-end yet. Check back in a few days.")
            trader.print_status()
    
    elif args.run:
        if not trader.state['initialized']:
            print("ERROR: Not initialized. Run --init first.")
            sys.exit(1)
        trader.run_rebalance()
    
    elif args.status:
        trader.print_status()
    
    else:
        print("Use --init, --check, --run, or --status")
        print("Run with --help for more info")


if __name__ == "__main__":
    main()