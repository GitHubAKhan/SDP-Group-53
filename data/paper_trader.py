#!/usr/bin/env python3
"""
Paper Trading System for Cross-Sectional Momentum Strategy
Integrated with cross_sectional_risk.py backtest logic

Usage:
    # Initialize with starting capital
    python paper_trading_system.py --capital 100000 --mode init
    
    # Generate monthly rebalance orders
    python paper_trading_system.py --mode rebalance --signal-date 2025-01-31
    
    # Manual trade execution (after you get real prices)
    python paper_trading_system.py --mode execute --trades-file orders_2025-01-31.csv --prices-file execution_prices.csv
    
    # View performance report
    python paper_trading_system.py --mode report
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

# Strategy configuration (matches cross_sectional_risk.py)
TOP_PCT_PER_SECTOR = 0.10
MAX_NAME_WEIGHT = 0.05
VOL_WINDOW = 20
TREND_MA_WINDOW = 200
TREND_TICKER = "SPY US Equity"
TARGET_VOL = 0.10
USE_TREND_FILTER = True
USE_VOL_TARGETING = True
COST_PER_SHARE = 0.005  # 0.5 cents per share commission

# Helper functions from your backtest
def to_parse_keyable(member_code: str) -> str:
    US_EXCHANGE_CODES = {"UN", "UW", "UQ", "UA", "UR", "UT", "UV"}
    parts = str(member_code).strip().split()
    if len(parts) == 1:
        return f"{parts[0]} US Equity"
    ticker, code = parts[0], parts[1]
    if code in US_EXCHANGE_CODES:
        return f"{ticker} US Equity"
    if len(code) == 2:
        return f"{ticker} {code} Equity"
    return f"{ticker} US Equity"


class PaperTradingSystem:
    def __init__(self, initial_capital: float, data_dir: str, output_dir: str):
        self.capital = initial_capital
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Trading state
        self.positions = {}  # {ticker: {"shares": N, "cost_basis": price, "entry_date": date}}
        self.cash = initial_capital
        self.nav_history = []
        self.trade_log = []
        
        # Performance tracking
        self.cumulative_costs = 0.0
        
        # Load existing state
        self.load_state()
        
    def load_state(self):
        """Load existing paper trading state"""
        state_file = self.output_dir / "paper_trading_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.positions = state.get('positions', {})
                self.cash = state.get('cash', self.capital)
                self.nav_history = state.get('nav_history', [])
                self.trade_log = state.get('trade_log', [])
                self.cumulative_costs = state.get('cumulative_costs', 0.0)
                print(f"Loaded existing state: ${self.get_nav():,.2f} NAV, {len(self.positions)} positions")
    
    def save_state(self):
        """Save current state"""
        state = {
            'positions': self.positions,
            'cash': self.cash,
            'nav_history': self.nav_history,
            'trade_log': self.trade_log,
            'cumulative_costs': self.cumulative_costs,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.output_dir / "paper_trading_state.json", 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_nav(self, current_prices: dict = None) -> float:
        """Calculate current NAV"""
        position_value = 0.0
        if current_prices:
            for ticker, pos in self.positions.items():
                if ticker in current_prices:
                    position_value += pos['shares'] * current_prices[ticker]
        return self.cash + position_value
    
    def load_prices(self):
        """Load price data"""
        parquet_path = self.data_dir / "prices_parquet"
        if parquet_path.is_dir():
            df = pd.read_parquet(parquet_path)
        else:
            files = list(self.data_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet data in {self.data_dir}")
            df = pd.read_parquet(files[0])
        
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        return df
    
    def load_constituents(self):
        """Load SPX constituents"""
        path = self.data_dir / "constituents_long.csv"
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    
    def load_sectors(self):
        """Load sector mappings"""
        path = self.data_dir / "sectors.csv"
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df
    
    def compute_momentum_signals(self, prices_df, asof_date):
        """Compute 12-1 momentum and 20-day volatility"""
        prices_df = prices_df.sort_values(["ticker", "date"]).copy()
        
        # Use total return index if available
        base_col = "tri_gross" if "tri_gross" in prices_df.columns else "px_last"
        
        # Daily returns
        prices_df["ret"] = prices_df.groupby("ticker")[base_col].pct_change()
        
        # 12-1 momentum
        g = prices_df.groupby("ticker")
        val_t_21 = g[base_col].shift(21)
        val_t_252 = g[base_col].shift(252)
        prices_df["mom_12_1"] = (val_t_21 / val_t_252) - 1.0
        
        # 20-day volatility - use apply to avoid index issues
        vol20_values = g["ret"].apply(lambda x: x.rolling(20).std())
        prices_df["vol20"] = vol20_values.values
        
        # Filter to asof date
        asof_data = prices_df[prices_df["date"] <= asof_date].groupby("ticker").tail(1)
        
        return asof_data.set_index("ticker")
    
    def check_market_regime(self, prices_df, asof_date):
        """Check if SPY is above 200-day MA"""
        if not USE_TREND_FILTER:
            return True
        
        spy_data = prices_df[prices_df["ticker"] == TREND_TICKER].copy()
        spy_data = spy_data[spy_data["date"] <= asof_date].sort_values("date")
        
        if len(spy_data) < TREND_MA_WINDOW:
            return True
        
        price_col = "px_last" if "px_last" in spy_data.columns else "tri_gross"
        current_price = spy_data[price_col].iloc[-1]
        ma = spy_data[price_col].rolling(TREND_MA_WINDOW).mean().iloc[-1]
        
        return current_price > ma
    
    def get_universe(self, const_df, asof_date):
        """Get SPX constituents as of date"""
        df = const_df[const_df["date"] <= asof_date].sort_values(["ticker_raw", "date"])
        last = df.groupby("ticker_raw").tail(1)
        U_raw = last.loc[last["in_spx"] == 1, "ticker_raw"]
        U = set(to_parse_keyable(x) for x in U_raw)
        return U
    
    def select_stocks(self, momentum_series, sectors_map, universe):
        """Select top 10% momentum stocks per sector"""
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
            k = max(1, int(np.ceil(len(sub) * TOP_PCT_PER_SECTOR)))
            top_names = sub.sort_values(ascending=False).head(k).index.tolist()
            picks.extend(top_names)
        
        return list(dict.fromkeys(picks))
    
    def compute_weights(self, vol_series, picks):
        """Inverse volatility weights with position cap"""
        if len(picks) == 0:
            return pd.Series(dtype=float)
        
        v = vol_series.loc[picks].replace([np.inf, -np.inf], np.nan)
        med = np.nanmedian(v.values) if np.isfinite(v.values).any() else 0.02
        v = v.fillna(med).clip(lower=1e-6)
        
        w = 1.0 / v
        w = w / w.sum()
        w = w.clip(upper=MAX_NAME_WEIGHT)
        w = w / w.sum()
        
        return w
    
    def apply_vol_targeting(self, weights):
        """Scale weights based on realized portfolio volatility"""
        if not USE_VOL_TARGETING or len(self.nav_history) < 60:
            return weights
        
        # Get recent returns
        recent_nav = pd.DataFrame(self.nav_history[-60:])
        recent_nav['return'] = recent_nav['nav'].pct_change()
        
        realized_vol = recent_nav['return'].std() * np.sqrt(252)
        
        if realized_vol < 0.01:
            return weights
        
        scalar = TARGET_VOL / realized_vol
        scalar = np.clip(scalar, 0.25, 1.0)
        
        return weights * scalar
    
    def generate_target_portfolio(self, signal_date):
        """
        Generate target portfolio weights using backtest logic.
        Returns: {ticker: target_weight}
        """
        print(f"\nGenerating signals for {signal_date}...")
        
        # Load data
        prices_df = self.load_prices()
        const_df = self.load_constituents()
        sectors_df = self.load_sectors()
        
        # Check market regime
        asof_date = pd.to_datetime(signal_date)
        is_uptrend = self.check_market_regime(prices_df, asof_date)
        
        if not is_uptrend:
            print(f"Market in downtrend (SPY < {TREND_MA_WINDOW}MA) - Going to cash")
            return {}
        
        # Get universe
        universe = self.get_universe(const_df, asof_date)
        print(f"Universe: {len(universe)} stocks")
        
        # Compute signals
        signals = self.compute_momentum_signals(prices_df, asof_date)
        
        # Build sector map
        sectors_map = sectors_df.drop_duplicates(subset=["ticker"]).set_index("ticker")["sector"]
        
        # Select stocks
        picks = self.select_stocks(signals["mom_12_1"], sectors_map, universe)
        print(f"Selected {len(picks)} stocks (top {TOP_PCT_PER_SECTOR:.0%} per sector)")
        
        if len(picks) == 0:
            return {}
        
        # Compute weights
        weights = self.compute_weights(signals["vol20"], picks)
        
        # Apply vol targeting
        weights = self.apply_vol_targeting(weights)
        
        target_weights = weights.to_dict()
        
        print(f"Total target exposure: {sum(target_weights.values()):.2%}")
        print(f"Top 5 positions:")
        for ticker, weight in sorted(target_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {ticker}: {weight:.2%}")
        
        return target_weights
    
    def calculate_trades(self, target_weights, current_prices):
        """Calculate trades needed to reach target portfolio"""
        current_nav = self.get_nav(current_prices)
        trades = []
        
        # Current weights
        current_weights = {}
        for ticker, pos in self.positions.items():
            if ticker in current_prices:
                position_value = pos['shares'] * current_prices[ticker]
                current_weights[ticker] = position_value / current_nav
        
        # Calculate trades
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        
        for ticker in all_tickers:
            current_weight = current_weights.get(ticker, 0.0)
            target_weight = target_weights.get(ticker, 0.0)
            
            # Skip if change is tiny
            if abs(target_weight - current_weight) < 0.001:
                continue
            
            target_dollars = target_weight * current_nav
            current_dollars = current_weight * current_nav
            trade_dollars = target_dollars - current_dollars
            
            if ticker not in current_prices:
                print(f"Warning: No price for {ticker}, skipping")
                continue
            
            price = current_prices[ticker]
            shares_to_trade = int(trade_dollars / price)
            
            if shares_to_trade != 0:
                trades.append({
                    'ticker': ticker,
                    'shares': shares_to_trade,
                    'side': 'BUY' if shares_to_trade > 0 else 'SELL',
                    'reference_price': price,
                    'dollar_amount': abs(shares_to_trade * price),
                    'current_weight': current_weight,
                    'target_weight': target_weight
                })
        
        return trades
    
    def execute_trades(self, trades, execution_date, execution_prices):
        """Execute trades and update positions"""
        print(f"\nExecuting {len(trades)} trades on {execution_date}...")
        
        total_bought = 0
        total_sold = 0
        total_cost = 0
        
        for trade in trades:
            ticker = trade['ticker']
            shares = trade['shares']
            
            # Get execution price
            if ticker not in execution_prices:
                print(f"Warning: No execution price for {ticker}, using reference price")
                exec_price = trade['reference_price']
            else:
                exec_price = execution_prices[ticker]
            
            # Calculate slippage
            ref_price = trade['reference_price']
            slippage_pct = abs(exec_price - ref_price) / ref_price
            
            # Commission
            commission = abs(shares) * COST_PER_SHARE
            
            # Execute
            if shares > 0:  # Buy
                cost = shares * exec_price + commission
                self.cash -= cost
                total_bought += shares * exec_price
                
                if ticker in self.positions:
                    old_shares = self.positions[ticker]['shares']
                    old_basis = self.positions[ticker]['cost_basis']
                    new_shares = old_shares + shares
                    new_basis = (old_shares * old_basis + shares * exec_price) / new_shares
                    self.positions[ticker]['shares'] = new_shares
                    self.positions[ticker]['cost_basis'] = new_basis
                else:
                    self.positions[ticker] = {
                        'shares': shares,
                        'cost_basis': exec_price,
                        'entry_date': execution_date
                    }
                
            else:  # Sell
                proceeds = abs(shares) * exec_price - commission
                self.cash += proceeds
                total_sold += abs(shares) * exec_price
                
                if ticker in self.positions:
                    self.positions[ticker]['shares'] += shares
                    if self.positions[ticker]['shares'] == 0:
                        del self.positions[ticker]
            
            # Log trade
            self.trade_log.append({
                'date': execution_date,
                'ticker': ticker,
                'side': 'BUY' if shares > 0 else 'SELL',
                'shares': abs(shares),
                'price': exec_price,
                'commission': commission,
                'slippage_pct': slippage_pct * 100,
                'reference_price': ref_price
            })
            
            total_cost += commission
            self.cumulative_costs += commission
        
        print(f"Bought: ${total_bought:,.2f}")
        print(f"Sold: ${total_sold:,.2f}")
        print(f"Commissions: ${total_cost:,.2f}")
        print(f"Cash remaining: ${self.cash:,.2f}")
        print(f"Positions: {len(self.positions)}")
    
    def record_nav(self, date, prices):
        """Record NAV snapshot"""
        nav = self.get_nav(prices)
        self.nav_history.append({
            'date': date,
            'nav': nav,
            'cash': self.cash,
            'positions_value': nav - self.cash,
            'num_positions': len(self.positions)
        })
    
    def export_orders(self, trades, execution_date):
        """Export trade orders to CSV"""
        if len(trades) == 0:
            print("No trades to export")
            return None
        
        orders_file = self.output_dir / f"orders_{execution_date}.csv"
        df = pd.DataFrame(trades)
        df['execution_date'] = execution_date
        df.to_csv(orders_file, index=False)
        
        print(f"\nOrders exported to: {orders_file}")
        return str(orders_file)
    
    def generate_performance_report(self):
        """Generate performance metrics"""
        if len(self.nav_history) < 2:
            return None
        
        df = pd.DataFrame(self.nav_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['return'] = df['nav'].pct_change()
        
        total_return = (df['nav'].iloc[-1] / df['nav'].iloc[0]) - 1
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        
        if days > 0:
            annualized_return = (1 + total_return) ** (365.25 / days) - 1
        else:
            annualized_return = 0
        
        volatility = df['return'].std() * np.sqrt(252) if len(df) > 1 else 0
        sharpe = (df['return'].mean() * 252) / volatility if volatility > 0 else 0
        
        df['cummax'] = df['nav'].cummax()
        df['drawdown'] = (df['nav'] / df['cummax']) - 1
        max_drawdown = df['drawdown'].min()
        
        return {
            'start_date': df['date'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'days': days,
            'starting_nav': df['nav'].iloc[0],
            'current_nav': df['nav'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trade_log),
            'total_costs': self.cumulative_costs,
            'avg_positions': df['num_positions'].mean()
        }
    
    def print_report(self):
        """Print performance report"""
        report = self.generate_performance_report()
        
        if not report:
            print("\nInsufficient data for performance report")
            return
        
        print("\n" + "="*70)
        print("PAPER TRADING PERFORMANCE REPORT")
        print("="*70)
        print(f"Period: {report['start_date']} to {report['end_date']} ({report['days']} days)")
        print(f"\nCapital:")
        print(f"  Starting NAV: ${report['starting_nav']:,.2f}")
        print(f"  Current NAV:  ${report['current_nav']:,.2f}")
        print(f"  P&L:          ${report['current_nav'] - report['starting_nav']:,.2f}")
        print(f"  Return:       {report['total_return']:.2%}")
        print(f"\nPerformance Metrics:")
        print(f"  Annualized Return: {report['annualized_return']:.2%}")
        print(f"  Volatility:        {report['volatility']:.2%}")
        print(f"  Sharpe Ratio:      {report['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:      {report['max_drawdown']:.2%}")
        print(f"\nTrading Activity:")
        print(f"  Total Trades:      {report['num_trades']}")
        print(f"  Avg Positions:     {report['avg_positions']:.1f}")
        print(f"  Total Costs:       ${report['total_costs']:,.2f}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Paper Trading System")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="paper_trading")
    parser.add_argument("--mode", required=True, 
                        choices=['init', 'rebalance', 'execute', 'report'])
    parser.add_argument("--signal-date", help="Signal date (YYYY-MM-DD)")
    parser.add_argument("--trades-file", help="CSV file with trades to execute")
    parser.add_argument("--prices-file", help="CSV file with execution prices")
    
    args = parser.parse_args()
    
    system = PaperTradingSystem(
        initial_capital=args.capital,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    if args.mode == 'init':
        print(f"Initialized paper trading with ${args.capital:,.2f}")
        system.save_state()
    
    elif args.mode == 'rebalance':
        if not args.signal_date:
            print("Error: --signal-date required for rebalance mode")
            sys.exit(1)
        
        # Generate target portfolio
        target_weights = system.generate_target_portfolio(args.signal_date)
        
        if not target_weights:
            print("\nNo positions - staying in cash")
            system.save_state()
            return
        
        # Load current prices for trade calculation
        prices_df = system.load_prices()
        asof = pd.to_datetime(args.signal_date)
        latest_prices = prices_df[prices_df["date"] <= asof].groupby("ticker").tail(1)
        price_col = "px_last" if "px_last" in latest_prices.columns else "tri_gross"
        current_prices = latest_prices.set_index("ticker")[price_col].to_dict()
        
        # Calculate trades
        trades = system.calculate_trades(target_weights, current_prices)
        
        if trades:
            print(f"\nGenerated {len(trades)} trades:")
            for i, trade in enumerate(trades[:10], 1):
                print(f"{i}. {trade['side']} {trade['shares']:,} {trade['ticker']} @ ${trade['reference_price']:.2f}")
            if len(trades) > 10:
                print(f"... and {len(trades)-10} more")
            
            # Export orders
            system.export_orders(trades, args.signal_date)
        else:
            print("\nNo rebalancing needed")
        
        system.save_state()
    
    elif args.mode == 'execute':
        if not args.trades_file or not args.prices_file:
            print("Error: --trades-file and --prices-file required for execute mode")
            sys.exit(1)
        
        # Load trades and prices
        trades_df = pd.read_csv(args.trades_file)
        prices_df = pd.read_csv(args.prices_file)
        
        execution_prices = dict(zip(prices_df['ticker'], prices_df['price']))
        trades = trades_df.to_dict('records')
        execution_date = trades_df['execution_date'].iloc[0]
        
        # Execute
        system.execute_trades(trades, execution_date, execution_prices)
        system.record_nav(execution_date, execution_prices)
        system.save_state()
    
    elif args.mode == 'report':
        system.print_report()


if __name__ == "__main__":
    main()