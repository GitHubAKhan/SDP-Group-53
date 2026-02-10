#!/usr/bin/env python3
"""
Alpaca Auto-Trader

Executes trades automatically using Alpaca API.
Supports both paper trading (testing) and live trading.

Usage:
    # Dry run (preview only, no execution)
    python3 alpaca_trader.py --portfolio todays_portfolio.csv --dry-run

    # Execute trades (paper trading)
    python3 alpaca_trader.py --portfolio todays_portfolio.csv

    # Execute trades (LIVE MONEY)
    python3 alpaca_trader.py --portfolio todays_portfolio.csv --live
"""

import os
import sys
import argparse
import pandas as pd
import requests
import time
from datetime import datetime

class AlpacaTrader:
    def __init__(self, api_key=None, api_secret=None, paper=True):
        """
        Initialize Alpaca trader.

        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            api_secret: Alpaca API secret (or set ALPACA_API_SECRET env var)
            paper: Use paper trading (True) or live trading (False)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')

        # Try loading from config file if not set
        if not self.api_key or not self.api_secret:
            try:
                import alpaca_config
                self.api_key = self.api_key or alpaca_config.ALPACA_API_KEY
                self.api_secret = self.api_secret or alpaca_config.ALPACA_API_SECRET
            except ImportError:
                pass

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials not found!\n\n"
                "Option 1 - Environment variables:\n"
                "  export ALPACA_API_KEY='your_key'\n"
                "  export ALPACA_API_SECRET='your_secret'\n\n"
                "Option 2 - Config file:\n"
                "  1. Copy alpaca_config_template.py to alpaca_config.py\n"
                "  2. Add your API credentials to alpaca_config.py\n"
                "  3. Add alpaca_config.py to .gitignore\n\n"
                "Get API keys at: https://app.alpaca.markets/paper/dashboard/overview"
            )

        if paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }

        self.paper = paper

    def get_account(self):
        """Get account information."""
        response = requests.get(
            f"{self.base_url}/v2/account",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_positions(self):
        """Get current positions."""
        response = requests.get(
            f"{self.base_url}/v2/positions",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_position(self, symbol):
        """Get position for specific symbol."""
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions/{symbol}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def close_all_positions(self):
        """Close all open positions."""
        response = requests.delete(
            f"{self.base_url}/v2/positions",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def close_position(self, symbol):
        """Close position for specific symbol."""
        response = requests.delete(
            f"{self.base_url}/v2/positions/{symbol}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def place_order(self, symbol, qty=None, notional=None, side='buy',
                    order_type='market', time_in_force='day'):
        """
        Place an order.

        Args:
            symbol: Stock ticker
            qty: Number of shares (use qty OR notional, not both)
            notional: Dollar amount (use qty OR notional, not both)
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', etc.
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
        """
        order = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'time_in_force': time_in_force
        }

        if qty is not None:
            order['qty'] = str(int(qty))  # Must be string integer
        elif notional is not None:
            order['notional'] = str(round(notional, 2))  # Must be string with 2 decimals
        else:
            raise ValueError("Must specify either qty or notional")

        response = requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json=order
        )

        if response.status_code != 200:
            # Print detailed error for debugging
            try:
                error_detail = response.json()
                raise requests.exceptions.HTTPError(
                    f"{response.status_code} {response.reason}: {error_detail.get('message', 'Unknown error')}"
                )
            except:
                response.raise_for_status()

        return response.json()

    def get_orders(self, status='all'):
        """Get orders (all, open, closed)."""
        response = requests.get(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            params={'status': status}
        )
        response.raise_for_status()
        return response.json()

    def cancel_all_orders(self):
        """Cancel all open orders."""
        response = requests.delete(
            f"{self.base_url}/v2/orders",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()


def load_portfolio(csv_path):
    """Load portfolio from CSV."""
    df = pd.read_csv(csv_path)

    # Expected columns: ticker, weight, dollar_amount
    required_cols = ['ticker', 'weight', 'dollar_amount']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Portfolio CSV missing column: {col}")

    return df


def clean_ticker(ticker):
    """
    Clean ticker for Alpaca.
    Remove ' US Equity' suffix if present.
    """
    if isinstance(ticker, str):
        ticker = ticker.replace(' US Equity', '').strip()
    return ticker


def generate_trade_list(current_positions, target_portfolio):
    """
    Generate list of trades to execute.

    Strategy:
    1. Sell all positions not in target portfolio
    2. Buy/rebalance positions in target portfolio

    Returns:
        List of trades: [{'action': 'buy/sell', 'symbol': 'AAPL', 'notional': 1000}, ...]
    """
    trades = []

    # Get current position symbols
    current_symbols = {pos['symbol']: float(pos['market_value'])
                       for pos in current_positions}

    # Get target symbols with dollar amounts
    target_symbols = {}
    for _, row in target_portfolio.iterrows():
        symbol = clean_ticker(row['ticker'])
        target_symbols[symbol] = float(row['dollar_amount'])

    # Sell positions not in target
    for symbol in current_symbols:
        if symbol not in target_symbols:
            trades.append({
                'action': 'sell',
                'symbol': symbol,
                'current_value': current_symbols[symbol],
                'target_value': 0,
                'notional': None  # Sell entire position
            })

    # Buy/adjust positions in target
    for symbol, target_value in target_symbols.items():
        current_value = current_symbols.get(symbol, 0)

        # Only trade if difference is significant (>$10)
        diff = target_value - current_value
        if abs(diff) > 10:
            if diff > 0:
                trades.append({
                    'action': 'buy',
                    'symbol': symbol,
                    'current_value': current_value,
                    'target_value': target_value,
                    'notional': diff
                })
            else:
                # Need to sell some shares (reduce position)
                trades.append({
                    'action': 'sell',
                    'symbol': symbol,
                    'current_value': current_value,
                    'target_value': target_value,
                    'notional': abs(diff)
                })

    return trades


def print_trade_summary(trades, account_value):
    """Print summary of trades to execute."""
    print("\n" + "="*80)
    print("📋 TRADE SUMMARY")
    print("="*80)
    print(f"\nAccount Value: ${account_value:,.2f}")
    print(f"Total Trades: {len(trades)}")

    sells = [t for t in trades if t['action'] == 'sell']
    buys = [t for t in trades if t['action'] == 'buy']

    if sells:
        print(f"\n🔴 SELLS ({len(sells)}):")
        print("-" * 80)
        for trade in sells:
            if trade['notional'] is None:
                print(f"  {trade['symbol']:<6} | Close entire position (${trade['current_value']:,.2f})")
            else:
                print(f"  {trade['symbol']:<6} | Reduce by ${trade['notional']:,.2f} "
                      f"(${trade['current_value']:,.2f} → ${trade['target_value']:,.2f})")

    if buys:
        print(f"\n🟢 BUYS ({len(buys)}):")
        print("-" * 80)
        for trade in buys:
            if trade['current_value'] > 0:
                print(f"  {trade['symbol']:<6} | Add ${trade['notional']:,.2f} "
                      f"(${trade['current_value']:,.2f} → ${trade['target_value']:,.2f})")
            else:
                print(f"  {trade['symbol']:<6} | Buy ${trade['notional']:,.2f} (new position)")

    print("\n" + "="*80)


def execute_trades(trader, trades, dry_run=False):
    """
    Execute trades.

    Args:
        trader: AlpacaTrader instance
        trades: List of trades to execute
        dry_run: If True, preview only (don't execute)

    Returns:
        List of results
    """
    if dry_run:
        print("\n🔍 DRY RUN MODE - No trades will be executed")
        return []

    print("\n⚡ EXECUTING TRADES...")
    print("-" * 80)

    results = []

    # First, cancel any open orders
    try:
        trader.cancel_all_orders()
        print("✅ Cancelled all open orders")
    except Exception as e:
        print(f"⚠️  Warning: Could not cancel orders: {e}")

    # Execute sells first
    sells = [t for t in trades if t['action'] == 'sell']
    for trade in sells:
        try:
            if trade['notional'] is None:
                # Close entire position
                result = trader.close_position(trade['symbol'])
                print(f"✅ SELL {trade['symbol']}: Closed entire position")
            else:
                # Sell partial (by notional value)
                result = trader.place_order(
                    symbol=trade['symbol'],
                    notional=trade['notional'],
                    side='sell',
                    order_type='market',
                    time_in_force='day'
                )
                print(f"✅ SELL {trade['symbol']}: ${trade['notional']:,.2f}")

            results.append({
                'action': 'sell',
                'symbol': trade['symbol'],
                'status': 'success',
                'result': result
            })

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"❌ SELL {trade['symbol']}: Failed - {e}")
            results.append({
                'action': 'sell',
                'symbol': trade['symbol'],
                'status': 'failed',
                'error': str(e)
            })

    # Wait a bit for sells to settle
    if sells:
        print("\n⏳ Waiting 2 seconds for sells to settle...")
        time.sleep(2)

    # Execute buys
    buys = [t for t in trades if t['action'] == 'buy']
    for trade in buys:
        try:
            result = trader.place_order(
                symbol=trade['symbol'],
                notional=trade['notional'],
                side='buy',
                order_type='market',
                time_in_force='day'
            )
            print(f"✅ BUY {trade['symbol']}: ${trade['notional']:,.2f}")

            results.append({
                'action': 'buy',
                'symbol': trade['symbol'],
                'status': 'success',
                'result': result
            })

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"❌ BUY {trade['symbol']}: Failed - {e}")
            results.append({
                'action': 'buy',
                'symbol': trade['symbol'],
                'status': 'failed',
                'error': str(e)
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Execute trades automatically using Alpaca',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview trades (dry run)
  python3 alpaca_trader.py --portfolio todays_portfolio.csv --dry-run

  # Execute on paper trading
  python3 alpaca_trader.py --portfolio todays_portfolio.csv

  # Execute on LIVE account
  python3 alpaca_trader.py --portfolio todays_portfolio.csv --live
        """
    )

    parser.add_argument('--portfolio', required=True,
                        help='Path to portfolio CSV file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview trades without executing')
    parser.add_argument('--live', action='store_true',
                        help='Use LIVE trading (default: paper trading)')
    parser.add_argument('--auto-confirm', action='store_true',
                        help='Skip confirmation prompt')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🤖 ALPACA AUTO-TRADER")
    print("="*80)

    # Determine mode
    if args.live:
        mode = "LIVE TRADING 💰"
        paper = False
    else:
        mode = "PAPER TRADING 📄"
        paper = True

    print(f"\nMode: {mode}")
    print(f"Portfolio: {args.portfolio}")

    # Load portfolio
    print("\n📊 Loading portfolio...")
    try:
        portfolio = load_portfolio(args.portfolio)
        print(f"✅ Loaded {len(portfolio)} positions")
    except Exception as e:
        print(f"❌ Error loading portfolio: {e}")
        sys.exit(1)

    # Initialize trader
    print("\n🔌 Connecting to Alpaca...")
    try:
        trader = AlpacaTrader(paper=paper)
        account = trader.get_account()
        print(f"✅ Connected to Alpaca")
        print(f"   Account: {account['account_number']}")
        print(f"   Cash: ${float(account['cash']):,.2f}")
        print(f"   Portfolio Value: ${float(account['portfolio_value']):,.2f}")
        print(f"   Buying Power: ${float(account['buying_power']):,.2f}")
    except Exception as e:
        print(f"❌ Error connecting to Alpaca: {e}")
        sys.exit(1)

    # Get current positions
    print("\n📦 Getting current positions...")
    try:
        positions = trader.get_positions()
        print(f"✅ Currently holding {len(positions)} positions")
    except Exception as e:
        print(f"❌ Error getting positions: {e}")
        sys.exit(1)

    # Generate trade list
    print("\n🔄 Generating trade list...")
    trades = generate_trade_list(positions, portfolio)

    # Print summary
    print_trade_summary(trades, float(account['portfolio_value']))

    if not trades:
        print("\n✅ Portfolio is already aligned - no trades needed!")
        sys.exit(0)

    # Confirmation
    if not args.dry_run and not args.auto_confirm:
        print("\n" + "="*80)
        if args.live:
            print("⚠️  WARNING: This will execute REAL trades with REAL money!")
        print("\nDo you want to proceed? (yes/no): ", end='')
        response = input().strip().lower()
        if response not in ['yes', 'y']:
            print("\n❌ Aborted by user")
            sys.exit(0)

    # Execute trades
    results = execute_trades(trader, trades, dry_run=args.dry_run)

    # Summary
    if not args.dry_run:
        successes = sum(1 for r in results if r['status'] == 'success')
        failures = sum(1 for r in results if r['status'] == 'failed')

        print("\n" + "="*80)
        print("✅ EXECUTION COMPLETE")
        print("="*80)
        print(f"\nSuccessful: {successes}/{len(trades)}")
        print(f"Failed: {failures}/{len(trades)}")

        if failures > 0:
            print("\n❌ Failed trades:")
            for r in results:
                if r['status'] == 'failed':
                    print(f"  {r['symbol']}: {r['error']}")

        # Get updated positions
        print("\n📦 Updated positions:")
        try:
            new_positions = trader.get_positions()
            print(f"   Now holding {len(new_positions)} positions")
        except Exception as e:
            print(f"   Could not fetch updated positions: {e}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
