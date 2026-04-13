#!/usr/bin/env python3
"""
Rebalancer

Takes target portfolio weights and generates the trade list to
transition from current positions to the target allocation.
Then executes the trades via the Alpaca broker.

Usage:
    from execution.rebalancer import Rebalancer
    rebalancer = Rebalancer(broker)
    rebalancer.rebalance(target_weights, dry_run=True)
"""

import time
import pandas as pd
from execution.alpaca_broker import AlpacaBroker
from data.universe import to_alpaca_ticker


class Rebalancer:
    """
    Handles portfolio rebalancing against Alpaca.
    """

    def __init__(self, broker=None, rate_limit_delay=0.5):
        if broker is None:
            broker = AlpacaBroker(paper=True)
        self.broker = broker
        self.rate_limit_delay = rate_limit_delay

    def get_current_positions(self):
        """Get current positions as {symbol: market_value} dict."""
        positions = self.broker.get_positions()
        return {
            pos["symbol"]: float(pos["market_value"])
            for pos in positions
        }

    def generate_trades(self, target_weights, total_capital=None):
        """
        Generate trade list to move from current positions to target weights.

        Args:
            target_weights: Series indexed by ticker (Alpaca format) with dollar amounts
                           OR weights (if total_capital is provided)
            total_capital: If provided, multiply weights by this to get dollar amounts

        Returns:
            List of trade dicts with keys: action, symbol, notional, current_value, target_value
        """
        current = self.get_current_positions()

        # Convert weights to dollar amounts if needed
        if total_capital is not None:
            target_dollars = {
                ticker: weight * total_capital
                for ticker, weight in target_weights.items()
            }
        else:
            target_dollars = dict(target_weights)

        # Clean tickers (remove Bloomberg suffixes)
        clean_targets = {}
        for ticker, value in target_dollars.items():
            clean_ticker = to_alpaca_ticker(ticker)
            clean_targets[clean_ticker] = value

        trades = []

        # Sell positions not in target
        for symbol, current_value in current.items():
            if symbol not in clean_targets:
                trades.append({
                    "action": "sell",
                    "symbol": symbol,
                    "current_value": current_value,
                    "target_value": 0,
                    "notional": None,  # Close entire position
                })

        # Buy/adjust positions in target
        for symbol, target_value in clean_targets.items():
            current_value = current.get(symbol, 0)
            diff = target_value - current_value

            if abs(diff) > 10:  # Only trade if difference > $10
                if diff > 0:
                    trades.append({
                        "action": "buy",
                        "symbol": symbol,
                        "current_value": current_value,
                        "target_value": target_value,
                        "notional": diff,
                    })
                else:
                    trades.append({
                        "action": "sell",
                        "symbol": symbol,
                        "current_value": current_value,
                        "target_value": target_value,
                        "notional": abs(diff),
                    })

        return trades

    def execute_trades(self, trades, dry_run=False):
        """
        Execute a list of trades.

        Args:
            trades: List of trade dicts from generate_trades()
            dry_run: If True, just print what would happen

        Returns:
            List of result dicts
        """
        if dry_run:
            self._print_trade_plan(trades)
            return []

        results = []

        # Cancel any open orders first
        try:
            self.broker.cancel_all_orders()
        except Exception as e:
            print(f"  Warning: Could not cancel orders: {e}")

        # Execute sells first (free up capital)
        sells = [t for t in trades if t["action"] == "sell"]
        for trade in sells:
            try:
                if trade["notional"] is None:
                    result = self.broker.close_position(trade["symbol"])
                    print(f"  SELL {trade['symbol']}: closed entire position")
                else:
                    result = self.broker.place_order(
                        symbol=trade["symbol"],
                        notional=trade["notional"],
                        side="sell",
                    )
                    print(f"  SELL {trade['symbol']}: ${trade['notional']:,.2f}")

                results.append({"action": "sell", "symbol": trade["symbol"], "status": "success"})
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                print(f"  SELL {trade['symbol']}: FAILED - {e}")
                results.append({"action": "sell", "symbol": trade["symbol"], "status": "failed", "error": str(e)})

        # Wait for sells to settle
        if sells:
            time.sleep(2)

        # Execute buys
        buys = [t for t in trades if t["action"] == "buy"]
        for trade in buys:
            try:
                result = self.broker.place_order(
                    symbol=trade["symbol"],
                    notional=trade["notional"],
                    side="buy",
                )
                print(f"  BUY  {trade['symbol']}: ${trade['notional']:,.2f}")
                results.append({"action": "buy", "symbol": trade["symbol"], "status": "success"})
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                print(f"  BUY  {trade['symbol']}: FAILED - {e}")
                results.append({"action": "buy", "symbol": trade["symbol"], "status": "failed", "error": str(e)})

        return results

    def rebalance(self, target_weights, total_capital=None, dry_run=False):
        """
        Full rebalance: generate trades and execute them.

        Args:
            target_weights: Target portfolio weights/dollars
            total_capital: Total capital (if weights are fractions)
            dry_run: Preview only, don't execute
        """
        print("Generating trade list...")
        trades = self.generate_trades(target_weights, total_capital)

        if not trades:
            print("Portfolio is already aligned - no trades needed.")
            return []

        print(f"Generated {len(trades)} trades")

        if dry_run:
            self._print_trade_plan(trades)
            return []

        print("Executing trades...")
        results = self.execute_trades(trades)

        successes = sum(1 for r in results if r["status"] == "success")
        failures = sum(1 for r in results if r["status"] == "failed")
        print(f"\nExecution complete: {successes} succeeded, {failures} failed")

        return results

    def _print_trade_plan(self, trades):
        """Print a summary of planned trades."""
        sells = [t for t in trades if t["action"] == "sell"]
        buys = [t for t in trades if t["action"] == "buy"]

        print(f"\n{'='*60}")
        print(f"TRADE PLAN (DRY RUN - No trades will execute)")
        print(f"{'='*60}")

        if sells:
            print(f"\nSELLS ({len(sells)}):")
            for t in sells:
                if t["notional"] is None:
                    print(f"  {t['symbol']:<6} Close entire position (${t['current_value']:,.2f})")
                else:
                    print(f"  {t['symbol']:<6} Reduce ${t['notional']:,.2f}")

        if buys:
            print(f"\nBUYS ({len(buys)}):")
            for t in buys:
                print(f"  {t['symbol']:<6} Buy ${t['notional']:,.2f}")

        print(f"{'='*60}")
