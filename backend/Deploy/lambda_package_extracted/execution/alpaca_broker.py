#!/usr/bin/env python3
"""
Alpaca Broker Interface

Wraps Alpaca API for paper and live trading.
Handles order placement, position management, and account queries.
"""

import os
import time
import requests


class AlpacaBroker:
    """
    Alpaca trading broker interface.
    Supports paper trading and live trading via REST API.
    """

    def __init__(self, api_key=None, api_secret=None, paper=True):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")

        # Try loading from config/credentials.env
        if not self.api_key or not self.api_secret:
            env_path = os.path.join(os.path.dirname(__file__), "..", "config", "credentials.env")
            if os.path.exists(env_path):
                try:
                    from dotenv import load_dotenv
                    load_dotenv(env_path)
                    self.api_key = self.api_key or os.getenv("ALPACA_API_KEY")
                    self.api_secret = self.api_secret or os.getenv("ALPACA_API_SECRET")
                except ImportError:
                    # Parse manually if python-dotenv not installed
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("#") or "=" not in line:
                                continue
                            key, val = line.split("=", 1)
                            key, val = key.strip(), val.strip()
                            if key == "ALPACA_API_KEY" and not self.api_key:
                                self.api_key = val
                            elif key == "ALPACA_API_SECRET" and not self.api_secret:
                                self.api_secret = val

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials not found.\n"
                "Option 1: Set environment variables:\n"
                "  export ALPACA_API_KEY='your_key'\n"
                "  export ALPACA_API_SECRET='your_secret'\n"
                "Option 2: Copy config/credentials.env.template to config/credentials.env\n"
                "  and fill in your keys."
            )

        self.paper = paper
        if paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    def get_account(self):
        """Get account information."""
        resp = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def get_positions(self):
        """Get all current positions."""
        resp = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def get_position(self, symbol):
        """Get position for a specific symbol. Returns None if no position."""
        try:
            resp = requests.get(
                f"{self.base_url}/v2/positions/{symbol}", headers=self.headers
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def place_order(self, symbol, qty=None, notional=None, side="buy",
                    order_type="market", time_in_force="day"):
        """
        Place an order. Use qty for share count or notional for dollar amount.
        """
        order = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }

        if qty is not None:
            order["qty"] = str(int(qty))
        elif notional is not None:
            order["notional"] = str(round(notional, 2))
        else:
            raise ValueError("Must specify either qty or notional")

        resp = requests.post(
            f"{self.base_url}/v2/orders", headers=self.headers, json=order
        )

        if resp.status_code not in (200, 201):
            try:
                detail = resp.json()
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code}: {detail.get('message', 'Unknown error')}"
                )
            except ValueError:
                resp.raise_for_status()

        return resp.json()

    def close_position(self, symbol):
        """Close entire position for a symbol."""
        resp = requests.delete(
            f"{self.base_url}/v2/positions/{symbol}", headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()

    def close_all_positions(self):
        """Close all open positions."""
        resp = requests.delete(
            f"{self.base_url}/v2/positions", headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_all_orders(self):
        """Cancel all open orders."""
        resp = requests.delete(
            f"{self.base_url}/v2/orders", headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()

    def get_orders(self, status="all"):
        """Get orders by status (all, open, closed)."""
        resp = requests.get(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            params={"status": status},
        )
        resp.raise_for_status()
        return resp.json()
