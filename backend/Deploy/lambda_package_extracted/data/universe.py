#!/usr/bin/env python3
"""
Universe Management

Manages S&P 500 constituent lists with point-in-time accuracy
to avoid survivorship bias. Loads from Bloomberg-pulled constituent data.
"""

import os
import pandas as pd
import numpy as np


# Exchange codes that map to 'US Equity' in Bloomberg format
US_EXCHANGE_CODES = {"UN", "UW", "UQ", "UA", "UR", "UT", "UV"}


def to_parse_keyable(member_code):
    """Convert Bloomberg 'TICKER CC' format to 'TICKER US Equity'."""
    parts = str(member_code).strip().split()
    if len(parts) == 1:
        return f"{parts[0]} US Equity"
    ticker, code = parts[0], parts[1]
    if code in US_EXCHANGE_CODES:
        return f"{ticker} US Equity"
    if len(code) == 2:
        return f"{ticker} {code} Equity"
    return f"{ticker} US Equity"


def to_alpaca_ticker(bbg_ticker):
    """Convert Bloomberg ticker 'AAPL US Equity' to Alpaca format 'AAPL'."""
    if isinstance(bbg_ticker, str):
        return bbg_ticker.replace(" US Equity", "").replace(" LN Equity", "").strip()
    return bbg_ticker


class UniverseManager:
    """
    Manages point-in-time S&P 500 constituent lists.

    Loads from data/constituents_long.csv which has columns:
        date, ticker_raw, in_spx (1 or 0)
    """

    def __init__(self, data_dir="data/data"):
        self.data_dir = data_dir
        self.constituents = None
        self._load()

    def _load(self):
        path = os.path.join(self.data_dir, "constituents_long.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Constituent file not found: {path}\n"
                "Run 12_1trade_datascraper.py with Bloomberg to generate it."
            )
        self.constituents = pd.read_csv(path)
        self.constituents["date"] = pd.to_datetime(self.constituents["date"]).dt.tz_localize(None)
        print(f"Loaded constituents: {len(self.constituents)} rows, "
              f"{self.constituents['date'].min().date()} to {self.constituents['date'].max().date()}")

    def get_members(self, as_of_date, as_bbg=True):
        """
        Get S&P 500 members as of a given date (point-in-time).

        Args:
            as_of_date: Date to look up constituents for
            as_bbg: If True, return Bloomberg format ('AAPL US Equity').
                    If False, return plain tickers ('AAPL').
        Returns:
            Set of ticker strings
        """
        as_of_date = pd.to_datetime(as_of_date)
        df = self.constituents[self.constituents["date"] <= as_of_date].sort_values(["ticker_raw", "date"])
        last = df.groupby("ticker_raw").tail(1)
        members_raw = last.loc[last["in_spx"] == 1, "ticker_raw"]

        if as_bbg:
            return set(to_parse_keyable(x) for x in members_raw)
        else:
            return set(to_alpaca_ticker(to_parse_keyable(x)) for x in members_raw)

    def get_all_historical_tickers(self, as_bbg=True):
        """Get all tickers that were ever in the S&P 500."""
        all_tickers = self.constituents.loc[
            self.constituents["in_spx"] == 1, "ticker_raw"
        ].unique()

        if as_bbg:
            return sorted(set(to_parse_keyable(x) for x in all_tickers))
        else:
            return sorted(set(to_alpaca_ticker(to_parse_keyable(x)) for x in all_tickers))
