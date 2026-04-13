#!/usr/bin/env python3
"""
Master Data Pipeline

Orchestrates data loading from all sources: Bloomberg (existing parquet),
FRED (macro), Yahoo Finance (gold), and Alpaca (backup).

Provides a unified interface for the backtest and signal engines.

Usage:
    from data.data_pipeline import DataPipeline
    pipeline = DataPipeline()
    prices = pipeline.load_prices()
    macro = pipeline.load_macro_data()
    gold = pipeline.load_gold_prices()
"""

import os
import io
import pandas as pd
import numpy as np
import yaml


def _get_s3_client():
    import boto3
    return boto3.client("s3")


def _s3_download_to_tmp(s3, bucket, s3_key, local_path):
    """Download a file from S3 to a local /tmp path. Returns local_path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, s3_key, local_path)
    return local_path


def _s3_key_exists(s3, bucket, s3_key):
    """Return True if the S3 key exists."""
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        return True
    except Exception:
        return False


def _upload_csv_to_s3(s3, bucket, s3_key, df, index=True):
    """Upload a DataFrame as CSV to S3."""
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=buf.getvalue(), ContentType="text/csv")


class DataPipeline:
    """
    Master ETL pipeline that loads and cleans data from all sources.

    When the S3_BUCKET environment variable is set (i.e., running in AWS Lambda),
    all data is read from / written to S3. Files are temporarily cached in /tmp
    during the Lambda invocation.
    """

    def __init__(self, settings_path=None, data_dir=None):
        if settings_path is None:
            settings_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")

        if os.path.exists(settings_path):
            with open(settings_path, "r") as f:
                self.settings = yaml.safe_load(f)
        else:
            self.settings = {}

        data_cfg = self.settings.get("data", {})
        self.data_dir = data_dir or data_cfg.get("data_dir", "data/data")
        self.cache_dir = data_cfg.get("cache_dir", "data/cache")

        # S3 mode: active when running in AWS Lambda
        self.s3_bucket = os.getenv("S3_BUCKET")
        if self.s3_bucket:
            self._s3 = _get_s3_client()
            # Use /tmp for all local file operations in Lambda
            self.data_dir = "/tmp/sdp/data"
            self.cache_dir = "/tmp/sdp/cache"
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"  [DataPipeline] S3 mode active. Bucket: {self.s3_bucket}")

    def _sync_from_s3(self, s3_prefix, local_dir):
        """
        Download all objects under an S3 prefix to a local directory.
        Only used when running in Lambda (self.s3_bucket is set).
        """
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative = key[len(s3_prefix):].lstrip("/")
                if not relative:
                    continue
                local_path = os.path.join(local_dir, relative)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                if not os.path.exists(local_path):
                    self._s3.download_file(self.s3_bucket, key, local_path)

    def _download_file_from_s3(self, s3_key, local_path):
        """Download a single file from S3 to local_path if not already present."""
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if _s3_key_exists(self._s3, self.s3_bucket, s3_key):
                self._s3.download_file(self.s3_bucket, s3_key, local_path)
                return True
        return os.path.exists(local_path)

    def load_prices(self, start_date=None, end_date=None):
        """
        Load equity prices from existing Bloomberg parquet data, with a
        yfinance top-up appended if data/cache/price_topup.parquet exists.

        Returns:
            DataFrame with columns: date, ticker, px_last, tri_gross, volume
        """
        parquet_path = os.path.join(self.data_dir, "prices_parquet")

        # In S3 mode, sync the parquet folder from S3 first
        if self.s3_bucket and not os.path.isdir(parquet_path):
            print("  [DataPipeline] Downloading price data from S3...")
            self._sync_from_s3("data/prices_parquet/", parquet_path)

        if os.path.isdir(parquet_path):
            df = pd.read_parquet(parquet_path)
        else:
            # Fallback: look for individual parquet files
            files = [f for f in os.listdir(self.data_dir) if f.endswith(".parquet")]
            if not files:
                raise FileNotFoundError(
                    f"No price data found in {self.data_dir}.\n"
                    "Run data/12_1trade_datascraper.py with Bloomberg to generate price data."
                )
            df = pd.read_parquet(os.path.join(self.data_dir, files[0]))

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # Drop the year partition column if present
        if "year" in df.columns:
            df = df.drop(columns=["year"])

        # Append yfinance top-up data if available (download from S3 if needed)
        topup_path = os.path.join(self.cache_dir, "price_topup.parquet")
        if self.s3_bucket:
            self._download_file_from_s3("cache/price_topup.parquet", topup_path)
        if os.path.exists(topup_path):
            df_topup = pd.read_parquet(topup_path)
            df_topup.columns = [c.lower() for c in df_topup.columns]
            df_topup["date"] = pd.to_datetime(df_topup["date"]).dt.tz_localize(None)
            bbg_last = df["date"].max()
            df_topup = df_topup[df_topup["date"] > bbg_last]
            if not df_topup.empty:
                df = pd.concat([df, df_topup], ignore_index=True)
                print(f"  Appended top-up: {df_topup['date'].min().date()} -> {df_topup['date'].max().date()} "
                      f"({df_topup['ticker'].nunique()} tickers)")

        # Filter date range
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        # Ensure tri_gross exists (fallback to px_last)
        if "tri_gross" not in df.columns or df["tri_gross"].isna().all():
            df["tri_gross"] = df["px_last"]

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    def load_sectors(self):
        """Load sector mappings."""
        path = os.path.join(self.data_dir, "sectors.csv")
        if self.s3_bucket:
            self._download_file_from_s3("data/sectors.csv", path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sector file not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df

    def load_constituents(self):
        """Load point-in-time constituent lists."""
        path = os.path.join(self.data_dir, "constituents_long.csv")
        if self.s3_bucket:
            self._download_file_from_s3("data/constituents_long.csv", path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Constituent file not found: {path}")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df

    def load_macro_data(self):
        """
        Load macro data from cached FRED pulls.
        Returns None if no macro data is available.
        """
        path = os.path.join(self.cache_dir, "macro_data.csv")
        if self.s3_bucket:
            self._download_file_from_s3("cache/macro_data.csv", path)
        if not os.path.exists(path):
            print(f"  No cached macro data at {path}. Run fred_fetcher.py first.")
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    def load_gold_prices(self):
        """
        Load gold (GLD) price data from cache.
        Returns None if no gold data is available.
        """
        gold_cfg = self.settings.get("gold", {})
        ticker = gold_cfg.get("ticker", "GLD").lower()
        path = os.path.join(self.cache_dir, f"{ticker}_prices.csv")
        if self.s3_bucket:
            self._download_file_from_s3(f"cache/{ticker}_prices.csv", path)
        if not os.path.exists(path):
            print(f"  No cached gold data at {path}. Run gold_fetcher.py first.")
            return None
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df

    def load_index_data(self, tickers=None):
        """
        Load index/ETF data from Bloomberg parquet files.
        """
        indices_dir = os.path.join(self.data_dir, "indices")
        if not os.path.isdir(indices_dir):
            return None

        dfs = []
        for f in sorted(os.listdir(indices_dir)):
            if f.endswith(".parquet"):
                dfs.append(pd.read_parquet(os.path.join(indices_dir, f)))

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        if tickers:
            df = df[df["ticker"].isin(tickers)]

        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def compute_liquidity_filter(self, prices, min_adv=None, lookback=20):
        """
        Filter stocks by minimum average daily volume ($).

        Args:
            prices: DataFrame with columns date, ticker, px_last, volume
            min_adv: Minimum average daily volume in dollars
            lookback: Number of trailing days for ADV calculation

        Returns:
            Set of tickers that pass the liquidity filter
        """
        if min_adv is None:
            risk_cfg = self.settings.get("risk", {})
            min_adv = risk_cfg.get("liquidity_min_adv", 5_000_000)

        prices = prices.copy()
        prices["dollar_volume"] = prices["px_last"] * prices["volume"]

        # Compute trailing ADV per ticker
        adv = (
            prices.groupby("ticker")["dollar_volume"]
            .rolling(lookback, min_periods=lookback)
            .mean()
            .reset_index(level=0, drop=True)
        )
        prices["adv"] = adv

        # Get latest ADV per ticker
        latest_adv = prices.groupby("ticker")["adv"].last()
        liquid_tickers = set(latest_adv[latest_adv >= min_adv].index)

        return liquid_tickers

    def get_full_dataset(self, start_date=None, end_date=None):
        """
        Load all data sources and return as a dictionary.
        Useful for the backtest engine.
        """
        data = {
            "prices": self.load_prices(start_date, end_date),
            "sectors": self.load_sectors(),
            "constituents": self.load_constituents(),
            "macro": self.load_macro_data(),
            "gold": self.load_gold_prices(),
        }
        return data
