#!/usr/bin/env python3
"""
Macro Regime Classifier

Simple rule-based regime detection using VIX, yield curve, and credit spreads.
Per the buildspec addendum: RISK-OFF if any TWO conditions are true.

This is intentionally simple and robust — no ML, no Markov switching.

Usage:
    from data.macro_regime import MacroRegimeClassifier
    regime = MacroRegimeClassifier(settings)
    is_risk_off = regime.classify(macro_df, portfolio_vol_series, as_of_date)
"""

import numpy as np
import pandas as pd
import yaml
import os


class MacroRegimeClassifier:
    """
    Rule-based macro regime classifier.

    RISK-OFF if any TWO of the following are true:
        1. VIX > 25 (elevated fear)
        2. Yield curve (T10Y2Y) < 0 (inverted)
        3. Credit spread (BAA10Y) > 3.0% (elevated stress)
        4. Portfolio trailing 21-day vol > 1.5x its 252-day average
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = self._load_settings()

        macro_cfg = settings.get("macro_regime", {})
        self.enabled = macro_cfg.get("enabled", True)
        self.vix_threshold = macro_cfg.get("vix_threshold", 25)
        self.yield_curve_threshold = macro_cfg.get("yield_curve_threshold", 0)
        self.credit_spread_threshold = macro_cfg.get("credit_spread_threshold", 3.0)
        self.vol_ratio_threshold = macro_cfg.get("vol_ratio_threshold", 1.5)
        self.risk_off_vol_target = macro_cfg.get("risk_off_vol_target", 0.06)

    def _load_settings(self):
        path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def classify(self, macro_df, portfolio_returns=None, as_of_date=None):
        """
        Classify current market regime.

        Args:
            macro_df: DataFrame with columns VIXCLS, T10Y2Y, BAA10Y (index=date)
            portfolio_returns: Series of daily portfolio returns (optional)
            as_of_date: Date to classify as of (defaults to latest)

        Returns:
            dict with keys: regime ('RISK-ON' or 'RISK-OFF'), signals, vol_target
        """
        if not self.enabled:
            return {
                "regime": "RISK-ON",
                "signals": {},
                "vol_target_multiplier": 1.0,
            }

        if as_of_date is not None:
            macro_df = macro_df.loc[:as_of_date]

        if macro_df.empty:
            return {
                "regime": "RISK-ON",
                "signals": {},
                "vol_target_multiplier": 1.0,
            }

        # Get latest values
        latest = macro_df.iloc[-1]
        signals = {}
        risk_off_count = 0

        # 1. VIX check
        if "VIXCLS" in macro_df.columns:
            vix_val = latest.get("VIXCLS", np.nan)
            if not np.isnan(vix_val):
                vix_elevated = vix_val > self.vix_threshold
                signals["vix"] = {"value": float(vix_val), "threshold": self.vix_threshold, "triggered": vix_elevated}
                if vix_elevated:
                    risk_off_count += 1

        # 2. Yield curve check
        if "T10Y2Y" in macro_df.columns:
            yc_val = latest.get("T10Y2Y", np.nan)
            if not np.isnan(yc_val):
                yc_inverted = yc_val < self.yield_curve_threshold
                signals["yield_curve"] = {"value": float(yc_val), "threshold": self.yield_curve_threshold, "triggered": yc_inverted}
                if yc_inverted:
                    risk_off_count += 1

        # 3. Credit spread check
        if "BAA10Y" in macro_df.columns:
            cs_val = latest.get("BAA10Y", np.nan)
            if not np.isnan(cs_val):
                cs_elevated = cs_val > self.credit_spread_threshold
                signals["credit_spread"] = {"value": float(cs_val), "threshold": self.credit_spread_threshold, "triggered": cs_elevated}
                if cs_elevated:
                    risk_off_count += 1

        # 4. Portfolio vol ratio check
        if portfolio_returns is not None and len(portfolio_returns) >= 252:
            if as_of_date is not None:
                port_rets = portfolio_returns.loc[:as_of_date]
            else:
                port_rets = portfolio_returns
            vol_21d = port_rets.tail(21).std() * np.sqrt(252)
            vol_252d = port_rets.tail(252).std() * np.sqrt(252)
            if vol_252d > 0.001:
                vol_ratio = vol_21d / vol_252d
                vol_elevated = vol_ratio > self.vol_ratio_threshold
                signals["portfolio_vol_ratio"] = {
                    "value": float(vol_ratio),
                    "threshold": self.vol_ratio_threshold,
                    "triggered": vol_elevated,
                    "vol_21d": float(vol_21d),
                    "vol_252d": float(vol_252d),
                }
                if vol_elevated:
                    risk_off_count += 1

        # RISK-OFF if 2+ conditions triggered
        is_risk_off = risk_off_count >= 2

        return {
            "regime": "RISK-OFF" if is_risk_off else "RISK-ON",
            "risk_off_count": risk_off_count,
            "signals": signals,
            "vol_target_multiplier": 0.5 if is_risk_off else 1.0,
        }

    def get_regime_history(self, macro_df, portfolio_returns=None):
        """
        Compute regime classification for every date in macro_df.
        Returns a DataFrame with regime labels.
        """
        results = []
        for date in macro_df.index:
            result = self.classify(macro_df, portfolio_returns, as_of_date=date)
            results.append({
                "date": date,
                "regime": result["regime"],
                "risk_off_count": result["risk_off_count"],
                "vol_target_multiplier": result["vol_target_multiplier"],
            })
        return pd.DataFrame(results).set_index("date")
