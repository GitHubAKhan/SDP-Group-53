#!/usr/bin/env python3
"""
Risk Manager

Implements all risk controls from the buildspec:
    - Volatility scaling (Barroso & Santa-Clara 2015)
    - Sector concentration caps (30%)
    - Single-name caps (5%)
    - Drawdown circuit breaker (-15% trailing 30d)
    - Liquidity filter ($5M ADV minimum)
    - Turnover monitoring (flag > 40%)
    - Macro regime overlay (addendum)
"""

import numpy as np
import pandas as pd
from portfolio.vol_scaler import VolatilityScaler


class RiskManager:
    """
    Unified risk management for the momentum strategy.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}

        risk_cfg = settings.get("risk", {})
        self.vol_target = risk_cfg.get("vol_target", 0.12)
        self.vol_lookback = risk_cfg.get("vol_lookback", 126)
        self.vol_scale_min = risk_cfg.get("vol_scale_min", 0.25)
        self.vol_scale_max = risk_cfg.get("vol_scale_max", 2.0)
        self.sector_cap = risk_cfg.get("sector_cap", 0.30)
        self.max_single_name = risk_cfg.get("max_single_name", 0.05)
        self.drawdown_breaker = risk_cfg.get("drawdown_breaker", -0.15)
        self.liquidity_min_adv = risk_cfg.get("liquidity_min_adv", 5_000_000)
        self.turnover_flag = risk_cfg.get("turnover_flag", 0.40)

        # Drawdown thresholds
        self.dd_threshold_1 = risk_cfg.get("dd_threshold_1", 0.10)
        self.dd_threshold_2 = risk_cfg.get("dd_threshold_2", 0.15)
        self.dd_threshold_3 = risk_cfg.get("dd_threshold_3", 0.25)

        # Vol scaler
        self.vol_scaler = VolatilityScaler(
            vol_target=self.vol_target,
            vol_lookback=self.vol_lookback,
            scale_min=self.vol_scale_min,
            scale_max=self.vol_scale_max,
        )

    def apply_all_risk_controls(self, weights, portfolio_returns, equity_curve,
                                 sectors_map=None, macro_regime=None, as_of_date=None):
        """
        Apply all risk controls to portfolio weights.

        Args:
            weights: Series of raw portfolio weights
            portfolio_returns: Series of daily portfolio returns
            equity_curve: Series of cumulative equity curve
            sectors_map: Series mapping ticker -> sector
            macro_regime: dict from MacroRegimeClassifier.classify()
            as_of_date: Current date

        Returns:
            dict with adjusted weights and risk report
        """
        report = {}

        # 1. Volatility scaling
        vol_factor = self.vol_scaler.compute_scaling_factor(portfolio_returns, as_of_date)
        adjusted_vol_target = self.vol_target

        # If macro regime is RISK-OFF, halve the vol target
        if macro_regime and macro_regime.get("regime") == "RISK-OFF":
            vol_multiplier = macro_regime.get("vol_target_multiplier", 0.5)
            adjusted_vol_target *= vol_multiplier
            # Recompute factor with adjusted target
            if portfolio_returns is not None and len(portfolio_returns) >= 21:
                rets = portfolio_returns.loc[:as_of_date] if as_of_date else portfolio_returns
                recent = rets.tail(self.vol_lookback)
                realized_vol = recent.std() * np.sqrt(252)
                if realized_vol > 0.001:
                    vol_factor = np.clip(
                        adjusted_vol_target / realized_vol,
                        self.vol_scale_min,
                        self.vol_scale_max,
                    )
            report["macro_regime"] = macro_regime["regime"]
            report["macro_signals"] = macro_regime.get("risk_off_count", 0)

        report["vol_scaling_factor"] = float(vol_factor)
        report["adjusted_vol_target"] = adjusted_vol_target

        # 2. Drawdown circuit breaker
        dd_factor = self._compute_drawdown_factor(equity_curve, as_of_date)
        report["drawdown_factor"] = float(dd_factor)

        # 3. Combined scaling
        combined_factor = vol_factor * dd_factor
        scaled_weights = weights * combined_factor
        report["combined_factor"] = float(combined_factor)

        # 4. Sector caps (applied per leg for long/short)
        if sectors_map is not None and not scaled_weights.empty:
            scaled_weights = self._apply_sector_caps(scaled_weights, sectors_map)

        # 5. Single-name caps
        scaled_weights = scaled_weights.clip(upper=self.max_single_name * combined_factor)

        report["total_exposure"] = float(scaled_weights.abs().sum())
        report["n_positions"] = int((scaled_weights != 0).sum())

        return {
            "weights": scaled_weights,
            "report": report,
        }

    def _compute_drawdown_factor(self, equity_curve, as_of_date=None):
        """Compute exposure reduction based on current drawdown."""
        if equity_curve is None or equity_curve.empty:
            return 1.0

        if as_of_date is not None:
            eq = equity_curve.loc[:as_of_date]
        else:
            eq = equity_curve

        if len(eq) == 0:
            return 1.0

        current = eq.iloc[-1]
        peak = eq.max()
        dd = (current / peak) - 1.0

        if dd <= -self.dd_threshold_3:
            return 0.0  # Exit completely
        elif dd <= -self.dd_threshold_2:
            return 0.5
        elif dd <= -self.dd_threshold_1:
            return 0.75
        else:
            return 1.0

    def _apply_sector_caps(self, weights, sectors_map):
        """Apply sector concentration cap."""
        w = weights.copy()
        sec = sectors_map.reindex(w.index).fillna("Unknown")

        for _ in range(10):
            sector_weights = w.abs().groupby(sec).sum()
            total = w.abs().sum()
            if total == 0:
                break
            sector_pcts = sector_weights / total
            over_limit = sector_pcts[sector_pcts > self.sector_cap]

            if over_limit.empty:
                break

            for sector, pct in over_limit.items():
                sector_tickers = sec[sec == sector].index
                scale = self.sector_cap / pct
                w.loc[sector_tickers] *= scale

        return w

    def check_turnover(self, turnover, gross_exposure):
        """Check if turnover exceeds the flag threshold."""
        if gross_exposure == 0:
            return False
        turnover_pct = turnover / gross_exposure
        if turnover_pct > self.turnover_flag:
            return True
        return False

    def filter_by_liquidity(self, tickers, prices_df, as_of_date=None, lookback=20):
        """
        Filter tickers by minimum average daily volume.

        Args:
            tickers: List/set of tickers to filter
            prices_df: DataFrame with date, ticker, px_last, volume
            as_of_date: Date to compute ADV as of
            lookback: Trailing days for ADV

        Returns:
            Set of liquid tickers
        """
        if as_of_date is not None:
            prices = prices_df[prices_df["date"] <= as_of_date]
        else:
            prices = prices_df

        prices = prices[prices["ticker"].isin(tickers)].copy()
        prices["dollar_vol"] = prices["px_last"] * prices["volume"]

        # Get trailing ADV per ticker
        adv = (
            prices.sort_values("date")
            .groupby("ticker")["dollar_vol"]
            .apply(lambda s: s.tail(lookback).mean())
        )

        liquid = set(adv[adv >= self.liquidity_min_adv].index)
        filtered_out = set(tickers) - liquid

        if filtered_out:
            print(f"  Liquidity filter removed {len(filtered_out)} stocks "
                  f"(ADV < ${self.liquidity_min_adv / 1e6:.0f}M)")

        return liquid
