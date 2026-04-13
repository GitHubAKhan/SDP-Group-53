#!/usr/bin/env python3
"""
Volatility Scaler

Implements the Barroso & Santa-Clara (2015) volatility scaling technique.
This nearly doubles the Sharpe ratio and cuts drawdowns roughly in half
compared to raw momentum.

Scaling Factor = sigma_target / sigma_realized

When vol is high, we reduce exposure. When vol is low, we increase it.
"""

import numpy as np
import pandas as pd


class VolatilityScaler:
    """
    Volatility scaling for momentum portfolios.

    Per Barroso & Santa-Clara (2015):
        - Target vol: 12% annualized (standard from literature)
        - Realized vol: trailing 126-day (6-month) realized vol
        - Scaling factor capped at [0.25x, 2.0x]
    """

    def __init__(self, vol_target=0.12, vol_lookback=126,
                 scale_min=0.25, scale_max=2.0):
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.scale_min = scale_min
        self.scale_max = scale_max

    def compute_scaling_factor(self, portfolio_returns, as_of_date=None):
        """
        Compute the volatility scaling factor.

        Args:
            portfolio_returns: Series of daily portfolio returns (index=date)
            as_of_date: Date to compute scaling as of

        Returns:
            float: scaling factor in [scale_min, scale_max]
        """
        if portfolio_returns is None or len(portfolio_returns) < 21:
            return 1.0

        if as_of_date is not None:
            rets = portfolio_returns.loc[:as_of_date]
        else:
            rets = portfolio_returns

        # Use trailing lookback window
        recent = rets.tail(self.vol_lookback)
        if len(recent) < 21:
            return 1.0

        # Annualized realized vol
        realized_vol = recent.std() * np.sqrt(252)

        if realized_vol < 0.001:
            return 1.0

        # Scaling factor
        factor = self.vol_target / realized_vol
        factor = np.clip(factor, self.scale_min, self.scale_max)

        return float(factor)

    def scale_weights(self, weights, portfolio_returns, as_of_date=None):
        """
        Scale portfolio weights by the volatility scaling factor.

        Args:
            weights: Series of portfolio weights
            portfolio_returns: Series of daily portfolio returns
            as_of_date: Date to compute scaling as of

        Returns:
            tuple: (scaled_weights, scaling_factor)
        """
        factor = self.compute_scaling_factor(portfolio_returns, as_of_date)
        scaled = weights * factor
        return scaled, factor

    def compute_scaling_history(self, portfolio_returns):
        """
        Compute the full history of scaling factors.

        Returns:
            Series indexed by date with scaling factor values
        """
        factors = pd.Series(index=portfolio_returns.index, dtype=float)

        for i in range(len(portfolio_returns)):
            date = portfolio_returns.index[i]
            rets_to_date = portfolio_returns.iloc[:i + 1]
            factors.loc[date] = self.compute_scaling_factor(rets_to_date)

        return factors
