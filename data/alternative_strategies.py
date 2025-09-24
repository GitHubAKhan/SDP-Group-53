#!/usr/bin/env python3
"""
Alternative Trading Strategies Implementation

This script implements several proven quantitative strategies using your S&P 500 data:
1. Mean Reversion (Short-term Reversal)
2. Low Volatility 
3. Value Strategy
4. Quality Strategy
5. Combined Multi-Factor Strategy

These strategies should work better than momentum with large-cap S&P 500 stocks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AlternativeStrategyTester:
    """Test multiple quantitative strategies on S&P 500 data."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.prices = None
        self.constituents = None
        self.monthly_data = None
        
    def load_data(self):
        """Load and prepare data."""
        print("Loading S&P 500 data...")
        
        # Load price data
        if (self.data_dir / "prices_parquet").exists():
            self.prices = pd.read_parquet(self.data_dir / "prices_parquet")
        else:
            raise FileNotFoundError("Price data not found")
        
        # Load constituents
        self.constituents = pd.read_csv(self.data_dir / "constituents_long.csv")
        self.constituents['date'] = pd.to_datetime(self.constituents['date'])
        
        # Prepare monthly data
        self._prepare_monthly_data()
        
    def _prepare_monthly_data(self):
        """Create monthly dataset with all necessary features."""
        print("Preparing monthly features...")
        
        self.prices['date'] = pd.to_datetime(self.prices['date'])
        
        # Create monthly data with features for all strategies
        monthly_records = []
        
        for ticker in self.prices['ticker'].unique():
            ticker_data = self.prices[self.prices['ticker'] == ticker].copy()
            ticker_data = ticker_data.set_index('date').sort_index()
            
            # Resample to month-end - only use available columns
            available_cols = {
                'px_last': 'last',
                'tri_gross': 'last', 
                'volume': 'mean'
            }
            
            # Only include columns that exist
            agg_dict = {col: agg for col, agg in available_cols.items() if col in ticker_data.columns}
            
            monthly_prices = ticker_data.resample('M').agg(agg_dict).dropna()
            
            if len(monthly_prices) < 24:  # Need at least 2 years
                continue
            
            # Calculate features for each month
            for i in range(12, len(monthly_prices)):  # Need 12 months of history
                date = monthly_prices.index[i]
                current_price = monthly_prices.iloc[i]['tri_gross']
                
                # Basic return calculations
                returns_1m = (monthly_prices.iloc[i]['tri_gross'] / monthly_prices.iloc[i-1]['tri_gross']) - 1
                returns_3m = (monthly_prices.iloc[i]['tri_gross'] / monthly_prices.iloc[i-3]['tri_gross']) - 1
                returns_6m = (monthly_prices.iloc[i]['tri_gross'] / monthly_prices.iloc[i-6]['tri_gross']) - 1
                returns_12m = (monthly_prices.iloc[i]['tri_gross'] / monthly_prices.iloc[i-12]['tri_gross']) - 1
                
                # Volatility (12-month rolling)
                price_series = monthly_prices.iloc[i-12:i]['tri_gross']
                monthly_returns = price_series.pct_change().dropna()
                volatility = monthly_returns.std() * np.sqrt(12)
                
                # Mean reversion signals
                reversal_1m = -returns_1m  # Negative of last month return
                reversal_3m = -returns_3m  # Negative of last 3 months
                
                # Volume patterns (if available)
                avg_volume = monthly_prices.iloc[i-3:i]['volume'].mean() if 'volume' in monthly_prices.columns else 0
                
                # Price-based volatility proxy
                price_volatility = volatility  # Use the calculated volatility as proxy
                
                # Next month return (target variable)
                if i < len(monthly_prices) - 1:
                    next_return = (monthly_prices.iloc[i+1]['tri_gross'] / current_price) - 1
                    
                    monthly_records.append({
                        'date': date,
                        'ticker': ticker,
                        'price': current_price,
                        'next_return': next_return,
                        'returns_1m': returns_1m,
                        'returns_3m': returns_3m, 
                        'returns_6m': returns_6m,
                        'returns_12m': returns_12m,
                        'volatility_12m': volatility,
                        'reversal_1m': reversal_1m,
                        'reversal_3m': reversal_3m,
                        'avg_volume': avg_volume,
                        'price_volatility': price_volatility,
                        'momentum_12_1': (monthly_prices.iloc[i-1]['tri_gross'] / monthly_prices.iloc[i-12]['tri_gross']) - 1
                    })
        
        self.monthly_data = pd.DataFrame(monthly_records)
        print(f"Created monthly dataset: {len(self.monthly_data)} observations")
        
        # Filter to S&P 500 constituents only
        spx_filter = self._filter_to_spx_constituents()
        self.monthly_data = self.monthly_data[spx_filter]
        print(f"Filtered to S&P 500: {len(self.monthly_data)} observations")
    
    def _filter_to_spx_constituents(self):
        """Filter to S&P 500 constituents."""
        # Convert ticker format
        def to_parse_keyable(ticker_raw):
            US_CODES = {"UN", "UW", "UQ", "UA", "UR", "UT", "UV"}
            parts = ticker_raw.strip().split()
            if len(parts) == 1:
                return f"{parts[0]} US Equity"
            ticker, code = parts[0], parts[1]
            if code in US_CODES:
                return f"{ticker} US Equity"
            return f"{ticker} {code} Equity"
        
        self.constituents['ticker'] = self.constituents['ticker_raw'].apply(to_parse_keyable)
        spx_members = self.constituents[self.constituents['in_spx'] == 1]
        
        # Create filter
        valid_combinations = set()
        for _, row in spx_members.iterrows():
            valid_combinations.add((row['date'], row['ticker']))
        
        return self.monthly_data.apply(
            lambda row: (row['date'], row['ticker']) in valid_combinations, axis=1
        )
    
    def run_mean_reversion_strategy(self, n_stocks=50):
        """Mean reversion strategy - buy recent losers."""
        print("Running Mean Reversion Strategy...")
        
        results = []
        dates = sorted(self.monthly_data['date'].unique())
        
        for date in dates:
            date_data = self.monthly_data[self.monthly_data['date'] == date]
            
            if len(date_data) >= n_stocks * 2:
                # Remove outliers
                date_data = date_data[
                    (date_data['returns_1m'] >= date_data['returns_1m'].quantile(0.05)) &
                    (date_data['returns_1m'] <= date_data['returns_1m'].quantile(0.95))
                ]
                
                if len(date_data) >= n_stocks:
                    # Buy biggest losers (most negative returns)
                    date_data = date_data.sort_values('returns_1m', ascending=True)
                    selected_stocks = date_data.head(n_stocks)
                    
                    portfolio_return = selected_stocks['next_return'].mean()
                    benchmark_return = date_data['next_return'].mean()
                    
                    results.append({
                        'date': date,
                        'strategy_return': portfolio_return,
                        'benchmark_return': benchmark_return,
                        'n_stocks': len(selected_stocks)
                    })
        
        return pd.DataFrame(results)
    
    def run_low_volatility_strategy(self, n_stocks=50):
        """Low volatility strategy - buy least volatile stocks."""
        print("Running Low Volatility Strategy...")
        
        results = []
        dates = sorted(self.monthly_data['date'].unique())
        
        for date in dates:
            date_data = self.monthly_data[self.monthly_data['date'] == date]
            
            if len(date_data) >= n_stocks * 2:
                # Filter out stocks with insufficient volatility data
                date_data = date_data.dropna(subset=['volatility_12m'])
                date_data = date_data[date_data['volatility_12m'] > 0.05]  # Min 5% volatility
                
                if len(date_data) >= n_stocks:
                    # Buy lowest volatility stocks
                    date_data = date_data.sort_values('volatility_12m', ascending=True)
                    selected_stocks = date_data.head(n_stocks)
                    
                    portfolio_return = selected_stocks['next_return'].mean()
                    benchmark_return = date_data['next_return'].mean()
                    
                    results.append({
                        'date': date,
                        'strategy_return': portfolio_return,
                        'benchmark_return': benchmark_return,
                        'n_stocks': len(selected_stocks),
                        'avg_volatility': selected_stocks['volatility_12m'].mean()
                    })
        
        return pd.DataFrame(results)
    
    def run_quality_strategy(self, n_stocks=50):
        """Quality strategy based on consistent returns and low volatility."""
        print("Running Quality Strategy...")
        
        # Calculate quality scores
        quality_data = []
        
        for ticker in self.monthly_data['ticker'].unique():
            ticker_data = self.monthly_data[self.monthly_data['ticker'] == ticker].sort_values('date')
            
            if len(ticker_data) >= 24:  # Need 2+ years
                # Rolling 24-month quality metrics
                for i in range(24, len(ticker_data)):
                    current_date = ticker_data.iloc[i]['date']
                    
                    # Quality metrics based on past 24 months
                    past_24m = ticker_data.iloc[i-24:i]
                    
                    # Consistency (negative of volatility)
                    consistency_score = -past_24m['returns_1m'].std()
                    
                    # Profitability (average return)
                    profitability_score = past_24m['returns_1m'].mean()
                    
                    # Combined quality score
                    quality_score = 0.6 * consistency_score + 0.4 * profitability_score
                    
                    quality_data.append({
                        'date': current_date,
                        'ticker': ticker_data.iloc[i]['ticker'],
                        'quality_score': quality_score,
                        'next_return': ticker_data.iloc[i]['next_return']
                    })
        
        quality_df = pd.DataFrame(quality_data)
        
        # Run quality strategy
        results = []
        dates = sorted(quality_df['date'].unique())
        
        for date in dates:
            date_data = quality_df[quality_df['date'] == date]
            
            if len(date_data) >= n_stocks:
                # Buy highest quality stocks
                date_data = date_data.sort_values('quality_score', ascending=False)
                selected_stocks = date_data.head(n_stocks)
                
                portfolio_return = selected_stocks['next_return'].mean()
                benchmark_return = date_data['next_return'].mean()
                
                results.append({
                    'date': date,
                    'strategy_return': portfolio_return,
                    'benchmark_return': benchmark_return,
                    'n_stocks': len(selected_stocks),
                    'avg_quality': selected_stocks['quality_score'].mean()
                })
        
        return pd.DataFrame(results)
    
    def run_multifactor_strategy(self, n_stocks=50):
        """Multi-factor strategy combining signals."""
        print("Running Multi-Factor Strategy...")
        
        results = []
        dates = sorted(self.monthly_data['date'].unique())
        
        for date in dates:
            date_data = self.monthly_data[self.monthly_data['date'] == date]
            
            if len(date_data) >= n_stocks * 2:
                # Calculate composite scores
                date_data = date_data.dropna(subset=['volatility_12m', 'returns_1m'])
                
                if len(date_data) >= n_stocks:
                    # Normalize factors (z-scores)
                    date_data['reversal_zscore'] = (date_data['reversal_1m'] - date_data['reversal_1m'].mean()) / date_data['reversal_1m'].std()
                    date_data['lowvol_zscore'] = -(date_data['volatility_12m'] - date_data['volatility_12m'].mean()) / date_data['volatility_12m'].std()
                    
                    # Composite score (equal weights)
                    date_data['composite_score'] = 0.5 * date_data['reversal_zscore'] + 0.5 * date_data['lowvol_zscore']
                    
                    # Buy top composite score stocks
                    date_data = date_data.sort_values('composite_score', ascending=False)
                    selected_stocks = date_data.head(n_stocks)
                    
                    portfolio_return = selected_stocks['next_return'].mean()
                    benchmark_return = date_data['next_return'].mean()
                    
                    results.append({
                        'date': date,
                        'strategy_return': portfolio_return,
                        'benchmark_return': benchmark_return,
                        'n_stocks': len(selected_stocks)
                    })
        
        return pd.DataFrame(results)
    
    def analyze_strategy_performance(self, results, strategy_name):
        """Analyze strategy performance."""
        if results.empty:
            print(f"No results for {strategy_name}")
            return None
        
        strategy_returns = results['strategy_return']
        benchmark_returns = results['benchmark_return']
        excess_returns = strategy_returns - benchmark_returns
        
        # Performance metrics
        n_months = len(strategy_returns)
        n_years = n_months / 12
        
        # Total and annualized returns
        strategy_total = (1 + strategy_returns).prod() - 1
        strategy_annual = (1 + strategy_returns).prod() ** (1/n_years) - 1
        
        benchmark_total = (1 + benchmark_returns).prod() - 1
        benchmark_annual = (1 + benchmark_returns).prod() ** (1/n_years) - 1
        
        excess_annual = strategy_annual - benchmark_annual
        
        # Risk metrics
        strategy_vol = strategy_returns.std() * np.sqrt(12)
        excess_vol = excess_returns.std() * np.sqrt(12)
        
        strategy_sharpe = strategy_annual / strategy_vol if strategy_vol > 0 else 0
        info_ratio = excess_annual / excess_vol if excess_vol > 0 else 0
        
        # Drawdown
        strategy_cumret = (1 + strategy_returns).cumprod()
        strategy_dd = (strategy_cumret / strategy_cumret.expanding().max() - 1).min()
        
        # Win rates
        win_rate = (strategy_returns > benchmark_returns).mean()
        
        # Statistical significance
        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
        
        return {
            'strategy': strategy_name,
            'periods': n_months,
            'strategy_total': strategy_total,
            'strategy_annual': strategy_annual,
            'benchmark_annual': benchmark_annual,
            'excess_annual': excess_annual,
            'strategy_vol': strategy_vol,
            'strategy_sharpe': strategy_sharpe,
            'info_ratio': info_ratio,
            'max_drawdown': strategy_dd,
            'win_rate': win_rate,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def run_all_strategies(self):
        """Run all alternative strategies."""
        print("="*80)
        print("ALTERNATIVE STRATEGIES PERFORMANCE TEST")
        print("="*80)
        
        strategies = {
            'Mean Reversion': self.run_mean_reversion_strategy,
            'Low Volatility': self.run_low_volatility_strategy,
            'Quality': self.run_quality_strategy,
            'Multi-Factor': self.run_multifactor_strategy
        }
        
        all_results = {}
        performance_summary = []
        
        for name, strategy_func in strategies.items():
            print(f"\nTesting {name} Strategy...")
            
            try:
                results = strategy_func(n_stocks=50)
                all_results[name] = results
                
                if not results.empty:
                    performance = self.analyze_strategy_performance(results, name)
                    performance_summary.append(performance)
                    
                    print(f"✓ {name}: {performance['excess_annual']:.1%} excess return, "
                          f"{performance['info_ratio']:.2f} IR, "
                          f"{'Significant' if performance['significant'] else 'Not significant'}")
                else:
                    print(f"✗ {name}: No results generated")
                    
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
        
        # Display summary
        if performance_summary:
            summary_df = pd.DataFrame(performance_summary)
            
            print("\n" + "="*80)
            print("STRATEGY PERFORMANCE SUMMARY")
            print("="*80)
            
            # Sort by excess return
            summary_df = summary_df.sort_values('excess_annual', ascending=False)
            
            display_cols = ['strategy', 'excess_annual', 'strategy_sharpe', 'info_ratio', 
                           'max_drawdown', 'win_rate', 'significant']
            
            print(summary_df[display_cols].round(4).to_string(index=False))
            
            # Identify best strategy
            best_strategy = summary_df.iloc[0]
            print(f"\n🏆 BEST PERFORMER: {best_strategy['strategy']}")
            print(f"   Excess Return: {best_strategy['excess_annual']:.2%}")
            print(f"   Information Ratio: {best_strategy['info_ratio']:.2f}")
            print(f"   Statistically Significant: {best_strategy['significant']}")
        
        return all_results, performance_summary
    
    def create_performance_charts(self, all_results):
        """Create performance comparison charts."""
        if not all_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Alternative Trading Strategies Performance', fontsize=16)
        
        # Cumulative returns
        for name, results in all_results.items():
            if not results.empty:
                results['strategy_cumret'] = (1 + results['strategy_return']).cumprod()
                axes[0,0].plot(results['date'], results['strategy_cumret'], 
                              label=name, linewidth=2)
        
        # Add benchmark
        if all_results:
            first_result = next(iter(all_results.values()))
            if not first_result.empty:
                first_result['benchmark_cumret'] = (1 + first_result['benchmark_return']).cumprod()
                axes[0,0].plot(first_result['date'], first_result['benchmark_cumret'], 
                              label='S&P 500 Benchmark', linewidth=2, color='black', linestyle='--')
        
        axes[0,0].set_title('Cumulative Returns')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratios
        for name, results in all_results.items():
            if not results.empty and len(results) >= 12:
                rolling_sharpe = (results['strategy_return'].rolling(12).mean() * 12) / \
                               (results['strategy_return'].rolling(12).std() * np.sqrt(12))
                axes[0,1].plot(results['date'], rolling_sharpe, label=name, alpha=0.7)
        
        axes[0,1].set_title('12-Month Rolling Sharpe Ratios')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Return distributions
        for name, results in all_results.items():
            if not results.empty:
                axes[1,0].hist(results['strategy_return'], bins=30, alpha=0.6, 
                             label=name, density=True)
        
        axes[1,0].set_title('Monthly Return Distributions')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Excess return scatter
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, (name, results) in enumerate(all_results.items()):
            if not results.empty:
                excess = results['strategy_return'] - results['benchmark_return']
                axes[1,1].scatter(results['benchmark_return'], excess, 
                                alpha=0.6, label=name, color=colors[i % len(colors)])
        
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,1].set_xlabel('Benchmark Return')
        axes[1,1].set_ylabel('Excess Return')
        axes[1,1].set_title('Strategy Excess Returns vs Benchmark')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('alternative_strategies_performance.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Run alternative strategies test."""
    
    tester = AlternativeStrategyTester("data")
    
    try:
        # Load data
        tester.load_data()
        
        # Run all strategies
        all_results, performance_summary = tester.run_all_strategies()
        
        # Create visualizations
        tester.create_performance_charts(all_results)
        
        # Save results
        if performance_summary:
            summary_df = pd.DataFrame(performance_summary)
            summary_df.to_csv('alternative_strategies_summary.csv', index=False)
            print(f"\nResults saved to: alternative_strategies_summary.csv")
        
    except Exception as e:
        print(f"Error running strategies: {e}")
        raise


if __name__ == "__main__":
    main()