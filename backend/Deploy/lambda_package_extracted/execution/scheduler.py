#!/usr/bin/env python3
"""
Monthly Rebalance Scheduler

Schedules the monthly rebalance job to run automatically.
Uses APScheduler to trigger on the first trading day of each month.

Usage:
    # Run as a persistent scheduler (keeps running)
    python -m execution.scheduler

    # Run a single rebalance now (for testing)
    python -m execution.scheduler --run-now --dry-run
"""

import os
import sys
import argparse
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _refresh_macro_cache(settings):
    """Pull fresh FRED and gold data into the cache before the rebalance runs."""
    from data.fred_fetcher import fetch_all_macro_data
    from data.gold_fetcher import fetch_gold_prices

    cache_dir = settings.get("data", {}).get("cache_dir", "data/cache")
    gold_cfg  = settings.get("gold", {})

    # Read FRED key from credentials.env
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        env_path = os.path.join(os.path.dirname(__file__), "..", "config", "credentials.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == "FRED_API_KEY":
                        fred_key = v.strip()
                        break

    print("\n[Data] Refreshing FRED macro data...")
    try:
        fetch_all_macro_data(api_key=fred_key, start_date="2010-01-01", output_dir=cache_dir)
    except Exception as e:
        print(f"  Warning: FRED refresh failed: {e} — using cached data")

    print("[Data] Refreshing gold prices...")
    try:
        fetch_gold_prices(ticker=gold_cfg.get("ticker", "GLD"),
                          start_date="2010-01-01", output_dir=cache_dir)
    except Exception as e:
        print(f"  Warning: Gold refresh failed: {e} — using cached data")


def _get_portfolio_returns(lookback=252):
    """
    Build a portfolio return series for vol scaling.
    Seeds with real backtest daily returns, then appends live Alpaca account returns.
    This avoids using SPY as a proxy entirely.
    """
    import pandas as pd

    # 1. Load real backtest returns (in-sample + OOS)
    frames = []
    for path in ["results/long_only/daily_returns.csv", "results/long_only_oos/daily_returns.csv"]:
        if os.path.exists(path):
            r = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            frames.append(r)

    if frames:
        backtest_rets = pd.concat(frames).sort_index()
        backtest_rets = backtest_rets[~backtest_rets.index.duplicated(keep="last")]
    else:
        backtest_rets = pd.Series(dtype=float)

    # 2. Append live Alpaca account returns if available
    live_rets, _ = _get_equity_history(None)  # handled below if broker not passed
    if live_rets is not None and not live_rets.empty:
        combined = pd.concat([backtest_rets, live_rets]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = backtest_rets

    if combined.empty:
        return None

    return combined.tail(lookback)


def _get_spy_trend(ma_window=200):
    """Return True if SPY is above its MA (risk-on), False if below (risk-off)."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="2y", auto_adjust=True, progress=False)
        close = spy["Close"].squeeze().dropna()  # ensure 1D Series
        if len(close) < ma_window:
            return True  # Not enough data, default to risk-on
        spy_price = close.iloc[-1].item()
        spy_ma = close.tail(ma_window).mean().item()
        above_ma = spy_price > spy_ma
        print(f"  Trend filter: SPY ${spy_price:.2f} vs {ma_window}d MA ${spy_ma:.2f} "
              f"-> {'ABOVE (risk-on)' if above_ma else 'BELOW (risk-off)'}")
        return above_ma
    except Exception as e:
        print(f"  Warning: Could not fetch SPY for trend filter: {e}")
        return True


def _get_equity_history(broker):
    """Fetch account equity history from Alpaca for drawdown computation."""
    import requests
    import pandas as pd
    import numpy as np
    try:
        resp = requests.get(
            f"{broker.base_url}/v2/account/portfolio/history",
            headers=broker.headers,
            params={"period": "6M", "timeframe": "1D"},
        )
        resp.raise_for_status()
        data = resp.json()
        equity = pd.Series(
            data["equity"],
            index=pd.to_datetime(data["timestamp"], unit="s").tz_localize(None),
        ).dropna()
        # Drop zero equity rows (pre-funding placeholders from new accounts)
        equity = equity[equity > 0]
        # Convert to daily returns; drop inf/nan (e.g. first day after funding)
        rets = equity.pct_change().dropna()
        rets = rets[np.isfinite(rets)]
        return rets, equity
    except Exception as e:
        print(f"  Warning: Could not fetch equity history: {e}")
        return None, None


def run_monthly_rebalance(dry_run=False):
    """
    Execute a single monthly rebalance with full risk controls:
        1. Load prices + compute momentum signals
        2. Liquidity filter
        3. Trend filter (SPY 200-day MA)
        4. Macro regime overlay (VIX, yield curve, credit spread)
        5. Rank and select (cross-sectional or industry-neutral)
        6. Build long-only inverse-vol weighted portfolio
        7. Volatility scaling (Barroso & Santa-Clara 2015)
        8. Drawdown circuit breaker
        9. Gold sleeve (5% allocation)
        10. Execute via Alpaca
    """
    from data.data_pipeline import DataPipeline
    from data.universe import UniverseManager
    from data.macro_regime import MacroRegimeClassifier
    from signals.momentum import compute_momentum_signal
    from signals.ranking import rank_cross_sectional, rank_industry_neutral
    from portfolio.constructor import build_long_only_portfolio
    from portfolio.risk_manager import RiskManager
    from execution.rebalancer import Rebalancer
    from execution.alpaca_broker import AlpacaBroker
    from data.universe import to_alpaca_ticker
    import yaml
    import pandas as pd

    # Load settings
    settings_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)

    strat_cfg = settings.get("strategy", {})
    risk_cfg  = settings.get("risk", {})
    gold_cfg  = settings.get("gold", {})
    macro_cfg = settings.get("macro_regime", {})

    print(f"\n{'='*60}")
    print(f"MONTHLY REBALANCE - {date.today()}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # 0. Refresh macro cache with latest FRED + gold data
    # ------------------------------------------------------------------ #
    _refresh_macro_cache(settings)

    # ------------------------------------------------------------------ #
    # 1. Load prices and compute signals
    # ------------------------------------------------------------------ #
    pipeline = DataPipeline()
    prices = pipeline.load_prices()
    prices = compute_momentum_signal(
        prices,
        formation_period=strat_cfg.get("formation_period", 252),
        skip_period=strat_cfg.get("skip_period", 21),
    )

    latest_date = prices["date"].max()
    latest = prices[prices["date"] == latest_date].set_index("ticker")
    mom_series = latest["mom_12_1"]
    vol_series  = latest["vol20"]

    sectors_df  = pipeline.load_sectors()
    sectors_map = sectors_df.drop_duplicates(subset=["ticker"]).set_index("ticker")["sector"]

    universe_mgr = UniverseManager(pipeline.data_dir)
    universe     = universe_mgr.get_members(latest_date, as_bbg=True)

    risk_mgr = RiskManager(settings)
    broker   = AlpacaBroker(paper=settings.get("execution", {}).get("paper_trading", True))

    print(f"Signal date: {latest_date.date()}")
    print(f"Universe size: {len(universe)} stocks")

    # ------------------------------------------------------------------ #
    # 2. Liquidity filter
    # ------------------------------------------------------------------ #
    print("\n[Risk] Applying liquidity filter...")
    liquid_universe = risk_mgr.filter_by_liquidity(universe, prices, latest_date)
    if liquid_universe:
        universe = liquid_universe

    # ------------------------------------------------------------------ #
    # 3. Trend filter — SPY 200-day MA
    # ------------------------------------------------------------------ #
    trend_on = risk_cfg.get("use_trend_filter", True)
    trend_risk_on = True
    if trend_on:
        print("\n[Risk] Checking trend filter...")
        trend_risk_on = _get_spy_trend(risk_cfg.get("trend_ma_window", 200))

    # ------------------------------------------------------------------ #
    # 4. Macro regime
    # ------------------------------------------------------------------ #
    macro_df     = pipeline.load_macro_data()
    macro_result = None
    if macro_cfg.get("enabled", True) and macro_df is not None:
        print("\n[Risk] Classifying macro regime...")
        classifier   = MacroRegimeClassifier(settings)
        macro_result = classifier.classify(macro_df)
        signals_triggered = macro_result.get("risk_off_count", 0)
        print(f"  Macro regime: {macro_result['regime']} "
              f"({signals_triggered}/4 risk-off signals)")
        for name, sig in macro_result.get("signals", {}).items():
            flag = "⚠" if sig["triggered"] else " "
            print(f"  {flag} {name}: {sig['value']:.2f} (threshold {sig['threshold']})")

    # ------------------------------------------------------------------ #
    # 5. Rank and select
    # ------------------------------------------------------------------ #
    print("\n[Signal] Ranking stocks...")
    if strat_cfg.get("use_industry_neutral", False):
        selection = rank_industry_neutral(mom_series, sectors_map, universe,
                                          strat_cfg.get("long_pct", 0.20),
                                          strat_cfg.get("short_pct", 0.20))
    else:
        selection = rank_cross_sectional(
            mom_series, universe,
            long_pct=strat_cfg.get("long_pct", 0.20),
            short_pct=strat_cfg.get("short_pct", 0.20),
        )
    long_picks = selection["long"]
    print(f"  Long picks: {len(long_picks)}")

    if not long_picks:
        print("No picks — aborting rebalance.")
        return []

    # ------------------------------------------------------------------ #
    # 6. Build raw long-only portfolio weights
    # ------------------------------------------------------------------ #
    portfolio = build_long_only_portfolio(
        long_picks, vol_series, sectors_map,
        max_weight=risk_cfg.get("max_single_name", 0.05),
        sector_cap=risk_cfg.get("sector_cap", 0.30),
    )
    long_w = portfolio["weights"]

    # ------------------------------------------------------------------ #
    # 7 & 8. Vol scaling + drawdown circuit breaker via RiskManager
    # ------------------------------------------------------------------ #
    print("\n[Risk] Applying vol scaling and drawdown controls...")
    port_rets_live, equity_curve = _get_equity_history(broker)

    # Build portfolio return series: real backtest returns + live Alpaca returns
    port_rets = _get_portfolio_returns(lookback=risk_cfg.get("vol_lookback", 126))
    if port_rets_live is not None and not port_rets_live.empty:
        port_rets = pd.concat([port_rets, port_rets_live]).sort_index()
        port_rets = port_rets[~port_rets.index.duplicated(keep="last")]
        print(f"  Return series: {len(port_rets)} days (backtest + {len(port_rets_live)} live days)")
    else:
        print(f"  Return series: {len(port_rets) if port_rets is not None else 0} days (backtest only — real strategy returns)")

    risk_result = risk_mgr.apply_all_risk_controls(
        long_w,
        portfolio_returns=port_rets,
        equity_curve=equity_curve if equity_curve is not None else pd.Series(dtype=float),
        sectors_map=sectors_map,
        macro_regime=macro_result,
        as_of_date=None,
    )
    long_w  = risk_result["weights"]
    report  = risk_result["report"]

    print(f"  Vol scaling factor:  {report.get('vol_scaling_factor', 1.0):.2f}x")
    print(f"  Drawdown factor:     {report.get('drawdown_factor', 1.0):.2f}x")
    print(f"  Combined factor:     {report.get('combined_factor', 1.0):.2f}x")
    print(f"  Total equity exposure: {report.get('total_exposure', 1.0):.1%}")

    # Trend filter: if SPY below 200-day MA, cut exposure in half
    if not trend_risk_on:
        print("  Trend filter ACTIVE: SPY below 200d MA — cutting exposure 50%")
        long_w = long_w * 0.5

    # ------------------------------------------------------------------ #
    # 9. Account value + gold sleeve
    # ------------------------------------------------------------------ #
    account       = broker.get_account()
    total_capital = float(account["portfolio_value"])
    print(f"\nAccount value: ${total_capital:,.2f}")

    gold_enabled = gold_cfg.get("enabled", False)
    gold_pct     = gold_cfg.get("allocation_pct", 0.05)
    equity_pct   = 1.0 - (gold_pct if gold_enabled else 0.0)

    long_dollars = long_w * total_capital * equity_pct

    target = {to_alpaca_ticker(t): v for t, v in long_dollars.items()}

    if gold_enabled:
        gold_dollars = total_capital * gold_pct
        target[gold_cfg.get("ticker", "GLD")] = gold_dollars
        print(f"Gold sleeve: ${gold_dollars:,.2f} ({gold_pct:.0%} of NAV)")

    # ------------------------------------------------------------------ #
    # 10. Execute
    # ------------------------------------------------------------------ #
    rebalancer   = Rebalancer(broker)
    target_series = pd.Series(target)
    results = rebalancer.rebalance(target_series, dry_run=dry_run)

    print(f"\nRebalance {'preview' if dry_run else 'complete'}.")
    return results


def start_scheduler():
    """
    Start a persistent scheduler that runs the rebalance monthly.
    Triggers on the first business day of each month at 10:00 AM ET.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        raise ImportError("Install apscheduler: pip install apscheduler")

    scheduler = BlockingScheduler()

    # Run on the first business day of each month at 10:00 AM
    scheduler.add_job(
        run_monthly_rebalance,
        CronTrigger(day="1-7", day_of_week="mon-fri", hour=10, minute=0),
        id="monthly_rebalance",
        name="Monthly Momentum Rebalance",
        misfire_grace_time=3600,
    )

    print("Scheduler started. Monthly rebalance will run on first business day at 10:00 AM.")
    print("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\nScheduler stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monthly rebalance scheduler")
    parser.add_argument("--run-now", action="store_true", help="Run rebalance immediately")
    parser.add_argument("--dry-run", action="store_true", help="Preview trades without executing")
    parser.add_argument("--schedule", action="store_true", help="Start persistent scheduler")
    args = parser.parse_args()

    if args.run_now:
        run_monthly_rebalance(dry_run=args.dry_run)
    elif args.schedule:
        start_scheduler()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
