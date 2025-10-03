
#!/usr/bin/env python3
import os
import argparse
import datetime as dt
import numpy as np
import pandas as pd

# ------------------ Config (edit as needed) ------------------
REBAL_FREQ = "M"               # Monthly rebalances at month-end (signals computed as-of M; trade next day)
TOP_PCT_PER_SECTOR = 0.10      # Top 20% by 12-1 momentum within each sector
MAX_NAME_WEIGHT = 0.02         # 2% cap per name after normalization
COST_PER_TURNOVER = 0.0010     # 5 bps per $ traded (one-way)
VOL_WINDOW = 20                # 20-day vol for inverse-vol sizing
USE_TRI_FIRST = True           # Prefer total return index if available
# --------------------------------------------------------------

def to_datetime(s):
    return pd.to_datetime(s).dt.tz_localize(None)

def load_prices(data_dir: str):
    # Support both a single parquet file or a directory dataset
    parquet_path = os.path.join(data_dir, "prices_parquet")
    if os.path.isdir(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        # Fallback: try a single parquet file in data_dir
        files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
        if not files:
            raise FileNotFoundError("No Parquet data found. Expected data/prices_parquet/ or a .parquet file.")
        df = pd.read_parquet(os.path.join(data_dir, files[0]))
    # Coerce dtypes
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    # Some parquet writers might save columns in uppercase—normalize expected cols
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc != c:
            rename[c] = lc
    if rename:
        df = df.rename(columns=rename)
    return df

def load_constituents(data_dir: str):
    path = os.path.join(data_dir, "constituents_long.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    # We only need the raw ticker mapping if 'sectors.csv' uses parse-keyable tickers
    # In your Bloomberg script, sectors.csv uses parse-keyable tickers ("AAPL US Equity").
    # constituents_long.csv stores 'ticker_raw' as "AAPL UW" etc. We'll convert to parse-keyable later.
    return df

def load_sectors(data_dir: str):
    path = os.path.join(data_dir, "sectors.csv")
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    # Expect columns: ticker (parse-keyable), sector
    return df

US_EXCHANGE_CODES = {"UN", "UW", "UQ", "UA", "UR", "UT", "UV"}
def to_parse_keyable(member_code: str) -> str:
    parts = str(member_code).strip().split()
    if len(parts) == 1:
        return f"{parts[0]} US Equity"
    ticker, code = parts[0], parts[1]
    if code in US_EXCHANGE_CODES:
        return f"{ticker} US Equity"
    if len(code) == 2:
        return f"{ticker} {code} Equity"
    return f"{ticker} US Equity"

def compute_daily_returns(prices: pd.DataFrame):
    # Choose return base: tri_gross (preferred) else px_last
    cols = list(prices.columns)
    has_tri = "tri_gross" in cols
    base_col = "tri_gross" if (has_tri and USE_TRI_FIRST) else "px_last"
    # If using px_last, we use simple price returns
    prices = prices.sort_values(["ticker", "date"])
    prices["ret"] = prices.groupby("ticker")[base_col].pct_change()
    return prices, base_col

def compute_momentum(prices: pd.DataFrame, base_col: str):
    # 12-1 momentum per stock at each date: TRI[t-21] / TRI[t-252] - 1
    # Using groupby rolling with integer window sizes in trading days
    g = prices.sort_values(["ticker", "date"]).groupby("ticker")
    # Rolling shift to get values at t-21 and t-252
    val_t_21  = g[base_col].shift(21)
    val_t_252 = g[base_col].shift(252)
    mom = (val_t_21 / val_t_252) - 1.0
    prices["mom_12_1"] = mom.values
    # 20-day vol for inverse-vol sizing
    ret = g["ret"].apply(lambda s: s.rolling(20).std())
    prices["vol20"] = ret.values
    return prices

def month_ends_in_range(dates: pd.DatetimeIndex):
    # Bussiness month end index from min to max
    start = dates.min().normalize()
    end   = dates.max().normalize()
    mes = pd.date_range(start=start, end=end, freq="BM")
    # Restrict to dates that exist in data (as-of dates). We'll asof() later
    return mes

def asof_slice(series: pd.Series, ts: pd.Timestamp):
    # Return the last value on or before ts (per index)
    return series.loc[:ts].iloc[-1] if not series.loc[:ts].empty else np.nan

def build_universe(const_df: pd.DataFrame, asof_date: pd.Timestamp):
    # Universe = SPX members with in_spx=1 as of the month-end date
    # constituents_long.csv uses 'date' month-ends and 'ticker_raw'
    df = const_df[const_df["date"] <= asof_date].sort_values(["ticker_raw", "date"])
    last = df.groupby("ticker_raw").tail(1)
    U_raw = last.loc[last["in_spx"] == 1, "ticker_raw"]
    # Convert to parse-keyable for joining with sectors & prices
    U = set(to_parse_keyable(x) for x in U_raw)
    return U

def pick_by_sector(momentum: pd.Series, sectors_map: pd.Series, universe: set, top_pct=TOP_PCT_PER_SECTOR):
    # momentum: index=ticker, value=score
    # sectors_map: index=ticker, value=sector
    # Select top pct within each sector; require finite momentum
    mom = momentum.copy().replace([np.inf, -np.inf], np.nan).dropna()
    mom = mom[mom.index.isin(universe)]
    sec = sectors_map.loc[mom.index].fillna("Unknown")
    picks = []
    for s, idx in sec.groupby(sec).groups.items():
        sub = mom.loc[idx].dropna()
        if len(sub) == 0:
            continue
        k = max(1, int(np.ceil(len(sub) * top_pct)))
        top_names = sub.sort_values(ascending=False).head(k).index.tolist()
        picks.extend(top_names)
    return list(dict.fromkeys(picks))  # preserve order unique

def inverse_vol_weights(vol_series: pd.Series, names: list, cap=MAX_NAME_WEIGHT):
    v = vol_series.loc[names].replace([np.inf, -np.inf], np.nan)
    # Replace missing/zero vol with median
    med = np.nanmedian(v.values) if np.isfinite(v.values).any() else 0.02
    v = v.fillna(med)
    v = v.clip(lower=1e-6)
    w = 1.0 / v
    w = w / w.sum()
    # Cap per name and renormalize
    w = w.clip(upper=cap)
    w = w / w.sum()
    return w

def compute_turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    # Sum absolute change (L1) / 2 isn't used here; we want $ traded over NAV for cost = sum|Δw|
    names = prev_w.index.union(new_w.index)
    prev = prev_w.reindex(names).fillna(0.0)
    new  = new_w.reindex(names).fillna(0.0)
    return float((new - prev).abs().sum())

def backtest(prices: pd.DataFrame, const_df: pd.DataFrame, sectors_df: pd.DataFrame, start: str, end: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Prep
    prices, base_col = compute_daily_returns(prices)
    prices = compute_momentum(prices, base_col)
    # Pivot helper: fast lookup by (date, ticker) for returns
    # We'll compute portfolio daily returns via weights * next-day returns
    prices = prices.sort_values(["date", "ticker"])
    # build sector map
    sectors_map = sectors_df.drop_duplicates(subset=["ticker"]).set_index("ticker")["sector"]

    # Filter date range
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    prices = prices[(prices["date"] >= start_dt) & (prices["date"] <= end_dt)]

    # Make daily return table for quick indexing
    daily_ret = prices.pivot(index="date", columns="ticker", values="ret")
    vol20_tbl = prices.pivot(index="date", columns="ticker", values="vol20")
    mom_tbl   = prices.pivot(index="date", columns="ticker", values="mom_12_1")

    # Month-ends (signal dates)
    mes = pd.date_range(start=daily_ret.index.min(), end=daily_ret.index.max(), freq="BM")
    mes = mes[(mes >= start_dt) & (mes <= end_dt)]
    if len(mes) == 0:
        raise RuntimeError("No month-ends in selected range.")

    # Trackers
    daily_index = daily_ret.index
    port_ret = pd.Series(index=daily_index, dtype=float)
    positions_by_month = {}

    prev_w = pd.Series(dtype=float)

    for M in mes:
        # Universe
        U = build_universe(const_df, M)

        # Signals as-of M: use .loc[:M].iloc[-1] behavior via asof (last valid obs on or before M)
        mom_M = mom_tbl.loc[:M].tail(1).T.squeeze()  # Series indexed by ticker
        vol_M = vol20_tbl.loc[:M].tail(1).T.squeeze()

        # Sector selection
        picks = pick_by_sector(momentum=mom_M, sectors_map=sectors_map, universe=U)

        if len(picks) == 0:
            # No picks? hold cash until next rebalance
            next_window = daily_ret.loc[(daily_ret.index > M) & (daily_ret.index <= (mes[mes.get_loc(M)+1] if M != mes[-1] else daily_index.max()))]
            port_ret.loc[next_window.index] = 0.0
            continue

        # Weights (inverse vol within picks)
        w = inverse_vol_weights(vol_M, picks, cap=MAX_NAME_WEIGHT)
        w.name = "weight"

        # Compute turnover and apply transaction cost on the rebalance date (charged on first day after M)
        turnover = compute_turnover(prev_w, w)
        cost = turnover * COST_PER_TURNOVER  # cost as fraction of NAV
        # --- Logging trade events ---
        if prev_w is None or prev_w.empty:
            print(f"\n[{M.date()}] Initial portfolio build:")
            for t, wt in w.sort_values(ascending=False).items():
                print(f"  LONG {t:<15} weight={wt:.2%}")
        else:
            new_names = set(w.index)
            old_names = set(prev_w.index)

            opened = new_names - old_names
            closed = old_names - new_names
            kept   = new_names & old_names

            print(f"\n[{M.date()}] Rebalance summary:")
            if opened:
                print("  Opened longs:")
                for t in opened:
                    print(f"    {t:<15} weight={w[t]:.2%}")
            if closed:
                print("  Closed longs:")
                for t in closed:
                    print(f"    {t}")
            if kept:
                print("  Continuing positions (updated weights):")
                for t in kept:
                    old_wt, new_wt = prev_w.get(t, 0), w.get(t, 0)
                    print(f"    {t:<15} {old_wt:.2%} -> {new_wt:.2%}")

            print(f"  Turnover this month: {turnover:.2%} (cost {cost:.3%})")

        prev_w = w

        # Trading window: from next trading day after M up to next month-end (inclusive or exclusive?)
        # We'll apply weights on days strictly > M up to and including the next month-end day.
        if M != mes[-1]:
            next_M = mes[mes.get_loc(M)+1]
        else:
            next_M = daily_index.max()

        window = daily_ret.loc[(daily_ret.index > M) & (daily_ret.index <= next_M)]
        # Daily portfolio returns = sum_i w_i * r_{i,t}
        sub = window[picks].copy()
        # Some tickers may be missing days; fill NaN returns with 0 for that day (equivalently, no position return)
        sub = sub.fillna(0.0)
        r_series = (sub @ w.reindex(sub.columns).fillna(0.0)).astype(float)
        # Apply one-time cost on the first day of the window
        if len(r_series) > 0 and cost != 0.0:
            first_day = r_series.index[0]
            r_series.loc[first_day] = r_series.loc[first_day] - cost

        port_ret.loc[r_series.index] = r_series.values

        # Store positions at this rebalance
        positions_by_month[pd.to_datetime(M).date()] = w.sort_values(ascending=False).to_dict()

    # Build equity curve
    port_ret = port_ret.fillna(0.0)
    equity = (1.0 + port_ret).cumprod()
    equity.name = "NAV"

    # Summary stats
    def ann_return(series):
        if len(series) == 0: return np.nan
        yrs = (series.index[-1] - series.index[0]).days / 365.25
        return series.iloc[-1]**(1/yrs) - 1 if yrs > 0 else np.nan

    def ann_vol(ret_daily):
        return ret_daily.std(ddof=0) * np.sqrt(252)

    cagr = ann_return(equity)
    vol = ann_vol(port_ret)
    sharpe = (port_ret.mean() * 252) / vol if vol and vol > 0 else np.nan
    mdd = (equity / equity.cummax() - 1.0).min()

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    equity.to_csv(os.path.join(args.out_dir, "mvp_momentum_longonly_equity_curve.csv"))
    pd.DataFrame.from_dict(positions_by_month, orient="index").to_csv(os.path.join(args.out_dir, "mvp_momentum_longonly_positions.csv"))
    with open(os.path.join(args.out_dir, "mvp_momentum_longonly_summary.txt"), "w") as f:
        f.write(f"CAGR:   {cagr:.4%}\n")
        f.write(f"Vol:    {vol:.4%}\n")
        f.write(f"Sharpe: {sharpe:.2f}\n")
        f.write(f"MaxDD:  {mdd:.2%}\n")
        f.write(f"Costs:  {COST_PER_TURNOVER*10000:.1f} bps per $ traded (one-way)\n")
        f.write(f"Top% per sector: {TOP_PCT_PER_SECTOR:.0%}\n")
        f.write(f"Vol window: {VOL_WINDOW} days\n")
        f.write(f"Name cap: {MAX_NAME_WEIGHT:.2%}\n")

    print("=== MVP Cross-Sectional Momentum (Long-Only) ===")
    print(f"CAGR:   {cagr:.2%}")
    print(f"Vol:    {vol:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"MaxDD:  {mdd:.2%}")
    print(f"Outputs written to: {args.out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory containing prices_parquet/, constituents_long.csv, sectors.csv")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out-dir", default="results", help="Output directory for results")
    args_parsed = parser.parse_args()

    # Load data
    prices = load_prices(args_parsed.data_dir)
    const  = load_constituents(args_parsed.data_dir)
    sectors = load_sectors(args_parsed.data_dir)

    # Keep only needed columns
    keep = [c for c in ["date","ticker","px_last","tri_gross","volume"] if c in prices.columns]
    prices = prices[keep].dropna(subset=["date","ticker"])

    # Run backtest
    backtest(prices, const, sectors, args_parsed.start, args_parsed.end, args_parsed.out_dir)

if __name__ == "__main__":
    # Make args visible inside backtest for saving outputs
    import argparse as _argparse
    _p = _argparse.ArgumentParser(add_help=False)
    _p.add_argument("--out-dir", default="results")
    try:
        args, _ = _p.parse_known_args()
    except SystemExit:
        class A: pass
        args = A()
        args.out_dir = "results"
    main()
