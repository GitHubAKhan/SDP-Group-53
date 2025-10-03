import pandas as pd
import json
from datetime import datetime

# Load your S&P 500 CSV
df = pd.read_csv("sp500_daily_1975-09-30_to_2025-09-17.csv")

# Rename and parse date column
df['Date'] = pd.to_datetime(df['index'])
df = df.sort_values('Date')

# Use the closing price column
df['Close'] = df['SPX Index_PX_LAST']

# Compute daily returns
df['Return'] = df['Close'].pct_change()

def compute_return(days):
    """Compute cumulative return for last N trading days"""
    if len(df) < days:
        return None
    recent = df['Return'].tail(days)
    cumulative = (1 + recent).prod() - 1
    return round(cumulative * 100, 2)

def compute_ytd():
    """Compute return since Jan 1 of current year"""
    today = df['Date'].max()
    start_of_year = datetime(today.year, 1, 1)
    ytd = df[df['Date'] >= start_of_year]['Return']
    cumulative = (1 + ytd).prod() - 1
    return round(cumulative * 100, 2)

def compute_itd():
    """Compute return from first date to last date in CSV"""
    start = df['Close'].iloc[0]
    end = df['Close'].iloc[-1]
    cumulative = (end / start) - 1
    return round(cumulative * 100, 2)

# Build JSON object
benchmark_data = {
    "1D": compute_return(1),
    "1W": compute_return(5),
    "1M": compute_return(21),
    "3M": compute_return(63),
    "YTD": compute_ytd(),
    "ITD": compute_itd()
}

# Save to file
with open("./src/benchmark_returns.json", "w") as f:
    json.dump(benchmark_data, f, indent=2)

print("✅ benchmark_returns.json created:", benchmark_data)
