from xbbg import blp
import pandas as pd
import os
from datetime import datetime, timedelta

# --- CONFIG ---
TICKER = 'SPX Index'   # For ETF alternative with fuller intraday history: 'SPY US Equity'
INTERVAL = 60          # hourly bars
DATA_DIR = r'C:\Users\ask20004\senior_design\SDP-Group-53\data'
DAYS_BACK = 100        # Bloomberg intraday history limit

# --- SETUP ---
os.makedirs(DATA_DIR, exist_ok=True)
end_dt = datetime.today() - timedelta(days=1)   # yesterday
start_dt = end_dt - timedelta(days=DAYS_BACK)
print(f"Downloading {INTERVAL}-min bars for {TICKER} from {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}")

all_chunks = []

# --- LOOP DAY BY DAY ---
current = start_dt
while current <= end_dt:
    day_start = current.replace(hour=9, minute=30)   # US equity market open
    day_end = current.replace(hour=16, minute=0)    # US equity market close

    try:
        df = blp.bdib(
            ticker=TICKER,
            dt=day_start,
            end_dt=day_end,
            interval=INTERVAL
        ).reset_index()

        # Flatten multi-index
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

        if not df.empty:
            all_chunks.append(df)
            print(f"✓ Got {len(df)} rows for {current:%Y-%m-%d}")
        else:
            print(f"✗ No trading data for {current:%Y-%m-%d}")

    except Exception as e:
        print(f"Error on {current:%Y-%m-%d}: {e}")

    current += timedelta(days=1)   # move to next day

# --- COMBINE ---
if all_chunks:
    full_df = pd.concat(all_chunks, ignore_index=True)

    # Technical indicators
    full_df['return'] = full_df[f'{TICKER}_close'].pct_change()
    full_df['momentum_3'] = full_df[f'{TICKER}_close'].pct_change(periods=3)
    full_df['momentum_10'] = full_df[f'{TICKER}_close'].pct_change(periods=10)
    full_df['ma_fast'] = full_df[f'{TICKER}_close'].rolling(window=5).mean()
    full_df['ma_slow'] = full_df[f'{TICKER}_close'].rolling(window=20).mean()
    full_df['volatility'] = full_df['return'].rolling(window=20).std()

    filename = f'sp500_intraday_{INTERVAL}min_{start_dt:%Y-%m-%d}_to_{end_dt:%Y-%m-%d}.csv'
    filepath = os.path.join(DATA_DIR, filename)
    full_df.to_csv(filepath, index=False)
    print(f"Saved {full_df.shape[0]} rows to {filepath}")
else:
    print("No data collected.")
