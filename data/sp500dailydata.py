# sp500_daily.py
from xbbg import blp
import pandas as pd
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
TICKER = 'SPX Index'
FIELDS = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']
DATA_DIR = r'C:\Users\ask20004\senior_design\SDP-Group-53\data'
YEARS_BACK = 50

# --- SETUP ---
os.makedirs(DATA_DIR, exist_ok=True)
end_date = datetime.today() - timedelta(days=1)
start_date = end_date - timedelta(days=YEARS_BACK * 365)
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# --- FETCH DAILY DATA ---
print(f"Downloading daily data for {TICKER} from {start_str} to {end_str}...")

df = blp.bdh(
    tickers=TICKER,
    flds=FIELDS,
    start_date=start_str,
    end_date=end_str
)

# Flatten multi-index
df.columns = ['_'.join(col).strip() for col in df.columns.values]
df.reset_index(inplace=True)

# --- ADD TECHNICAL INDICATORS ---
df['return'] = df['SPX Index_PX_LAST'].pct_change()
df['momentum_5d'] = df['SPX Index_PX_LAST'].pct_change(periods=5)
df['momentum_21d'] = df['SPX Index_PX_LAST'].pct_change(periods=21)
df['ma_21d'] = df['SPX Index_PX_LAST'].rolling(window=21).mean()
df['ma_50d'] = df['SPX Index_PX_LAST'].rolling(window=50).mean()
df['volatility_21d'] = df['return'].rolling(window=21).std()
df['rsi_14'] = 100 - (100 / (1 + df['return'].clip(lower=0).rolling(14).mean() / df['return'].clip(upper=0).abs().rolling(14).mean()))

# --- SAVE ---
filename = f'sp500_daily_{start_str}_to_{end_str}.csv'
df.to_csv(os.path.join(DATA_DIR, filename), index=False)
print(f"Saved to {os.path.join(DATA_DIR, filename)}")
