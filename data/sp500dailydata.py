# sp500_daily_all.py
from xbbg import blp
import pandas as pd
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
FIELDS = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']
DATA_DIR = r'C:\Users\ask20004\senior_design\SDP-Group-53\data'
YEARS_BACK = 30

# --- SETUP ---
os.makedirs(DATA_DIR, exist_ok=True)
end_date = datetime.today() - timedelta(days=1)
start_date = end_date - timedelta(days=YEARS_BACK * 365)
start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

# --- FETCH S&P 500 MEMBERS FROM BLOOMBERG ---
members = blp.bds('SPX Index', 'INDX_MEMBERS')
tickers = [f"{t} Equity" for t in members['member_ticker_and_exchange_code'].dropna()]

print(f"Found {len(tickers)} tickers in S&P 500")

all_data = []

# --- LOOP THROUGH ALL TICKERS ---
for t in tickers:
    print(f"Downloading {t} from {start_str} to {end_str}...")
    df = blp.bdh(
        tickers=t,
        flds=FIELDS,
        start_date=start_str,
        end_date=end_str
    )

    # Flatten multi-index
    df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns.values]
    df.reset_index(inplace=True)
    df.insert(1, 'ticker', t)   # add ticker column

    # --- ADD TECHNICAL INDICATORS ---
    df['return'] = df['PX_LAST'].pct_change()
    df['momentum_5d'] = df['PX_LAST'].pct_change(periods=5)
    df['momentum_21d'] = df['PX_LAST'].pct_change(periods=21)
    df['ma_21d'] = df['PX_LAST'].rolling(window=21).mean()
    df['ma_50d'] = df['PX_LAST'].rolling(window=50).mean()
    df['volatility_21d'] = df['return'].rolling(window=21).std()
    df['rsi_14'] = 100 - (100 / (1 + df['return'].clip(lower=0).rolling(14).mean() /
                                 df['return'].clip(upper=0).abs().rolling(14).mean()))

    all_data.append(df)

# --- CONCATENATE ALL RESULTS ---
final_df = pd.concat(all_data, ignore_index=True)

# --- SAVE ---
filename = f'sp500_constituents_{start_str}_to_{end_str}.csv'
final_df.to_csv(os.path.join(DATA_DIR, filename), index=False)
print(f"Saved to {os.path.join(DATA_DIR, filename)}")
