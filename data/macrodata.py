# CODE IS INCOMPLETE WILL NEED TO COME BACK TO IT LATER
from xbbg import blp
import pandas as pd
import os
from datetime import datetime

# --- Config ---
DATA_DIR = r"C:\Users\ask20004\senior_design\SDP-Group-53\data"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_XLSX = os.path.join(DATA_DIR, "macrodata.xlsx")

START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# --- Macro tickers ---
macro_tickers = {
    "CPI_YoY": "CPI YOY:IND",
    "Core_PCE_YoY": "PCE CYOY:IND",
    "Unemployment_Rate_U3": "USURTOT:IND",
    "GDP_QoQ_Ann": "GDP CQOQ:IND",
    "Fed_Funds_Rate": "FDFD:IND",
    # BLOOMBERG DOES NOT HAVE A DISCOUNT RATE
    "UST_2Y_Yield": "USGG2YR:IND",
    "UST_5Y_Yield": "USGG5YR:IND",
    "UST_10Y_Yield": "USGG10YR:IND",
    "Gold_Spot_USD": "XAU:CUR",
}

FIELD_CANDIDATES = ["PX_LAST", "VALUE", "INDX_VAL", "LAST_PRICE"]

def fetch_series(ticker: str):
    """Try multiple fields to get a Bloomberg time series."""
    for fld in FIELD_CANDIDATES:
        try:
            df = blp.bdh(
                tickers=ticker,
                flds=[fld],
                start_date=START_DATE,
                end_date=END_DATE
            )
            if df is not None and not df.empty:
                df = df.reset_index()
                col = (fld, ticker) if (fld, ticker) in df.columns else df.columns[-1]
                s = df[col]
                s.index = pd.to_datetime(df["index"])
                return s, fld
        except Exception:
            continue
    return None, None

def main():
    combined = pd.DataFrame()
    meta = []

    for name, ticker in macro_tickers.items():
        series, fld = fetch_series(ticker)
        if series is None:
            print(f"No data for {name} ({ticker})")
            meta.append([name, ticker, "FAILED"])
            continue

        series = series.rename(name)
        combined = pd.concat([combined, series], axis=1)
        print(f"{name}: {ticker} ({fld}) → {series.dropna().shape[0]} rows")
        meta.append([name, ticker, fld])

    combined.index.name = "date"
    combined = combined.sort_index()

    if combined.empty:
        print("No data pulled.")
        return

    # Momentum features
    feats = pd.DataFrame(index=combined.index)
    for col in combined.columns:
        feats[f"{col}_chg"]   = combined[col].pct_change()
        feats[f"{col}_ma3"]   = combined[col].rolling(3).mean()
        feats[f"{col}_ma12"]  = combined[col].rolling(12).mean()
        feats[f"{col}_vol12"] = combined[col].pct_change().rolling(12).std()

    meta_df = pd.DataFrame(meta, columns=["Series", "Ticker", "Field"])

    # Save
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xw:
        combined.to_excel(xw, sheet_name="raw")
        feats.to_excel(xw, sheet_name="features")
        meta_df.to_excel(xw, sheet_name="meta", index=False)

    print(f"\n Saved {OUT_XLSX}")
    print(f"   raw:      {combined.shape}")
    print(f"   features: {feats.shape}")

if __name__ == "__main__":
    main()
