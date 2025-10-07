#!/usr/bin/env python3
"""
Survivorship-bias-free S&P 500 pull via Bloomberg blpapi.

Key behavior:
- Convert 'Member Ticker and Exchange Code' (e.g., 'CMCSA UW') to parse-keyable:
    US codes {UN, UW, UQ, UA, UR, UT, UV} -> 'TICKER US Equity'
    others   -> 'TICKER CC Equity'  (e.g., 'VOD LN Equity')
- Robust date coercion for Bloomberg HistoricalDataResponse (handles either
  datetime.date or datetime.datetime).

Outputs
  - data/constituents_long.csv  (date, ticker_raw, in_spx)
  - data/sectors.csv            (ticker, sector)
  - data/prices_parquet/        (Parquet dataset, partitioned by year)
  - data/prices_csv/            (CSV, one file per year)
  - data/summary.xlsx           (small Excel summary)

Usage
  pip install blpapi pandas pyarrow xlsxwriter
  python 12_1trade.py --start 1995-01-01 --end 2025-09-19
"""

import argparse
import datetime as dt
import os
from typing import Iterable, List

import blpapi
import pandas as pd

REFDATA_SVC = "//blp/refdata"
SPX_INDEX = "SPX Index"

HIST_FIELDS = [
    "PX_LAST",
    "TOT_RETURN_INDEX_GROSS_DVDS",
    "PX_VOLUME",
]

SECTOR_FIELDS_PREF = ["GICS_SECTOR_NAME", "INDUSTRY_SECTOR"]

# Exchange codes that map to 'US Equity'
US_EXCHANGE_CODES = {"UN", "UW", "UQ", "UA", "UR", "UT", "UV"}


def yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def month_ends(start_date: dt.date, end_date: dt.date):
    cur = dt.date(start_date.year, start_date.month, 1)
    while cur <= end_date:
        ny = cur.year + (1 if cur.month == 12 else 0)
        nm = 1 if cur.month == 12 else (cur.month + 1)
        first_of_next = dt.date(ny, nm, 1)
        me = first_of_next - dt.timedelta(days=1)
        if me >= start_date and me <= end_date:
            yield me
        cur = first_of_next


def to_parse_keyable(member_code: str) -> str:
    """
    Convert 'TICKER CC' to a parse-keyable equity string.
    US codes map to 'US Equity'. Others keep CC with ' Equity' appended.
    """
    parts = member_code.strip().split()
    if len(parts) == 1:
        return f"{parts[0]} US Equity"
    ticker, code = parts[0], parts[1]
    if code in US_EXCHANGE_CODES:
        return f"{ticker} US Equity"
    if len(code) == 2:
        return f"{ticker} {code} Equity"
    return f"{ticker} US Equity"


def _as_date(x):
    """Coerce Bloomberg date element to a Python date."""
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.datetime):
        return x.date()
    # Fallback: parse string
    return pd.to_datetime(x).date()


class Bbg:
    def __init__(self, host="localhost", port=8194):
        self.host = host
        self.port = port
        self.sess = None

    def __enter__(self):
        opts = blpapi.SessionOptions()
        opts.setServerAddress(self.host, self.port, 0)
        opts.setNumStartAttempts(1)
        self.sess = blpapi.Session(opts)
        if not self.sess.start():
            raise RuntimeError("Failed to start Bloomberg session")
        if not self.sess.openService(REFDATA_SVC):
            raise RuntimeError("Failed to open //blp/refdata")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.sess:
            self.sess.stop()

    @property
    def refdata(self):
        return self.sess.getService(REFDATA_SVC)

    def _send(self, req):
        self.sess.sendRequest(req)
        out = []
        while True:
            ev = self.sess.nextEvent()
            for msg in ev:
                out.append(msg)
            if ev.eventType() == blpapi.Event.RESPONSE:
                break
        return out

    # ---------- constituents as-of ----------
    def get_index_members_asof(self, index_ticker: str, asof_date: dt.date) -> List[str]:
        req = self.refdata.createRequest("ReferenceDataRequest")
        req.getElement("securities").appendValue(index_ticker)
        req.getElement("fields").appendValue("INDX_MEMBERS")

        ov = req.getElement("overrides")
        o = ov.appendElement()
        o.setElement("fieldId", "START_DT")
        o.setElement("value", yyyymmdd(asof_date))
        o = ov.appendElement()
        o.setElement("fieldId", "END_DT")
        o.setElement("value", yyyymmdd(asof_date))

        msgs = self._send(req)
        members = []
        for m in msgs:
            if m.hasElement("securityData"):
                sd = m.getElement("securityData").getValue(0)
                if sd.hasElement("fieldData") and sd.getElement("fieldData").hasElement("INDX_MEMBERS"):
                    bulk = sd.getElement("fieldData").getElement("INDX_MEMBERS")
                    for i in range(bulk.numValues()):
                        row = bulk.getValueAsElement(i)
                        if row.hasElement("Member Ticker and Exchange Code"):
                            members.append(row.getElementAsString("Member Ticker and Exchange Code"))
                        elif row.hasElement("Member Ticker"):
                            members.append(row.getElementAsString("Member Ticker"))
        return sorted(set(members))

    # ---------- sectors ----------
    def get_sectors(self, tickers: Iterable[str]):
        req = self.refdata.createRequest("ReferenceDataRequest")
        sec = req.getElement("securities")
        for t in tickers:
            sec.appendValue(t)
        flds = req.getElement("fields")
        for f in SECTOR_FIELDS_PREF:
            flds.appendValue(f)

        msgs = self._send(req)

        out = {}
        for msg in msgs:
            if not msg.hasElement("securityData"):
                continue
            sd_arr = msg.getElement("securityData")
            for i in range(sd_arr.numValues()):
                sd = sd_arr.getValueAsElement(i)
                tkr = sd.getElementAsString("security")
                fd = sd.getElement("fieldData") if sd.hasElement("fieldData") else None
                sector_val = "Unknown"
                if fd:
                    for f in SECTOR_FIELDS_PREF:
                        if fd.hasElement(f):
                            v = fd.getElement(f)
                            if v.isValid():
                                sval = str(v.getValue())
                                if sval and sval.upper() != "N.A.":
                                    sector_val = sval
                                    break
                out[tkr] = sector_val
        return out

    # ---------- historical daily ----------
    def get_history(self, tickers, start_date, end_date, fields=HIST_FIELDS, periodicity="DAILY"):
        req = self.refdata.createRequest("HistoricalDataRequest")
        sec = req.getElement("securities")
        for t in tickers:
            sec.appendValue(t)
        flds = req.getElement("fields")
        for f in fields:
            flds.appendValue(f)
        req.set("startDate", yyyymmdd(start_date))
        req.set("endDate", yyyymmdd(end_date))
        req.set("periodicitySelection", periodicity)
        req.set("adjustmentSplit", True)
        req.set("adjustmentNormal", True)
        req.set("adjustmentAbnormal", True)

        msgs = self._send(req)

        rows = []
        for msg in msgs:
            if not msg.hasElement("securityData"):
                continue
            sd = msg.getElement("securityData")
            ticker = sd.getElementAsString("security")
            if not sd.hasElement("fieldData"):
                continue
            fd = sd.getElement("fieldData")
            for i in range(fd.numValues()):
                r = fd.getValueAsElement(i)
                # Robust date coercion (handles date or datetime)
                d_elem = r.getElementAsDatetime("date")
                rec = {"date": _as_date(d_elem), "ticker": ticker}
                for f in fields:
                    rec[f] = r.getElement(f).getValue() if r.hasElement(f) else None
                rows.append(rec)
        return pd.DataFrame(rows)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--host",  default="localhost")
    ap.add_argument("--port",  type=int, default=8194)
    ap.add_argument("--batch", type=int, default=175, help="Batch size for historical/reference calls")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end   = dt.date.fromisoformat(args.end)

    outdir = args.outdir
    ensure_dirs(outdir, f"{outdir}/prices_parquet", f"{outdir}/prices_csv")

    with Bbg(args.host, args.port) as bb:
        # 1) month-end constituents
        print("[1/5] Pulling month-end constituents…")
        me_dates = list(month_ends(start, end))
        date_to_members = {}
        universe_raw = set()
        for me in me_dates:
            members = bb.get_index_members_asof(SPX_INDEX, me)
            date_to_members[me] = members
            universe_raw.update(members)
        if not universe_raw:
            raise RuntimeError("No constituents returned. Check entitlements for INDX_MEMBERS.")

        # Save raw constituents long
        const_rows = []
        uni_sorted = sorted(universe_raw)
        for me in me_dates:
            s = set(date_to_members[me])
            for t in uni_sorted:
                const_rows.append({"date": me.isoformat(), "ticker_raw": t, "in_spx": 1 if t in s else 0})
        df_const = pd.DataFrame(const_rows)
        df_const.to_csv(f"{outdir}/constituents_long.csv", index=False)
        print(f"Saved {outdir}/constituents_long.csv  (rows={len(df_const):,})")

        # 1b) convert to parse-keyable without reference call
        print("Converting 'ticker exchange' codes to parse-keyable tickers…")
        tickers = sorted({to_parse_keyable(t) for t in uni_sorted})

        # 2) sectors
        print("[2/5] Pulling sector tags (GICS preferred)…")
        sectors = {}
        for i in range(0, len(tickers), args.batch):
            chunk = tickers[i:i+args.batch]
            sectors.update(bb.get_sectors(chunk))
        df_sec = pd.DataFrame({"ticker": list(sectors.keys()), "sector": list(sectors.values())})
        df_sec.to_csv(f"{outdir}/sectors.csv", index=False)
        print(f"Saved {outdir}/sectors.csv  (rows={len(df_sec):,})")

        # 3) daily history
        print("[3/5] Pulling daily history (this can take a while)…")
        price_frames = []
        for i in range(0, len(tickers), args.batch):
            chunk = tickers[i:i+args.batch]
            df = bb.get_history(chunk, start, end, fields=HIST_FIELDS, periodicity="DAILY")
            price_frames.append(df)
            rows_so_far = sum(len(x) for x in price_frames)
            print(f"  pulled {i+len(chunk)}/{len(tickers)} tickers; rows so far: {rows_so_far:,}")

        df_px = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
        if df_px.empty:
            raise RuntimeError(
                "Historical pull returned 0 rows. Possible causes:\n"
                "  - Missing history entitlements\n"
                "  - Date range out of bounds\n"
            )

        df_px.rename(columns={
            "PX_LAST": "px_last",
            "TOT_RETURN_INDEX_GROSS_DVDS": "tri_gross",
            "PX_VOLUME": "volume"
        }, inplace=True)
        df_px["year"] = pd.to_datetime(df_px["date"]).dt.year

        # 4) Parquet by year
        print("[4/5] Writing Parquet dataset partitioned by year…")
        (df_px
         .astype({"px_last": "float64", "tri_gross": "float64", "volume": "float64"})
         .to_parquet(f"{outdir}/prices_parquet", partition_cols=["year"], index=False))
        print(f"Parquet dataset written to {outdir}/prices_parquet/")

        # 5) CSV by year
        print("[5/5] Writing per-year CSV shards…")
        for y, grp in df_px.groupby("year"):
            grp.drop(columns=["year"]).sort_values(["date", "ticker"]).to_csv(f"{outdir}/prices_csv/prices_{y}.csv", index=False)
        print(f"Year-sharded CSVs written to {outdir}/prices_csv/")

        # Excel summary
        n_rows = len(df_px)
        n_tickers = df_px["ticker"].nunique()
        n_dates = df_px["date"].nunique()
        summary = pd.DataFrame({
            "metric": ["rows", "unique_tickers", "unique_dates", "start", "end"],
            "value": [n_rows, n_tickers, n_dates, str(df_px['date'].min()), str(df_px['date'].max())]
        })
        with pd.ExcelWriter(f"{outdir}/summary.xlsx", engine="xlsxwriter") as xw:
            summary.to_excel(xw, sheet_name="summary", index=False)
            df_const.head(1000).to_excel(xw, sheet_name="constituents_sample", index=False)
            df_px.head(1000).to_excel(xw, sheet_name="prices_sample", index=False)
        print(f"Excel summary written to {outdir}/summary.xlsx")

        print("Done. Next step: compute 12–1 momentum on tri_gross (or price-only if tri_gross unavailable).")


if __name__ == "__main__":
    main()
