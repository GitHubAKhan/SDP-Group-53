#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import time
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import blpapi
except ImportError:
    print("ERROR: Bloomberg API (blpapi) not installed")
    print("Install with: pip install blpapi")
    raise

INDEX_TICKERS = {
    'SPY US Equity': 'S&P 500 ETF',
    'SPX Index': 'S&P 500 Index',
    'IWM US Equity': 'Russell 2000 ETF',
    'RTY Index': 'Russell 2000 Index',
    'QQQ US Equity': 'NASDAQ 100 ETF',
    'NDX Index': 'NASDAQ 100 Index',
    'NKY Index': 'Nikkei 225',
    'DIA US Equity': 'Dow Jones ETF',
    'INDU Index': 'Dow Jones Index',
    'IWD US Equity': 'Russell 1000 Value',
    'IWF US Equity': 'Russell 1000 Growth',
    'VTI US Equity': 'Total Stock Market ETF',
}

MOMENTUM_ETFS = {
    'MTUM US Equity': 'iShares Momentum Factor',
    'PDP US Equity': 'Invesco DWA Momentum',
    'SPMO US Equity': 'Invesco S&P 500 Momentum',
}

SECTOR_ETFS = {
    'XLK US Equity': 'Technology',
    'XLF US Equity': 'Financials',
    'XLV US Equity': 'Health Care',
    'XLE US Equity': 'Energy',
    'XLI US Equity': 'Industrials',
    'XLY US Equity': 'Consumer Discretionary',
    'XLP US Equity': 'Consumer Staples',
    'XLU US Equity': 'Utilities',
    'XLRE US Equity': 'Real Estate',
    'XLC US Equity': 'Communication Services',
    'XLB US Equity': 'Materials',
}

BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 2

class BloombergDataFetcher:
    def __init__(self):
        self.session = None
        self.refDataService = None

    def start_session(self):
        print("Starting Bloomberg session...")
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost("localhost")
        sessionOptions.setServerPort(8194)
        self.session = blpapi.Session(sessionOptions)

        if not self.session.start():
            raise Exception("Failed to start Bloomberg session. Is Bloomberg Terminal running?")
        if not self.session.openService("//blp/refdata"):
            raise Exception("Failed to open //blp/refdata service")

        self.refDataService = self.session.getService("//blp/refdata")
        print("Bloomberg session started successfully")

    def stop_session(self):
        if self.session:
            self.session.stop()
            print("Bloomberg session stopped")

    def fetch_historical_data(self, tickers: List[str], start_date: str, end_date: str,
                              fields: List[str] = None) -> pd.DataFrame:
        if fields is None:
            fields = ['PX_LAST', 'TOT_RETURN_INDEX_GROSS', 'PX_VOLUME']

        start_date_bbg = start_date.replace('-', '')
        end_date_bbg = end_date.replace('-', '')

        request = self.refDataService.createRequest("HistoricalDataRequest")
        for ticker in tickers:
            request.append("securities", ticker)
        for field in fields:
            request.append("fields", field)

        request.set("startDate", start_date_bbg)
        request.set("endDate", end_date_bbg)
        request.set("periodicitySelection", "DAILY")
        self.session.sendRequest(request)

        all_data = []
        while True:
            event = self.session.nextEvent(500)
            if event.eventType() == blpapi.Event.RESPONSE or \
               event.eventType() == blpapi.Event.PARTIAL_RESPONSE:

                for msg in event:
                    securityData = msg.getElement("securityData")
                    ticker = securityData.getElementAsString("security")

                    if securityData.hasElement("securityError"):
                        error = securityData.getElement("securityError")
                        print(f"   Error for {ticker}: {error.getElementAsString('message')}")
                        continue

                    fieldData = securityData.getElement("fieldData")
                    ticker_data = []
                    for i in range(fieldData.numValues()):
                        fields_dict = {}
                        field_element = fieldData.getValueAsElement(i)

                        if field_element.hasElement("date"):
                            date_val = field_element.getElementAsDatetime("date")
                            fields_dict['date'] = pd.Timestamp(date_val.year, date_val.month, date_val.day)
                        if field_element.hasElement("PX_LAST"):
                            fields_dict['px_last'] = field_element.getElementAsFloat("PX_LAST")
                        if field_element.hasElement("TOT_RETURN_INDEX_GROSS"):
                            fields_dict['tri_gross'] = field_element.getElementAsFloat("TOT_RETURN_INDEX_GROSS")
                        if field_element.hasElement("PX_VOLUME"):
                            fields_dict['volume'] = field_element.getElementAsFloat("PX_VOLUME")

                        fields_dict['ticker'] = ticker
                        ticker_data.append(fields_dict)

                    if ticker_data:
                        all_data.extend(ticker_data)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        if all_data:
            df = pd.DataFrame(all_data)
            cols = ['date', 'ticker', 'px_last', 'tri_gross', 'volume']
            df = df[[col for col in cols if col in df.columns]]

            if 'tri_gross' not in df.columns or df['tri_gross'].isna().all():
                df['tri_gross'] = df['px_last']
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0)
            return df
        else:
            return pd.DataFrame()

    def _convert_ticker_format(self, ticker: str) -> str:
        if not ticker:
            return None
        parts = ticker.split()
        if len(parts) >= 2:
            symbol = parts[0]
            exchange_code = parts[1]
            if exchange_code.startswith('U'):
                return f"{symbol} US Equity"
            elif exchange_code in ['LN', 'LI']:
                return f"{symbol} LN Equity"
            elif exchange_code in ['JP', 'JT']:
                return f"{symbol} JP Equity"
            else:
                return f"{symbol} US Equity"
        return ticker

    def fetch_index_members(self, index_ticker: str, date: str = None) -> List[str]:
        print(f"  Attempting to fetch members for {index_ticker}...")
        request = self.refDataService.createRequest("ReferenceDataRequest")
        request.append("securities", index_ticker)
        request.append("fields", "INDX_MWEIGHT")

        if date:
            overrides = request.getElement("overrides")
            override1 = overrides.appendElement()
            override1.setElement("fieldId", "END_DATE_OVERRIDE")
            override1.setElement("value", date)

        self.session.sendRequest(request)
        members = []

        while True:
            event = self.session.nextEvent(500)
            if event.eventType() == blpapi.Event.RESPONSE or \
               event.eventType() == blpapi.Event.PARTIAL_RESPONSE:

                for msg in event:
                    securityDataArray = msg.getElement("securityData")
                    for i in range(securityDataArray.numValues()):
                        securityData = securityDataArray.getValueAsElement(i)

                        if securityData.hasElement("securityError"):
                            error = securityData.getElement("securityError")
                            print(f"   Bloomberg API Error for {index_ticker}: {error.getElementAsString('message')}")
                            continue

                        if securityData.hasElement("fieldData"):
                            fieldData = securityData.getElement("fieldData")
                            if fieldData.hasElement("INDX_MWEIGHT"):
                                mweightArray = fieldData.getElement("INDX_MWEIGHT")
                                for j in range(mweightArray.numValues()):
                                    memberElement = mweightArray.getValueAsElement(j)
                                    if memberElement.hasElement("Member Ticker and Exchange Code"):
                                        ticker = memberElement.getElementAsString("Member Ticker and Exchange Code")
                                        if ticker:
                                            ticker_converted = self._convert_ticker_format(ticker)
                                            if ticker_converted and ticker_converted not in members:
                                                members.append(ticker_converted)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        if not members:
            print(f"   No members found with INDX_MWEIGHT, trying INDX_MEMBERS3...")
            members = self._fetch_index_members_alternative(index_ticker, date)

        if members:
            print(f"  ✓ Successfully fetched {len(members)} members from {index_ticker}")
        else:
            print(f"  ✗ Could not fetch members from {index_ticker}")
        return members

    def _fetch_index_members_alternative(self, index_ticker: str, date: str = None) -> List[str]:
        request = self.refDataService.createRequest("ReferenceDataRequest")
        request.append("securities", index_ticker)
        request.append("fields", "INDX_MEMBERS3")

        if date:
            overrides = request.getElement("overrides")
            override1 = overrides.appendElement()
            override1.setElement("fieldId", "END_DATE_OVERRIDE")
            override1.setElement("value", date)

        self.session.sendRequest(request)
        members = []

        while True:
            event = self.session.nextEvent(500)
            if event.eventType() == blpapi.Event.RESPONSE or \
               event.eventType() == blpapi.Event.PARTIAL_RESPONSE:

                for msg in event:
                    securityDataArray = msg.getElement("securityData")
                    for i in range(securityDataArray.numValues()):
                        securityData = securityDataArray.getValueAsElement(i)
                        if securityData.hasElement("fieldData"):
                            fieldData = securityData.getElement("fieldData")
                            if fieldData.hasElement("INDX_MEMBERS3"):
                                membersArray = fieldData.getElement("INDX_MEMBERS3")
                                for j in range(membersArray.numValues()):
                                    try:
                                        memberElement = membersArray.getValueAsElement(j)
                                        ticker = None
                                        for field_name in ["Member Ticker and Exchange Code", "Security", "Ticker"]:
                                            if memberElement.hasElement(field_name):
                                                ticker = memberElement.getElementAsString(field_name)
                                                break
                                        if ticker:
                                            ticker_converted = self._convert_ticker_format(ticker)
                                            if ticker_converted and ticker_converted not in members:
                                                members.append(ticker_converted)
                                    except Exception as e:
                                        continue

            if event.eventType() == blpapi.Event.RESPONSE:
                break
        return members


def get_nasdaq_tickers_bloomberg(fetcher: BloombergDataFetcher, use_index: bool = True) -> List[str]:
    if use_index:
        print("  Fetching NASDAQ 100 members from Bloomberg...")
        try:
            members = fetcher.fetch_index_members('NDX Index')
            if members and len(members) > 50:
                print(f"  ✓ Found {len(members)} NASDAQ 100 members")
                return members
            else:
                print(f"  ⚠ Only found {len(members)} members, falling back to hardcoded list...")
        except Exception as e:
            print(f"  Error fetching index members: {e}")
            print("  Falling back to hardcoded list...")

    major_nasdaq = [
        'AAPL US Equity', 'MSFT US Equity', 'AMZN US Equity', 'NVDA US Equity',
        'META US Equity', 'GOOGL US Equity', 'GOOG US Equity', 'TSLA US Equity',
        'AVGO US Equity', 'COST US Equity', 'NFLX US Equity', 'AMD US Equity',
        'PEP US Equity', 'ADBE US Equity', 'CSCO US Equity', 'TMUS US Equity',
        'INTC US Equity', 'CMCSA US Equity', 'TXN US Equity', 'QCOM US Equity',
        'AMGN US Equity', 'HON US Equity', 'INTU US Equity', 'AMAT US Equity',
        'SBUX US Equity', 'ISRG US Equity', 'BKNG US Equity', 'ADP US Equity',
        'GILD US Equity', 'ADI US Equity', 'VRTX US Equity', 'REGN US Equity',
        'MDLZ US Equity', 'MU US Equity', 'LRCX US Equity', 'PANW US Equity',
        'KLAC US Equity', 'MELI US Equity', 'PYPL US Equity', 'SNPS US Equity',
        'ASML US Equity', 'CDNS US Equity', 'ORLY US Equity', 'CHTR US Equity',
        'ABNB US Equity', 'CTAS US Equity', 'MAR US Equity', 'NXPI US Equity',
        'MRNA US Equity', 'FTNT US Equity', 'MRVL US Equity', 'CSX US Equity',
    ]
    return major_nasdaq


def get_russell_2000_tickers_bloomberg(fetcher: BloombergDataFetcher, use_index: bool = True,
                                       limit: Optional[int] = None) -> List[str]:
    if use_index:
        print("  Fetching Russell 2000 members from Bloomberg...")
        try:
            members = fetcher.fetch_index_members('RTY Index')
            if members and len(members) > 100:
                print(f"  ✓ Found {len(members)} Russell 2000 members")
                if limit:
                    print(f"  Limiting to {limit} tickers")
                    return members[:limit]
                return members
            else:
                print(f"  ⚠ Only found {len(members)} members, falling back to sample list...")
        except Exception as e:
            print(f"  Error fetching index members: {e}")
            print("  Falling back to sample list...")

    russell_2000_sample = [
        'SAIA US Equity', 'EXLS US Equity', 'KNSL US Equity', 'MTSI US Equity',
        'AEIS US Equity', 'CEIX US Equity', 'FIVE US Equity', 'PRIM US Equity',
        'INSM US Equity', 'CRVL US Equity', 'CASY US Equity', 'OMCL US Equity',
        'ENSG US Equity', 'LFUS US Equity', 'SITM US Equity', 'ALKS US Equity',
        'PFSI US Equity', 'CTRE US Equity', 'EXPO US Equity', 'CBZ US Equity',
        'CALM US Equity', 'SANM US Equity', 'HQY US Equity', 'CADE US Equity',
        'BOOT US Equity', 'NEOG US Equity', 'UFPI US Equity', 'GTLS US Equity',
        'CVCO US Equity', 'NOVT US Equity', 'BCC US Equity', 'SHAK US Equity',
    ]
    if limit:
        return russell_2000_sample[:limit]
    return russell_2000_sample


def fetch_batch_data_bloomberg(fetcher: BloombergDataFetcher, tickers: List[str],
                               start_date: str, end_date: str, batch_name: str = "Batch") -> pd.DataFrame:
    print(f"\nFetching {batch_name} ({len(tickers)} tickers)...")
    all_data = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches}: Processing {len(batch)} tickers...")

        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                df = fetcher.fetch_historical_data(batch, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
                    print(f"  ✓ Fetched {len(df['ticker'].unique())} tickers, {len(df)} rows")
                else:
                    print(f"  ⚠ No data returned for batch")
                break
            except Exception as e:
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    print(f"  ⚠ Error (attempt {retry_count}/{MAX_RETRIES}): {str(e)}")
                    print(f"  Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  ✗ Failed after {MAX_RETRIES} attempts: {str(e)}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"  ✓ Completed: {combined['ticker'].nunique()} tickers, {len(combined)} total rows")
        return combined
    else:
        print(f"  ✗ No data fetched for {batch_name}")
        return pd.DataFrame()


def save_data_by_year(df: pd.DataFrame, output_dir: str, data_type: str = "prices"):
    if df.empty:
        print(f"  No data to save for {data_type}")
        return

    parquet_dir = os.path.join(output_dir, f"{data_type}_parquet_v2")
    os.makedirs(parquet_dir, exist_ok=True)
    df['year'] = pd.to_datetime(df['date']).dt.year

    for year, year_df in df.groupby('year'):
        year_df_clean = year_df.drop('year', axis=1)
        output_path = os.path.join(parquet_dir, f"year={year}")
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "data.parquet")
        year_df_clean.to_parquet(file_path, index=False)
        print(f"  Saved {len(year_df_clean)} rows for year {year}")

    single_file = os.path.join(output_dir, f"{data_type}_all_v2.parquet")
    df.drop('year', axis=1).to_parquet(single_file, index=False)
    print(f"  Saved combined file: {single_file}")


def create_summary_report(df: pd.DataFrame, output_dir: str):
    if df.empty:
        return

    report = []
    report.append("=" * 80)
    report.append("BLOOMBERG MARKET DATA FETCH SUMMARY")
    report.append("=" * 80)
    report.append(f"\nDate Range: {df['date'].min().date()} to {df['date'].max().date()}")
    report.append(f"Total Tickers: {df['ticker'].nunique()}")
    report.append(f"Total Records: {len(df):,}")
    report.append(f"\nTickers by Type:")

    index_count = df[df['ticker'].str.contains(' Index', regex=False)]['ticker'].nunique()
    etf_count = df[df['ticker'].str.contains('SPY|IWM|QQQ|DIA|IWF|IWD|VTI|MTUM|PDP|SPMO|XL', regex=True)]['ticker'].nunique()
    equity_count = df[df['ticker'].str.contains(' US Equity', regex=False)]['ticker'].nunique() - etf_count

    report.append(f"  Indices: {index_count}")
    report.append(f"  ETFs: {etf_count}")
    report.append(f"  Individual Equities: {equity_count}")
    report.append(f"\nData Coverage:")
    report.append(f"  Average records per ticker: {len(df) / df['ticker'].nunique():.0f}")
    report.append(f"\nSample Tickers (first 20):")
    
    for ticker in sorted(df['ticker'].unique())[:20]:
        ticker_data = df[df['ticker'] == ticker]
        report.append(f"  {ticker}: {len(ticker_data)} records ({ticker_data['date'].min().date()} to {ticker_data['date'].max().date()})")

    report.append("\n" + "=" * 80)
    report_path = os.path.join(output_dir, "fetch_summary_v2.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print('\n'.join(report))


def fetch_all_data_bloomberg(start_date: str, end_date: str, output_dir: str,
                             include_nasdaq: bool = True, include_russell: bool = True,
                             nasdaq_limit: Optional[int] = None, russell_limit: Optional[int] = None,
                             fetch_from_index: bool = True):
    print("\n" + "=" * 80)
    print("BLOOMBERG MARKET DATA FETCHER")
    print("=" * 80)
    print(f"\nDate Range: {start_date} to {end_date}")
    print(f"Output Directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    fetcher = BloombergDataFetcher()

    try:
        fetcher.start_session()
        all_data = []

        print("\n" + "=" * 80)
        print("1. FETCHING INDICES AND INDEX ETFs")
        print("=" * 80)
        index_tickers = list(INDEX_TICKERS.keys())
        index_data = fetch_batch_data_bloomberg(fetcher, index_tickers, start_date, end_date, "Indices")
        if not index_data.empty:
            all_data.append(index_data)

        print("\n" + "=" * 80)
        print("2. FETCHING MOMENTUM ETFs")
        print("=" * 80)
        momentum_tickers = list(MOMENTUM_ETFS.keys())
        momentum_data = fetch_batch_data_bloomberg(fetcher, momentum_tickers, start_date, end_date, "Momentum ETFs")
        if not momentum_data.empty:
            all_data.append(momentum_data)

        print("\n" + "=" * 80)
        print("3. FETCHING SECTOR ETFs")
        print("=" * 80)
        sector_tickers = list(SECTOR_ETFS.keys())
        sector_data = fetch_batch_data_bloomberg(fetcher, sector_tickers, start_date, end_date, "Sector ETFs")
        if not sector_data.empty:
            all_data.append(sector_data)

        if include_nasdaq:
            print("\n" + "=" * 80)
            print("4. FETCHING NASDAQ STOCKS")
            print("=" * 80)
            nasdaq_tickers = get_nasdaq_tickers_bloomberg(fetcher, use_index=fetch_from_index)
            if nasdaq_limit and len(nasdaq_tickers) > nasdaq_limit:
                nasdaq_tickers = nasdaq_tickers[:nasdaq_limit]
                print(f"  Limited to {nasdaq_limit} NASDAQ tickers")
            nasdaq_data = fetch_batch_data_bloomberg(fetcher, nasdaq_tickers, start_date, end_date, "NASDAQ Stocks")
            if not nasdaq_data.empty:
                all_data.append(nasdaq_data)

        if include_russell:
            print("\n" + "=" * 80)
            print("5. FETCHING RUSSELL 2000 STOCKS")
            print("=" * 80)
            russell_tickers = get_russell_2000_tickers_bloomberg(fetcher, use_index=fetch_from_index, limit=russell_limit)
            russell_data = fetch_batch_data_bloomberg(fetcher, russell_tickers, start_date, end_date, "Russell 2000 Stocks")
            if not russell_data.empty:
                all_data.append(russell_data)

        if all_data:
            print("\n" + "=" * 80)
            print("COMBINING AND SAVING DATA")
            print("=" * 80)
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
            combined_df = combined_df.drop_duplicates(subset=['date', 'ticker'], keep='first')
            save_data_by_year(combined_df, output_dir, data_type="prices")
            create_summary_report(combined_df, output_dir)

            print("\n" + "=" * 80)
            print("DATA FETCH COMPLETE!")
            print("=" * 80)
            print(f"\nTotal tickers fetched: {combined_df['ticker'].nunique()}")
            print(f"Total records: {len(combined_df):,}")
            print(f"Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
            print(f"\nData saved to: {output_dir}")
            return combined_df
        else:
            print("\n" + "=" * 80)
            print("ERROR: No data was fetched successfully.")
            print("=" * 80)
            return pd.DataFrame()

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        raise
    finally:
        fetcher.stop_session()


def main():
    parser = argparse.ArgumentParser(description='Fetch market data from Bloomberg Terminal')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data/data', help='Output directory')
    parser.add_argument('--no-nasdaq', action='store_true', help='Skip NASDAQ stocks')
    parser.add_argument('--no-russell', action='store_true', help='Skip Russell 2000 stocks')
    parser.add_argument('--nasdaq-limit', type=int, default=None, help='Limit NASDAQ stocks')
    parser.add_argument('--russell-limit', type=int, default=None, help='Limit Russell 2000 stocks')
    parser.add_argument('--no-fetch-from-index', action='store_true', help='Use hardcoded lists')
    args = parser.parse_args()

    try:
        start_dt = pd.to_datetime(args.start)
        end_dt = pd.to_datetime(args.end)
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")
    except Exception as e:
        print(f"Error: Invalid date format. {str(e)}")
        return

    try:
        fetch_all_data_bloomberg(
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output_dir,
            include_nasdaq=not args.no_nasdaq,
            include_russell=not args.no_russell,
            nasdaq_limit=args.nasdaq_limit,
            russell_limit=args.russell_limit,
            fetch_from_index=not args.no_fetch_from_index
        )
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        print("\nPlease ensure:")
        print("  1. Bloomberg Terminal is running")
        print("  2. You have an active Bloomberg subscription")
        print("  3. blpapi package is installed (pip install blpapi)")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())