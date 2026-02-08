import pandas as pd
import yfinance as yf
from edgar import set_identity, Company
from tqdm.auto import tqdm
import os
import re

from src.config import EDGAR_IDENTITY, NUM_QUARTERS_TO_FETCH, TICKERS, RAW_FILINGS_PATH, DATA_DIR, MDNA_REGEX_PATTERN
from src.blacklist import get_blacklist

# Set EDGAR identity for edgar-tools
set_identity(EDGAR_IDENTITY)

def fetch_and_extract_filing_data(ticker: str, filing: object) -> dict:
    """Extracts financials and MD&A text from a single filing object."""
    revenue, net_income = None, None
    tenq = filing.obj()
    if tenq and hasattr(tenq, 'financials') and tenq.financials:
        revenue = tenq.financials.get_revenue()
        net_income = tenq.financials.get_net_income()

    mda_text = ""
    full_text = filing.markdown()
    if full_text:
        pattern = re.compile(MDNA_REGEX_PATTERN, re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(full_text)
        if matches:
            mda_text = max(matches, key=len).strip()

    return {
        'ticker': ticker,
        'filing_date': pd.to_datetime(filing.filing_date),
        'revenue': revenue,
        'net_income': net_income,
        'mda_text': mda_text,
        'accession_number': filing.accession_number
    }

def fetch_stock_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical daily close prices for a given ticker, flattening columns if needed."""
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty: 
        print(f"Warning: No yfinance data for {ticker} from {start_date} to {end_date}")
        return pd.DataFrame()

    # Flatten the columns if they are a MultiIndex (e.g., ('Close', 'MSFT'))
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    data = data[['Close']].reset_index()
    data.rename(columns={'Date': 'date'}, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    return data

def load_all_raw_data(tickers: list = TICKERS, num_quarters: int = NUM_QUARTERS_TO_FETCH) -> pd.DataFrame:
    """Orchestrates fetching raw data with intelligent caching and no error suppression."""
    blacklist = get_blacklist()
    filtered_tickers = [t.upper() for t in tickers if t.upper() not in blacklist]
    print(f"Starting fetch for {len(filtered_tickers)} tickers (after removing {len(tickers) - len(filtered_tickers)} blacklisted).")

    existing_df = pd.DataFrame()
    if os.path.exists(RAW_FILINGS_PATH):
        existing_df = pd.read_excel(RAW_FILINGS_PATH)
        if 'filing_date' in existing_df.columns:
             existing_df['filing_date'] = pd.to_datetime(existing_df['filing_date'])
        print(f"Loaded {len(existing_df)} cached filings from {RAW_FILINGS_PATH}")

    all_new_filings_data = []

    for ticker in tqdm(filtered_tickers, desc="Checking tickers"):
        try:
            company = Company(ticker)
            filings = company.get_filings(form="10-Q")
            if not filings:
                print(f"No 10-Q filings found for {ticker}.")
                continue

            meta_df = filings.to_pandas()
            target_indices = meta_df.head(num_quarters).index

            cached_accession_numbers = set()
            if not existing_df.empty and ticker in existing_df['ticker'].values:
                cached_accession_numbers = set(existing_df[existing_df['ticker'] == ticker]['accession_number'])

            filings_to_process = [
                filings[int(idx)] for idx in target_indices 
                if filings[int(idx)].accession_number not in cached_accession_numbers
            ]

            if not filings_to_process:
                print(f"No new filings to fetch for {ticker}.")
                continue

            print(f"Fetching {len(filings_to_process)} new filing(s) for {ticker}...")
            for filing in tqdm(filings_to_process, desc=f"Downloading {ticker}", leave=False):
                filing_data = fetch_and_extract_filing_data(ticker, filing)
                all_new_filings_data.append(filing_data)
        except Exception as e:
            print(f"An error occurred while processing ticker {ticker}: {e}")


    if all_new_filings_data:
        new_data_df = pd.DataFrame(all_new_filings_data)
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['ticker', 'accession_number'], keep='last', inplace=True)
    else:
        print("No new filings found for any ticker.")
        combined_df = existing_df

    if not combined_df.empty:
        combined_df.sort_values(by=['ticker', 'filing_date'], ascending=[True, False], inplace=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        combined_df.to_excel(RAW_FILINGS_PATH, index=False)
        print(f"Updated raw filings data saved to {RAW_FILINGS_PATH} ({len(combined_df)} total filings).")
    
    return combined_df


