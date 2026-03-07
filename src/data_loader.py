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

def get_current_stock_prices(tickers: list) -> pd.Series:
    """Fetches the latest daily close prices for a list of tickers."""
    if not tickers:
        return pd.Series(dtype='float64')
    
    # yfinance.download can return MultiIndex for multiple tickers, simplify to get last Close price
    # Use period="1d" to get the latest day's data, and interval="1m" for intraday if needed,
    # but for daily close, "1d" is sufficient and less prone to API limits.
    data = yf.download(tickers, period="5d", interval="1d", progress=False)

    if data.empty:
        return pd.Series(dtype='float64')

    latest_close_prices = pd.Series(dtype='float64')

    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers, MultiIndex columns ('Close', 'Ticker')
        for ticker in tickers:
            if ('Close', ticker) in data.columns:
                latest_close_prices[ticker] = data['Close'][ticker].iloc[-1]
    else:
        # Single ticker, or yfinance flattened columns
        if 'Close' in data.columns:
            latest_close_prices[tickers[0]] = data['Close'].iloc[-1]
        elif len(tickers) == 1 and tickers[0] in data.columns: # Sometimes yfinance returns column directly for single ticker
            latest_close_prices[tickers[0]] = data[tickers[0]].iloc[-1]
    
    return latest_close_prices

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
    """
    Orchestrates fetching raw data. For each ticker, it fetches the num_quarters most recent
    10-Q filings, prioritizing up-to-date data over strict caching.
    """
    blacklist = get_blacklist()
    filtered_tickers = [t.upper() for t in tickers if t.upper() not in blacklist]
    print(f"Starting fetch for {len(filtered_tickers)} tickers (after removing {len(tickers) - len(filtered_tickers)} blacklisted).")

    all_fetched_filings_data = []

    for ticker in tqdm(filtered_tickers, desc="Fetching latest filings per ticker"):
        try:
            company = Company(ticker)
            filings = company.get_filings(form="10-Q")
            if not filings:
                print(f"No 10-Q filings found for {ticker}.")
                continue

            # Always fetch and process the 'num_quarters' most recent filings
            # This bypasses the strict caching based on accession numbers to ensure freshness.
            meta_df = filings.to_pandas()
            target_indices = meta_df.head(num_quarters).index

            print(f"Fetching {len(target_indices)} latest filing(s) for {ticker}...")
            for idx in tqdm(target_indices, desc=f"Downloading {ticker}", leave=False):
                filing_data = fetch_and_extract_filing_data(ticker, filings[int(idx)])
                all_fetched_filings_data.append(filing_data)
        except Exception as e:
            print(f"An error occurred while processing ticker {ticker}: {e}")

    if not all_fetched_filings_data:
        print("No filings were fetched for any ticker.")
        return pd.DataFrame()

    combined_df = pd.DataFrame(all_fetched_filings_data)
    combined_df.drop_duplicates(subset=['ticker', 'accession_number'], keep='last', inplace=True)
    combined_df.sort_values(by=['ticker', 'filing_date'], ascending=[True, False], inplace=True)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    combined_df.to_excel(RAW_FILINGS_PATH, index=False)
    print(f"Updated raw filings data saved to {RAW_FILINGS_PATH} ({len(combined_df)} total filings).")
    
    return combined_df


