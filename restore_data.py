import pandas as pd
import re
from edgar import set_identity, Company
from tqdm.auto import tqdm
import os
import yfinance as yf

# Set identity
set_identity("Jan Tous honza.tous@seznam.com")

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", 
           "LLY", "AVGO", "WMT", "JPM", "V", "XOM", "JNJ", "MA", "COST", "MU", 
           "ORCL", "BAC", "ABBV", "HD", "PG", "CVX", "NFLX", "KO", "CAT", "AMD", 
           "GE", "CSCO", "PLTR", "MRK", "WFC", "LRCX", "MS", "PM", "IBM", "GS",
            "RTX", "AMAT", "INTC", "UNH", "AXP", "PEP", "MCD", "TMUS", "C", 
            "GEV", "LIN", "AMGN"]
NUM_QUARTERS = 8
RAW_FILINGS_PATH = 'data/fetched/raw_filings.xlsx'
MDNA_REGEX_PATTERN = r'(Item\s+2[.:]?\s+.*?(?:Management.*?Discussion.*?Analysis).*?)(?=Item\s+3[.:]?|Item\s+4[.:]?|PART\s+II|Signatures)'

def extract_mda(text):
    if not text: return ""
    pattern = re.compile(MDNA_REGEX_PATTERN, re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(text)
    if not matches: return ""
    return max(matches, key=len).strip()

def restore_data():
    all_data = []
    
    for ticker in tqdm(TICKERS, desc="Restoring Tickers"):
        try:
            company = Company(ticker)
            filings = company.get_filings(form="10-Q")
            if not filings: continue
            
            # Fetch num_quarters
            meta_df = filings.to_pandas()
            target_indices = meta_df.head(NUM_QUARTERS).index
            
            for idx in target_indices:
                f = filings[int(idx)]
                tenq = f.obj()
                revenue, net_income = None, None
                if tenq and hasattr(tenq, 'financials') and tenq.financials:
                    revenue = tenq.financials.get_revenue()
                    net_income = tenq.financials.get_net_income()
                
                md_text = f.markdown()
                mda = extract_mda(md_text)
                
                all_data.append({
                    'ticker': ticker,
                    'filing_date': f.filing_date,
                    'revenue': revenue,
                    'net_income': net_income,
                    'mda_text': mda,
                    'accession_number': f.accession_number
                })
        except Exception as e:
            print(f"Error restoring {ticker}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_excel(RAW_FILINGS_PATH, index=False)
        print(f"Restoration finished. Saved {len(df)} filings.")

if __name__ == "__main__":
    restore_data()
