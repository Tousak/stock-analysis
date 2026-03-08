import pandas as pd
from edgar import Company, set_identity
import sys
import os
from tqdm import tqdm
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

from src.config import EDGAR_IDENTITY, RAW_FILINGS_PATH
from src.data_loader import fetch_and_extract_filing_data

def re_extract_all():
    set_identity(EDGAR_IDENTITY)
    if not os.path.exists(RAW_FILINGS_PATH):
        print(f"No raw filings found at {RAW_FILINGS_PATH}")
        return

    df = pd.read_excel(RAW_FILINGS_PATH)
    
    # Process all tickers - it's fast now with skip_mda
    print(f"Loaded {len(df)} filings. Re-extracting financials for ALL tickers...")
    
    # We need the company objects to get the actual filing objects
    tickers = df['ticker'].unique()
    company_map = {t: Company(t) for t in tqdm(tickers, desc="Loading Companies")}
    
    new_rows = []
    
    for ticker, group in tqdm(df.groupby('ticker'), desc="Processing Tickers"):
        company = company_map[ticker]
        filings = company.get_filings(form="10-Q")
        acc_to_filing = {f.accession_number: f for f in filings}
        
        for _, row in group.iterrows():
            acc = row['accession_number']
            existing_mda = row.get('mda_text', "")
            
            if acc in acc_to_filing:
                filing_obj = acc_to_filing[acc]
                # Skip MD&A parsing if we already have it
                skip = bool(existing_mda and len(str(existing_mda)) > 100)
                extracted = fetch_and_extract_filing_data(ticker, filing_obj, skip_mda=skip)
                
                # Restore MD&A if it was skipped
                if skip:
                    extracted['mda_text'] = existing_mda
                
                new_rows.append(extracted)
            else:
                new_rows.append(row.to_dict())

    new_df = pd.DataFrame(new_rows)
    new_df.to_excel(RAW_FILINGS_PATH, index=False)
    print(f"Successfully re-extracted and saved {len(new_df)} filings to {RAW_FILINGS_PATH}")

if __name__ == "__main__":
    re_extract_all()
