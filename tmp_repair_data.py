import pandas as pd
import re
from edgar import set_identity, Company
from tqdm.auto import tqdm
import os
import numpy as np

# Set identity
set_identity("Jan Tous honza.tous@seznam.com")

RAW_FILINGS_PATH = 'data/fetched/raw_filings.xlsx'
MDNA_REGEX_PATTERN = r'(Item\s+2[.:]?\s+.*?(?:Management.*?Discussion.*?Analysis).*?)(?=Item\s+3[.:]?|Item\s+4[.:]?|PART\s+II|Signatures)'

def extract_mda(text):
    if not text: return ""
    pattern = re.compile(MDNA_REGEX_PATTERN, re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(text)
    if not matches: return ""
    return max(matches, key=len).strip()

def repair_data():
    if not os.path.exists(RAW_FILINGS_PATH):
        print("Raw filings path not found.")
        return

    df = pd.read_excel(RAW_FILINGS_PATH)
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    
    # Get indices of the latest filing for each ticker
    latest_indices = df.sort_values('filing_date').groupby('ticker').tail(1).index
    
    print(f"Aggressively repairing {len(latest_indices)} latest filings...")
    
    for idx in tqdm(latest_indices):
        ticker = df.at[idx, 'ticker']
        acc_num = df.at[idx, 'accession_number']
        try:
            company = Company(ticker)
            filings = company.get_filings(form="10-Q")
            # Find the specific filing (it might not be the absolute latest on SEC if local is old, but usually is)
            for f in filings:
                if f.accession_number == acc_num:
                    md_text = f.markdown()
                    mda = extract_mda(md_text)
                    if mda:
                        df.at[idx, 'mda_text'] = mda
                        print(f"Fixed {ticker} ({len(mda)} chars)")
                    else:
                        print(f"Warning: Regex still failed for {ticker}")
                    break
        except Exception as e:
            print(f"Error repairing {ticker}: {e}")

    df.to_excel(RAW_FILINGS_PATH, index=False)
    print(f"Aggressive repair finished.")

if __name__ == "__main__":
    repair_data()
