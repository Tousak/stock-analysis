import pandas as pd
import os

RAW_FILINGS_PATH = 'data/fetched/raw_filings.xlsx'
PROCESSED_FILINGS_PATH = 'data/fetched/processed_filings.xlsx'

def repair():
    if not os.path.exists(RAW_FILINGS_PATH) or not os.path.exists(PROCESSED_FILINGS_PATH):
        print("Missing files.")
        return

    raw_df = pd.read_excel(RAW_FILINGS_PATH)
    processed_df = pd.read_excel(PROCESSED_FILINGS_PATH)
    
    print(f"Original raw: {len(raw_df)} filings.")
    
    # Create a mapping of accession_number -> mda_text from processed data
    mda_map = processed_df.set_index('accession_number')['mda_text'].to_dict()
    
    # Fill in mda_text in raw if missing
    def fill_mda(row):
        if pd.isna(row['mda_text']) or str(row['mda_text']).strip() == "":
            return mda_map.get(row['accession_number'], "")
        return row['mda_text']
        
    raw_df['mda_text'] = raw_df.apply(fill_mda, axis=1)
    
    # Also ensure revenue/net_income aren't NaN if they exist in processed
    # (though fetch should have gotten them)
    rev_map = processed_df.set_index('accession_number')['revenue'].to_dict()
    val_map = processed_df.set_index('accession_number')['net_income'].to_dict()
    
    def fill_rev(row):
        if pd.isna(row['revenue']) or row['revenue'] == 0:
            return rev_map.get(row['accession_number'], row['revenue'])
        return row['revenue']
        
    raw_df['revenue'] = raw_df.apply(fill_rev, axis=1)
    
    raw_df.to_excel(RAW_FILINGS_PATH, index=False)
    print(f"Repaired raw filings saved with {len(raw_df)} rows.")

if __name__ == "__main__":
    repair()
