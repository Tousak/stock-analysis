from edgar import set_identity, Company
import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from src.config import EDGAR_IDENTITY

def debug_cat():
    set_identity(EDGAR_IDENTITY)
    print(f"Using identity: {EDGAR_IDENTITY}")
    company = Company('CAT')
    filings = company.get_filings(form='10-Q')
    f = filings[0]
    print(f"Latest Filing Date: {f.filing_date}")
    
    obj = f.obj()
    if obj.financials and obj.financials.income_statement:
        df = obj.financials.income_statement().to_dataframe()
        print("\nIncome Statement Columns:")
        print(df.columns.tolist())
        print("\nIncome Statement Rows:")
        print(df.head(10))
    
    from src.data_loader import fetch_and_extract_filing_data
    data = fetch_and_extract_filing_data('CAT', f)
    print("\nExtracted Data:")
    print(f"Revenue: {data['revenue']}")
    print(f"Net Income: {data['net_income']}")
    print(f"MDA Length: {len(data['mda_text'])}")

if __name__ == "__main__":
    debug_cat()
