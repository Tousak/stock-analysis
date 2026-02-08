import pandas as pd
from datetime import timedelta
import numpy as np
import os
from tqdm.auto import tqdm

from src.config import FEATURES_PATH, DATA_DIR
from src.data_loader import fetch_stock_prices

def calculate_financial_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates revenue growth and net margin for each ticker."""
    df['revenue_growth'] = df.groupby('ticker')['revenue'].pct_change(fill_method=None)
    df['net_margin'] = df['net_income'] / df['revenue']
    return df

def calculate_sentiment_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates sentiment change (QoQ difference) for each ticker."""
    df['sentiment_change'] = df.groupby('ticker')['sentiment_score'].diff()
    return df

def calculate_next_quarter_return(filings_df: pd.DataFrame, market_data_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the next quarter's stock return."""
    filings_df['filing_date'] = pd.to_datetime(filings_df['filing_date'])
    market_data_df['date'] = pd.to_datetime(market_data_df['date']).dt.date

    def get_price_on_or_after_date(ticker_market_data, target_date):
        target_date = target_date.date()
        price_series = ticker_market_data[ticker_market_data['date'] == target_date]['Close']
        if not price_series.empty:
            return price_series.values[0]
        future_data_series = ticker_market_data[ticker_market_data['date'] > target_date]['Close']
        if not future_data_series.empty:
            return future_data_series.values[0]
        return np.nan

    returns = []
    market_data_grouped = market_data_df.groupby('ticker')
    
    for _, row in tqdm(filings_df.iterrows(), total=filings_df.shape[0], desc="Calculating Returns"):
        ticker = row['ticker']
        filing_date = row['filing_date']
        
        try:
            current_ticker_market_data = market_data_grouped.get_group(ticker)
        except KeyError:
            returns.append(np.nan)
            continue

        price_on_filing_date = get_price_on_or_after_date(current_ticker_market_data, filing_date)
        future_date = filing_date + timedelta(days=90)
        price_in_90_days = get_price_on_or_after_date(current_ticker_market_data, future_date)

        if pd.notna(price_on_filing_date) and pd.notna(price_in_90_days) and price_on_filing_date != 0:
            returns.append((price_in_90_days - price_on_filing_date) / price_on_filing_date)
        else:
            returns.append(np.nan)
            
    filings_df['next_quarter_return'] = returns
    return filings_df

def engineer_features(processed_filings_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates feature engineering, replicating the notebook's aggressive cleaning."""
    df = processed_filings_df.copy()
    
    # --- Notebook-style Aggressive Cleaning ---
    print(f"Original number of companies: {df['ticker'].nunique()}")
    # 1. Filter out companies with any missing revenue/income or empty mda_text
    # Replace empty strings with NaN to be dropped
    df['mda_text'].replace('', np.nan, inplace=True)
    
    # Filter groups (tickers) that have no nulls in these critical columns
    df_clean = df.groupby('ticker').filter(lambda x: x[['revenue', 'net_income', 'mda_text']].notna().all().all())
    
    print(f"Companies remaining after aggressive cleaning: {df_clean['ticker'].nunique()}")
    print(f"Remaining companies: {df_clean['ticker'].unique().tolist()}")
    
    if df_clean.empty:
        print("No companies remaining after aggressive data cleaning. Aborting.")
        return pd.DataFrame()
    
    df = df_clean.copy()
    # --- End Cleaning ---

    df.sort_values(by=['ticker', 'filing_date'], inplace=True)
    df = calculate_financial_growth(df)
    df = calculate_sentiment_change(df)

    if df.empty or df['filing_date'].isna().all():
        print("No valid data remaining. Skipping return calculation.")
        return pd.DataFrame()
        
    min_market_date = df['filing_date'].min() - timedelta(days=120)
    max_market_date = df['filing_date'].max() + timedelta(days=120)
    
    all_market_data = []
    print("Fetching market data for remaining tickers...")
    for ticker in tqdm(df['ticker'].unique(), desc="Fetching Market Data"):
        market_df = fetch_stock_prices(
            ticker, 
            start_date=min_market_date.strftime('%Y-%m-%d'), 
            end_date=max_market_date.strftime('%Y-%m-%d')
        )
        if not market_df.empty:
            market_df['ticker'] = ticker
            all_market_data.append(market_df)
    
    if not all_market_data:
        print("CRITICAL: No market data fetched. Cannot calculate returns.")
        return pd.DataFrame()
    
    full_market_data_df = pd.concat(all_market_data, ignore_index=True)
    full_market_data_df.drop_duplicates(subset=['ticker', 'date'], keep='first', inplace=True)
    df = calculate_next_quarter_return(df, full_market_data_df)
    
    # Final NaN drop for features that couldn't be calculated (e.g., first pct_change row)
    df.dropna(subset=['revenue_growth', 'net_margin', 'sentiment_change', 'next_quarter_return'], inplace=True)

    final_feature_columns = [
        'ticker', 'filing_date', 'revenue', 'net_income', 'sentiment_score',
        'revenue_growth', 'net_margin', 'sentiment_change', 'next_quarter_return'
    ]
    # Ensure columns exist before selection
    for col in final_feature_columns:
        if col not in df.columns:
            df[col] = np.nan

    final_features_df = df[final_feature_columns].copy()
    
    if final_features_df.empty:
        print("Warning: Feature engineering resulted in an empty DataFrame after all cleaning steps.")
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        final_features_df.to_excel(FEATURES_PATH, index=False)
        print(f"Engineered features saved to {FEATURES_PATH}")
    
    return final_features_df
