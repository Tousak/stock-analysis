import pandas as pd
from datetime import timedelta
import numpy as np
import os
from tqdm.auto import tqdm

from src.config import DATA_DIR
from src.data_loader import fetch_stock_prices

def calculate_financial_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates revenue growth and net margin for each ticker with minor filling."""
    # Forward fill financials within ticker group to handle rare missing points
    df['revenue'] = df.groupby('ticker')['revenue'].ffill(limit=1)
    df['net_income'] = df.groupby('ticker')['net_income'].ffill(limit=1)
    
    df['revenue_growth'] = df.groupby('ticker')['revenue'].pct_change(fill_method=None)
    df['net_margin'] = df['net_income'] / df['revenue']
    return df

def calculate_sentiment_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates sentiment change (QoQ difference) for each ticker."""
    df['sentiment_change'] = df.groupby('ticker')['sentiment_score'].diff()
    
    # Also calculate changes for the triplet features if they exist
    for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu']:
        if col in df.columns:
            df[f'{col}_change'] = df.groupby('ticker')[col].diff()
            
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

def engineer_features(processed_filings_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Orchestrates feature engineering, with differentiated cleaning for training and prediction data.
    Aggressively cleans historical data for model training.
    Retains latest data points (where next_quarter_return is NaN) for prediction,
    applying more lenient cleaning to ensure they are available for forecasting.
    """
    df = processed_filings_df.copy()
    
    df['mda_text'].replace('', np.nan, inplace=True)
    df.sort_values(by=['ticker', 'filing_date'], inplace=True)

    # --- Initial Feature Calculation (before splitting for cleaning) ---
    df = calculate_financial_growth(df)
    df = calculate_sentiment_change(df)

    if df.empty or df['filing_date'].isna().all():
        print("No valid data remaining after initial processing. Skipping market data fetching.")
        return pd.DataFrame()

    # --- Fetch Market Data for all potential periods ---
    min_market_date = df['filing_date'].min() - timedelta(days=120)
    max_market_date = df['filing_date'].max() + timedelta(days=120) + timedelta(days=90) # Add buffer for prediction period
    
    all_market_data = []
    print("Fetching market data for all relevant tickers...")
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
        print("CRITICAL: No market data fetched for any ticker. Cannot calculate returns.")
        return pd.DataFrame()
    
    full_market_data_df = pd.concat(all_market_data, ignore_index=True)
    full_market_data_df.drop_duplicates(subset=['ticker', 'date'], keep='first', inplace=True)
    
    df = calculate_next_quarter_return(df, full_market_data_df)

    # --- Split into data with known returns (for training) and unknown returns (for prediction) ---
    df_has_return = df.dropna(subset=['next_quarter_return']).copy()
    df_no_return = df[df['next_quarter_return'].isna()].copy()

    final_cleaned_dfs = []

    # --- Granular Cleaning for Training Data (df_has_return) ---
    if not df_has_return.empty:
        print(f"\nApplying granular cleaning to {len(df_has_return)} historical rows...")
        # Only drop rows that are actually missing critical features
        critical_cols = ['revenue', 'net_income', 'sentiment_score', 'revenue_growth', 'net_margin', 'sentiment_change']
        df_has_return_clean = df_has_return.dropna(subset=critical_cols).copy()
        
        if not df_has_return_clean.empty:
            print(f"{len(df_has_return_clean['ticker'].unique())} tickers / {len(df_has_return_clean)} rows remaining for training.")
            final_cleaned_dfs.append(df_has_return_clean)
        else:
            print("No data remaining after granular cleaning for training data.")

    # --- Lenient Cleaning for Prediction Data (df_no_return) ---
    if not df_no_return.empty:
        print(f"\nApplying lenient cleaning to {len(df_no_return)} tickers for prediction...")
        # For prediction data, ensure sentiment features are present
        df_no_return_clean = df_no_return.dropna(subset=['sentiment_score', 'mda_text']).copy()
        
        # Ensure latest unique entries for prediction for each ticker
        df_no_return_clean = df_no_return_clean.groupby('ticker').tail(1).copy()

        if not df_no_return_clean.empty:
            print(f"{len(df_no_return_clean['ticker'].unique())} tickers remaining for prediction.")
            final_cleaned_dfs.append(df_no_return_clean)
        else:
            print("No data remaining after lenient cleaning for prediction data.")

    if not final_cleaned_dfs:
        print("Warning: Feature engineering resulted in an empty DataFrame after all cleaning steps.")
        return pd.DataFrame()

    final_features_df = pd.concat(final_cleaned_dfs, ignore_index=True)

    # --- Final Column Selection and Save ---
    final_feature_columns = [
        'ticker', 'filing_date', 'revenue', 'net_income', 'sentiment_score',
        'revenue_growth', 'net_margin', 'sentiment_change', 'next_quarter_return',
        'sentiment_pos', 'sentiment_neg', 'sentiment_neu',
        'sentiment_pos_change', 'sentiment_neg_change', 'sentiment_neu_change'
    ]
    # Ensure columns exist before selection
    for col in final_feature_columns:
        if col not in final_features_df.columns:
            final_features_df[col] = np.nan

    final_features_df = final_features_df[final_feature_columns].copy()
    
    if final_features_df.empty:
        print("Warning: Final DataFrame is empty after column selection.")
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        final_features_df.to_excel(output_path, index=False)
        print(f"Engineered features saved to {output_path}")
    
    return final_features_df
