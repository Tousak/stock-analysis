import pandas as pd
from datetime import timedelta
import numpy as np
import os
from tqdm.auto import tqdm

from src.config import DATA_DIR
from src.data_loader import fetch_stock_prices

def _consecutive_quarters_mask(df: pd.DataFrame) -> pd.Series:
    """True where a ticker's previous row is exactly one quarter earlier."""
    quarters = pd.to_datetime(df['filing_date']).dt.to_period('Q')
    prev_quarter = quarters.groupby(df['ticker']).shift(1)
    return prev_quarter.eq(quarters - 1)

def calculate_financial_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates revenue growth and net margin for each ticker with minor filling.
    Growth is only valid between consecutive quarterly filings."""
    df = df.sort_values(['ticker', 'filing_date'])
    df['revenue'] = df.groupby('ticker')['revenue'].ffill(limit=1)
    df['net_income'] = df.groupby('ticker')['net_income'].ffill(limit=1)

    consecutive = _consecutive_quarters_mask(df)
    df['revenue_growth'] = df.groupby('ticker')['revenue'].pct_change(fill_method=None).where(consecutive)
    df['net_margin'] = df['net_income'] / df['revenue']
    return df

def calculate_sentiment_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates sentiment change (QoQ difference) for each ticker.
    Deltas are only valid between consecutive quarterly filings."""
    consecutive = _consecutive_quarters_mask(df)
    df['sentiment_change'] = df.groupby('ticker')['sentiment_score'].diff().where(consecutive)

    # Also calculate changes for the triplet features if they exist
    for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu']:
        if col in df.columns:
            df[f'{col}_change'] = df.groupby('ticker')[col].diff().where(consecutive)

    return df

def calculate_technical_indicators(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates RSI (Wilder), MACD, and Volatility (KISS implementation).
    Each filing is matched with the last trading day on or before its filing date."""
    # Group by ticker to iterate through market data
    market_grouped = market_df.groupby('ticker')
    all_ta_features = []

    for ticker, ticker_data in tqdm(market_grouped, desc="Engineering TA Features"):
        td = ticker_data.sort_values('date').copy()
        delta = td['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        td['rsi'] = 100 - (100 / (1 + rs))

        ema12 = td['Close'].ewm(span=12, adjust=False).mean()
        ema26 = td['Close'].ewm(span=26, adjust=False).mean()
        td['macd'] = ema12 - ema26

        td['volatility'] = (td['High'] - td['Low']) / (td['Open'] + 1e-9)
        all_ta_features.append(td[['ticker', 'date', 'rsi', 'macd', 'volatility']])

    ta_df = pd.concat(all_ta_features)
    ta_df['date'] = pd.to_datetime(ta_df['date'])

    # Backward as-of join: no more NaNs for filings dated on market holidays
    df = pd.merge_asof(
        df.sort_values('filing_date'), ta_df.sort_values('date'),
        left_on='filing_date', right_on='date', by='ticker', direction='backward'
    )
    return df.drop(columns=['date'])

def calculate_future_return(filings_df: pd.DataFrame, market_data_df: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    """Calculates stock return over a given horizon (default 90 days)."""
    filings_df['filing_date'] = pd.to_datetime(filings_df['filing_date'])
    market_data_df['date'] = pd.to_datetime(market_data_df['date']).dt.date

    def get_price_on_or_after_date(ticker_market_data, target_date):
        target_date = target_date.date()
        price_series = ticker_market_data[ticker_market_data['date'] == target_date]['Close']
        if not price_series.empty: return price_series.values[0]
        future_data_series = ticker_market_data[ticker_market_data['date'] > target_date]['Close']
        return future_data_series.values[0] if not future_data_series.empty else np.nan

    returns = []
    market_data_grouped = market_data_df.groupby('ticker')
    
    for _, row in tqdm(filings_df.iterrows(), total=filings_df.shape[0], desc=f"Calculating {horizon_days}-Day Returns"):
        ticker = row['ticker']
        filing_date = row['filing_date']

        if ticker not in market_data_grouped.groups:
            returns.append(np.nan)
            continue

        current_ticker_market_data = market_data_grouped.get_group(ticker)

        price_on_filing_date = get_price_on_or_after_date(current_ticker_market_data, filing_date)
        future_date = filing_date + timedelta(days=horizon_days)
        price_at_horizon = get_price_on_or_after_date(current_ticker_market_data, future_date)

        if pd.notna(price_on_filing_date) and pd.notna(price_at_horizon) and price_on_filing_date != 0:
            returns.append((price_at_horizon - price_on_filing_date) / price_on_filing_date)
        else:
            returns.append(np.nan)
            
    col_name = 'next_quarter_return' if horizon_days == 90 else f'return_{horizon_days}d'
    filings_df[col_name] = returns
    return filings_df

def engineer_features(processed_filings_df: pd.DataFrame, output_path: str, horizon_days: int = 90, include_ta: bool = False) -> pd.DataFrame:
    """
    Orchestrates feature engineering, with differentiated cleaning for training and prediction data.
    Aggressively cleans historical data for model training.
    Retains latest data points (where next_quarter_return is NaN) for prediction,
    applying more lenient cleaning to ensure they are available for forecasting.
    """
    df = processed_filings_df.copy()
    
    if 'mda_text' in df.columns:
        df['mda_text'] = df['mda_text'].replace('', np.nan)
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
    
    if include_ta:
        print("Calculating Technical indicators (RSI, MACD, Volatility)...")
        df = calculate_technical_indicators(df, full_market_data_df)
    
    df = calculate_future_return(df, full_market_data_df, horizon_days=horizon_days)

    # --- Split into data with known returns (for training) and unknown returns (for prediction) ---
    target_col = 'next_quarter_return' if horizon_days == 90 else f'return_{horizon_days}d'
    df_has_return = df.dropna(subset=[target_col]).copy()
    df_no_return = df[df[target_col].isna()].copy()

    final_cleaned_dfs = []

    # --- Granular Cleaning for Training Data (df_has_return) ---
    if not df_has_return.empty:
        print(f"\nApplying granular cleaning to {len(df_has_return)} historical rows...")
        # Only drop rows that are actually missing critical features
        critical_cols = ['revenue', 'net_income', 'sentiment_score', 'revenue_growth', 'net_margin', 'sentiment_change']
        if include_ta:
            critical_cols += ['rsi', 'macd', 'volatility']
        
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
        cols_to_check = ['sentiment_score']
        if 'mda_text' in df_no_return.columns:
            cols_to_check.append('mda_text')
        if include_ta:
            cols_to_check += ['rsi', 'macd', 'volatility']
        
        df_no_return_clean = df_no_return.dropna(subset=cols_to_check).copy()
        
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
        'revenue_growth', 'net_margin', 'sentiment_change', target_col,
        'sentiment_pos', 'sentiment_neg', 'sentiment_neu',
        'sentiment_pos_change', 'sentiment_neg_change', 'sentiment_neu_change'
    ]
    if include_ta:
        final_feature_columns += ['rsi', 'macd', 'volatility']
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
