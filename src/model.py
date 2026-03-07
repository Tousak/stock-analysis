import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
from tqdm.auto import tqdm
from datetime import timedelta

from src.config import (
    MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH, MODEL_RANDOM_STATE, 
    FEATURES_PATH, PREDICTIONS_PATH, DATA_DIR, LATEST_PREDICTIONS_PATH
)

def run_walk_forward_predictions(features_df: pd.DataFrame, start_year: int = 2021, lookahead_days: int = 90) -> pd.DataFrame:
    """
    Performs a robust walk-forward validation with data purging to prevent look-ahead bias.
    Trains one global model at each time step.
    """
    if features_df.empty:
        print("Features DataFrame is empty. Cannot run walk-forward.")
        return pd.DataFrame()

    # --- 1. Prepare Data ---
    df = features_df.copy()
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['quarter'] = df['filing_date'].dt.to_period('Q')
    df = df.sort_values('filing_date')

    all_quarters = sorted(df['quarter'].unique())
    trading_quarters = [q for q in all_quarters if q.year >= start_year]

    all_out_of_sample_preds = []

    print(f"Starting Robust Walk-Forward (Purging Overlaps)...")

    for current_q in tqdm(trading_quarters, desc="Walk-Forward Quarters"):
        test_df = df[df['quarter'] == current_q]
        if test_df.empty:
            continue
        
        decision_date = test_df['filing_date'].min()
        potential_train_df = df[df['quarter'] < current_q]
        
        # Purge training data where the outcome is not yet known
        valid_mask = (potential_train_df['filing_date'] + pd.Timedelta(days=lookahead_days)) < decision_date
        train_df = potential_train_df[valid_mask].copy()
        
        if train_df.empty or len(train_df) < 10:
            print(f"Skipping {current_q}: Not enough historical closed trades to train a model.")
            continue

        # --- 3. Training & Prediction ---
        features = ['sentiment_score', 'sentiment_change', 'revenue_growth', 'net_margin']
        target = 'next_quarter_return'

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]

        # Train a global model on all available past data
        model = RandomForestRegressor(n_estimators=MODEL_N_ESTIMATORS, max_depth=MODEL_MAX_DEPTH, random_state=MODEL_RANDOM_STATE)
        model.fit(X_train, y_train)
        
        # Generate predictions for the current quarter
        test_df['predicted_return'] = model.predict(X_test)
        
        all_out_of_sample_preds.append(test_df)

    if not all_out_of_sample_preds:
        print("No out-of-sample predictions were generated.")
        return pd.DataFrame()

    # --- 4. Finalize and Save ---
    predictions_df = pd.concat(all_out_of_sample_preds, ignore_index=True)
    output_df = predictions_df[['ticker', 'filing_date', 'quarter', 'next_quarter_return', 'predicted_return']].copy()
    
    os.makedirs(DATA_DIR, exist_ok=True)
    output_df.to_excel(PREDICTIONS_PATH, index=False)
    print(f"Walk-forward predictions saved to {PREDICTIONS_PATH}")

    return output_df


def generate_next_quarter_prediction(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a single forward-looking prediction for the next unseen quarter
    for each stock, using a model trained on that stock's historical data.
    """
    if features_df.empty:
        print("Features DataFrame is empty. Cannot generate next quarter prediction.")
        return pd.DataFrame()

    df = features_df.copy()
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['quarter'] = df['filing_date'].dt.to_period('Q')
    df = df.sort_values('filing_date')
    
    all_latest_predictions = []
    features = ['sentiment_score', 'sentiment_change', 'revenue_growth', 'net_margin']
    target = 'next_quarter_return'

    unique_tickers = df['ticker'].unique()
    for ticker in tqdm(unique_tickers, desc="Generating Next Quarter Predictions per Ticker"):
        ticker_df = df[df['ticker'] == ticker].copy()

        # Data where 'next_quarter_return' is known (for training)
        train_data_ticker = ticker_df.dropna(subset=['next_quarter_return']).copy()
        
        # Data for which 'next_quarter_return' is not known (for prediction)
        predict_data_ticker = ticker_df[ticker_df['next_quarter_return'].isna()].copy()

        if train_data_ticker.empty or len(train_data_ticker) < 5: # Reduced min data points for per-stock
            # print(f"Skipping {ticker}: Not enough historical data with known outcomes to train a model.")
            continue
        
        if predict_data_ticker.empty:
            # print(f"Skipping {ticker}: No new data available for forward-looking prediction.")
            continue

        # Ensure predict_data_ticker is only the latest available features
        predict_data_ticker = predict_data_ticker.sort_values('filing_date', ascending=False).head(1)

        X_train = train_data_ticker[features]
        y_train = train_data_ticker[target]
        X_predict = predict_data_ticker[features]
        
        # Fill NaNs in prediction features with the mean of the training features
        # This handles cases where latest quarter features like sentiment_change might be NaN
        for feature in features:
            if X_predict[feature].isnull().any():
                mean_val = X_train[feature].mean()
                X_predict[feature] = X_predict[feature].fillna(mean_val)

        # Train a model specifically for this ticker
        model = RandomForestRegressor(n_estimators=MODEL_N_ESTIMATORS, max_depth=MODEL_MAX_DEPTH, random_state=MODEL_RANDOM_STATE)
        model.fit(X_train, y_train)
        
        # Generate prediction for this ticker's next quarter
        predict_data_ticker['predicted_return'] = model.predict(X_predict)
        all_latest_predictions.append(predict_data_ticker)
    
    if not all_latest_predictions:
        print("No next quarter predictions were generated for any ticker.")
        return pd.DataFrame()

    output_df = pd.concat(all_latest_predictions, ignore_index=True)
    output_df = output_df[['ticker', 'filing_date', 'quarter', 'predicted_return']].copy()
    
    os.makedirs(DATA_DIR, exist_ok=True)
    output_df.to_excel(LATEST_PREDICTIONS_PATH, index=False)
    print(f"Latest next quarter predictions saved to {LATEST_PREDICTIONS_PATH}")

    return output_df

if __name__ == "__main__":
    print("Running model.py example (Robust Walk-Forward)...")
    try:
        features_data = pd.read_excel(FEATURES_PATH)
        run_walk_forward_predictions(features_data.copy())
    except FileNotFoundError:
        print(f"Error: {FEATURES_PATH} not found. Please run --engineer first.")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
