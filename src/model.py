import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
from tqdm.auto import tqdm
from datetime import timedelta

from src.config import (
    MODEL_N_ESTIMATORS, MODEL_MAX_DEPTH, MODEL_RANDOM_STATE, 
    FEATURES_PATH, PREDICTIONS_PATH, DATA_DIR
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
        model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
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

if __name__ == "__main__":
    print("Running model.py example (Robust Walk-Forward)...")
    try:
        features_data = pd.read_excel(FEATURES_PATH)
        run_walk_forward_predictions(features_data.copy())
    except FileNotFoundError:
        print(f"Error: {FEATURES_PATH} not found. Please run --engineer first.")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
