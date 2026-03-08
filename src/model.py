import pandas as pd
from xgboost import XGBRegressor
import os
from tqdm.auto import tqdm
from datetime import timedelta
import optuna
from sklearn.metrics import mean_squared_error

from src.config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_RANDOM_STATE, DATA_DIR
)

def optimize_xgboost_params(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame, n_trials: int = 20) -> dict:
    """Uses Optuna to find the best XGBoost hyperparameters for the given split (KISS implementation)."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'random_state': XGB_RANDOM_STATE
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        mse = mean_squared_error(y_valid, preds)
        return mse

    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def run_walk_forward_predictions(features_df: pd.DataFrame, output_path: str, start_year: int = 2021, lookahead_days: int = 90, 
                                 n_estimators: int = XGB_N_ESTIMATORS, max_depth: int = XGB_MAX_DEPTH, 
                                 learning_rate: float = XGB_LEARNING_RATE, use_optuna: bool = False,
                                 use_triplet: bool = False) -> pd.DataFrame:
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
        if use_triplet:
            features = [
                'sentiment_pos', 'sentiment_neg', 'sentiment_neu',
                'sentiment_pos_change', 'sentiment_neg_change', 'sentiment_neu_change',
                'revenue_growth', 'net_margin'
            ]
        else:
            features = ['sentiment_score', 'sentiment_change', 'revenue_growth', 'net_margin']
            
        target = 'next_quarter_return'

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]

        if use_optuna:
            # --- Fix: Avoid Leakage ---
            # Use only the training data to tune hyperparameters. 
            # We split the training data 80/20 (chronological) to create a proxy validation set.
            split_idx = int(len(X_train) * 0.8)
            if split_idx >= 5: # Ensure we have enough data to split
                X_tune_train, X_tune_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
                y_tune_train, y_tune_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
                
                best_params = optimize_xgboost_params(X_tune_train, y_tune_train, X_tune_val, y_tune_val)
                model = XGBRegressor(**best_params, random_state=XGB_RANDOM_STATE)
                # print(f"[{current_q}] Tuned (Leak-free): {best_params}")
            else:
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=XGB_RANDOM_STATE)
        else:
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=XGB_RANDOM_STATE)

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
    output_df.to_excel(output_path, index=False)
    print(f"Walk-forward predictions saved to {output_path}")

    return output_df


def generate_next_quarter_prediction(features_df: pd.DataFrame, output_path: str, n_estimators: int = XGB_N_ESTIMATORS, 
                                     max_depth: int = XGB_MAX_DEPTH, learning_rate: float = XGB_LEARNING_RATE, 
                                     use_optuna: bool = False, use_triplet: bool = False) -> pd.DataFrame:
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
    
    if use_triplet:
        features = [
            'sentiment_pos', 'sentiment_neg', 'sentiment_neu',
            'sentiment_pos_change', 'sentiment_neg_change', 'sentiment_neu_change',
            'revenue_growth', 'net_margin'
        ]
    else:
        features = ['sentiment_score', 'sentiment_change', 'revenue_growth', 'net_margin']
        
    target = 'next_quarter_return'

    unique_tickers = df['ticker'].unique()
    for ticker in tqdm(unique_tickers, desc="Generating Next Quarter Predictions per Ticker"):
        ticker_df = df[df['ticker'] == ticker].copy()

        # Data where 'next_quarter_return' is known (for training)
        train_data_ticker = ticker_df.dropna(subset=['next_quarter_return']).copy()
        
        # Data for which we want a forward-looking prediction: the absolute latest record for this ticker
        predict_data_ticker = ticker_df.sort_values('filing_date', ascending=False).head(1).copy()

        if train_data_ticker.empty or len(train_data_ticker) < 5: 
            # print(f"Skipping {ticker}: Not enough historical data with known outcomes to train a model.")
            continue
        
        if predict_data_ticker.empty:
            continue

        # In case the latest record is also in the training set (return is known), 
        # that's fine - we still want to see the AI's latest "opinion" on it.

        X_train = train_data_ticker[features]
        y_train = train_data_ticker[target]
        X_predict = predict_data_ticker[features]
        
        # Fill NaNs in prediction features with the mean of the training features
        # This handles cases where latest quarter features like sentiment_change might be NaN
        for feature in features:
            if X_predict[feature].isnull().any():
                mean_val = X_train[feature].mean()
                X_predict[feature] = X_predict[feature].fillna(mean_val)

        if use_optuna and len(train_data_ticker) >= 10:
            # Split train data 80/20 for tuning
            split_idx = int(len(train_data_ticker) * 0.8)
            X_tune_train, X_tune_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
            y_tune_train, y_tune_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
            
            best_params = optimize_xgboost_params(X_tune_train, y_tune_train, X_tune_val, y_tune_val)
            model = XGBRegressor(**best_params, random_state=XGB_RANDOM_STATE)
        else:
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=XGB_RANDOM_STATE)
            
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
    output_df.to_excel(output_path, index=False)
    print(f"Latest next quarter predictions saved to {output_path}")

    return output_df

if __name__ == "__main__":
    print("Running model.py example (Robust Walk-Forward)...")
    try:
        features_data = pd.read_excel("data/fetched/features_finbert.xlsx")
        run_walk_forward_predictions(features_data.copy(), output_path="data/fetched/predictions_finbert.xlsx")
    except FileNotFoundError:
        print("Error: Features file not found. Please run --engineer first.")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
