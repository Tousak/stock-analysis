import pandas as pd
from xgboost import XGBRegressor
import os
from tqdm.auto import tqdm
from datetime import timedelta
import optuna
from sklearn.metrics import mean_squared_error
import numpy as np

from src.config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_RANDOM_STATE, DATA_DIR
)

def optimize_xgboost_params(X_train, y_train, X_valid, y_valid, n_trials=20):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'random_state': XGB_RANDOM_STATE
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        return mean_squared_error(y_valid, model.predict(X_valid))

    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def purged_cross_validation(df, features, target, n_folds=5, lookahead_days=90):
    """Calculates cross-validation score using a purged walk-forward K-fold approach."""
    # Only use rows where the target is known
    df = df.dropna(subset=[target]).copy()
    
    if df.empty: return 0.0, 0.0
    
    df = df.sort_values('filing_date').reset_index(drop=True)
    all_dates = df['filing_date'].unique()
    if len(all_dates) < n_folds * 2: return 0.0, 0.0
    
    date_chunks = np.array_split(all_dates, n_folds)
    errors = []
    
    for i in range(1, n_folds):
        train_dates = np.concatenate(date_chunks[:i])
        test_dates = date_chunks[i]
        
        test_start = test_dates.min()
        train_df = df[df['filing_date'].isin(train_dates)].copy()
        test_df = df[df['filing_date'].isin(test_dates)].copy()
        
        # Purge: Remove training samples that overlap with test period outcomes
        purged_train_df = train_df[(train_df['filing_date'] + pd.Timedelta(days=lookahead_days)) < test_start]
        
        if purged_train_df.empty: continue
        
        model = XGBRegressor(random_state=XGB_RANDOM_STATE)
        model.fit(purged_train_df[features], purged_train_df[target])
        preds = model.predict(test_df[features])
        errors.append(mean_squared_error(test_df[target], preds))
        
    return np.mean(errors) if errors else 0.0, np.std(errors) if errors else 0.0

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
        default_features = ['sentiment_score', 'sentiment_change', 'revenue_growth', 'net_margin']
        triplet_features = [
            'sentiment_pos', 'sentiment_neg', 'sentiment_neu',
            'sentiment_pos_change', 'sentiment_neg_change', 'sentiment_neu_change',
            'revenue_growth', 'net_margin'
        ]
        features = (triplet_features if use_triplet else default_features).copy()
        
        # Automatically add TA features if present
        ta_features = [f for f in ['rsi', 'macd', 'volatility'] if f in df.columns]
        features.extend(ta_features)
            
        target = 'next_quarter_return' if lookahead_days == 90 else f'return_{lookahead_days}d'

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
                print(f"[{current_q}] Optuna Best Params: {best_params}")
                model = XGBRegressor(**best_params, random_state=XGB_RANDOM_STATE)
            else:
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=XGB_RANDOM_STATE)
        else:
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=XGB_RANDOM_STATE)

        model.fit(X_train.values, y_train.values)
        
        # Generate predictions for the current quarter
        test_df['predicted_return'] = model.predict(X_test.values)
        all_out_of_sample_preds.append(test_df)

    if not all_out_of_sample_preds:
        print("No out-of-sample predictions were generated.")
        return pd.DataFrame()

    # --- 4. Finalize and Save ---
    predictions_df = pd.concat(all_out_of_sample_preds, ignore_index=True)
    
    # Calculate Purged Cross-Validation Score for the entire history
    print(f"Calculating Final Purged CV Score...")
    cv_base = triplet_features if use_triplet else default_features
    cv_features = cv_base.copy()
    cv_features.extend(ta_features)
    
    cv_mean, cv_std = purged_cross_validation(df, cv_features, target, n_folds=5, lookahead_days=lookahead_days)
    print(f"Purged CV MSE: {cv_mean:.6f} (std: {cv_std:.6f})")

    # Save metrics
    metrics_path = output_path.replace("predictions_", "metrics_")
    pd.DataFrame([{
        'cv_mse_mean': cv_mean,
        'cv_mse_std': cv_std,
        'last_updated': pd.Timestamp.now()
    }]).to_excel(metrics_path, index=False)
    print(f"Strategy metrics saved to {metrics_path}")

    output_df = predictions_df[['ticker', 'filing_date', 'quarter', target, 'predicted_return']].copy()
    
    os.makedirs(DATA_DIR, exist_ok=True)
    output_df.to_excel(output_path, index=False)
    print(f"Walk-forward predictions saved to {output_path}")

    return output_df


def generate_next_quarter_prediction(features_df: pd.DataFrame, output_path: str, lookahead_days: int = 90,
                                     n_estimators: int = XGB_N_ESTIMATORS, 
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
    
    default_features = ['sentiment_score', 'sentiment_change', 'revenue_growth', 'net_margin']
    triplet_features = [
        'sentiment_pos', 'sentiment_neg', 'sentiment_neu',
        'sentiment_pos_change', 'sentiment_neg_change', 'sentiment_neu_change',
        'revenue_growth', 'net_margin'
    ]
    features = triplet_features if use_triplet else default_features
    
    # Automatically add TA features if present
    ta_features = [f for f in ['rsi', 'macd', 'volatility'] if f in df.columns]
    features += ta_features
        
    target = 'next_quarter_return' if lookahead_days == 90 else f'return_{lookahead_days}d'

    unique_tickers = df['ticker'].unique()
    for ticker in tqdm(unique_tickers, desc="Generating Next Quarter Predictions per Ticker"):
        ticker_df = df[df['ticker'] == ticker].copy()

        # Data where target is known (for training)
        train_data_ticker = ticker_df.dropna(subset=[target]).copy()
        
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
            print(f"[{ticker}] Optuna Best Params: {best_params}")
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
