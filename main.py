import argparse
import os
import pandas as pd

from src.config import (
    RAW_FILINGS_PATH, PROCESSED_FILINGS_PATH, FEATURES_PATH, PREDICTIONS_PATH,
    TICKERS, NUM_QUARTERS_TO_FETCH, EDGAR_IDENTITY
)
from src.data_loader import load_all_raw_data
from src.processor import process_filings_for_sentiment
from src.feature_eng import engineer_features
from src.model import run_walk_forward_predictions, generate_next_quarter_prediction
from src.backtester import simulate_portfolio

def run_fetch_step(tickers, num_quarters):
    """Executes the data fetching step."""
    print("\n--- Step 1: Fetching Raw Data ---")
    if EDGAR_IDENTITY == "Your Name Your.Email@example.com":
        print("ERROR: Please update EDGAR_IDENTITY in src.config.py.")
        return False
    load_all_raw_data(tickers=tickers, num_quarters=num_quarters)
    return True

def run_process_step():
    """Executes the sentiment analysis processing step with caching."""
    print("\n--- Step 2: Processing Data (Sentiment Analysis) ---")
    if not os.path.exists(RAW_FILINGS_PATH):
        print(f"ERROR: Raw filings not found at {RAW_FILINGS_PATH}. Please run --fetch first.")
        return False

    raw_df = pd.read_excel(RAW_FILINGS_PATH)
    
    processed_df_existing = pd.DataFrame()
    if os.path.exists(PROCESSED_FILINGS_PATH):
        try:
            processed_df_existing = pd.read_excel(PROCESSED_FILINGS_PATH)
            print(f"Loaded {len(processed_df_existing)} already processed filings from cache.")
        except Exception as e:
            print(f"Warning: Could not load existing processed filings. Reprocessing all. Error: {e}")

    if not processed_df_existing.empty:
        existing_accession_numbers = set(processed_df_existing['accession_number'])
        filings_to_process = raw_df[~raw_df['accession_number'].isin(existing_accession_numbers)]
    else:
        filings_to_process = raw_df

    if filings_to_process.empty:
        print("No new filings to process for sentiment.")
        return True

    print(f"Found {len(filings_to_process)} new filings to process for sentiment...")
    processed_df_new = process_filings_for_sentiment(filings_to_process.copy())
    
    final_processed_df = pd.concat([processed_df_existing, processed_df_new], ignore_index=True)
    final_processed_df.drop_duplicates(subset=['accession_number'], keep='last', inplace=True)
    
    final_processed_df.to_excel(PROCESSED_FILINGS_PATH, index=False)
    print(f"Updated processed filings saved to {PROCESSED_FILINGS_PATH} ({len(final_processed_df)} total).")
    return True

def run_engineer_step():
    """Executes the feature engineering step."""
    print("\n--- Step 3: Engineering Features ---")
    if not os.path.exists(PROCESSED_FILINGS_PATH):
        print(f"ERROR: Processed filings not found at {PROCESSED_FILINGS_PATH}. Please run --process first.")
        return False
    
    processed_df = pd.read_excel(PROCESSED_FILINGS_PATH)
    engineer_features(processed_df.copy())
    return True

def run_train_step():
    """Executes the model training and validation step using robust walk-forward."""
    print("\n--- Step 4: Training Model (Robust Walk-Forward) ---")
    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: Features file not found at {FEATURES_PATH}. Please run --engineer first.")
        return False

    features_df = pd.read_excel(FEATURES_PATH)
    run_walk_forward_predictions(features_df.copy())
    return True

def run_backtest_step():
    """Executes the backtesting step."""
    print("\n--- Step 5: Running Portfolio Backtest Simulation ---")
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"ERROR: Predictions file not found at {PREDICTIONS_PATH}. Please run --train first.")
        return False

    predictions_df = pd.read_excel(PREDICTIONS_PATH)
    
    if predictions_df.empty:
        print("Predictions file is empty. Cannot backtest.")
        return False
        
    simulate_portfolio(predictions_df.copy())
    return True

def run_predict_latest_step():
    """Executes the step to generate prediction for the next unseen quarter."""
    print("\n--- Step 6: Generating Latest Next Quarter Predictions ---")
    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: Features file not found at {FEATURES_PATH}. Please run --engineer first.")
        return False

    features_df = pd.read_excel(FEATURES_PATH)
    generate_next_quarter_prediction(features_df.copy())
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Stock Analysis Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--fetch", action="store_true", help="Step 1: Fetch raw 10-Q filings from EDGAR.")
    parser.add_argument("--process", action="store_true", help="Step 2: Process filings for sentiment analysis.")
    parser.add_argument("--engineer", action="store_true", help="Step 3: Engineer features from processed data.")
    parser.add_argument("--train", action="store_true", help="Step 4: Train model using robust walk-forward validation.")
    parser.add_argument("--backtest", action="store_true", help="Step 5: Run portfolio backtest simulation.")
    parser.add_argument("--predict-latest", action="store_true", help="Step 6: Generate prediction for the next unseen quarter.")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline from fetch to backtest.")
    
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers (e.g., AAPL,MSFT).")
    parser.add_argument("--num-quarters", type=int, help="Number of past quarters to fetch.")
    
    args = parser.parse_args()

    # Determine tickers and num_quarters
    current_tickers = args.tickers.split(',') if args.tickers else TICKERS
    current_num_quarters = args.num_quarters if args.num_quarters is not None else NUM_QUARTERS_TO_FETCH

    if args.all or args.fetch:
        if not run_fetch_step(current_tickers, current_num_quarters): return

    if args.all or args.process:
        if not run_process_step(): return
        
    if args.all or args.engineer:
        if not run_engineer_step(): return

    if args.all or args.train:
        if not run_train_step(): return

    if args.all or args.backtest:
        if not run_backtest_step(): return
    
    if args.all or args.predict_latest:
        if not run_predict_latest_step(): return

    if not any(vars(args).values()):
        print("No steps selected. Use --all or specify steps like --fetch, --process, etc.")
        print("Use --help for more information.")

    print("\n--- Pipeline execution finished ---")

if __name__ == "__main__":
    main()