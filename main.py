import argparse
import os
import pandas as pd

from src.config import (
    RAW_FILINGS_PATH, get_data_paths,
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

def run_process_step(nlp_method="finbert"):
    """Executes the sentiment analysis processing step with caching."""
    print(f"\n--- Step 2: Processing Data (Sentiment Analysis: {nlp_method}) ---")
    
    paths = get_data_paths(nlp_method)
    PROCESSED_FILINGS_PATH = paths["PROCESSED_FILINGS_PATH"]
    
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

    print(f"Found {len(filings_to_process)} new filings to process for sentiment using {nlp_method}...")
    processed_df_new = process_filings_for_sentiment(filings_to_process.copy(), nlp_method=nlp_method)
    
    # DROP REDUNDANT COLUMNS TO SAVE SPACE (99% reduction)
    # mda_text, revenue, net_income are already in raw_filings.xlsx
    cols_to_drop = [c for c in ['mda_text', 'revenue', 'net_income'] if c in processed_df_new.columns]
    processed_df_new = processed_df_new.drop(columns=cols_to_drop)
    
    final_processed_df = pd.concat([processed_df_existing, processed_df_new], ignore_index=True)
    final_processed_df.drop_duplicates(subset=['accession_number'], keep='last', inplace=True)
    
    # Also ensure existing rows don't have these columns if they were somehow re-added
    final_processed_df = final_processed_df.drop(columns=[c for c in cols_to_drop if c in final_processed_df.columns])
    
    final_processed_df.to_excel(PROCESSED_FILINGS_PATH, index=False)
    print(f"Updated processed filings saved to {PROCESSED_FILINGS_PATH} ({len(final_processed_df)} total).")
    return True

def run_engineer_step(nlp_method="finbert"):
    """Executes the feature engineering step."""
    print(f"\n--- Step 3: Engineering Features ({nlp_method}) ---")
    
    paths = get_data_paths(nlp_method)
    PROCESSED_FILINGS_PATH = paths["PROCESSED_FILINGS_PATH"]
    FEATURES_PATH = paths["FEATURES_PATH"]
    
    if not os.path.exists(PROCESSED_FILINGS_PATH):
        print(f"ERROR: Processed filings not found at {PROCESSED_FILINGS_PATH}. Please run --process first.")
        return False
    
    if not os.path.exists(RAW_FILINGS_PATH):
        print(f"ERROR: Raw filings not found at {RAW_FILINGS_PATH}. Cannot merge MDA text.")
        return False

    processed_df = pd.read_excel(PROCESSED_FILINGS_PATH)
    raw_df = pd.read_excel(RAW_FILINGS_PATH)
    
    # Merge sentiment cache with raw data to get mda_text and financials back for engineering
    # We join on accession_number
    print("Merging sentiment cache with raw filing data context...")
    merged_df = pd.merge(
        processed_df, 
        raw_df[['accession_number', 'mda_text', 'revenue', 'net_income']], 
        on='accession_number', 
        how='left'
    )
    
    # the engineer function will need the output path
    engineer_features(merged_df, output_path=FEATURES_PATH)
    return True

def run_train_step(nlp_method="finbert"):
    """Executes the model training and validation step using robust walk-forward."""
    print(f"\n--- Step 4: Training Model ({nlp_method}) ---")
    
    paths = get_data_paths(nlp_method)
    FEATURES_PATH = paths["FEATURES_PATH"]
    PREDICTIONS_PATH = paths["PREDICTIONS_PATH"]
    
    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: Features file not found at {FEATURES_PATH}. Please run --engineer first.")
        return False

    features_df = pd.read_excel(FEATURES_PATH)
    run_walk_forward_predictions(features_df.copy(), output_path=PREDICTIONS_PATH)
    return True

def run_backtest_step(nlp_method="finbert"):
    """Executes the backtesting step."""
    print(f"\n--- Step 5: Running Portfolio Backtest Simulation ({nlp_method}) ---")
    
    paths = get_data_paths(nlp_method)
    PREDICTIONS_PATH = paths["PREDICTIONS_PATH"]
    BACKTEST_RESULTS_PATH = paths["BACKTEST_RESULTS_PATH"]
    
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"ERROR: Predictions file not found at {PREDICTIONS_PATH}. Please run --train first.")
        return False

    predictions_df = pd.read_excel(PREDICTIONS_PATH)
    
    if predictions_df.empty:
        print("Predictions file is empty. Cannot backtest.")
        return False
        
    simulate_portfolio(predictions_df.copy(), output_path=BACKTEST_RESULTS_PATH)
    return True

def run_predict_latest_step(nlp_method="finbert"):
    """Executes the step to generate prediction for the next unseen quarter."""
    print(f"\n--- Step 6: Generating Latest Next Quarter Predictions ({nlp_method}) ---")
    
    paths = get_data_paths(nlp_method)
    FEATURES_PATH = paths["FEATURES_PATH"]
    LATEST_PREDICTIONS_PATH = paths["LATEST_PREDICTIONS_PATH"]
    
    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: Features file not found at {FEATURES_PATH}. Please run --engineer first.")
        return False

    features_df = pd.read_excel(FEATURES_PATH)
    generate_next_quarter_prediction(features_df.copy(), output_path=LATEST_PREDICTIONS_PATH)
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
    parser.add_argument("--nlp", type=str, choices=["openai", "finbert"], default="finbert", help="NLP engine for sentiment analysis.")
    
    args = parser.parse_args()

    # Determine tickers and num_quarters
    current_tickers = args.tickers.split(',') if args.tickers else TICKERS
    current_num_quarters = args.num_quarters if args.num_quarters is not None else NUM_QUARTERS_TO_FETCH

    if args.all or args.fetch:
        if not run_fetch_step(current_tickers, current_num_quarters): return

    if args.all or args.process:
        if not run_process_step(nlp_method=args.nlp): return
        
    if args.all or args.engineer:
        if not run_engineer_step(nlp_method=args.nlp): return

    if args.all or args.train:
        if not run_train_step(nlp_method=args.nlp): return

    if args.all or args.backtest:
        if not run_backtest_step(nlp_method=args.nlp): return
    
    if args.all or args.predict_latest:
        if not run_predict_latest_step(nlp_method=args.nlp): return

    if not any(vars(args).values()):
        print("No steps selected. Use --all or specify steps like --fetch, --process, etc.")
        print("Use --help for more information.")

    print("\n--- Pipeline execution finished ---")

if __name__ == "__main__":
    main()