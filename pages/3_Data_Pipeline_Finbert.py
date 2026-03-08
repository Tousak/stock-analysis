import streamlit as st
import pandas as pd
import io
import contextlib
import os

# Import pipeline functions and config
from src.data_loader import load_all_raw_data
from src.processor import process_filings_for_sentiment
from src.feature_eng import engineer_features
from src.config import TICKERS, NUM_QUARTERS_TO_FETCH, RAW_FILINGS_PATH, get_data_paths

# Get paths explicitly for the finbert pipeline
paths = get_data_paths("finbert")
PROCESSED_FILINGS_PATH = paths["PROCESSED_FILINGS_PATH"]
FEATURES_PATH = paths["FEATURES_PATH"]

st.set_page_config(
    page_title="Data Pipeline (FinBERT)",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Data Pipeline (FinBERT)")
st.write("This is the alternative 'engine room' using the local HuaggingFace **ProsusAI/finbert** model. This NLP pipeline is completely free of future-looking data leakage, making it optimal for robust historical backtesting.")

# --- Ticker Input ---
st.header("1. Configure Tickers")
ticker_list_str = st.text_area(
    "Enter stock tickers, separated by commas",
    value=", ".join(TICKERS)
)
# Parse the string into a list of tickers
try:
    current_tickers = [ticker.strip().upper() for ticker in ticker_list_str.split(',') if ticker.strip()]
    st.write(f"**{len(current_tickers)}** tickers to process: `{', '.join(current_tickers)}`")
except Exception:
    st.error("Invalid ticker format. Please use a comma-separated list.")
    current_tickers = []

# --- Action Buttons ---
st.header("2. Execute Pipeline Steps")

# Helper function to run a step and capture its print output
def run_step_with_logging(step_function, *args, **kwargs):
    log_stream = io.StringIO()
    with st.spinner(f"Running {step_function.__name__}... This may take several minutes."):
        with contextlib.redirect_stdout(log_stream):
            step_function(*args, **kwargs)
    
    st.success(f"{step_function.__name__} completed!")
    log_output = log_stream.getvalue()
    st.code(log_output, language='log')

# Step 1: Fetch Filings
if st.button("Step 1: Fetch New Filings", use_container_width=True):
    if current_tickers:
        run_step_with_logging(load_all_raw_data, tickers=current_tickers, num_quarters=NUM_QUARTERS_TO_FETCH)
    else:
        st.warning("Please enter at least one ticker.")

# Step 2: Run Sentiment Analysis
if st.button("Step 2: Run Sentiment Analysis (FinBERT)", use_container_width=True):
    try:
        raw_df = pd.read_excel(RAW_FILINGS_PATH)
        
        # We explicitly use the finbert method here
        method_str = "finbert"
        
        # This simplified call re-uses the logic from main.py's run_process_step
        # A more advanced version would be a dedicated function.
        # For simplicity, we create a temporary function here.
        def process_sentiment():
            processed_df_existing = pd.DataFrame()
            if os.path.exists(PROCESSED_FILINGS_PATH):
                processed_df_existing = pd.read_excel(PROCESSED_FILINGS_PATH)
            
            existing_accession_numbers = set(processed_df_existing['accession_number']) if not processed_df_existing.empty else set()
            filings_to_process = raw_df[~raw_df['accession_number'].isin(existing_accession_numbers)]

            if filings_to_process.empty:
                print("No new filings to process for sentiment.")
                return

            processed_df_new = process_filings_for_sentiment(filings_to_process.copy(), nlp_method=method_str)
            final_processed_df = pd.concat([processed_df_existing, processed_df_new], ignore_index=True)
            final_processed_df.to_excel(PROCESSED_FILINGS_PATH, index=False)
            print(f"Sentiment analysis complete. Saved to {PROCESSED_FILINGS_PATH}.")

        run_step_with_logging(process_sentiment)
    except FileNotFoundError:
        st.error(f"Error: Raw filings file not found at {RAW_FILINGS_PATH}. Please run Step 1 first.")

# Step 3: Engineer Features
if st.button("Step 3: Engineer Features", use_container_width=True):
    try:
        processed_df = pd.read_excel(PROCESSED_FILINGS_PATH)
        run_step_with_logging(engineer_features, processed_filings_df=processed_df, output_path=FEATURES_PATH)
    except FileNotFoundError:
        st.error(f"Error: Processed filings file not found at {PROCESSED_FILINGS_PATH}. Please run Step 2 first.")
