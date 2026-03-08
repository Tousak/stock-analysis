import os
import openai
from dotenv import load_dotenv

# --- General Settings ---
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", 
           "LLY", "AVGO", "WMT", "JPM", "V", "XOM", "JNJ", "MA", "COST", "MU", 
           "ORCL", "BAC", "ABBV", "HD", "PG", "CVX", "NFLX", "KO", "CAT", "AMD", 
           "GE", "CSCO", "PLTR", "MRK", "WFC", "LRCX", "MS", "PM", "IBM", "GS",
            "RTX", "AMAT", "INTC", "UNH", "AXP", "PEP", "MCD", "TMUS", "C", 
            "GEV", "LIN", "AMGN"] # User-specified list
NUM_QUARTERS_TO_FETCH = 24 # Approx 6 years to ensure we get data from start of 2020

# --- API Keys and Client Initialization ---
load_dotenv()

# Initialize OpenAI Client (Single Source of Truth)
# The client will be imported by other modules.
api_key = os.getenv("API_KEY_OPENAI")
if not api_key:
    # This will stop the program if the API key is not configured.
    raise ValueError("API_KEY_OPENAI environment variable not found. Please set it in your .env file.")
client = openai.OpenAI(api_key=api_key)

# --- SEC EDGAR Settings ---
# Identity for edgar-tools (replace with your actual name and email)
EDGAR_IDENTITY = "Jan Tous honza.tous@seznam.com" 

# --- Data Paths ---
DATA_DIR = "data/fetched"
BLACKLIST_PATH = os.path.join(DATA_DIR, "blacklist.txt")
RAW_FILINGS_PATH = os.path.join(DATA_DIR, "raw_filings.xlsx")

def get_data_paths(nlp_method: str = "openai") -> dict:
    """Returns a dictionary of dynamic file paths based on the chosen NLP method."""
    suffix = f"_{nlp_method}.xlsx"
    return {
        "PROCESSED_FILINGS_PATH": os.path.join(DATA_DIR, f"processed_filings{suffix}"),
        "FEATURES_PATH": os.path.join(DATA_DIR, f"features{suffix}"),
        "PREDICTIONS_PATH": os.path.join(DATA_DIR, f"predictions{suffix}"),
        "BACKTEST_RESULTS_PATH": os.path.join(DATA_DIR, f"backtest_results{suffix}"),
        "LATEST_PREDICTIONS_PATH": os.path.join(DATA_DIR, f"latest_predictions{suffix}"),
    }

# --- Processor Settings ---
# Regex pattern for MD&A extraction (Item 2 to Item 3)
MDNA_REGEX_PATTERN = r'(Item\s+2[.:]?\s+Management.*?)(?=Item\s+3[.:]?)'
MDNA_MIN_LENGTH = 1000 # Minimum characters for MD&A to be considered valid

# --- Model Settings (XGBoost) ---
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_RANDOM_STATE = 42

# --- Backtester Settings ---
INITIAL_CAPITAL = 100.0 # Starting capital for portfolio simulation

