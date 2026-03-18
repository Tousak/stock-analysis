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
NUM_QUARTERS_TO_FETCH = 48

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

def get_data_paths(nlp_method: str = "finbert") -> dict:
    """Returns a dictionary of dynamic file paths based on the chosen NLP method."""
    suffix = f"_{nlp_method}.xlsx"
    
    # We define the paths with the suffix
    paths = {
        "PROCESSED_FILINGS_PATH": os.path.join(DATA_DIR, f"processed_filings{suffix}"),
        "FEATURES_PATH": os.path.join(DATA_DIR, f"features{suffix}"),
        "PREDICTIONS_PATH": os.path.join(DATA_DIR, f"predictions{suffix}"),
        "BACKTEST_RESULTS_PATH": os.path.join(DATA_DIR, f"backtest_results{suffix}"),
        "LATEST_PREDICTIONS_PATH": os.path.join(DATA_DIR, f"latest_predictions{suffix}"),
    }
    
    # SMART FALLBACK: If the unsuffixed 'processed_filings.xlsx' exists and is 
    # much larger than the suffixed one (or the suffixed one is missing), use it.
    master_path = os.path.join(DATA_DIR, "processed_filings.xlsx")
    suffixed_path = paths["PROCESSED_FILINGS_PATH"]
    
    if os.path.exists(master_path):
        if not os.path.exists(suffixed_path) or os.path.getsize(master_path) > os.path.getsize(suffixed_path) * 2:
            # If master exists and is much bigger/only one, use it
            paths["PROCESSED_FILINGS_PATH"] = master_path
            
    return paths
# --- Default Paths for Compatibility ---
# These constants are exported for modules that expect a single source of truth 
# without calling get_data_paths() (like app.py and dashboard components).
_default_paths = get_data_paths("finbert")
PROCESSED_FILINGS_PATH = _default_paths["PROCESSED_FILINGS_PATH"]
FEATURES_PATH = _default_paths["FEATURES_PATH"]
PREDICTIONS_PATH = _default_paths["PREDICTIONS_PATH"]
BACKTEST_RESULTS_PATH = _default_paths["BACKTEST_RESULTS_PATH"]

# --- Processor Settings ---
# Regex pattern for MD&A extraction (Item 2 to Item 3/4 or Part II)
MDNA_REGEX_PATTERN = r'(Item\s+2[.:]?\s+.*?(?:Management.*?Discussion.*?Analysis).*?)(?=Item\s+3[.:]?|Item\s+4[.:]?|PART\s+II|Signatures)'
MDNA_MIN_LENGTH = 1000 # Minimum characters for MD&A to be considered valid

# --- Model Settings (XGBoost) ---
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_RANDOM_STATE = 42

# --- Backtester Settings ---
INITIAL_CAPITAL = 100.0 # Starting capital for portfolio simulation

