# Project Roadmap: Jupyter Notebook to Production Backend

This document outlines the step-by-step plan to transform the existing Jupyter Notebook (`research/notebooks/quarterly_prediction.ipynb`) into a modular, production-ready Python backend system.

## Goal of the System
The primary goal is to predict stock prices and simulate a trading strategy based on financial filings and market data, using a robust, modular architecture.

The goal of the system is to:
1. Fetch 10-Q filings for a specific list of tickers. (Configurable via CLI arguments).
2. Extract the "Management's Discussion and Analysis" (MD&A) section.
3. Analyze the sentiment of that text using OpenAI's API.
4. Fetch market data (stock prices) to calculate quarterly returns.
5. Train a Random Forest model using a Walk-Forward validation approach (purging overlapping trades to prevent data leakage).
6. Output a portfolio simulation and trade log.

## Architecture & Folder Structure

```
project_root/
│
├── data/
│   └── fetched/          # Store all intermediate and final XLSX files here
│
├── src/
│   ├── config.py         # API keys, Ticker List, Date settings
│   ├── data_loader.py    # SEC EDGAR fetching + yfinance logic
│   ├── processor.py      # Regex extraction + Sentiment Analysis
│   ├── feature_eng.py    # Merging data, calculating growth/margins/returns
│   ├── model.py          # Random Forest Walk-Forward Training logic
│   └── backtester.py     # Portfolio simulation logic
│
├── main.py               # Entry point to run the full pipeline
├── requirements.txt      # All dependencies
└── README.md             # Instructions
```

## Detailed Implementation Plan

The pipeline is designed with intelligent caching and blacklisting to ensure efficiency and robustness.

### 1. `data/` Directory
- Create the `data/` directory and its `fetched/` subdirectory.
- This directory will now store:
    - `blacklist.txt`: List of tickers to ignore due to data quality issues.
    - `raw_filings.xlsx`: Cache for raw fetched SEC filing data.
    - `processed_filings.xlsx`: Cache for filings with sentiment analysis scores.
    - `features.xlsx`: Final engineered features for model training.
    - `predictions.xlsx`: Output from the model's walk-forward validation.
    - `backtest_results.xlsx`: Final results of the portfolio simulation.

### 2. `src/blacklist.py` (New)
- Manages the `blacklist.txt` file.
- **Functions**: `get_blacklist`, `add_to_blacklist`.

### 3. `src/data_loader.py` (Enhanced)
- Handles fetching raw data with intelligent caching.
- **Logic**:
    - Ignores any tickers present in `blacklist.txt`.
    - Checks `raw_filings.xlsx` to see what filings are already cached.
    - Only fetches new filings from EDGAR, minimizing redundant downloads.
    - Merges new data with the cache.

### 4. `src/processor.py` (Enhanced)
- Handles sentiment analysis with intelligent caching.
- **Logic**:
    - Takes raw filings as input.
    - Checks `processed_filings.xlsx` to see what filings have already been analyzed.
    - Only sends new, un-analyzed filings to the OpenAI API, saving time and cost.
    - Merges new results with the cache.

### 5. `src/feature_eng.py` (Enhanced)
- Calculates features and now includes automatic blacklisting.
- **Logic**:
    - After calculating features, it identifies any tickers that are dropped due to missing or inconsistent data (resulting in NaN values).
    - Automatically adds these problematic tickers to `blacklist.txt` to exclude them from future runs.

### 6. `src/model.py` (Enhanced)
- Implements the Walk-Forward training loop.
- **Output**: Now saves its predictions to a dedicated `predictions.xlsx` file, which serves as the direct input for the backtester.

### 7. `src/backtester.py`
- Simulates the trading strategy.
- **Input**: Reads its required data directly from `predictions.xlsx`.

### 8. `main.py` (Enhanced)
- Orchestrates the new modular, file-based pipeline.
- **Logic**:
    - Each step (`--fetch`, `--process`, etc.) is now self-contained.
    - The script checks for the required input file at the beginning of each step and provides a clear error if it's missing, ensuring a robust and predictable workflow.

### 9. `requirements.txt` / `README.md`
- These files are updated to reflect the new modules and improved workflow.

## Technologies
- Python 3.10+
- `pandas`, `numpy`
- `scikit-learn` (for `RandomForestRegressor`)
- `edgar-tools` (for SEC filings)
- `yfinance` (for market data)
- `openai` (for sentiment analysis)
- `python-dotenv` (for environment variables)
- `openpyxl` (for Excel export)
- `tqdm` (for progress bars)

---
This roadmap provides a comprehensive guide for the development. I will now proceed to implement these steps.
