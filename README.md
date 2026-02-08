# Stock Analysis Pipeline

This project converts a Jupyter Notebook research prototype into a modular, production-ready Python backend for analyzing stock market data and predicting quarterly returns. It integrates SEC Edgar filings, OpenAI for sentiment analysis, `yfinance` for market data, and a Random Forest model with walk-forward validation for backtesting a trading strategy.

## Goal of the System
The system aims to:
1.  Fetch 10-Q filings for a specific list of tickers.
2.  Extract the "Management's Discussion and Analysis" (MD&A) section.
3.  Analyze the sentiment of that text using OpenAI's `gpt-4o-mini` API.
4.  Fetch market data (stock prices) to calculate quarterly returns.
5.  Train a Random Forest model using a Walk-Forward validation approach (with data leakage prevention).
6.  Output a portfolio simulation and trade log.

## Architecture & Folder Structure

```
project_root/
│
├── data/
│   └── fetched/          # Stores all intermediate and final XLSX files
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
└── README.md             # This file
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd stock-analysis
    ```

2.  **Set up Python Environment (using `uv`):**
    This project uses `uv` for dependency management. If you don't have `uv` installed, you can install it via pip: `pip install uv`.
    ```bash
    uv venv
    uv pip install -r requirements.txt
    ```
    Alternatively, if you prefer `pip`:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    Create a `.env` file in the `project_root/` directory (where `main.py` is located) and add your OpenAI API key:
    ```
    API_KEY_OPENAI=your_openai_api_key_here
    ```
    **Important**: Remember to replace `your_openai_api_key_here` with your actual OpenAI API key.

4.  **Configure EDGAR Identity:**
    Open `src/config.py` and update the `EDGAR_IDENTITY` variable with your name and email address. This is required by SEC Edgar's `edgar-tools`.
    ```python
    EDGAR_IDENTITY = "Your Name Your.Email@example.com"
    ```

## Caching and Idempotency

This pipeline is designed to be efficient and cost-effective by avoiding redundant work. It implements intelligent caching at its most expensive steps:

*   **Data Fetching**: When you run the `--fetch` step, the pipeline checks for existing data in `raw_filings.xlsx`. It only fetches new filings that are not already in your local cache, saving significant time on subsequent runs.
*   **Sentiment Analysis**: The `--process` step checks for `processed_filings.xlsx`. It identifies which filings have already been analyzed and only sends new, unprocessed filings to the OpenAI API. This saves both time and API costs.
*   **Blacklisting**: Tickers that are dropped during the feature engineering step due to inconsistent or missing data are automatically added to `blacklist.txt`. These tickers are ignored in all future pipeline runs, preventing repeated processing of bad data.

## Usage

The `main.py` script provides a command-line interface to run different parts of the pipeline. The steps are designed to be run in order, as each step depends on the output of the previous one.

```bash
python main.py --help
```

### Pipeline Steps

1.  **`--fetch`**: Fetches new 10-Q filings from EDGAR, ignoring blacklisted tickers and skipping filings already in `data/fetched/raw_filings.xlsx`.
2.  **`--process`**: Analyzes sentiment for new filings. Requires `raw_filings.xlsx`. It uses `processed_filings.xlsx` as a cache and only processes new filings.
3.  **`--engineer`**: Engineers features for all processed filings. Requires `processed_filings.xlsx`. Tickers with bad data are blacklisted.
4.  **`--train`**: Trains the model using walk-forward validation. Requires `features.xlsx`. Saves model output to `predictions.xlsx`.
5.  **`--backtest`**: Simulates the portfolio strategy. Requires `predictions.xlsx`.
6.  **`--all`**: Runs the entire pipeline in sequence.

### Examples

1.  **Run the entire pipeline for specific tickers:**
    ```bash
    python main.py --all --tickers AAPL,GOOGL
    ```

2.  **Run only the data fetching step:**
    ```bash
    python main.py --fetch
    ```
    *   If you run this again, it will only fetch filings that are new since the last run.

3.  **Run the sentiment analysis step:**
    ```bash
    python main.py --process
    ```
    *   This will only process filings that have not been analyzed before.

## Output Files

All generated data and results are stored in the `data/fetched/` directory:

*   `blacklist.txt`: A simple text file listing tickers that are automatically ignored by the pipeline due to data quality issues.
*   `raw_filings.xlsx`: Contains fetched 10-Q filing metadata and the raw MD&A text. This is the main cache for the `--fetch` step.
*   `processed_filings.xlsx`: Contains all data from `raw_filings.xlsx` plus the sentiment score and justification from OpenAI. This is the cache for the `--process` step.
*   `features.xlsx`: Contains the fully engineered features used for model training.
*   `predictions.xlsx`: Contains the output from the model's walk-forward validation, including the actual return and the predicted return.
*   `backtest_results.xlsx`: Details the portfolio simulation, including trade logs, portfolio value over time, and final performance.

## Customization

*   **Model Parameters**: Experiment with `MODEL_N_ESTIMATORS` and `MODEL_MAX_DEPTH` in `src/config.py` for the Random Forest model.
*   **Backtesting Strategy**: Adjust `INITIAL_CAPITAL` and `TOP_N_STOCKS_TO_INVEST` in `src/config.py` to test different portfolio sizes and investment strategies.
*   **OpenAI Model**: You can change the `model` parameter in `src/processor.py` for `get_sentiment_openai` if you wish to use a different OpenAI model.
