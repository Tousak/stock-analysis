# Roadmap: Streamlit Application

This document outlines the plan to build a 4-page Streamlit web application on top of the existing data pipeline.

**Core Principle:** The code will be kept as simple and readable as possible, with error handling omitted during this development phase to ensure full transparency and easier debugging, as requested.

## File Structure

The application will use Streamlit's multi-page file structure:
```
project_root/
├── streamlit_app.py      # The "Home" page (renamed to avoid conflict with main.py)
├── pages/
│   ├── 1_Data_Pipeline.py
│   ├── 2_Strategy_Lab.py
│   └── 3_Filing_Inspector.py
└── src/
    └── ... (existing backend code)
```
*(Note: I will use `streamlit_app.py` as the main entry point to avoid confusion with the pipeline's `main.py`)*

---

## Page-by-Page Implementation Plan

### 1. 🏠 Home (`streamlit_app.py`)

**Purpose:** A high-level executive summary of the latest backtest run.

**Implementation Steps:**
1.  **Title:** Set the page title to "🏠 Home Dashboard".
2.  **Load Data:** Load the most recent `backtest_results.xlsx` and `predictions.xlsx`.
3.  **Display KPIs:**
    *   Use `st.metric` to display the "Final Portfolio Value" from the last row of the backtest results.
    *   Calculate and display the "Total Return (%)".
4.  **Show Latest Picks:**
    *   Find the most recent quarter in the backtest results.
    *   Display the "Selection" for that quarter in a subheader.
5.  **Display System Status:**
    *   Check the modification times of key files (`raw_filings.xlsx`, `processed_filings.xlsx`, `features.xlsx`).
    *   Display these timestamps using `st.text` to show how fresh the data is.

### 2. ⬇️ Data Pipeline (`pages/1_Data_Pipeline.py`)

**Purpose:** An "engine room" to execute the slow and expensive data gathering and processing steps.

**Implementation Steps:**
1.  **Title:** Set the page title to "⬇️ Data Pipeline".
2.  **Ticker Input:** Use `st.text_area` to allow the user to modify the list of tickers.
3.  **Action Buttons:** Create separate `st.button`s for each pipeline step:
    *   "Run Step 1: Fetch Filings"
    *   "Run Step 2: Run Sentiment Analysis"
    *   "Run Step 3: Engineer Features"
4.  **Execution Logic:**
    *   When a button is pressed, call the corresponding backend function (e.g., `run_fetch_step`, `run_process_step`).
    *   Use `st.spinner` to show that a process is running.
    *   Use `st.info`, `st.success`, and `st.code` to display the log output from the backend functions in real-time.

### 3. 🧪 Strategy Lab (`pages/2_Strategy_Lab.py`)

**Purpose:** An interactive and fast backtesting playground that reads pre-calculated feature files.

**Implementation Steps:**
1.  **Title:** Set the page title to "🧪 Strategy Lab".
2.  **Inputs in Sidebar:** Use `st.sidebar` to create interactive widgets:
    *   `st.number_input` for "Initial Capital".
    *   `st.slider` for "Top N Stocks" to invest in.
    *   `st.number_input` for "Backtest Start Year".
3.  **Run Simulation Button:** A main button "Run New Simulation" will trigger the backtest.
4.  **Execution Logic:**
    *   The function will load the `features.xlsx` file.
    *   It will call re-written `train` and `backtest` functions that accept the UI parameters. **It will NOT re-fetch or re-process data**.
5.  **Outputs:**
    *   **Equity Curve:** Use `st.line_chart` to plot the `portfolio_value` column from the results. I will also fetch S&P 500 data using `yfinance` to plot as a benchmark comparison.
    *   **Metrics:** Use `st.metric` in columns to display key performance indicators: Total Return, Sharpe Ratio, Max Drawdown.
    *   **Trade Log:** Display the full `backtest_results.xlsx` DataFrame using `st.dataframe` to show the quarterly decisions.

### 4. 🔍 Filing Inspector (`pages/3_Filing_Inspector.py`)

**Purpose:** A tool to "trust but verify" the data and sentiment for any specific filing.

**Implementation Steps:**
1.  **Title:** Set the page title to "🔍 Filing Inspector".
2.  **Load Data:** Load both `processed_filings.xlsx` (for MD&A text) and `features.xlsx` (for calculated features).
3.  **Selectors:**
    *   Create a `st.selectbox` for the user to choose a `ticker`.
    *   Create another `st.selectbox` for the user to choose a `filing_date` for that ticker.
4.  **Display Data:**
    *   Once a filing is selected, use `st.metric` in columns to display the key data points: `Sentiment Score`, `Sentiment Change`, `Revenue Growth`, and `Next Quarter Return`.
    *   Use `st.expander` to show the full, raw `mda_text` for that filing, allowing for direct inspection.
