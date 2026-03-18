# Page 4: Strategic Recommendations

## Overview
This page provides the "Live" output of the system. While the Strategy Lab focuses on historical backtesting, the Recommendations page generates forecasts for the most recent filings that do not yet have known market outcomes.

## Forecast Engine

### 1. Ticker-Specific Modeling
- **Logic**: For each stock, the system trains a dedicated `XGBRegressor` on that specific ticker's entire historical sequence of sentiment and financials.
- **Goal**: To understand how the market typically reacts to *this specific company's* tone and growth metrics.
- **Execution**: Triggered by the "Generate Fresh Forecasts" button.

### 2. Live Data Integration
- **Prices**: Fetches current stock prices via `yfinance` to display real-time 6-month charts for recommended buys.
- **Predictions**: Uses the latest available record from the feature set (where the target return is NaN) to generate a forward-looking prediction.

## Interactive Portfolio Manager

### 1. Virtual Portfolio Tracking
- **State Management**: Uses `st.session_state` to track a persistent virtual portfolio within the browser session.
- **Initial Capital**: Defaults to the setting in `src/config.py` (typically $100.00).

### 2. AI-Driven Rebalancing
- **Method**: The "Apply AI Rebalance" button automatically distributes the entire portfolio value across all "Buy" recommendations (predicted return > 0).
- **Weighting**: Uses the model's predicted return to determine the allocation% for each stock.

## Data Visualisation
- **Recommendations**: Sorted tables highlighting "Top Conviction Picks".
- **Charts**: 
    - **Historical Price**: High-level 6-month close price trends.
    - **Allocation Pie Chart**: Built with `Altair`, visualizing current portfolio weights.

## Data Storage
- **Output**: `data/fetched/latest_predictions_finbert.xlsx` (or `_alpha` variant).
- **Format**: Contains tickers and their predicted return for the next period (90 days or 5 days).

## Technology & Libraries
- **Libraries**: `streamlit`, `altair`, `yfinance`, `pandas`.
- **Method**: Direct model inference using the latest trained weights.
