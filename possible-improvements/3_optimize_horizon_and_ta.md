# Roadmap: Optimize Forecasting Horizon & Integrate Technical Analysis

## 🎯 Objective
Shorten the prediction window from 90 days to a 1-day or 5-day horizon to capture transient alpha before it decays. Integrate Short-Term Momentum algorithms (RSI, MACD) and Intraday Volatility into the feature set.

## ⚠️ Critical Coding Guidelines (MANDATORY)
*   **KISS (Keep It Simple, Stupid):** No over-engineering. Write bare-bones implementations.
*   **Human-Readable & Short:** Keep the code as concise as absolutely possible.
*   **NO Error Blocks / NO Try Blocks:** The `try...except` pattern is strictly forbidden. Assume data is correct or let the program fail loudly.

## 📂 Architecture
*   **Backend (`src/`):** Update `src/data_loader.py` and `src/feature_eng.py`.
*   **Frontend (`pages/`):** Update `pages/2_Strategy_Lab.py` and `pages/1_Data_Pipeline.py`.

## 🚀 Implementation Steps

1.  **Shorten Forecast Horizon:**
    *   In `src/model.py` and `src/backtester.py`, update the target prediction variable (`next_quarter_return`).
    *   Define a new function `calculate_next_week_return` (5-day or 21-day holding period) or `calculate_next_day_return` (1-day).
    *   Ensure the forward-looking logic correctly calculates the return at the new horizon relative to the `filing_date`.
2.  **Integrate YFinance Daily Data:**
    *   In `src/data_loader.py`, fetch the daily stock history (Open, High, Low, Close, Volume) covering the same time periods as your 10-Q filings (with padding for the forward return calculation).
3.  **Engineer Technical Features:**
    *   In `src/feature_eng.py`, write bare-bones functions to calculate the Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Intraday Volatility (`High - Low` / `Open`).
    *   Join these pre-calculated technical features with the fundamental and sentiment data before model training.
4.  **Update UI Labels & Charts:**
    *   Modify `pages/2_Strategy_Lab.py` and `pages/1_Data_Pipeline.py` to reflect the newly calculated technical indicators and the new short-term return predictions.
