# Page 2: Strategy Lab

## Overview
The Strategy Lab is an interactive research environment for training machine learning models and simulating portfolio performance. It converts engineered features into actionable trade logs and equity curves.

## Machine Learning Methodology

### 1. Robust Walk-Forward Validation
Unlike standard K-fold cross-validation, the system uses a **Walk-Forward** approach to mimic real-world trading:
1. The model is trained on a historical window (e.g., all data before 2021).
2. It makes predictions for the next available quarter.
3. The "trading date" moves forward by one quarter, and the process repeats.
This ensures the model never "sees" the future during training.

### 2. Data Purging
To handle the "overlap" issue (where a 90-day return period starting in December overlaps with training data from January), the system implements **Purging**:
- Any training sample whose "outcome window" intersects with the test date is removed from the training set.
- This prevents information leakage and ensures institutional-grade validation.

### 3. Hyperparameter Optimization
- **Library**: `Optuna`.
- **Method**: Bayesian optimization using a Tree-structured Parzen Estimator (TPE).
- **Process**: For every quarter in the walk-forward, the system can optionally run 20+ trials to find the optimal `max_depth`, `learning_rate`, and `n_estimators` for the `XGBRegressor`.

## Backtesting Logic

### 1. Portfolio Simulation
- **Frequency**: Customizable (Quarterly for Fundamentals, Daily/Interval for Alpha).
- **Stock Selection**: Selects the `Top N` stocks with the highest positive predicted returns.
- **Weighting**: Positions are weighted proportionally to their predicted conviction (return).
- **Rebalancing**:
    - **Fundamental**: Rebalances at the start of every fiscal quarter.
    - **Alpha**: Rebalances every X days (e.g., 5 days) using the latest available text sentiment and technical indicators.

### 2. Performance Metrics
- **Equity Curve**: Visual comparison against the S&P 500 (`^GSPC`) benchmark.
- **Sharpe Ratio**: Annualized return divided by annualized volatility.
- **Max Drawdown**: The largest peak-to-trough decline in portfolio value.
- **Purged CV MSE**: Mean Squared Error calculated via the purged cross-validation folds.

## Data Storage
- **Predictions**: `data/fetched/predictions_finbert.xlsx` (contains actual vs. predicted returns).
- **Backtest Results**: `data/fetched/backtest_results_finbert.xlsx` (contains the periodic portfolio value and trade log).
- **Metrics**: `data/fetched/metrics_finbert.xlsx` (performance statistics from model training).

## Technology & Libraries
- **Libraries**: `xgboost`, `optuna`, `numpy`, `plotly` (via Streamlit `line_chart`).
- **Models**: `XGBRegressor` (Gradient Boosted Trees).
